"""Hedging market: simulated FX paths plus the tradable hedge instruments.

Instruments (all settle at the option expiry ``T``, so P&L is measured in
time-``T`` domestic currency and no discounting is needed):

* **FX forward** to ``T``: ``F_t = S_t exp((rd - rf)(T - t))``.
* **Variance swap** on the model's integrated variance ``dt * sum v_k``. Its
  value process ``G_k = accrued_k + E_k[remaining]`` is computed *exactly*
  under the discrete regime-switching Heston simulator via a linear recursion
  on ``(E[v 1{regime=i}], P(regime=i))``, so ``G`` is a true martingale and a
  learning agent cannot exploit pricing errors as fake arbitrage.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from deltasigma.pricing import garman_kohlhagen as gk
from deltasigma.sim.regime import RegimeSwitchingHeston

INSTRUMENTS = ("forward", "variance_swap")


@dataclass(frozen=True)
class VanillaOption:
    strike: float
    call: bool = True
    quantity: float = 1.0


@dataclass(frozen=True)
class OptionBook:
    """Options the desk is **short** (owes the payoff at expiry)."""

    options: tuple[VanillaOption, ...]

    @classmethod
    def of(cls, options: Sequence[VanillaOption]) -> OptionBook:
        return cls(tuple(options))

    def payoff(self, F_T: np.ndarray) -> np.ndarray:
        out = np.zeros_like(F_T, dtype=float)
        for o in self.options:
            phi = 1.0 if o.call else -1.0
            out += o.quantity * np.maximum(phi * (F_T - o.strike), 0.0)
        return out

    def black_greeks(
        self, F: np.ndarray, tau: np.ndarray, sigma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Undiscounted Black (value, delta, gamma, vega) of the book w.r.t. the forward."""
        value = np.zeros(np.broadcast_shapes(np.shape(F), np.shape(tau), np.shape(sigma)))
        delta, gamma, vega = (np.zeros_like(value) for _ in range(3))
        for o in self.options:
            g = gk.greeks(F, o.strike, tau, 0.0, 0.0, sigma, o.call)
            value += o.quantity * g.price
            delta += o.quantity * g.spot_delta
            gamma += o.quantity * g.gamma
            vega += o.quantity * g.vega
        return value, delta, gamma, vega


def expected_variance_operator(model: RegimeSwitchingHeston, dt: float) -> np.ndarray:
    """Matrix ``M`` with ``x_{k+1} = M x_k`` for ``x = [E[v 1{r=i}], P(r=i)]``.

    Exact for the simulator: the QE step preserves the conditional mean
    ``theta_i + (v - theta_i) exp(-kappa_i dt)``, then the regime jumps with
    transition matrix ``P = expm(Q dt)``.
    """
    n = len(model.regimes)
    kappa = np.array([r.kappa for r in model.regimes])
    theta = np.array([r.theta for r in model.regimes])
    P = model.transition_matrix(dt)
    e = np.exp(-kappa * dt)
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = P.T @ np.diag(e)
    M[:n, n:] = P.T @ np.diag(theta * (1.0 - e))
    M[n:, n:] = P.T
    return M


def remaining_variance_weights(
    model: RegimeSwitchingHeston, dt: float, n_steps: int
) -> np.ndarray:
    """Row ``j`` holds ``dt * sum_{l<j} u^T M^l``: dotted with ``x_k`` it gives the
    expected variance still to accrue over the next ``j`` steps.
    """
    n = len(model.regimes)
    M = expected_variance_operator(model, dt)
    W = np.zeros((n_steps + 1, 2 * n))
    row = np.concatenate([np.ones(n), np.zeros(n)])
    for j in range(1, n_steps + 1):
        W[j] = W[j - 1] + dt * row
        row = row @ M
    return W


@dataclass(frozen=True)
class HedgingMarket:
    times: np.ndarray            # (N+1,)
    forward: np.ndarray          # (P, N+1)
    variance: np.ndarray         # (P, N+1)
    regime: np.ndarray           # (P, N+1)
    variance_swap: np.ndarray    # (P, N+1) value process G_k (total variance units)
    implied_vol: np.ndarray      # (P, N)   fair vol of the remaining variance, k < N
    payoff: np.ndarray           # (P,)
    book: OptionBook

    @property
    def maturity(self) -> float:
        return float(self.times[-1])

    @property
    def n_paths(self) -> int:
        return self.forward.shape[0]

    @property
    def n_steps(self) -> int:
        return self.forward.shape[1] - 1

    @property
    def tau(self) -> np.ndarray:
        """Time to expiry at each rebalancing date, shape (N,)."""
        return self.maturity - self.times[:-1]

    def prices(self, instruments: Sequence[str] = INSTRUMENTS) -> np.ndarray:
        cols = {"forward": self.forward, "variance_swap": self.variance_swap}
        return np.stack([cols[i] for i in instruments], axis=-1)

    def book_greeks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.book.black_greeks(self.forward[:, :-1], self.tau, self.implied_vol)

    @property
    def fair_price(self) -> float:
        """Risk-neutral (Monte Carlo) value of the book, paid at ``T``."""
        return float(self.payoff.mean())

    def visited_stress(self, stress_regime: int = 1) -> np.ndarray:
        return (self.regime == stress_regime).any(axis=1)


def generate_market(
    model: RegimeSwitchingHeston,
    book: OptionBook,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    start_regime: int = 0,
    v0: float | None = None,
    seed: int | None = None,
) -> HedgingMarket:
    sim = model.simulate(S0, T, n_steps, n_paths, start_regime=start_regime, v0=v0, seed=seed)
    dt = T / n_steps
    tau_all = T - sim.times
    forward = sim.spot * np.exp((model.rd - model.rf) * tau_all)

    n = len(model.regimes)
    W = remaining_variance_weights(model, dt, n_steps)
    steps_left = n_steps - np.arange(n_steps + 1)
    w_v = W[steps_left[None, :], sim.regime]
    w_p = W[steps_left[None, :], n + sim.regime]
    remaining = w_v * sim.variance + w_p
    accrued = np.concatenate(
        [np.zeros((n_paths, 1)), dt * np.cumsum(sim.variance[:, :-1], axis=1)], axis=1
    )
    implied_vol = np.sqrt(np.maximum(remaining[:, :-1], 1e-16) / tau_all[:-1])

    return HedgingMarket(
        times=sim.times,
        forward=forward,
        variance=sim.variance,
        regime=sim.regime,
        variance_swap=accrued + remaining,
        implied_vol=implied_vol,
        payoff=book.payoff(forward[:, -1]),
        book=book,
    )
