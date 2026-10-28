"""Regime-switching Heston market simulator.

Volatility dynamics switch between regimes (e.g. *calm* and *stress*) driven by
a continuous-time Markov chain. This gives the non-stationary, regime-changing
environment needed to stress-test ML pricing, hedging and quoting policies
before they meet real data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from deltasigma.sim.heston import HestonParams, log_spot_step, qe_step


@dataclass(frozen=True)
class SimulationResult:
    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray
    regime: np.ndarray
    regime_names: tuple[str, ...]

    @property
    def realized_vol(self) -> np.ndarray:
        """Annualised realised vol of each path."""
        log_ret = np.diff(np.log(self.spot), axis=1)
        return np.sqrt(np.sum(log_ret**2, axis=1) / self.times[-1])


class RegimeSwitchingHeston:
    def __init__(
        self,
        regimes: Sequence[HestonParams],
        generator: np.ndarray,
        rd: float,
        rf: float,
        regime_names: Sequence[str] | None = None,
    ) -> None:
        Q = np.asarray(generator, dtype=float)
        n = len(regimes)
        if Q.shape != (n, n):
            raise ValueError("generator must be n_regimes x n_regimes")
        if np.any(Q - np.diag(np.diag(Q)) < 0) or not np.allclose(Q.sum(axis=1), 0.0):
            raise ValueError("generator needs non-negative off-diagonals and zero row sums")
        self.regimes, self.generator, self.rd, self.rf = tuple(regimes), Q, rd, rf
        self.regime_names = tuple(regime_names or (f"regime_{i}" for i in range(n)))
        self._params = {
            name: np.array([getattr(r, name) for r in self.regimes])
            for name in ("kappa", "theta", "xi", "rho")
        }

    def transition_matrix(self, dt: float) -> np.ndarray:
        return expm(self.generator * dt)

    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        start_regime: int = 0,
        v0: float | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        cum_p = np.cumsum(self.transition_matrix(dt), axis=1)
        cum_p[:, -1] = 1.0
        mu = self.rd - self.rf

        log_s = np.empty((n_paths, n_steps + 1))
        var = np.empty_like(log_s)
        reg = np.empty((n_paths, n_steps + 1), dtype=np.int8)
        log_s[:, 0] = np.log(S0)
        var[:, 0] = self.regimes[start_regime].v0 if v0 is None else v0
        reg[:, 0] = start_regime

        for t in range(n_steps):
            r = reg[:, t]
            kappa, theta, xi, rho = (self._params[k][r] for k in ("kappa", "theta", "xi", "rho"))
            z_v, z_s = rng.standard_normal((2, n_paths))
            u_v, u_r = rng.random((2, n_paths))
            var[:, t + 1] = qe_step(var[:, t], kappa, theta, xi, dt, z_v, u_v)
            log_s[:, t + 1] = log_spot_step(
                log_s[:, t], var[:, t], var[:, t + 1], mu, kappa, theta, xi, rho, dt, z_s
            )
            reg[:, t + 1] = (u_r[:, None] > cum_p[r]).sum(axis=1)

        return SimulationResult(
            times=np.linspace(0.0, T, n_steps + 1),
            spot=np.exp(log_s),
            variance=var,
            regime=reg,
            regime_names=self.regime_names,
        )


def simulate_heston(
    S0: float,
    params: HestonParams,
    T: float,
    n_steps: int,
    n_paths: int,
    rd: float,
    rf: float,
    seed: int | None = None,
) -> SimulationResult:
    """Single-regime Heston paths."""
    model = RegimeSwitchingHeston([params], np.zeros((1, 1)), rd, rf, ("heston",))
    return model.simulate(S0, T, n_steps, n_paths, seed=seed)


def eurusd_calm_stress_model(rd: float = 0.043, rf: float = 0.025) -> RegimeSwitchingHeston:
    """Two-regime EURUSD-like market: ~7% vol calm regime, ~14% vol stress regime.

    Stress arrives roughly every two years and lasts about four months on average.
    """
    calm = HestonParams(kappa=4.0, theta=0.07**2, xi=0.15, rho=-0.10, v0=0.07**2)
    stress = HestonParams(kappa=2.0, theta=0.14**2, xi=0.30, rho=-0.40, v0=0.14**2)
    generator = np.array([[-0.5, 0.5], [3.0, -3.0]])
    return RegimeSwitchingHeston([calm, stress], generator, rd, rf, ("calm", "stress"))
