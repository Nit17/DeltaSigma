"""Transaction costs, terminal hedging P&L and tail-risk metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from deltasigma.hedging.market import INSTRUMENTS, HedgingMarket

PIP = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Proportional half-spread costs.

    ``forward_bps`` is charged on forward notional ``F * |trade|``.
    ``varswap_vol_pts`` is a half-spread in vol points on the *remaining* variance,
    i.e. ``2 * sigma * tau * (vol_pts / 100) * |trade|``.
    """

    forward_bps: float = 2.0
    varswap_vol_pts: float = 0.2


def cost_rates(
    market: HedgingMarket, costs: CostModel, instruments: Sequence[str] = INSTRUMENTS
) -> np.ndarray:
    """Cost per unit traded at each rebalancing date, shape (P, N, n_instruments)."""
    cols = {
        "forward": costs.forward_bps * 1e-4 * market.forward[:, :-1],
        "variance_swap": costs.varswap_vol_pts
        * 1e-2
        * 2.0
        * market.implied_vol
        * market.tau[None, :],
    }
    return np.stack([cols[i] for i in instruments], axis=-1)


@dataclass(frozen=True)
class HedgeOutcome:
    loss: np.ndarray      # payoff - hedge gains + costs (what the desk needs to have charged)
    gains: np.ndarray
    costs: np.ndarray
    turnover: np.ndarray  # (P, n_instruments) total absolute units traded


def hedge_outcome(
    market: HedgingMarket,
    positions: np.ndarray,
    costs: CostModel,
    instruments: Sequence[str] = INSTRUMENTS,
) -> HedgeOutcome:
    """Terminal P&L of holding ``positions[:, k]`` over ``[t_k, t_{k+1})``.

    Trading costs apply to the initial trade and every rebalance; instruments
    expire with the option, so there is no unwind cost at ``T``.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (market.n_paths, market.n_steps, len(instruments)):
        raise ValueError(f"positions shape {positions.shape} does not match market/instruments")
    prices = market.prices(instruments)
    gains = np.sum(positions * np.diff(prices, axis=1), axis=(1, 2))
    trades = np.diff(positions, axis=1, prepend=0.0)
    cost = np.sum(cost_rates(market, costs, instruments) * np.abs(trades), axis=(1, 2))
    return HedgeOutcome(
        loss=market.payoff - gains + cost,
        gains=gains,
        costs=cost,
        turnover=np.abs(trades).sum(axis=1),
    )


def cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """Expected shortfall: mean of the worst ``1 - alpha`` fraction of losses."""
    x = np.sort(np.asarray(losses, dtype=float))
    n_tail = max(1, int(np.ceil(round((1.0 - alpha) * x.size, 9))))
    return float(x[-n_tail:].mean())


def summarize(outcome: HedgeOutcome, fair_price: float, scale: float = PIP) -> dict[str, float]:
    """Risk metrics in units of ``scale`` (default: pips per unit foreign notional).

    ``cvar95_price`` is the premium a CVaR(95%)-constrained desk must charge;
    ``risk_premium95`` is how far that sits above the risk-neutral fair value,
    i.e. the bid-offer the hedging strategy forces the market-maker to quote.
    """
    loss = outcome.loss
    return {
        "mean": float(loss.mean()) / scale,
        "std": float(loss.std()) / scale,
        "cvar95_price": cvar(loss, 0.95) / scale,
        "cvar99_price": cvar(loss, 0.99) / scale,
        "risk_premium95": (cvar(loss, 0.95) - fair_price) / scale,
        "cost": float(outcome.costs.mean()) / scale,
        "turnover": float(outcome.turnover[:, 0].mean()),
    }
