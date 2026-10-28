"""Classical hedging benchmarks the deep hedger has to beat.

All strategies return positions of shape ``(P, N, 2)`` in
``(forward, variance_swap)`` units, held over ``[t_k, t_{k+1})``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from deltasigma.hedging.market import HedgingMarket
from deltasigma.hedging.pnl import CostModel, cvar, hedge_outcome


def _zeros(market: HedgingMarket) -> np.ndarray:
    return np.zeros((market.n_paths, market.n_steps, 2))


def no_hedge(market: HedgingMarket) -> np.ndarray:
    return _zeros(market)


def black_delta(market: HedgingMarket, vol: str = "live") -> np.ndarray:
    """Black delta hedge in forwards with ``live`` implied vol or the ``inception`` vol."""
    if vol not in ("live", "inception"):
        raise ValueError("vol must be 'live' or 'inception'")
    sigma = market.implied_vol
    if vol == "inception":
        sigma = np.broadcast_to(sigma[:, :1], sigma.shape)
    _, delta, _, _ = market.book.black_greeks(market.forward[:, :-1], market.tau, sigma)
    h = _zeros(market)
    h[..., 0] = delta
    return h


def vega_hedge_ratio(market: HedgingMarket) -> np.ndarray:
    """Variance-swap units neutralising the book's sensitivity to the variance state.

    With ``sigma^2 tau = R`` (expected remaining variance) and ``dG/dx = dR/dx``
    for the swap, matching ``dC/dx = vega * dR/dx / (2 sigma tau)`` gives
    ``h = vega / (2 sigma tau)``.
    """
    _, _, _, vega = market.book_greeks()
    return vega / (2.0 * market.implied_vol * market.tau[None, :])


def delta_vega(market: HedgingMarket) -> np.ndarray:
    h = black_delta(market)
    h[..., 1] = vega_hedge_ratio(market)
    return h


def _band_positions(delta: np.ndarray, half_width: np.ndarray) -> np.ndarray:
    h = np.zeros((*delta.shape, 2))
    pos = np.zeros(delta.shape[0])
    for k in range(delta.shape[1]):
        pos = np.clip(pos, delta[:, k] - half_width[:, k], delta[:, k] + half_width[:, k])
        h[:, k, 0] = pos
    return h


def _ww_inputs(market: HedgingMarket, costs: CostModel) -> tuple[np.ndarray, np.ndarray]:
    _, delta, gamma, _ = market.book_greeks()
    return delta, 1.5 * costs.forward_bps * 1e-4 * market.forward[:, :-1] * gamma**2


def whalley_wilmott(market: HedgingMarket, costs: CostModel, risk_aversion: float) -> np.ndarray:
    """Asymptotically optimal no-trade band around the live Black delta
    (Whalley & Wilmott, 1997): ``H = (3 c F Gamma^2 / (2 gamma))^(1/3)``.
    """
    delta, band_scale = _ww_inputs(market, costs)
    return _band_positions(delta, np.cbrt(band_scale / risk_aversion))


def tune_whalley_wilmott(
    market: HedgingMarket,
    costs: CostModel,
    grid: Sequence[float] = tuple(np.geomspace(1.0, 1e5, 21)),
    alpha: float = 0.95,
) -> float:
    """Pick the risk aversion minimising CVaR on the (training) market."""
    delta, band_scale = _ww_inputs(market, costs)
    scores = [
        cvar(hedge_outcome(market, _band_positions(delta, np.cbrt(band_scale / g)), costs).loss,
             alpha)
        for g in grid
    ]
    return float(grid[int(np.argmin(scores))])
