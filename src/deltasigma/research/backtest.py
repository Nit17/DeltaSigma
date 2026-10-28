"""Variance-risk-premium (VRP) backtest: selling 1M EURUSD vol when it is rich.

A trade entered at ``t`` is a vega-notional vol swap struck at implied vol
``iv_t`` that pays realised vol over ``t+1 .. t+horizon``; P&L is in vol points per
unit vega notional. Positions: ``-1`` sell vol, ``0`` flat, ``+1`` buy vol.
Simplification: the 30-day vol index is used as the tradable strike (no convexity
adjustment), with a fixed half-spread cost per trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_swap_pnl(
    iv: pd.Series, rv_fwd: pd.Series, position: pd.Series, cost_vol_pts: float = 0.25
) -> pd.Series:
    return position * (rv_fwd - iv) * 100.0 - cost_vol_pts * position.abs()


def predicted_vrp(iv: pd.Series, pred_log_rv: pd.Series) -> pd.Series:
    """Implied minus forecast realised vol, in vol points."""
    return (iv - np.exp(pred_log_rv)) * 100.0


def timing_positions(pred_vrp: pd.Series, threshold: float, allow_long: bool = False) -> pd.Series:
    pos = pd.Series(0.0, index=pred_vrp.index)
    pos[pred_vrp > threshold] = -1.0
    if allow_long:
        pos[pred_vrp < -threshold] = 1.0
    return pos.where(pred_vrp.notna(), 0.0)


def max_drawdown(pnl: pd.Series) -> float:
    equity = np.concatenate([[0.0], pnl.cumsum().to_numpy()])
    return float((np.maximum.accumulate(equity) - equity).max())


def strategy_stats(trade_pnl: pd.Series, position: pd.Series, horizon: int) -> dict[str, float]:
    """Statistics on non-overlapping trades, averaged over all ``horizon`` start offsets
    so results do not depend on an arbitrary roll date.
    """
    sharpes, drawdowns, totals = [], [], []
    for offset in range(horizon):
        s = trade_pnl.iloc[offset::horizon]
        sd = s.std()
        sharpes.append(s.mean() / sd * np.sqrt(252.0 / horizon) if sd > 0 else 0.0)
        drawdowns.append(max_drawdown(s))
        totals.append(s.sum())
    active = position != 0
    act = trade_pnl[active]
    n_tail = max(1, int(np.ceil(0.05 * act.size))) if act.size else 0
    return {
        "Active %": 100.0 * float(active.mean()),
        "Avg P&L / trade": float(act.mean()) if act.size else 0.0,
        "Hit rate %": 100.0 * float((act > 0).mean()) if act.size else 0.0,
        "Sharpe": float(np.mean(sharpes)),
        "Max drawdown": float(np.mean(drawdowns)),
        "Worst 5% trades (avg)": float(np.sort(act.to_numpy())[:n_tail].mean()) if n_tail else 0.0,
        "Total P&L / yr": float(np.mean(totals)) / (len(trade_pnl) / 252.0),
    }


def ladder_equity(trade_pnl: pd.Series, horizon: int) -> pd.Series:
    """Cumulative P&L of a ladder entering ``1/horizon`` notional every day,
    booked when each trade matures.
    """
    return (trade_pnl.shift(horizon).fillna(0.0) / horizon).cumsum()
