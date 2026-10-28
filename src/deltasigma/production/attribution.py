"""Daily P&L attribution ("P&L explain") for an FX options book.

The actual P&L from full revaluation is decomposed into the Taylor terms risk
managers sign off on every day:

    dV ~ delta dS + 1/2 gamma dS^2 + vega dsigma + vanna dS dsigma
         + 1/2 volga dsigma^2 + theta dt + rho_d d(rd) + rho_f d(rf)

and an *unexplained* residual. A large or growing residual means the risk model
(Greeks, vol dynamics, smile assumption) no longer describes the book, which is
itself a production model-risk signal. The residual is third order for spot/vol
moves but second order once time or rates move too (charm, veta and rate cross
terms are not in the explain).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from deltasigma.pricing import garman_kohlhagen as gk

TERMS = ("delta", "gamma", "vega", "vanna", "volga", "theta", "rho_d", "rho_f")


@dataclass(frozen=True)
class MarketState:
    t: float          # current time in years
    spot: float
    atm_vol: float
    rd: float
    rf: float


@dataclass(frozen=True)
class OptionPosition:
    strike: float
    expiry: float     # absolute expiry time in years
    call: bool
    notional: float   # foreign-currency notional; negative = short
    vol_offset: float = 0.0  # fixed smile offset vs ATM (sticky-strike smile, parallel shifts)


def _arrays(book: Sequence[OptionPosition], state: MarketState):
    K = np.array([p.strike for p in book])
    tau = np.array([p.expiry for p in book]) - state.t
    if np.any(tau <= 0):
        raise ValueError("all positions must be unexpired")
    call = np.array([p.call for p in book])
    notional = np.array([p.notional for p in book])
    vol = state.atm_vol + np.array([p.vol_offset for p in book])
    return K, tau, call, notional, vol


def book_value(book: Sequence[OptionPosition], state: MarketState) -> float:
    K, tau, call, n, vol = _arrays(book, state)
    return float(np.sum(n * gk.price(state.spot, K, tau, state.rd, state.rf, vol, call)))


def book_greeks(book: Sequence[OptionPosition], state: MarketState) -> dict[str, float]:
    K, tau, call, n, vol = _arrays(book, state)
    g = gk.greeks(state.spot, K, tau, state.rd, state.rf, vol, call)
    return {
        "delta": float(np.sum(n * g.spot_delta)),
        "gamma": float(np.sum(n * g.gamma)),
        "vega": float(np.sum(n * g.vega)),
        "vanna": float(np.sum(n * g.vanna)),
        "volga": float(np.sum(n * g.volga)),
        "theta": float(np.sum(n * g.theta)),
        "rho_d": float(np.sum(n * g.rho_d)),
        "rho_f": float(np.sum(n * g.rho_f)),
    }


def explain(
    book: Sequence[OptionPosition], start: MarketState, end: MarketState
) -> dict[str, float]:
    """Greek attribution of the book's P&L between two market states."""
    g = book_greeks(book, start)
    dS, dv = end.spot - start.spot, end.atm_vol - start.atm_vol
    terms = {
        "delta": g["delta"] * dS,
        "gamma": 0.5 * g["gamma"] * dS**2,
        "vega": g["vega"] * dv,
        "vanna": g["vanna"] * dS * dv,
        "volga": 0.5 * g["volga"] * dv**2,
        "theta": g["theta"] * (end.t - start.t),
        "rho_d": g["rho_d"] * (end.rd - start.rd),
        "rho_f": g["rho_f"] * (end.rf - start.rf),
    }
    actual = book_value(book, end) - book_value(book, start)
    terms["explained"] = float(sum(terms[k] for k in TERMS))
    terms["actual"] = actual
    terms["unexplained"] = actual - terms["explained"]
    return terms


def attribute_path(
    book: Sequence[OptionPosition],
    states: Sequence[MarketState],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Day-by-day attribution along a path of market states (one row per interval)."""
    rows = [explain(book, a, b) for a, b in zip(states[:-1], states[1:], strict=True)]
    return pd.DataFrame(rows, index=index[1:] if index is not None else None)


def explain_ratio(attribution: pd.DataFrame) -> float:
    """Share of actual P&L variance captured by the Greek explain (1 = perfect)."""
    resid = attribution["unexplained"].var()
    total = attribution["actual"].var()
    return float(1.0 - resid / total) if total > 0 else 1.0
