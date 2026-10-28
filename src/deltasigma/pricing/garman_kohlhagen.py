"""Garman-Kohlhagen pricing and Greeks for European FX options.

Notation follows the FX market: spot ``S`` is quoted as units of domestic (terms)
currency per unit of foreign (base) currency, e.g. EURUSD = USD per EUR.
``rd`` / ``rf`` are continuously compounded domestic / foreign rates and ``T`` is
the time to expiry in years. All functions broadcast over NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm

_N = norm.cdf
_n = norm.pdf


def _phi(call: ArrayLike) -> np.ndarray:
    return np.where(np.asarray(call, dtype=bool), 1.0, -1.0)


def forward(S: ArrayLike, T: ArrayLike, rd: ArrayLike, rf: ArrayLike) -> np.ndarray:
    """Outright forward ``F = S * exp((rd - rf) T)`` (covered interest parity)."""
    return np.asarray(S) * np.exp((np.asarray(rd) - np.asarray(rf)) * np.asarray(T))


def d1_d2(F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike) -> tuple[np.ndarray, ...]:
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return d1, d1 - sd


def price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    rd: ArrayLike,
    rf: ArrayLike,
    sigma: ArrayLike,
    call: ArrayLike = True,
) -> np.ndarray:
    """Option value in domestic currency per unit of foreign notional."""
    phi = _phi(call)
    F = forward(S, T, rd, rf)
    d1, d2 = d1_d2(F, K, T, sigma)
    return np.exp(-np.asarray(rd) * T) * phi * (F * _N(phi * d1) - np.asarray(K) * _N(phi * d2))


@dataclass(frozen=True)
class Greeks:
    """Sensitivities of a Garman-Kohlhagen option (per unit foreign notional).

    ``vega``/``rho`` are per unit (1.00 = 100%) moves; ``theta`` is per year.
    """

    price: np.ndarray
    spot_delta: np.ndarray
    forward_delta: np.ndarray
    spot_delta_pa: np.ndarray
    forward_delta_pa: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray
    theta: np.ndarray
    rho_d: np.ndarray
    rho_f: np.ndarray


def greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    rd: ArrayLike,
    rf: ArrayLike,
    sigma: ArrayLike,
    call: ArrayLike = True,
) -> Greeks:
    """Analytic Greeks, including the FX-specific premium-adjusted deltas."""
    S, K, T, rd, rf, sigma = map(np.asarray, (S, K, T, rd, rf, sigma))
    phi = _phi(call)
    df_d, df_f = np.exp(-rd * T), np.exp(-rf * T)
    F = S * df_f / df_d
    d1, d2 = d1_d2(F, K, T, sigma)
    sqrt_t = np.sqrt(T)
    nd1 = _n(d1)
    Nd1, Nd2 = _N(phi * d1), _N(phi * d2)

    value = df_d * phi * (F * Nd1 - K * Nd2)
    fwd_delta = phi * Nd1
    fwd_delta_pa = phi * (K / F) * Nd2
    vega = S * df_f * nd1 * sqrt_t
    return Greeks(
        price=value,
        spot_delta=df_f * fwd_delta,
        forward_delta=fwd_delta,
        spot_delta_pa=df_f * fwd_delta_pa,
        forward_delta_pa=fwd_delta_pa,
        gamma=df_f * nd1 / (S * sigma * sqrt_t),
        vega=vega,
        vanna=-df_f * nd1 * d2 / sigma,
        volga=vega * d1 * d2 / sigma,
        theta=(
            -S * df_f * nd1 * sigma / (2.0 * sqrt_t)
            + phi * rf * S * df_f * Nd1
            - phi * rd * K * df_d * Nd2
        ),
        rho_d=phi * K * T * df_d * Nd2,
        rho_f=-phi * S * T * df_f * Nd1,
    )


def implied_vol(
    target_price: float,
    S: float,
    K: float,
    T: float,
    rd: float,
    rf: float,
    call: bool = True,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Invert Garman-Kohlhagen for volatility with a bracketed Brent solve."""
    intrinsic = float(price(S, K, T, rd, rf, lo, call))
    upper = float(price(S, K, T, rd, rf, hi, call))
    if not intrinsic - 1e-14 <= target_price <= upper + 1e-14:
        raise ValueError(f"price {target_price} outside no-arbitrage bounds [{intrinsic}, {upper}]")
    return brentq(
        lambda s: float(price(S, K, T, rd, rf, s, call)) - target_price, lo, hi, xtol=1e-14
    )
