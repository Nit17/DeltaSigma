"""FX option market conventions: delta types, ATM definitions and delta -> strike.

FX smiles are quoted in delta space, and the delta definition depends on the
currency pair (premium currency) and tenor. Getting this wrong mis-places every
smile pillar, so it is modelled explicitly here following Clark (2011),
*Foreign Exchange Option Pricing*, and Reiswich & Wystup (2010),
*A Guide to FX Options Quoting Conventions*.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from deltasigma.pricing.garman_kohlhagen import d1_d2, forward


class DeltaType(str, Enum):
    SPOT = "spot"
    FORWARD = "forward"
    SPOT_PA = "spot_pa"
    FORWARD_PA = "forward_pa"

    @property
    def premium_adjusted(self) -> bool:
        return self in (DeltaType.SPOT_PA, DeltaType.FORWARD_PA)

    @property
    def is_spot(self) -> bool:
        return self in (DeltaType.SPOT, DeltaType.SPOT_PA)


class AtmType(str, Enum):
    FORWARD = "atm_forward"
    DNS = "delta_neutral_straddle"


@dataclass(frozen=True)
class FXConvention:
    """Simplified G10 quoting convention for a currency pair."""

    pair: str
    delta_type_short: DeltaType
    delta_type_long: DeltaType
    atm_type: AtmType = AtmType.DNS
    cutoff_years: float = 1.0

    def delta_type(self, T: float) -> DeltaType:
        return self.delta_type_short if T <= self.cutoff_years + 1e-12 else self.delta_type_long


# Premium-adjusted when the premium is paid in the foreign (base) currency, e.g. USD in USDJPY.
CONVENTIONS: dict[str, FXConvention] = {
    "EURUSD": FXConvention("EURUSD", DeltaType.SPOT, DeltaType.FORWARD),
    "GBPUSD": FXConvention("GBPUSD", DeltaType.SPOT, DeltaType.FORWARD),
    "AUDUSD": FXConvention("AUDUSD", DeltaType.SPOT, DeltaType.FORWARD),
    "USDJPY": FXConvention("USDJPY", DeltaType.SPOT_PA, DeltaType.FORWARD_PA),
    "USDCHF": FXConvention("USDCHF", DeltaType.SPOT_PA, DeltaType.FORWARD_PA),
    "USDCAD": FXConvention("USDCAD", DeltaType.SPOT_PA, DeltaType.FORWARD_PA),
}


def fx_delta(
    S: float,
    K: float,
    T: float,
    rd: float,
    rf: float,
    sigma: float,
    call: bool,
    delta_type: DeltaType,
) -> float:
    """Option delta under the requested market convention."""
    phi = 1.0 if call else -1.0
    F = float(forward(S, T, rd, rf))
    d1, d2 = d1_d2(F, K, T, sigma)
    if delta_type.premium_adjusted:
        fwd_delta = phi * (K / F) * norm.cdf(phi * d2)
    else:
        fwd_delta = phi * norm.cdf(phi * d1)
    scale = np.exp(-rf * T) if delta_type.is_spot else 1.0
    return float(scale * fwd_delta)


def atm_strike(
    S: float, T: float, rd: float, rf: float, sigma: float, atm_type: AtmType, delta_type: DeltaType
) -> float:
    """ATM strike: the forward, or the delta-neutral-straddle strike."""
    F = float(forward(S, T, rd, rf))
    if atm_type is AtmType.FORWARD:
        return F
    sign = -1.0 if delta_type.premium_adjusted else 1.0
    return F * float(np.exp(sign * 0.5 * sigma**2 * T))


def strike_from_delta(
    delta: float,
    S: float,
    T: float,
    rd: float,
    rf: float,
    sigma: float,
    call: bool,
    delta_type: DeltaType,
) -> float:
    """Strike whose ``delta_type`` delta equals ``delta`` (puts take negative deltas).

    Non-premium-adjusted deltas invert in closed form. Premium-adjusted deltas
    have no closed form and the call delta is not monotone in strike, so the
    root is bracketed between the maximum-delta strike and the unadjusted strike.
    """
    phi = 1.0 if call else -1.0
    if phi * delta <= 0:
        raise ValueError("call deltas must be positive and put deltas negative")
    F = float(forward(S, T, rd, rf))
    sd = sigma * np.sqrt(T)
    fwd_delta = delta / np.exp(-rf * T) if delta_type.is_spot else delta
    if abs(fwd_delta) >= 1.0:
        raise ValueError(f"forward delta {fwd_delta} is not attainable")

    k_plain = F * float(np.exp(-phi * sd * norm.ppf(phi * fwd_delta) + 0.5 * sd**2))
    if not delta_type.premium_adjusted:
        return k_plain

    def pa_delta_gap(log_k: float) -> float:
        k = np.exp(log_k)
        _, d2 = d1_d2(F, k, T, sigma)
        return float(phi * (k / F) * norm.cdf(phi * d2) - fwd_delta)

    if call:
        # PA call delta peaks where sd * N(d2) = n(d2); below that strike it decreases again.
        # The +/-10 bracket guarantees a sign change without both terms underflowing to 0.
        d2_star = brentq(lambda d2: sd * norm.cdf(d2) - norm.pdf(d2), -10.0, 10.0)
        k_min = F * np.exp(-d2_star * sd - 0.5 * sd**2)
        if pa_delta_gap(np.log(k_min)) < 0:
            raise ValueError(f"premium-adjusted call delta {delta} exceeds the attainable maximum")
        lo, hi = np.log(k_min), np.log(k_plain)
    else:
        lo, hi = np.log(F) - 12.0 * sd - 1.0, np.log(k_plain)
    return float(np.exp(brentq(pa_delta_gap, lo, hi, xtol=1e-14)))
