"""Illustrative EURUSD market snapshot (hand-built, not real market data)."""

from __future__ import annotations

from deltasigma.pricing.conventions import CONVENTIONS
from deltasigma.vol.smile import SmileModel, SmileQuote
from deltasigma.vol.surface import FXVolSurface, build_surface

EURUSD_SPOT = 1.0850
USD_RATE = 0.0430
EUR_RATE = 0.0250

# tenor label -> (T in years, ATM, 25D RR, 25D BF)
EURUSD_QUOTES: dict[str, tuple[float, float, float, float]] = {
    "1W": (7 / 365, 0.0640, -0.0020, 0.0012),
    "1M": (30 / 365, 0.0685, -0.0030, 0.0016),
    "3M": (91 / 365, 0.0720, -0.0042, 0.0020),
    "6M": (182 / 365, 0.0745, -0.0050, 0.0024),
    "1Y": (1.0, 0.0770, -0.0058, 0.0027),
}


def sample_eurusd_surface(model: SmileModel = "sabr") -> FXVolSurface:
    pillars = [
        (T, USD_RATE, EUR_RATE, SmileQuote(atm, rr, bf))
        for T, atm, rr, bf in EURUSD_QUOTES.values()
    ]
    return build_surface(EURUSD_SPOT, pillars, CONVENTIONS["EURUSD"], model=model)
