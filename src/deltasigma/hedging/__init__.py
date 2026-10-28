"""Hedging an FX options book: simulated market, classical benchmarks and deep hedging.

``deltasigma.hedging.deep`` requires PyTorch (``pip install -e ".[ml]"``) and is
not imported here so the rest of the package works without it.
"""

from deltasigma.hedging.market import (
    INSTRUMENTS,
    HedgingMarket,
    OptionBook,
    VanillaOption,
    generate_market,
)
from deltasigma.hedging.pnl import CostModel, HedgeOutcome, cvar, hedge_outcome, summarize
from deltasigma.hedging.strategies import (
    black_delta,
    delta_vega,
    no_hedge,
    tune_whalley_wilmott,
    whalley_wilmott,
)

__all__ = [
    "INSTRUMENTS",
    "CostModel",
    "HedgeOutcome",
    "HedgingMarket",
    "OptionBook",
    "VanillaOption",
    "black_delta",
    "cvar",
    "delta_vega",
    "generate_market",
    "hedge_outcome",
    "no_hedge",
    "summarize",
    "tune_whalley_wilmott",
    "whalley_wilmott",
]
