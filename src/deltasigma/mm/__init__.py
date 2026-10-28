"""FX options RFQ market-making: simulator, win-probability model, classical quoters
and a PPO agent (``deltasigma.mm.ppo`` requires PyTorch and is imported lazily).
"""

from deltasigma.mm.env import OBS_NAMES, MarketConfig, RFQMarket, episode_summary
from deltasigma.mm.policies import (
    EvalResult,
    FixedSpreadQuoter,
    ModelQuoter,
    band_hedge,
    evaluate,
    tune,
)
from deltasigma.mm.winprob import (
    WinProbabilityModel,
    calibration_report,
    collect_rfq_history,
    estimate_flow_markout,
    estimate_markout,
)

__all__ = [
    "OBS_NAMES",
    "EvalResult",
    "FixedSpreadQuoter",
    "MarketConfig",
    "ModelQuoter",
    "PPOConfig",
    "PPOMarketMaker",
    "RFQMarket",
    "WinProbabilityModel",
    "band_hedge",
    "calibration_report",
    "collect_rfq_history",
    "episode_summary",
    "estimate_flow_markout",
    "estimate_markout",
    "evaluate",
    "tune",
]


def __getattr__(name: str):
    if name in ("PPOConfig", "PPOMarketMaker"):
        from deltasigma.mm import ppo

        return getattr(ppo, name)
    raise AttributeError(name)
