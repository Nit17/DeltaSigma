"""Volatility signal research: leakage-safe features, purged walk-forward CV,
regime detection, forecast comparison and a variance-risk-premium backtest.

Requires the ``research`` extra (pandas, scikit-learn); the GRU model needs PyTorch.
"""

from deltasigma.research.backtest import (
    ladder_equity,
    predicted_vrp,
    strategy_stats,
    timing_positions,
    vol_swap_pnl,
)
from deltasigma.research.cv import PurgedWalkForward, purged_kfold
from deltasigma.research.data import load_eurusd_vol, synthetic_vol_dataset
from deltasigma.research.evaluation import (
    diebold_mariano,
    forecast_report,
    qlike,
    walk_forward_predict,
)
from deltasigma.research.features import HAR_COLUMNS, build_dataset, build_features
from deltasigma.research.leakage import check_no_lookahead
from deltasigma.research.models import (
    ColumnForecast,
    GBMForecast,
    GRUForecast,
    LinearForecast,
)
from deltasigma.research.regime import GaussianHMM, causal_regime_probability

__all__ = [
    "HAR_COLUMNS",
    "ColumnForecast",
    "GBMForecast",
    "GRUForecast",
    "GaussianHMM",
    "LinearForecast",
    "PurgedWalkForward",
    "build_dataset",
    "build_features",
    "causal_regime_probability",
    "check_no_lookahead",
    "diebold_mariano",
    "forecast_report",
    "ladder_equity",
    "load_eurusd_vol",
    "predicted_vrp",
    "purged_kfold",
    "qlike",
    "strategy_stats",
    "synthetic_vol_dataset",
    "timing_positions",
    "vol_swap_pnl",
    "walk_forward_predict",
]
