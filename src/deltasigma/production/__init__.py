"""Production tooling: P&L attribution, model monitoring, model registry and
regime-gated policy deployment."""

from deltasigma.production.attribution import (
    TERMS,
    MarketState,
    OptionPosition,
    attribute_path,
    book_greeks,
    book_value,
    explain,
    explain_ratio,
)
from deltasigma.production.monitoring import (
    CUSUM,
    Alert,
    PageHinkley,
    alert_summary,
    detection_delay,
    ks_drift,
    out_of_range_rate,
    psi,
    rolling_feature_drift,
    run_detector,
)
from deltasigma.production.registry import ModelRegistry, config_hash

__all__ = [
    "CUSUM",
    "TERMS",
    "Alert",
    "MarketState",
    "ModelRegistry",
    "OptionPosition",
    "PageHinkley",
    "alert_summary",
    "attribute_path",
    "book_greeks",
    "book_value",
    "config_hash",
    "detection_delay",
    "explain",
    "explain_ratio",
    "ks_drift",
    "out_of_range_rate",
    "psi",
    "rolling_feature_drift",
    "run_detector",
]
