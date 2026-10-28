"""Leakage-safe features and forward realised-vol targets.

Every feature at date ``t`` uses only data observed up to and including ``t``
(audited by :func:`deltasigma.research.leakage.check_no_lookahead`). The target
is the log of annualised realised vol over the *next* ``horizon`` days,
``t+1 .. t+horizon``, so the last ``horizon`` rows have no label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_DAILY_VAR_FLOOR = 0.01**2 / TRADING_DAYS  # 1% annualised vol floor before taking logs

HAR_COLUMNS = ["log_rv_1", "log_rv_5", "log_rv_22"]


def log_returns(spot: pd.Series) -> pd.Series:
    return np.log(spot).diff()


def forward_realized_vol(returns: pd.Series, horizon: int) -> pd.Series:
    """Annualised realised vol over ``t+1 .. t+horizon`` (the forecasting target)."""
    future_sq = (returns**2).rolling(horizon).sum().shift(-horizon)
    return np.sqrt(TRADING_DAYS / horizon * future_sq)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking features from ``spot`` and ``iv`` columns."""
    ret = log_returns(df["spot"])
    sq = ret**2
    log_spot = np.log(df["spot"])
    log_iv = np.log(df["iv"])

    def log_rv(window: int) -> pd.Series:
        return 0.5 * np.log(TRADING_DAYS * np.maximum(sq.rolling(window).mean(), _DAILY_VAR_FLOOR))

    X = pd.DataFrame(index=df.index)
    X["ret_1"] = ret
    for w in (1, 5, 22, 66):
        X[f"log_rv_{w}"] = log_rv(w)
    X["log_iv"] = log_iv
    X["iv_rv_spread"] = df["iv"] - np.sqrt(TRADING_DAYS * sq.rolling(22).mean())
    X["d_log_iv_5"] = log_iv.diff(5)
    X["d_log_iv_22"] = log_iv.diff(22)
    X["ret_5"] = log_spot.diff(5)
    X["ret_22"] = log_spot.diff(22)
    X["iv_z_252"] = (log_iv - log_iv.rolling(252).mean()) / log_iv.rolling(252).std()
    return X


def build_dataset(
    df: pd.DataFrame, horizon: int = 21
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Features ``X``, log-vol target ``y`` and an ``info`` frame (iv, realised vol)
    restricted to rows where every feature and the target are available.
    """
    X = build_features(df)
    rv_fwd = forward_realized_vol(log_returns(df["spot"]), horizon)
    y = np.log(rv_fwd).rename("target")
    info = pd.DataFrame({"iv": df["iv"], "rv_fwd": rv_fwd})
    valid = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    return X[valid], y[valid], info[valid]
