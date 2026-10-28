"""Walk-forward evaluation and forecast-comparison statistics."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from deltasigma.research.cv import PurgedWalkForward
from deltasigma.research.models import Forecaster


def walk_forward_predict(
    factory: Callable[[], Forecaster],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: PurgedWalkForward,
) -> pd.Series:
    """Out-of-sample forecasts: refit a fresh model per fold, predict the next block."""
    preds = pd.Series(np.nan, index=y.index, name=factory().name)
    for train_idx, test_idx in splitter.split(len(y)):
        model = factory().fit(X.iloc[train_idx], y.iloc[train_idx])
        history = X.iloc[: test_idx[-1] + 1]
        preds.iloc[test_idx] = model.predict(history)[test_idx]
    return preds


def qlike(y_log: np.ndarray, pred_log: np.ndarray) -> np.ndarray:
    """QLIKE loss on variances (Patton, 2011): robust to noise in the realised-vol proxy."""
    ratio = np.exp(2.0 * (np.asarray(y_log) - np.asarray(pred_log)))
    return ratio - np.log(ratio) - 1.0


def squared_error(y_log: np.ndarray, pred_log: np.ndarray) -> np.ndarray:
    return (np.asarray(y_log) - np.asarray(pred_log)) ** 2


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """DM test of equal predictive accuracy with a Bartlett/Newey-West long-run variance
    (``horizon - 1`` lags for overlapping forecasts) and the Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic => ``a`` has lower loss.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = d.size
    dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, horizon):
        lrv += 2.0 * (1.0 - k / horizon) * (dc[k:] @ dc[:-k]) / T
    if lrv <= 0:
        lrv = dc @ dc / T
    if lrv == 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(lrv / T)
    stat *= np.sqrt((T + 1 - 2 * horizon + horizon * (horizon - 1) / T) / T)
    p_value = 2.0 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p_value)


def forecast_report(
    y: pd.Series,
    preds: Mapping[str, pd.Series],
    benchmark: str,
    horizon: int,
    mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Out-of-sample accuracy table on rows where every model has a forecast."""
    frame = pd.DataFrame(preds)
    ok = frame.notna().all(axis=1) & y.notna()
    if mask is not None:
        ok &= mask.reindex(y.index, fill_value=False)
    yt, frame = y[ok].to_numpy(), frame[ok]
    bench = frame[benchmark].to_numpy()
    rows = {}
    for name in frame.columns:
        p = frame[name].to_numpy()
        ql, se = qlike(yt, p), squared_error(yt, p)
        dm_stat, dm_p = (np.nan, np.nan)
        if name != benchmark:
            dm_stat, dm_p = diebold_mariano(ql, qlike(yt, bench), horizon)
        rows[name] = {
            "RMSE (log vol)": float(np.sqrt(se.mean())),
            "QLIKE": float(ql.mean()),
            f"R2 oos vs {benchmark}": float(1.0 - se.sum() / squared_error(yt, bench).sum()),
            "DM stat (QLIKE)": dm_stat,
            "DM p-value": dm_p,
        }
    report = pd.DataFrame(rows).T
    report.attrs["n_obs"] = int(ok.sum())
    return report
