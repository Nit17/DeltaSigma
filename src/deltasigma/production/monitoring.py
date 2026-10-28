"""Model monitoring: input drift (PSI, KS) and performance drift (CUSUM, Page-Hinkley).

Typical wiring in production: the monitors are fitted on a *reference* window
(the model's validation period) and then fed live observations; an alert means
"the model is operating outside the conditions it was validated on" and should
trigger review, retraining or a fallback to a simpler model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with reference-quantile bins.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / reference.size + eps
    q = np.histogram(current, edges)[0] / current.size + eps
    return float(np.sum((q - p) * np.log(q / p)))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    res = stats.ks_2samp(reference, current)
    return float(res.statistic), float(res.pvalue)


@dataclass(frozen=True)
class Alert:
    time: object
    monitor: str
    value: float
    threshold: float

    def __str__(self) -> str:
        return f"[{self.time}] {self.monitor}: {self.value:.3f} > {self.threshold:.3f}"


class CUSUM:
    """One-sided standardised CUSUM for an *increase* in mean loss.

    ``S_t = max(0, S_{t-1} + (x_t - mu0) / sigma0 - k)``, alarm when ``S_t > h``,
    then reset. With ``k = 0.5`` it is tuned to detect a one-sigma mean shift.
    """

    def __init__(self, mu0: float, sigma0: float, k: float = 0.5, h: float = 8.0) -> None:
        if sigma0 <= 0:
            raise ValueError("sigma0 must be positive")
        self.mu0, self.sigma0, self.k, self.h = mu0, sigma0, k, h
        self.s = 0.0
        self.last_stat = 0.0

    @classmethod
    def fit(cls, reference: Iterable[float], **kw) -> CUSUM:
        ref = np.asarray(list(reference), dtype=float)
        return cls(float(ref.mean()), float(ref.std()), **kw)

    def update(self, x: float) -> bool:
        self.s = max(0.0, self.s + (x - self.mu0) / self.sigma0 - self.k)
        self.last_stat = self.s
        if self.s > self.h:
            self.s = 0.0
            return True
        return False


class PageHinkley:
    """Page-Hinkley test for an increase in the mean of a stream (no reference needed)."""

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, burn_in: int = 30) -> None:
        self.delta, self.threshold, self.burn_in = delta, threshold, burn_in
        self.reset()

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.min_cum, self.last_stat = 0, 0.0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        stat = self.cum - self.min_cum
        self.last_stat = stat
        if self.n > self.burn_in and stat > self.threshold:
            self.reset()
            self.last_stat = stat
            return True
        return False


def run_detector(detector, stream: pd.Series, name: str, cooldown: int = 0) -> list[Alert]:
    """Feed a time-indexed stream through a detector and collect alerts
    (alert value = the detector statistic that crossed its threshold).

    ``cooldown`` suppresses repeat alerts for that many observations after an alert,
    so one degradation episode raises one ticket instead of hundreds.
    """
    alerts = []
    threshold = float(getattr(detector, "h", getattr(detector, "threshold", np.nan)))
    quiet_until = -1
    for i, (t, x) in enumerate(stream.items()):
        if detector.update(float(x)) and i > quiet_until:
            alerts.append(Alert(t, name, float(detector.last_stat), threshold))
            quiet_until = i + cooldown
    return alerts


def out_of_range_rate(reference: np.ndarray, current: np.ndarray, tail: float = 0.005) -> float:
    """Share of ``current`` outside the reference ``[tail, 1 - tail]`` quantile range."""
    lo, hi = np.quantile(np.asarray(reference, dtype=float), [tail, 1.0 - tail])
    current = np.asarray(current, dtype=float)
    return float(np.mean((current < lo) | (current > hi)))


def rolling_feature_drift(
    reference: pd.DataFrame,
    live: pd.DataFrame,
    window: int = 63,
    step: int = 21,
    psi_threshold: float = 0.25,
    expanding: bool = False,
    metric: str = "psi",
    ood_threshold: float = 0.10,
) -> tuple[pd.DataFrame, list[Alert]]:
    """Drift of each feature on rolling live windows vs the reference distribution.

    ``metric="psi"`` alerts on the population stability index; ``metric="ood"`` alerts on
    the share of the window outside the reference 0.5%-99.5% range. PSI assumes i.i.d.
    samples and flags almost every window of a persistent *level* series (e.g. implied
    vol), so the out-of-range rate is the more actionable alert for such features.
    With ``expanding=True`` the reference grows with all live data before each window,
    matching a model that is refit on an expanding window: an alert then means the
    inputs are unlike *anything* the current model was trained on.
    """
    if metric not in ("psi", "ood"):
        raise ValueError("metric must be 'psi' or 'ood'")
    rows, alerts = {}, []
    for end in range(window, len(live) + 1, step):
        chunk = live.iloc[end - window : end]
        ref = pd.concat([reference, live.iloc[: end - window]]) if expanding else reference
        t = chunk.index[-1]
        if metric == "psi":
            rows[t] = {c: psi(ref[c].to_numpy(), chunk[c].to_numpy()) for c in live.columns}
            limit = psi_threshold
        else:
            rows[t] = {c: out_of_range_rate(ref[c].to_numpy(), chunk[c].to_numpy())
                       for c in live.columns}
            limit = ood_threshold
        for c, v in rows[t].items():
            if v > limit:
                alerts.append(Alert(t, f"{metric.upper()}[{c}]", v, limit))
    return pd.DataFrame(rows).T, alerts


def detection_delay(alerts: Iterable[Alert], change_time, index: pd.Index) -> float:
    """Observations between a known change point and the first alert after it."""
    after = [a.time for a in alerts if a.time >= change_time]
    if not after:
        return np.inf
    return float(index.get_loc(after[0]) - index.get_loc(change_time))


def alert_summary(alerts: Iterable[Alert]) -> Mapping[str, int]:
    out: dict[str, int] = {}
    for a in alerts:
        out[a.monitor] = out.get(a.monitor, 0) + 1
    return out
