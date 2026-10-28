"""Look-ahead (data leakage) audit for feature pipelines.

A feature is causal if its value at row ``t`` is unchanged when every row after
``t`` is deleted. The audit recomputes the pipeline on random prefixes of the
data and compares the last row with the full-sample computation. It catches
centred windows, full-sample normalisation, back-filled values, smoothed (rather
than filtered) regime probabilities and similar silent leaks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def check_no_lookahead(
    feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df: pd.DataFrame,
    n_checks: int = 25,
    min_rows: int = 300,
    seed: int = 0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> list[tuple[pd.Timestamp, str]]:
    """Return ``(date, column)`` pairs whose value depends on future rows."""
    full = feature_fn(df)
    rng = np.random.default_rng(seed)
    cut_points = rng.choice(np.arange(min_rows, len(df)), size=min(n_checks, len(df) - min_rows),
                            replace=False)
    violations: list[tuple[pd.Timestamp, str]] = []
    for t in sorted(cut_points):
        prefix_row = feature_fn(df.iloc[: t + 1]).iloc[-1]
        full_row = full.iloc[t]
        for col in full.columns:
            a, b = prefix_row[col], full_row[col]
            both_nan = pd.isna(a) and pd.isna(b)
            if not both_nan and not np.isclose(a, b, rtol=rtol, atol=atol):
                violations.append((df.index[t], col))
    return violations
