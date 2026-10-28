"""RFQ win-probability modelling and post-trade markout (adverse selection) estimates.

The dealer never sees the true acceptance curve; it estimates one from an RFQ
log in which quotes were randomised (exploration). The simulator knows the
true probabilities, so calibration can be measured against ground truth as
well as against outcomes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from deltasigma.mm.env import MarketConfig, RFQMarket

FEATURES = ("edge", "edge_x_tier", "tier", "vol_level", "vol_of_vol")
EDGE_GRID = np.linspace(-0.1, 1.0, 221)


def collect_rfq_history(
    config: MarketConfig,
    n_episodes: int,
    seed: int,
    edge_range: tuple[float, float] = (-0.05, 0.6),
) -> pd.DataFrame:
    """RFQ log with uniformly randomised quotes and no hedging."""
    env = RFQMarket(config, n_episodes, seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(seed + 10_000)
    H = config.informed_horizon
    frames = []
    for t in range(config.n_steps):
        rfq = env.current_rfq()
        edge = rng.uniform(*edge_range, n_episodes)
        stress = env.stress.copy()
        p_true = config.win_probability(edge, rfq["tier"], stress)
        frame = pd.DataFrame({
            "episode": np.arange(n_episodes),
            "step": t,
            "edge": edge,
            "tier": rfq["tier"],
            "side": rfq["side"],
            "size": rfq["size"],
            "vol_level": obs[:, 1],
            "vol_of_vol": obs[:, 2],
            "flow_tier1": obs[:, 3],
            "stress": stress,
            "p_true": p_true,
            # dealer P&L per vega unit from the vol move after the trade (sold vol if side=+1)
            "markout": -rfq["side"] * (env.vol[:, t + H] - env.vol[:, t]),
        })
        obs, _, _, info = env.step(edge, np.zeros(n_episodes))
        frame["accepted"] = info["filled"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _design(edge, tier, vol_level, vol_of_vol) -> np.ndarray:
    edge, tier = np.asarray(edge, float), np.asarray(tier, float)
    edge, tier, vol_level, vol_of_vol = np.broadcast_arrays(edge, tier, vol_level, vol_of_vol)
    return np.stack([edge, edge * tier, tier, vol_level, vol_of_vol], axis=-1)


class WinProbabilityModel:
    """Logistic acceptance model with tier-specific price sensitivity and
    regime proxies (vol level, vol-of-vol)."""

    def __init__(self, C: float = 100.0) -> None:
        self.model = LogisticRegression(C=C, max_iter=1000)

    def fit(self, history: pd.DataFrame) -> WinProbabilityModel:
        X = _design(history["edge"], history["tier"], history["vol_level"], history["vol_of_vol"])
        self.model.fit(X, history["accepted"].astype(int))
        return self

    def predict_proba(self, edge, tier, vol_level, vol_of_vol) -> np.ndarray:
        X = _design(edge, tier, vol_level, vol_of_vol)
        z = X @ self.model.coef_[0] + self.model.intercept_[0]
        return 1.0 / (1.0 + np.exp(-z))

    def optimal_edge(self, tier, vol_level, vol_of_vol, cost=0.0) -> np.ndarray:
        """``argmax_delta (delta - cost) * P(win | delta)`` per RFQ, on a fixed grid."""
        tier, vol_level, vol_of_vol, cost = (np.asarray(a, float)[:, None] for a in
                                             np.broadcast_arrays(tier, vol_level, vol_of_vol, cost))
        p = self.predict_proba(EDGE_GRID[None, :], tier, vol_level, vol_of_vol)
        return EDGE_GRID[np.argmax((EDGE_GRID[None, :] - cost) * p, axis=1)]


def calibration_report(model: WinProbabilityModel, test: pd.DataFrame, n_bins: int = 10):
    """Scores on held-out RFQs plus a reliability table."""
    p = model.predict_proba(test["edge"], test["tier"], test["vol_level"], test["vol_of_vol"])
    y = test["accepted"].astype(int).to_numpy()
    scores = {
        "Brier": brier_score_loss(y, p),
        "Brier (true probabilities)": brier_score_loss(y, test["p_true"]),
        "Log loss": log_loss(y, p),
        "AUC": roc_auc_score(y, p),
        "Mean |p_hat - p_true|": float(np.mean(np.abs(p - test["p_true"]))),
    }
    bins = pd.cut(p, np.linspace(0, 1, n_bins + 1), include_lowest=True)
    table = pd.DataFrame({"predicted": p, "observed": y, "true": test["p_true"]}).groupby(
        bins, observed=True).mean()
    table["count"] = pd.Series(bins).value_counts().reindex(table.index).to_numpy()
    return scores, table


def estimate_markout(history: pd.DataFrame) -> np.ndarray:
    """Average adverse markout cost per vega unit for each client tier (filled trades)."""
    filled = history[history["accepted"]]
    cost = -filled.groupby("tier")["markout"].mean()
    return cost.reindex([0, 1], fill_value=0.0).to_numpy()


def estimate_flow_markout(history: pd.DataFrame) -> tuple[np.ndarray, float]:
    """OLS of markout cost on tier dummies and ``side * flow_tier1``.

    Returns ``(cost_by_tier, flow_coef)``: a positive ``flow_coef`` means trading in
    the same direction as recent tier-1 flow is followed by adverse vol moves.
    """
    filled = history[history["accepted"]]
    X = np.column_stack([
        (filled["tier"] == 0).to_numpy(float),
        (filled["tier"] == 1).to_numpy(float),
        (filled["side"] * filled["flow_tier1"]).to_numpy(float),
    ])
    coef, *_ = np.linalg.lstsq(X, -filled["markout"].to_numpy(float), rcond=None)
    return coef[:2], float(coef[2])
