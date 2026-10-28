"""Regime-gated policy switching for the RFQ market-maker.

Phase 4 found that the PPO quoter wins in stress but not in calm markets, so the
desk deploys it *conditionally*: an online Gaussian HMM on observed vol changes
estimates the probability of the stress regime, and each RFQ is routed to the
stress policy when that probability exceeds a threshold fixed ex ante.
"""

from __future__ import annotations

import numpy as np

from deltasigma.mm.env import MarketConfig, RFQMarket
from deltasigma.research.regime import GaussianHMM


def collect_vol_changes(config: MarketConfig, n_episodes: int, seed: int) -> np.ndarray:
    """Observed per-step ATM vol changes from simulated history (one series per episode)."""
    env = RFQMarket(config, n_episodes, seed=seed)
    env.reset()
    return np.diff(env.vol[:, : config.n_steps + 1], axis=1)


def fit_regime_hmm(vol_changes: np.ndarray) -> GaussianHMM:
    """Fit a 2-state HMM on concatenated training episodes."""
    return GaussianHMM(2).fit(vol_changes.reshape(-1))


class OnlineRegimeFilter:
    """Vectorised forward filter: ``p_t ∝ (p_{t-1} P) * N(x_t; 0, sigma_state^2)``."""

    def __init__(self, hmm: GaussianHMM, n: int) -> None:
        self.hmm = hmm
        self.p = np.tile(self._stationary(hmm.transition), (n, 1))

    @staticmethod
    def _stationary(P: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eig(P.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        return pi / pi.sum()

    def update(self, x: np.ndarray) -> np.ndarray:
        var = self.hmm.sigma**2
        lik = np.exp(-0.5 * x[:, None] ** 2 / var) / np.sqrt(2 * np.pi * var)
        p = (self.p @ self.hmm.transition) * np.maximum(lik, 1e-300)
        self.p = p / p.sum(axis=1, keepdims=True)
        return self.p[:, -1]


class RegimeGatedPolicy:
    """Route each RFQ to ``stress_policy`` when ``P(stress) > threshold``."""

    def __init__(self, calm_policy, stress_policy, hmm: GaussianHMM, threshold: float = 0.5):
        self.calm, self.stress = calm_policy, stress_policy
        self.hmm, self.threshold = hmm, threshold
        self._env_id: int | None = None
        self.share_stress: list[float] = []

    def __call__(self, env: RFQMarket):
        if self._env_id != id(env) or env.t == 0:
            self._env_id = id(env)
            self.filter = OnlineRegimeFilter(self.hmm, env.n_envs)
        if env.t > 0:
            self.filter.update(env.vol[:, env.t] - env.vol[:, env.t - 1])
        use_stress = self.filter.p[:, -1] > self.threshold
        self.share_stress.append(float(use_stress.mean()))
        e_calm, h_calm = self.calm(env)
        e_stress, h_stress = self.stress(env)
        return np.where(use_stress, e_stress, e_calm), np.where(use_stress, h_stress, h_calm)
