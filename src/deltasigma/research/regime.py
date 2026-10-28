"""Volatility regime detection with a Gaussian hidden Markov model.

Returns are modelled as zero-mean Gaussian with a state-dependent vol and a
Markov state. Two leakage traps are avoided:

* only **filtered** probabilities ``P(state_t | r_1..r_t)`` are used, never the
  smoothed ``P(state_t | r_1..r_T)`` that EM produces;
* parameters are re-estimated on an expanding window at fixed dates, so the
  probability at ``t`` never depends on parameters fitted with data after ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-7) -> None:
        self.n_states, self.n_iter, self.tol = n_states, n_iter, tol
        self.sigma: np.ndarray | None = None
        self.transition: np.ndarray | None = None
        self.initial: np.ndarray | None = None
        self.log_likelihood = -np.inf

    def _emission(self, x: np.ndarray) -> np.ndarray:
        var = self.sigma**2
        return np.exp(-0.5 * x[:, None] ** 2 / var) / np.sqrt(2.0 * np.pi * var)

    def _forward(self, B: np.ndarray, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, K = B.shape
        alpha, scale = np.empty((T, K)), np.empty(T)
        a = p0 * B[0]
        P = self.transition
        for t in range(T):
            if t:
                a = (alpha[t - 1] @ P) * B[t]
            scale[t] = a.sum()
            alpha[t] = a / scale[t]
        return alpha, scale

    def _init_params(self, x: np.ndarray) -> None:
        K = self.n_states
        sq = np.sort(x**2)
        self.sigma = np.sqrt([chunk.mean() for chunk in np.array_split(sq, K)])
        self.transition = np.full((K, K), 0.02 / max(K - 1, 1))
        np.fill_diagonal(self.transition, 0.98)
        self.initial = np.full(K, 1.0 / K)

    def fit(self, x: np.ndarray, warm_start: GaussianHMM | None = None) -> GaussianHMM:
        """Baum-Welch EM with scaled forward-backward recursions."""
        x = np.asarray(x, dtype=float)
        if warm_start is not None and warm_start.sigma is not None:
            self.sigma = warm_start.sigma.copy()
            self.transition = warm_start.transition.copy()
            self.initial = warm_start.initial.copy()
        else:
            self._init_params(x)
        prev = -np.inf
        for _ in range(self.n_iter):
            B = np.maximum(self._emission(x), 1e-300)
            alpha, scale = self._forward(B, self.initial)
            beta = np.ones_like(alpha)
            for t in range(len(x) - 2, -1, -1):
                beta[t] = self.transition @ (B[t + 1] * beta[t + 1]) / scale[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (alpha[:-1].T @ (B[1:] * beta[1:] / scale[1:, None])) * self.transition
            self.initial = gamma[0]
            self.transition = xi / xi.sum(axis=1, keepdims=True)
            self.sigma = np.sqrt((gamma * x[:, None] ** 2).sum(axis=0) / gamma.sum(axis=0))
            ll = float(np.log(scale).sum())
            converged = ll - prev < self.tol * abs(ll)
            prev = ll
            if converged:
                break
        self.log_likelihood = prev
        order = np.argsort(self.sigma)
        self.sigma = self.sigma[order]
        self.transition = self.transition[np.ix_(order, order)]
        self.initial = self.initial[order]
        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Filtered state probabilities ``P(state_t | x_1..x_t)``, states sorted by vol."""
        if self.sigma is None:
            raise RuntimeError("fit the model first")
        B = np.maximum(self._emission(np.asarray(x, dtype=float)), 1e-300)
        return self._forward(B, self.initial)[0]


def causal_regime_probability(
    returns: pd.Series,
    n_states: int = 2,
    refit_every: int = 252,
    min_history: int = 504,
) -> pd.Series:
    """Filtered probability of the highest-vol state, refit on an expanding window
    at rows ``min_history, min_history + refit_every, ...`` (NaN before that).
    """
    x = returns.fillna(0.0).to_numpy(dtype=float)
    out = np.full(x.size, np.nan)
    hmm: GaussianHMM | None = None
    for start in range(min_history, x.size, refit_every):
        stop = min(start + refit_every, x.size)
        hmm = GaussianHMM(n_states).fit(x[:start], warm_start=hmm)
        out[start:stop] = hmm.filter(x[:stop])[start:stop, -1]
    return pd.Series(out, index=returns.index, name="p_stress")
