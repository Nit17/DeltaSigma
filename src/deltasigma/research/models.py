"""Realised-volatility forecasters with a common ``fit`` / ``predict`` interface.

``predict(X)`` returns one forecast per row of ``X`` using only rows up to that
row, so the walk-forward driver can pass the full history up to the end of a
test block (needed by sequence models) without leaking future information.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from deltasigma.research.cv import purged_kfold


class Forecaster(ABC):
    name: str = "forecaster"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class ColumnForecast(Forecaster):
    """Use a feature directly, e.g. trailing realised vol (random walk) or implied vol."""

    def __init__(self, column: str, name: str) -> None:
        self.column, self.name = column, name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ColumnForecast:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].to_numpy(dtype=float)


class LinearForecast(Forecaster):
    """OLS on selected columns, e.g. HAR-RV (Corsi, 2009) or HAR-RV-IV."""

    def __init__(self, columns: Sequence[str], name: str) -> None:
        self.columns, self.name = list(columns), name
        self.coef_: np.ndarray | None = None

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X[self.columns].to_numpy(dtype=float)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearForecast:
        self.coef_, *_ = np.linalg.lstsq(self._design(X), y.to_numpy(dtype=float), rcond=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._design(X) @ self.coef_


DEFAULT_GBM_GRID: Mapping[str, Sequence] = {
    "learning_rate": (0.03, 0.08),
    "max_leaf_nodes": (7, 15),
}


class GBMForecast(Forecaster):
    """Gradient-boosted trees with hyper-parameters picked by *nested* purged K-fold
    inside each training window (the outer test block is never touched).
    """

    def __init__(
        self,
        horizon: int,
        columns: Sequence[str] | None = None,
        grid: Mapping[str, Sequence] = DEFAULT_GBM_GRID,
        inner_splits: int = 4,
        name: str = "Gradient boosting",
        seed: int = 0,
    ) -> None:
        self.horizon, self.columns, self.grid = horizon, columns, grid
        self.inner_splits, self.name, self.seed = inner_splits, name, seed
        self.model: HistGradientBoostingRegressor | None = None
        self.best_params_: dict | None = None

    def _make(self, **params) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=250, min_samples_leaf=60, l2_regularization=1.0,
            early_stopping=False, random_state=self.seed, **params,
        )

    def _cols(self, X: pd.DataFrame) -> np.ndarray:
        return (X if self.columns is None else X[list(self.columns)]).to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GBMForecast:
        Xa, ya = self._cols(X), y.to_numpy(dtype=float)
        keys = list(self.grid)
        candidates = [dict(zip(keys, vals, strict=True)) for vals in product(*self.grid.values())]
        if len(candidates) > 1:
            folds = list(purged_kfold(len(ya), self.inner_splits, self.horizon, self.horizon))
            scores = []
            for params in candidates:
                errs = []
                for tr, te in folds:
                    fitted = self._make(**params).fit(Xa[tr], ya[tr])
                    errs.append(np.mean((fitted.predict(Xa[te]) - ya[te]) ** 2))
                scores.append(np.mean(errs))
            self.best_params_ = candidates[int(np.argmin(scores))]
        else:
            self.best_params_ = candidates[0]
        self.model = self._make(**self.best_params_).fit(Xa, ya)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._cols(X))


class GRUForecast(Forecaster):
    """GRU over the last ``seq_len`` days of features plus a linear "highway" on the
    latest features (LSTNet-style), so the network only has to learn what a linear
    HAR-type model misses. Early stopping uses a purged, time-ordered validation tail.
    """

    def __init__(
        self,
        horizon: int,
        seq_len: int = 22,
        hidden: int = 16,
        epochs: int = 300,
        lr: float = 3e-3,
        weight_decay: float = 1e-3,
        val_frac: float = 0.2,
        patience: int = 30,
        name: str = "GRU + linear highway",
        seed: int = 0,
    ) -> None:
        self.horizon, self.seq_len, self.hidden = horizon, seq_len, hidden
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.val_frac, self.patience, self.name, self.seed = val_frac, patience, name, seed

    def _standardize(self, Xa: np.ndarray) -> np.ndarray:
        return np.clip((Xa - self.mu_) / self.sd_, -8.0, 8.0)

    def _sequences(self, Z: np.ndarray):
        import torch

        windows = np.lib.stride_tricks.sliding_window_view(Z, self.seq_len, axis=0)
        return torch.as_tensor(np.ascontiguousarray(windows.transpose(0, 2, 1)),
                               dtype=torch.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GRUForecast:
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        Xa = X.to_numpy(dtype=float)
        sd = Xa.std(axis=0)
        # Constant-in-training features (e.g. a regime filter still warming up) must not
        # be blown up by a ~0 standard deviation when they start moving out of sample.
        self.mu_, self.sd_ = Xa.mean(axis=0), np.where(sd > 1e-6, sd, 1.0)
        self.y_mu_, self.y_sd_ = float(y.mean()), float(y.std()) + 1e-8
        seqs = self._sequences(self._standardize(Xa))
        target = torch.as_tensor(
            (y.to_numpy(dtype=float)[self.seq_len - 1:] - self.y_mu_) / self.y_sd_,
            dtype=torch.float32,
        )
        n = len(target)
        n_val = int(self.val_frac * n)
        n_tr = n - n_val - self.horizon
        if n_tr < 50:
            raise ValueError("not enough training data for the GRU")

        class Net(nn.Module):
            def __init__(self, n_in: int, hidden: int) -> None:
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)
                self.highway = nn.Linear(n_in, 1)

            def forward(self, s: torch.Tensor) -> torch.Tensor:
                out, _ = self.gru(s)
                return (self.head(out[:, -1]) + self.highway(s[:, -1])).squeeze(-1)

        self.net_ = Net(seqs.shape[-1], self.hidden)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        tr_x, tr_y = seqs[:n_tr], target[:n_tr]
        va_x, va_y = seqs[n - n_val:], target[n - n_val:]
        best, best_state, bad = np.inf, None, 0
        for _ in range(self.epochs):
            self.net_.train()
            loss = ((self.net_(tr_x) - tr_y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.net_.eval()
            with torch.no_grad():
                val = float(((self.net_(va_x) - va_y) ** 2).mean())
            if val < best - 1e-6:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        out = np.full(len(X), np.nan)
        if len(X) < self.seq_len:
            return out
        seqs = self._sequences(self._standardize(X.to_numpy(dtype=float)))
        self.net_.eval()
        with torch.no_grad():
            out[self.seq_len - 1:] = self.net_(seqs).numpy() * self.y_sd_ + self.y_mu_
        return out
