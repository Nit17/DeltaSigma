"""Cross-validation for overlapping financial labels (Lopez de Prado, 2018, ch. 7).

A label at row ``i`` spans returns ``i+1 .. i+horizon``. Two samples whose label
windows overlap share information, so a model trained on one and tested on the
other is evaluated on data it has partly seen. *Purging* removes such training
rows; an *embargo* additionally drops rows just after the test block to guard
against serial correlation in features.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PurgedWalkForward:
    """Expanding (or rolling) window walk-forward with purged training sets.

    Test blocks of ``test_size`` rows follow an initial ``min_train_size``. For a test
    block starting at ``a``, training rows are ``i <= a - horizon - embargo`` so no
    training label reaches into the test period.
    """

    horizon: int
    min_train_size: int
    test_size: int
    embargo: int = 0
    max_train_size: int | None = None

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        start = self.min_train_size
        while start < n_samples:
            stop = min(start + self.test_size, n_samples)
            train_end = start - self.horizon - self.embargo + 1
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_end - self.max_train_size)
            if train_end > train_start:
                yield np.arange(train_start, train_end), np.arange(start, stop)
            start = stop

    def n_splits(self, n_samples: int) -> int:
        return sum(1 for _ in self.split(n_samples))


def purged_kfold(
    n_samples: int, n_splits: int, horizon: int, embargo: int = 0
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Contiguous K-fold with purging on both sides and an embargo after each test block.

    Used for hyper-parameter selection *inside* a walk-forward training window.
    """
    bounds = np.linspace(0, n_samples, n_splits + 1).astype(int)
    all_idx = np.arange(n_samples)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        lo = max(0, a - horizon + 1)
        hi = min(n_samples, b + horizon - 1 + embargo)
        train = all_idx[(all_idx < lo) | (all_idx >= hi)]
        yield train, all_idx[a:b]
