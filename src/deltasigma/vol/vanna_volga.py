"""Vanna-Volga smile construction (Castagna & Mercurio, 2007).

The FX market-maker's classic: price the smile cost of any strike by hedging its
vega, vanna and volga with the three liquid pillars (25D put, ATM, 25D call).
The second-order closed-form implied-vol approximation is used, which
reproduces the three pillar vols exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from deltasigma.pricing.garman_kohlhagen import d1_d2


class VannaVolgaSmile:
    def __init__(self, F: float, T: float, strikes: ArrayLike, vols: ArrayLike) -> None:
        strikes, vols = np.asarray(strikes, float), np.asarray(vols, float)
        if strikes.shape != (3,) or not np.all(np.diff(strikes) > 0):
            raise ValueError("Vanna-Volga needs three increasing strikes (25P, ATM, 25C)")
        self.F, self.T = F, T
        self.strikes, self.vols = strikes, vols

    def _d1d2(self, K: np.ndarray) -> np.ndarray:
        d1, d2 = d1_d2(self.F, K, self.T, self.vols[1])
        return d1 * d2

    def vol(self, K: ArrayLike) -> np.ndarray:
        K = np.asarray(K, dtype=float)
        (k1, k2, k3), (s1, s2, s3) = self.strikes, self.vols
        y1 = np.log(k2 / K) * np.log(k3 / K) / (np.log(k2 / k1) * np.log(k3 / k1))
        y2 = np.log(K / k1) * np.log(k3 / K) / (np.log(k2 / k1) * np.log(k3 / k2))
        y3 = np.log(K / k1) * np.log(K / k2) / (np.log(k3 / k1) * np.log(k3 / k2))

        first_order = y1 * s1 + y2 * s2 + y3 * s3
        dd = self._d1d2(K)
        d_one = first_order - s2
        d_two = (
            y1 * self._d1d2(k1) * (s1 - s2) ** 2 + y3 * self._d1d2(k3) * (s3 - s2) ** 2
        )
        disc = s2**2 + dd * (2.0 * s2 * d_one + d_two)
        tiny = np.abs(dd) < 1e-10
        safe_dd = np.where(tiny, 1.0, dd)
        second_order = s2 + (-s2 + np.sqrt(np.maximum(disc, 0.0))) / safe_dd
        limit = s2 + d_one + d_two / (2.0 * s2)
        out = np.where(tiny, limit, second_order)
        # Far wings can make the square root ill-defined; fall back to first order there.
        return np.where(disc > 0, out, first_order)
