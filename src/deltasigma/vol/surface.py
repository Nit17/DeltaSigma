"""FX vol surface: smiles per tenor joined by total-variance interpolation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from deltasigma.pricing.conventions import FXConvention
from deltasigma.vol.smile import FXSmile, SmileModel, SmileQuote


class FXVolSurface:
    """Interpolates total implied variance ``w = vol^2 T`` linearly in time along
    constant forward log-moneyness ``k = ln(K / F(T))``. Flat-vol extrapolation
    is used outside the quoted tenors.
    """

    def __init__(self, smiles: Sequence[FXSmile]) -> None:
        if not smiles:
            raise ValueError("at least one smile is required")
        self.smiles = sorted(smiles, key=lambda s: s.T)
        self.tenors = np.array([s.T for s in self.smiles])
        if np.any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be distinct")
        self.spot = self.smiles[0].spot
        self._log_fwd = np.log([s.forward for s in self.smiles])

    def forward(self, T: float) -> float:
        times = np.concatenate([[0.0], self.tenors])
        log_f = np.concatenate([[np.log(self.spot)], self._log_fwd])
        if T > times[-1]:
            slope = (log_f[-1] - log_f[-2]) / (times[-1] - times[-2])
            return float(np.exp(log_f[-1] + slope * (T - times[-1])))
        return float(np.exp(np.interp(T, times, log_f)))

    def _pillar_vol(self, i: int, k: np.ndarray) -> np.ndarray:
        s = self.smiles[i]
        return s.vol(s.forward * np.exp(k))

    def vol(self, K: ArrayLike, T: float) -> np.ndarray:
        if T <= 0:
            raise ValueError("T must be positive")
        k = np.log(np.asarray(K, dtype=float) / self.forward(T))
        if T <= self.tenors[0]:
            return self._pillar_vol(0, k)
        if T >= self.tenors[-1]:
            return self._pillar_vol(len(self.tenors) - 1, k)
        j = int(np.searchsorted(self.tenors, T))
        t0, t1 = self.tenors[j - 1], self.tenors[j]
        w0 = self._pillar_vol(j - 1, k) ** 2 * t0
        w1 = self._pillar_vol(j, k) ** 2 * t1
        w = w0 + (w1 - w0) * (T - t0) / (t1 - t0)
        return np.sqrt(w / T)

    def calendar_arbitrage_violations(
        self, moneyness: ArrayLike | None = None, tol: float = 1e-10
    ) -> list[tuple[float, float, float]]:
        """``(k, T_i, T_{i+1})`` where total variance decreases between pillars."""
        k = np.linspace(-0.25, 0.25, 51) if moneyness is None else np.asarray(moneyness, float)
        w = np.array([self._pillar_vol(i, k) ** 2 * t for i, t in enumerate(self.tenors)])
        bad = np.argwhere(np.diff(w, axis=0) < -tol)
        return [(float(k[c]), float(self.tenors[r]), float(self.tenors[r + 1])) for r, c in bad]


def build_surface(
    spot: float,
    pillars: Sequence[tuple[float, float, float, SmileQuote]],
    convention: FXConvention,
    model: SmileModel = "sabr",
) -> FXVolSurface:
    """Build a surface from ``(T, rd, rf, quote)`` pillars."""
    return FXVolSurface(
        [FXSmile(spot, T, rd, rf, q, convention, model=model) for T, rd, rf, q in pillars]
    )
