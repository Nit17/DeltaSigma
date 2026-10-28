"""SABR stochastic-volatility smile (Hagan et al., 2002, lognormal expansion)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_vol(
    F: float, K: ArrayLike, T: float, alpha: float, beta: float, rho: float, nu: float
) -> np.ndarray:
    """Hagan's lognormal implied-volatility approximation."""
    K = np.asarray(K, dtype=float)
    omb = 1.0 - beta
    fk = F * K
    fk_pow = fk ** (0.5 * omb)
    log_fk = np.log(F / K)
    denom = fk_pow * (1.0 + omb**2 / 24.0 * log_fk**2 + omb**4 / 1920.0 * log_fk**4)
    z = nu / alpha * fk_pow * log_fk
    x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    small = np.abs(z) < 1e-8
    z_over_x = np.where(small, 1.0 - 0.5 * rho * z, z / np.where(small, 1.0, x))
    correction = 1.0 + T * (
        omb**2 / 24.0 * alpha**2 / fk**omb
        + 0.25 * rho * beta * nu * alpha / fk_pow
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )
    return alpha / denom * z_over_x * correction


def calibrate_sabr(
    F: float,
    T: float,
    strikes: ArrayLike,
    vols: ArrayLike,
    beta: float = 1.0,
    weights: ArrayLike | None = None,
) -> SABRParams:
    """Least-squares fit of (alpha, rho, nu) with ``beta`` fixed (FX desks typically use 1)."""
    strikes, vols = np.asarray(strikes, float), np.asarray(vols, float)
    w = np.ones_like(vols) if weights is None else np.asarray(weights, float)
    atm_vol = float(np.interp(F, strikes, vols)) if strikes.size > 1 else float(vols[0])

    def residuals(x: np.ndarray) -> np.ndarray:
        return w * (sabr_vol(F, strikes, T, x[0], beta, x[1], x[2]) - vols)

    x0 = np.array([atm_vol * F ** (1.0 - beta), 0.0, 0.5])
    fit = least_squares(
        residuals,
        x0,
        bounds=([1e-6, -0.999, 1e-4], [10.0, 0.999, 10.0]),
        x_scale=[x0[0], 0.5, 0.5],
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
    )
    return SABRParams(alpha=float(fit.x[0]), beta=beta, rho=float(fit.x[1]), nu=float(fit.x[2]))


class SABRSmile:
    """Calibrated SABR smile for a single expiry."""

    def __init__(self, F: float, T: float, params: SABRParams) -> None:
        self.F, self.T, self.params = F, T, params

    def vol(self, K: ArrayLike) -> np.ndarray:
        p = self.params
        return sabr_vol(self.F, K, self.T, p.alpha, p.beta, p.rho, p.nu)
