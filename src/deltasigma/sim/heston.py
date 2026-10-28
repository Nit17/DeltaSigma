"""Heston stochastic volatility for FX: semi-analytic pricing and QE Monte Carlo.

    dS/S = (rd - rf) dt + sqrt(v) dW1
    dv   = kappa (theta - v) dt + xi sqrt(v) dW2,     d<W1, W2> = rho dt
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        if min(self.kappa, self.theta, self.xi) <= 0 or self.v0 < 0 or not -1 < self.rho < 1:
            raise ValueError(f"invalid Heston parameters: {self}")

    @property
    def feller_satisfied(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.xi**2


def heston_char_func(
    u: ArrayLike, S: float, T: float, rd: float, rf: float, p: HestonParams
) -> np.ndarray:
    """Characteristic function of ``ln S_T`` ("little trap" form, Albrecher et al. 2007)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b * b + p.xi**2 * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (rd - rf) * iu * T + p.kappa * p.theta / p.xi**2 * (
        (b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
    )
    D = (b - d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * p.v0 + iu * np.log(S))


def heston_price(
    S: float, K: float, T: float, rd: float, rf: float, p: HestonParams, call: bool = True
) -> float:
    """European FX option price via Gil-Pelaez inversion of the characteristic function."""
    log_k = np.log(K)
    fwd = S * np.exp((rd - rf) * T)

    def integrand(u: float, shift: bool) -> float:
        arg = u - 1j if shift else u
        val = heston_char_func(arg, S, T, rd, rf, p) * np.exp(-1j * u * log_k) / (1j * u)
        return float(np.real(val / fwd if shift else val))

    p1 = 0.5 + quad(integrand, 0.0, np.inf, args=(True,), limit=500)[0] / np.pi
    p2 = 0.5 + quad(integrand, 0.0, np.inf, args=(False,), limit=500)[0] / np.pi
    call_px = S * np.exp(-rf * T) * p1 - K * np.exp(-rd * T) * p2
    if call:
        return float(call_px)
    return float(call_px - S * np.exp(-rf * T) + K * np.exp(-rd * T))


def qe_step(
    v: np.ndarray,
    kappa: ArrayLike,
    theta: ArrayLike,
    xi: ArrayLike,
    dt: float,
    z: np.ndarray,
    u: np.ndarray,
    psi_crit: float = 1.5,
) -> np.ndarray:
    """Andersen (2008) quadratic-exponential step for the CIR variance process.

    Stays non-negative and accurate even when the Feller condition fails, which
    is the norm for calibrated FX parameters.
    """
    ekt = np.exp(-kappa * dt)
    m = theta + (v - theta) * ekt
    s2 = v * xi**2 * ekt * (1.0 - ekt) / kappa + theta * xi**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi = np.maximum(s2 / m**2, 1e-12)

    inv = 2.0 / psi
    b2 = np.maximum(inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0)), 0.0)
    quad_branch = m / (1.0 + b2) * (np.sqrt(b2) + z) ** 2

    p_zero = (psi - 1.0) / (psi + 1.0)
    beta = (1.0 - p_zero) / m
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_branch = np.where(
            u <= p_zero, 0.0, np.log((1.0 - p_zero) / np.maximum(1.0 - u, 1e-300)) / beta
        )
    return np.where(psi <= psi_crit, quad_branch, exp_branch)


def log_spot_step(
    log_s: np.ndarray,
    v: np.ndarray,
    v_next: np.ndarray,
    mu: float,
    kappa: ArrayLike,
    theta: ArrayLike,
    xi: ArrayLike,
    rho: ArrayLike,
    dt: float,
    z: np.ndarray,
) -> np.ndarray:
    """Andersen's log-spot update consistent with the QE variance step (gamma1 = gamma2 = 1/2)."""
    k0 = -rho * kappa * theta * dt / xi
    k_half = 0.5 * dt * (kappa * rho / xi - 0.5)
    k1, k2 = k_half - rho / xi, k_half + rho / xi
    k34 = 0.5 * dt * (1.0 - rho**2)
    diffusion = np.sqrt(np.maximum(k34 * (v + v_next), 0.0))
    return log_s + mu * dt + k0 + k1 * v + k2 * v_next + diffusion * z
