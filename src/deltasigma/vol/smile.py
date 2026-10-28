"""Build an arbitrage-checked FX smile from market quotes (ATM, 25D RR, 25D BF).

FX vol is quoted as ATM, risk reversal ``RR = vol(25C) - vol(25P)`` and
butterfly. The interbank butterfly is a *market (broker) strangle*: a single
vol ``ATM + BF`` that prices the 25-delta strangle. It is **not** the same as
the *smile strangle* ``(vol(25C) + vol(25P)) / 2 - ATM`` of the fitted smile,
so this module solves for the smile strangle that reproduces the market
strangle premium (Clark, 2011, ch. 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar

from deltasigma.pricing import garman_kohlhagen as gk
from deltasigma.pricing.conventions import FXConvention, atm_strike, strike_from_delta
from deltasigma.vol.sabr import SABRSmile, calibrate_sabr
from deltasigma.vol.vanna_volga import VannaVolgaSmile

SmileModel = Literal["sabr", "vanna_volga"]


@dataclass(frozen=True)
class SmileQuote:
    """Market quote for one expiry, as decimals (0.075 = 7.5 vol)."""

    atm: float
    rr25: float
    bf25: float
    strangle: Literal["broker", "smile"] = "broker"


class FXSmile:
    """Single-expiry FX smile calibrated to ATM / 25D RR / 25D BF quotes."""

    def __init__(
        self,
        spot: float,
        T: float,
        rd: float,
        rf: float,
        quote: SmileQuote,
        convention: FXConvention,
        model: SmileModel = "sabr",
        sabr_beta: float = 1.0,
    ) -> None:
        if model not in ("sabr", "vanna_volga"):
            raise ValueError(f"unknown smile model {model!r}")
        self.spot, self.T, self.rd, self.rf = spot, T, rd, rf
        self.quote, self.convention, self.model = quote, convention, model
        self.sabr_beta = sabr_beta
        self.forward = float(gk.forward(spot, T, rd, rf))
        self.delta_type = convention.delta_type(T)

        if quote.strangle == "smile":
            self.smile_bf25 = quote.bf25
            self.market_strangle_strikes = None
        else:
            self.smile_bf25, self.market_strangle_strikes = self._solve_broker_strangle()
        self.pillar_strikes, self.pillar_vols, self._interp = self._fit(self.smile_bf25)

    def _strike(self, delta: float, sigma: float, call: bool) -> float:
        return strike_from_delta(
            delta, self.spot, self.T, self.rd, self.rf, sigma, call, self.delta_type
        )

    def _fit(self, bf_smile: float) -> tuple[np.ndarray, np.ndarray, SABRSmile | VannaVolgaSmile]:
        q = self.quote
        vols = np.array([q.atm + bf_smile - 0.5 * q.rr25, q.atm, q.atm + bf_smile + 0.5 * q.rr25])
        strikes = np.array(
            [
                self._strike(-0.25, vols[0], call=False),
                atm_strike(
                    self.spot, self.T, self.rd, self.rf, q.atm,
                    self.convention.atm_type, self.delta_type,
                ),
                self._strike(0.25, vols[2], call=True),
            ]
        )
        if self.model == "sabr":
            params = calibrate_sabr(self.forward, self.T, strikes, vols, beta=self.sabr_beta)
            return strikes, vols, SABRSmile(self.forward, self.T, params)
        return strikes, vols, VannaVolgaSmile(self.forward, self.T, strikes, vols)

    def _gk(self, K: float, sigma: float, call: bool) -> float:
        return float(gk.price(self.spot, K, self.T, self.rd, self.rf, sigma, call))

    def _solve_broker_strangle(self) -> tuple[float, tuple[float, float]]:
        q = self.quote
        vol_ms = q.atm + q.bf25
        k_put, k_call = self._strike(-0.25, vol_ms, False), self._strike(0.25, vol_ms, True)
        target = self._gk(k_put, vol_ms, False) + self._gk(k_call, vol_ms, True)

        def premium_gap(bf_smile: float) -> float:
            _, _, interp = self._fit(bf_smile)
            vp, vc = interp.vol(np.array([k_put, k_call]))
            return self._gk(k_put, vp, False) + self._gk(k_call, vc, True) - target

        sol = root_scalar(
            premium_gap, x0=q.bf25, x1=q.bf25 + 5e-4, method="secant", xtol=1e-12, maxiter=100
        )
        if not sol.converged:
            raise RuntimeError("market-strangle calibration did not converge")
        return float(sol.root), (k_put, k_call)

    def vol(self, K: ArrayLike) -> np.ndarray:
        return self._interp.vol(K)

    def price(self, K: ArrayLike, call: ArrayLike = True) -> np.ndarray:
        return gk.price(self.spot, K, self.T, self.rd, self.rf, self.vol(K), call)

    def strike_for_delta(self, delta: float, call: bool, tol: float = 1e-12) -> float:
        """Smile-consistent strike: solves ``K = K(delta, vol(K))`` by fixed-point iteration."""
        k = self.forward
        for _ in range(200):
            k_new = self._strike(delta, float(self.vol(k)), call)
            if abs(k_new - k) < tol * k:
                return k_new
            k = k_new
        raise RuntimeError("delta-to-strike fixed point did not converge")

    def risk_neutral_density(self, strikes: ArrayLike) -> np.ndarray:
        """Breeden-Litzenberger density ``exp(rd T) d2C/dK2`` by central differences."""
        K = np.asarray(strikes, dtype=float)
        h = 1e-4 * self.forward
        c = [self.price(K + s * h, True) for s in (-1.0, 0.0, 1.0)]
        return np.exp(self.rd * self.T) * (c[0] - 2.0 * c[1] + c[2]) / h**2

    def strike_grid(self, n: int = 201, n_std: float = 4.0) -> np.ndarray:
        sd = self.quote.atm * np.sqrt(self.T)
        return self.forward * np.exp(np.linspace(-n_std * sd, n_std * sd, n))

    def is_butterfly_arbitrage_free(self, n: int = 201, n_std: float = 4.0) -> bool:
        return bool(np.all(self.risk_neutral_density(self.strike_grid(n, n_std)) >= -1e-8))
