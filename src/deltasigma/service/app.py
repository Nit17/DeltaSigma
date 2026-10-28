"""FastAPI pricing and RFQ-quoting service.

Endpoints
    GET  /health       liveness + version
    GET  /model-info   quoter configuration and provenance
    POST /price        smile-consistent FX vanilla price and Greeks from ATM/RR/BF quotes
    POST /quote        RFQ quote (edge, quoted vol, win probability, expected P&L)

Run locally:  uvicorn deltasigma.service.app:app --port 8000
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from deltasigma import __version__
from deltasigma.mm import MarketConfig, ModelQuoter, WinProbabilityModel
from deltasigma.mm.winprob import collect_rfq_history, estimate_flow_markout
from deltasigma.pricing import CONVENTIONS
from deltasigma.pricing import garman_kohlhagen as gk
from deltasigma.vol import FXSmile, SmileQuote


class PriceRequest(BaseModel):
    pair: Literal[tuple(CONVENTIONS)] = "EURUSD"  # type: ignore[valid-type]
    spot: float = Field(gt=0)
    rd: float = Field(ge=-0.1, le=0.5, description="domestic rate, continuous")
    rf: float = Field(ge=-0.1, le=0.5, description="foreign rate, continuous")
    tenor_years: float = Field(gt=0, le=10)
    atm: float = Field(gt=0, lt=2, description="ATM vol, decimal")
    rr25: float = Field(ge=-0.5, le=0.5)
    bf25: float = Field(ge=-0.1, le=0.5)
    call: bool = True
    strike: float | None = Field(default=None, gt=0)
    delta: float | None = Field(default=None, gt=-1, lt=1, description="e.g. 0.25 / -0.10")
    notional: float = 1.0
    model: Literal["sabr", "vanna_volga"] = "sabr"

    @model_validator(mode="after")
    def _strike_or_delta(self) -> PriceRequest:
        if (self.strike is None) == (self.delta is None):
            raise ValueError("provide exactly one of strike or delta")
        if self.delta is not None and (self.delta > 0) != self.call:
            raise ValueError("call deltas must be positive and put deltas negative")
        return self


class PriceResponse(BaseModel):
    forward: float
    strike: float
    vol: float
    premium: float
    premium_pct_foreign: float
    greeks: dict[str, float]
    arbitrage_free: bool


class QuoteRequest(BaseModel):
    tier: Literal[0, 1]
    side: Literal[-1, 1] = Field(description="+1 client buys vol, -1 client sells vol")
    size: float = Field(gt=0, le=50, description="vega units")
    inventory: float = Field(default=0.0, description="dealer vega inventory (vega units)")
    mid_vol: float = Field(gt=0, lt=100, description="ATM mid vol in vol points")
    vol_of_vol: float = Field(default=0.05, gt=0, description="recent |d vol| per step, vol pts")
    flow_tier1: float = Field(default=0.0, ge=-1, le=1)


class QuoteResponse(BaseModel):
    edge_vol_pts: float
    quoted_vol: float
    win_probability: float
    expected_pnl: float


@lru_cache(maxsize=512)
def calibrated_smile(pair: str, spot: float, T: float, rd: float, rf: float, atm: float,
                     rr25: float, bf25: float, model: str) -> tuple[FXSmile, bool]:
    """Smile calibration and its arbitrage check are the expensive steps (~tens of ms);
    requests against the same market snapshot reuse them, so only the first request
    after a market update pays."""
    smile = FXSmile(spot, T, rd, rf, SmileQuote(atm, rr25, bf25), CONVENTIONS[pair], model=model)
    return smile, smile.is_butterfly_arbitrage_free()


def default_quoter(seed: int = 1, n_episodes: int = 150) -> ModelQuoter:
    """Classical quoter fitted on simulated RFQ history (the Phase 4 tuned B4 setup)."""
    cfg = MarketConfig()
    history = collect_rfq_history(cfg, n_episodes, seed=seed)
    win = WinProbabilityModel().fit(history)
    markout, flow_coef = estimate_flow_markout(history)
    return ModelQuoter(win, markout, inventory_aversion=0.05, band=5.0, flow_coef=flow_coef,
                       markout_scale=0.25)


def create_app(quoter: ModelQuoter | None = None, model_info: dict | None = None) -> FastAPI:
    app = FastAPI(title="DeltaSigma pricing & quoting", version=__version__)
    app.state.quoter = quoter or default_quoter()
    app.state.model_info = model_info or {"quoter": "B4 classical (default fit at startup)"}

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{1e3 * (time.perf_counter() - t0):.3f}"
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/model-info")
    def model_info_endpoint() -> dict:
        q = app.state.quoter
        return {
            **app.state.model_info,
            "inventory_aversion": q.inventory_aversion,
            "markout_scale": q.markout_scale,
            "flow_coef": q.flow_coef,
            "hedge_band": q.band,
        }

    @app.post("/price", response_model=PriceResponse)
    def price(req: PriceRequest) -> PriceResponse:
        try:
            smile, arb_free = calibrated_smile(req.pair, req.spot, req.tenor_years, req.rd,
                                               req.rf, req.atm, req.rr25, req.bf25, req.model)
            K = req.strike if req.strike is not None else smile.strike_for_delta(req.delta,
                                                                                req.call)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        vol = float(smile.vol(K))
        g = gk.greeks(req.spot, K, req.tenor_years, req.rd, req.rf, vol, req.call)
        n = req.notional
        greeks = {k: float(getattr(g, k)) * n for k in
                  ("spot_delta", "forward_delta", "spot_delta_pa", "gamma", "vega", "vanna",
                   "volga", "theta", "rho_d", "rho_f")}
        prem = float(g.price) * n
        return PriceResponse(forward=smile.forward, strike=float(K), vol=vol, premium=prem,
                             premium_pct_foreign=prem / (req.spot * n) * 100 if n else 0.0,
                             greeks=greeks, arbitrage_free=arb_free)

    @app.post("/quote", response_model=QuoteResponse)
    def quote(req: QuoteRequest) -> QuoteResponse:
        q: ModelQuoter = app.state.quoter
        vol_level, vov = (req.mid_vol - 8.0) / 3.0, req.vol_of_vol / 0.05 - 1.0
        edge = float(q.quote_state(np.array([req.tier]), np.array([req.side]),
                                   np.array([req.size]), np.array([req.inventory]),
                                   vol_level, vov, np.array([req.flow_tier1]))[0])
        p_win = float(q.win_model.predict_proba(edge, req.tier, vol_level, vov))
        cost = float(q.unit_cost(np.array([req.tier]), req.side, req.size, req.inventory,
                                 req.flow_tier1)[0])
        return QuoteResponse(edge_vol_pts=edge, quoted_vol=req.mid_vol + req.side * edge,
                             win_probability=p_win,
                             expected_pnl=p_win * req.size * (edge - cost))

    return app


app = create_app()
