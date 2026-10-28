import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from deltasigma.pricing import CONVENTIONS, DeltaType, fx_delta  # noqa: E402
from deltasigma.pricing import garman_kohlhagen as gk  # noqa: E402
from deltasigma.service.app import create_app, default_quoter  # noqa: E402

MARKET = {"spot": 1.085, "rd": 0.043, "rf": 0.025, "tenor_years": 0.25, "atm": 0.072,
          "rr25": -0.0042, "bf25": 0.002}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(default_quoter(n_episodes=40), {"quoter": "test"}))


def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["quoter"] == "test" and info["markout_scale"] == 0.25


def test_price_by_delta_returns_consistent_strike(client):
    r = client.post("/price", json={**MARKET, "delta": -0.25, "call": False, "notional": 1e6})
    assert r.status_code == 200 and "X-Process-Time-ms" in r.headers
    body = r.json()
    d = fx_delta(1.085, body["strike"], 0.25, 0.043, 0.025, body["vol"], False, DeltaType.SPOT)
    assert d == pytest.approx(-0.25, abs=1e-9)
    expected = float(gk.price(1.085, body["strike"], 0.25, 0.043, 0.025, body["vol"], False))
    assert body["premium"] == pytest.approx(expected * 1e6)
    assert body["arbitrage_free"] and body["greeks"]["vega"] > 0


def test_put_call_parity_through_the_api(client):
    c = client.post("/price", json={**MARKET, "strike": 1.10, "call": True}).json()
    p = client.post("/price", json={**MARKET, "strike": 1.10, "call": False}).json()
    parity = 1.085 * np.exp(-0.025 * 0.25) - 1.10 * np.exp(-0.043 * 0.25)
    assert c["premium"] - p["premium"] == pytest.approx(parity, abs=1e-12)


def test_premium_adjusted_pair(client):
    body = {**MARKET, "pair": "USDJPY", "spot": 150.0, "rd": 0.005, "rf": 0.043,
            "atm": 0.10, "rr25": -0.015, "bf25": 0.004, "delta": 0.25, "call": True}
    r = client.post("/price", json=body).json()
    d = fx_delta(150.0, r["strike"], 0.25, 0.005, 0.043, r["vol"], True,
                 CONVENTIONS["USDJPY"].delta_type(0.25))
    assert d == pytest.approx(0.25, abs=1e-9)


@pytest.mark.parametrize("bad", [
    {"strike": 1.1, "delta": 0.25},          # both
    {},                                       # neither
    {"delta": 0.25, "call": False},           # sign mismatch
    {"delta": 0.25, "atm": -0.1},             # negative vol
    {"delta": 0.25, "pair": "XXXYYY"},        # unknown pair
])
def test_price_validation_errors(client, bad):
    assert client.post("/price", json={**MARKET, **bad}).status_code == 422


def test_quote_skews_with_inventory_and_is_consistent(client):
    base = {"tier": 0, "size": 3, "mid_vol": 7.5}
    sell_more = client.post("/quote", json={**base, "side": 1, "inventory": -10}).json()
    buy_back = client.post("/quote", json={**base, "side": -1, "inventory": -10}).json()
    assert sell_more["edge_vol_pts"] > buy_back["edge_vol_pts"]
    assert sell_more["quoted_vol"] == pytest.approx(7.5 + sell_more["edge_vol_pts"])
    assert buy_back["quoted_vol"] == pytest.approx(7.5 - buy_back["edge_vol_pts"])
    for q in (sell_more, buy_back):
        assert 0.0 <= q["win_probability"] <= 1.0


def test_informed_tier_is_quoted_wider_against_its_flow(client):
    base = {"side": 1, "size": 3, "mid_vol": 7.5, "inventory": 0}
    with_flow = client.post("/quote", json={**base, "tier": 1, "flow_tier1": 0.8}).json()
    against = client.post("/quote", json={**base, "tier": 1, "flow_tier1": -0.8}).json()
    assert with_flow["edge_vol_pts"] >= against["edge_vol_pts"]


def test_quote_validation(client):
    assert client.post("/quote", json={"tier": 2, "side": 1, "size": 1,
                                       "mid_vol": 7}).status_code == 422
