import numpy as np
import pytest

from deltasigma.pricing import forward, greeks, implied_vol, price

S, T, RD, RF = 1.0850, 0.5, 0.043, 0.025
STRIKES = np.array([0.95, 1.05, 1.0850, 1.12, 1.25])


@pytest.mark.parametrize("sigma", [0.03, 0.08, 0.25])
def test_put_call_parity(sigma):
    c = price(S, STRIKES, T, RD, RF, sigma, True)
    p = price(S, STRIKES, T, RD, RF, sigma, False)
    np.testing.assert_allclose(c - p, S * np.exp(-RF * T) - STRIKES * np.exp(-RD * T), atol=1e-14)


def test_forward_is_covered_interest_parity():
    assert forward(S, T, RD, RF) == pytest.approx(S * np.exp((RD - RF) * T))


def test_zero_vol_limit_is_discounted_intrinsic():
    c = price(S, STRIKES, T, RD, RF, 1e-8, True)
    intrinsic = np.exp(-RD * T) * np.maximum(forward(S, T, RD, RF) - STRIKES, 0.0)
    np.testing.assert_allclose(c, intrinsic, atol=1e-12)


@pytest.mark.parametrize("call", [True, False])
def test_greeks_match_finite_differences(call):
    K, sigma = 1.10, 0.09
    g = greeks(S, K, T, RD, RF, sigma, call)

    def px(s=S, t=T, rd=RD, rf=RF, vol=sigma):
        return float(price(s, K, t, rd, rf, vol, call))

    h_s, h_v, h_r, h_t = 1e-4, 1e-5, 1e-6, 1e-6
    assert g.spot_delta == pytest.approx((px(s=S + h_s) - px(s=S - h_s)) / (2 * h_s), rel=1e-6)
    assert g.gamma == pytest.approx(
        (px(s=S + h_s) - 2 * px() + px(s=S - h_s)) / h_s**2, rel=1e-4
    )
    assert g.vega == pytest.approx(
        (px(vol=sigma + h_v) - px(vol=sigma - h_v)) / (2 * h_v), rel=1e-6
    )
    assert g.rho_d == pytest.approx((px(rd=RD + h_r) - px(rd=RD - h_r)) / (2 * h_r), rel=1e-6)
    assert g.rho_f == pytest.approx((px(rf=RF + h_r) - px(rf=RF - h_r)) / (2 * h_r), rel=1e-6)
    assert g.theta == pytest.approx(-(px(t=T + h_t) - px(t=T - h_t)) / (2 * h_t), rel=1e-5)

    def delta(vol):
        return float(greeks(S, K, T, RD, RF, vol, call).spot_delta)

    def vega(vol):
        return float(greeks(S, K, T, RD, RF, vol, call).vega)

    assert g.vanna == pytest.approx((delta(sigma + h_v) - delta(sigma - h_v)) / (2 * h_v), rel=1e-5)
    assert g.volga == pytest.approx((vega(sigma + h_v) - vega(sigma - h_v)) / (2 * h_v), rel=1e-5)


def test_premium_adjusted_delta_subtracts_premium_in_foreign_units():
    K, sigma = 1.10, 0.09
    for call in (True, False):
        g = greeks(S, K, T, RD, RF, sigma, call)
        assert g.spot_delta_pa == pytest.approx(g.spot_delta - g.price / S, abs=1e-14)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("K", [0.9, 1.0850, 1.3])
@pytest.mark.parametrize("sigma", [0.02, 0.1, 0.6])
def test_implied_vol_round_trip(call, K, sigma):
    target = float(price(S, K, T, RD, RF, sigma, call))
    iv = implied_vol(target, S, K, T, RD, RF, call)
    assert float(price(S, K, T, RD, RF, iv, call)) == pytest.approx(target, abs=1e-13)
    # Deep ITM / low-vol options have ~zero vega, so vol is identifiable only when vega is material.
    if float(greeks(S, K, T, RD, RF, sigma, call).vega) > 1e-6:
        assert iv == pytest.approx(sigma, abs=1e-8)


def test_implied_vol_rejects_arbitrageable_price():
    with pytest.raises(ValueError):
        implied_vol(-0.01, S, 1.1, T, RD, RF)
