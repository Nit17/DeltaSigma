import numpy as np
import pytest
from scipy.integrate import trapezoid

from deltasigma.pricing import CONVENTIONS
from deltasigma.pricing import garman_kohlhagen as gk
from deltasigma.vol import (
    FXSmile,
    SmileQuote,
    VannaVolgaSmile,
    calibrate_sabr,
    sabr_vol,
    sample_eurusd_surface,
)

F, T = 1.10, 0.5


def test_sabr_atm_continuity():
    k = F * np.array([1 - 1e-9, 1.0, 1 + 1e-9])
    vols = sabr_vol(F, k, T, alpha=0.08, beta=1.0, rho=-0.3, nu=0.6)
    assert np.ptp(vols) < 1e-9


def test_sabr_zero_volvol_is_flat_for_lognormal_beta():
    k = F * np.exp(np.linspace(-0.2, 0.2, 9))
    np.testing.assert_allclose(sabr_vol(F, k, T, 0.08, 1.0, 0.0, 1e-8), 0.08, atol=1e-9)


def test_sabr_calibration_recovers_parameters():
    k = F * np.exp(np.linspace(-0.15, 0.15, 11))
    vols = sabr_vol(F, k, T, alpha=0.075, beta=1.0, rho=-0.25, nu=0.7)
    p = calibrate_sabr(F, T, k, vols, beta=1.0)
    assert (p.alpha, p.rho, p.nu) == pytest.approx((0.075, -0.25, 0.7), abs=1e-6)


def test_vanna_volga_reproduces_pillars():
    strikes, vols = np.array([1.05, 1.10, 1.15]), np.array([0.085, 0.075, 0.080])
    vv = VannaVolgaSmile(F, T, strikes, vols)
    np.testing.assert_allclose(vv.vol(strikes), vols, atol=1e-12)


def test_vanna_volga_requires_ordered_strikes():
    with pytest.raises(ValueError):
        VannaVolgaSmile(F, T, [1.1, 1.05, 1.15], [0.08, 0.07, 0.08])


@pytest.fixture(params=["sabr", "vanna_volga"])
def model(request):
    return request.param


@pytest.mark.parametrize("pair,spot,rd,rf", [("EURUSD", 1.085, 0.043, 0.025),
                                             ("USDJPY", 150.0, 0.005, 0.043)])
def test_smile_hits_quoted_pillars(model, pair, spot, rd, rf):
    quote = SmileQuote(atm=0.09, rr25=-0.012, bf25=0.003)
    smile = FXSmile(spot, 0.5, rd, rf, quote, CONVENTIONS[pair], model=model)
    np.testing.assert_allclose(smile.vol(smile.pillar_strikes), smile.pillar_vols, atol=1e-8)
    p_vol, atm, c_vol = smile.pillar_vols
    assert c_vol - p_vol == pytest.approx(quote.rr25, abs=1e-12)
    assert 0.5 * (c_vol + p_vol) - atm == pytest.approx(smile.smile_bf25, abs=1e-12)


def test_broker_strangle_premium_is_matched(model):
    quote = SmileQuote(atm=0.09, rr25=-0.02, bf25=0.004)
    spot, T_, rd, rf = 1.085, 0.5, 0.043, 0.025
    smile = FXSmile(spot, T_, rd, rf, quote, CONVENTIONS["EURUSD"], model=model)
    k_put, k_call = smile.market_strangle_strikes
    vol_ms = quote.atm + quote.bf25
    target = gk.price(spot, k_put, T_, rd, rf, vol_ms, False) + gk.price(
        spot, k_call, T_, rd, rf, vol_ms, True
    )
    on_smile = smile.price(k_put, False) + smile.price(k_call, True)
    assert float(on_smile) == pytest.approx(float(target), abs=1e-12)
    # With a skew the smile strangle differs from the broker strangle.
    assert smile.smile_bf25 != pytest.approx(quote.bf25, abs=1e-6)


def test_smile_strangle_quote_is_used_directly():
    quote = SmileQuote(atm=0.09, rr25=-0.01, bf25=0.003, strangle="smile")
    smile = FXSmile(1.085, 0.5, 0.043, 0.025, quote, CONVENTIONS["EURUSD"])
    assert smile.smile_bf25 == quote.bf25
    assert smile.market_strangle_strikes is None


def test_negative_risk_reversal_gives_put_skew(model):
    smile = FXSmile(1.085, 0.5, 0.043, 0.025, SmileQuote(0.08, -0.01, 0.002),
                    CONVENTIONS["EURUSD"], model=model)
    k_put, k_atm, k_call = smile.pillar_strikes
    assert smile.vol(k_put * 0.97) > smile.vol(k_call * 1.03)


def test_strike_for_delta_is_smile_consistent(model):
    smile = FXSmile(1.085, 0.5, 0.043, 0.025, SmileQuote(0.08, -0.01, 0.002),
                    CONVENTIONS["EURUSD"], model=model)
    k = smile.strike_for_delta(0.25, call=True)
    assert k == pytest.approx(smile.pillar_strikes[2], rel=1e-10)


def test_density_integrates_to_one_and_is_positive(model):
    smile = sample_eurusd_surface(model).smiles[2]
    k = smile.strike_grid(n=801, n_std=7.0)
    dens = smile.risk_neutral_density(k)
    assert smile.is_butterfly_arbitrage_free()
    assert trapezoid(dens, k) == pytest.approx(1.0, abs=2e-3)
    assert trapezoid(dens * k, k) == pytest.approx(smile.forward, rel=1e-3)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        FXSmile(1.085, 0.5, 0.043, 0.025, SmileQuote(0.08, 0.0, 0.002),
                CONVENTIONS["EURUSD"], model="svi")
