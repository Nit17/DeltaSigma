import numpy as np
import pytest

from deltasigma.pricing import (
    CONVENTIONS,
    AtmType,
    DeltaType,
    atm_strike,
    fx_delta,
    greeks,
    strike_from_delta,
)

S, T, RD, RF, SIGMA = 150.0, 0.5, 0.043, 0.005, 0.10


@pytest.mark.parametrize("delta_type", list(DeltaType))
@pytest.mark.parametrize("delta,call", [(0.25, True), (0.10, True), (-0.25, False), (-0.10, False)])
def test_strike_from_delta_round_trip(delta_type, delta, call):
    K = strike_from_delta(delta, S, T, RD, RF, SIGMA, call, delta_type)
    assert fx_delta(S, K, T, RD, RF, SIGMA, call, delta_type) == pytest.approx(delta, abs=1e-12)


def test_premium_adjusted_call_strike_is_lower():
    k_pa = strike_from_delta(0.25, S, T, RD, RF, SIGMA, True, DeltaType.SPOT_PA)
    k_plain = strike_from_delta(0.25, S, T, RD, RF, SIGMA, True, DeltaType.SPOT)
    assert k_pa < k_plain


def test_premium_adjusted_call_takes_the_otm_root():
    k = strike_from_delta(0.25, S, T, RD, RF, SIGMA, True, DeltaType.FORWARD_PA)
    assert k > S * np.exp((RD - RF) * T)


def test_unattainable_premium_adjusted_call_delta_raises():
    with pytest.raises(ValueError):
        strike_from_delta(0.99, S, T, RD, RF, SIGMA, True, DeltaType.FORWARD_PA)


def test_wrong_delta_sign_raises():
    with pytest.raises(ValueError):
        strike_from_delta(0.25, S, T, RD, RF, SIGMA, False, DeltaType.SPOT)


@pytest.mark.parametrize("delta_type", list(DeltaType))
def test_dns_strike_is_delta_neutral(delta_type):
    K = atm_strike(S, T, RD, RF, SIGMA, AtmType.DNS, delta_type)
    straddle = fx_delta(S, K, T, RD, RF, SIGMA, True, delta_type) + fx_delta(
        S, K, T, RD, RF, SIGMA, False, delta_type
    )
    assert straddle == pytest.approx(0.0, abs=1e-12)


def test_fx_delta_matches_analytic_greeks():
    g = greeks(S, 155.0, T, RD, RF, SIGMA, True)
    assert fx_delta(S, 155.0, T, RD, RF, SIGMA, True, DeltaType.SPOT) == pytest.approx(
        float(g.spot_delta)
    )
    assert fx_delta(S, 155.0, T, RD, RF, SIGMA, True, DeltaType.FORWARD_PA) == pytest.approx(
        float(g.forward_delta_pa)
    )


def test_pair_conventions_switch_to_forward_delta_after_one_year():
    eur = CONVENTIONS["EURUSD"]
    assert eur.delta_type(0.5) is DeltaType.SPOT
    assert eur.delta_type(2.0) is DeltaType.FORWARD
    assert CONVENTIONS["USDJPY"].delta_type(0.5).premium_adjusted
