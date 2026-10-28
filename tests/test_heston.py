import numpy as np
import pytest

from deltasigma.pricing import price
from deltasigma.sim import (
    HestonParams,
    RegimeSwitchingHeston,
    eurusd_calm_stress_model,
    heston_char_func,
    heston_price,
    simulate_heston,
)

S, T, RD, RF = 1.085, 0.5, 0.043, 0.025
PARAMS = HestonParams(kappa=2.0, theta=0.01, xi=0.3, rho=-0.3, v0=0.012)


def test_char_func_normalised_and_martingale():
    assert heston_char_func(0.0, S, T, RD, RF, PARAMS) == pytest.approx(1.0)
    expected_fwd = S * np.exp((RD - RF) * T)
    assert heston_char_func(-1j, S, T, RD, RF, PARAMS).real == pytest.approx(expected_fwd)


def test_heston_collapses_to_black_scholes():
    flat = HestonParams(kappa=2.0, theta=0.01, xi=1e-4, rho=0.0, v0=0.01)
    for K in (1.0, 1.085, 1.2):
        assert heston_price(S, K, T, RD, RF, flat) == pytest.approx(
            float(price(S, K, T, RD, RF, 0.1)), abs=1e-7
        )


def test_heston_put_call_parity():
    c = heston_price(S, 1.1, T, RD, RF, PARAMS, True)
    p = heston_price(S, 1.1, T, RD, RF, PARAMS, False)
    assert c - p == pytest.approx(S * np.exp(-RF * T) - 1.1 * np.exp(-RD * T), abs=1e-12)


def test_negative_correlation_produces_put_skew():
    from deltasigma.pricing import implied_vol

    ivs = [implied_vol(heston_price(S, K, T, RD, RF, PARAMS), S, K, T, RD, RF) for K in (1.0, 1.2)]
    assert ivs[0] > ivs[1]


@pytest.mark.parametrize("K", [1.0, 1.085, 1.18])
def test_qe_monte_carlo_matches_semi_analytic(K):
    sim = simulate_heston(S, PARAMS, T, n_steps=50, n_paths=100_000, rd=RD, rf=RF, seed=7)
    payoff = np.exp(-RD * T) * np.maximum(sim.spot[:, -1] - K, 0.0)
    se = payoff.std() / np.sqrt(payoff.size)
    assert abs(payoff.mean() - heston_price(S, K, T, RD, RF, PARAMS)) < 4 * se


def test_qe_handles_feller_violation_without_negative_variance():
    harsh = HestonParams(kappa=1.0, theta=0.005, xi=1.0, rho=-0.7, v0=0.005)
    assert not harsh.feller_satisfied
    sim = simulate_heston(S, harsh, 1.0, 100, 20_000, RD, RF, seed=1)
    assert sim.variance.min() >= 0.0
    assert np.isfinite(sim.spot).all()


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        HestonParams(kappa=1.0, theta=0.01, xi=0.3, rho=-1.0, v0=0.01)


def test_regime_occupancy_matches_markov_chain():
    model = eurusd_calm_stress_model()
    horizon = 2.0
    res = model.simulate(S, horizon, n_steps=200, n_paths=20_000, seed=11)
    p_stress = model.transition_matrix(horizon)[0, 1]
    assert (res.regime[:, -1] == 1).mean() == pytest.approx(p_stress, abs=0.015)


def test_stress_regime_is_more_volatile():
    model = eurusd_calm_stress_model()
    calm = model.simulate(S, 0.25, 63, 5_000, start_regime=0, seed=2)
    stress = model.simulate(S, 0.25, 63, 5_000, start_regime=1, seed=2)
    assert stress.realized_vol.mean() > 1.5 * calm.realized_vol.mean()


def test_simulation_is_reproducible():
    model = eurusd_calm_stress_model()
    a = model.simulate(S, 0.1, 10, 100, seed=5)
    b = model.simulate(S, 0.1, 10, 100, seed=5)
    np.testing.assert_array_equal(a.spot, b.spot)
    np.testing.assert_array_equal(a.regime, b.regime)


def test_bad_generator_rejected():
    with pytest.raises(ValueError):
        RegimeSwitchingHeston([PARAMS, PARAMS], np.array([[-1.0, 0.5], [1.0, -1.0]]), RD, RF)
