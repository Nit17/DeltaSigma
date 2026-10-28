import numpy as np
import pytest

from deltasigma.hedging import (
    CostModel,
    OptionBook,
    VanillaOption,
    black_delta,
    cvar,
    delta_vega,
    generate_market,
    hedge_outcome,
    no_hedge,
    summarize,
    whalley_wilmott,
)
from deltasigma.hedging.market import remaining_variance_weights
from deltasigma.pricing import garman_kohlhagen as gk
from deltasigma.sim import HestonParams, RegimeSwitchingHeston, eurusd_calm_stress_model

S0, T, N = 1.085, 0.25, 21
MODEL = eurusd_calm_stress_model()
F0 = S0 * np.exp((MODEL.rd - MODEL.rf) * T)
BOOK = OptionBook.of([VanillaOption(F0, call=True)])


@pytest.fixture(scope="module")
def market():
    return generate_market(MODEL, BOOK, S0, T, N, 40_000, seed=3)


def _near_bs_model() -> RegimeSwitchingHeston:
    flat = HestonParams(kappa=2.0, theta=0.01, xi=1e-4, rho=0.0, v0=0.01)
    return RegimeSwitchingHeston([flat], np.zeros((1, 1)), MODEL.rd, MODEL.rf)


def test_variance_swap_and_forward_are_martingales(market):
    for prices in (market.variance_swap, market.forward):
        d = prices[:, -1] - prices[:, 0]
        assert abs(d.mean()) < 4 * d.std() / np.sqrt(d.size)


def test_variance_swap_settles_on_realised_variance(market):
    dt = T / N
    realised = dt * market.variance[:, :-1].sum(axis=1)
    np.testing.assert_allclose(market.variance_swap[:, -1], realised, rtol=1e-12)


def test_expected_variance_matches_single_regime_closed_form():
    p = HestonParams(kappa=3.0, theta=0.006, xi=0.3, rho=0.0, v0=0.01)
    model = RegimeSwitchingHeston([p], np.zeros((1, 1)), 0.0, 0.0)
    dt = T / N
    W = remaining_variance_weights(model, dt, N)
    expected = dt * sum(p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * j * dt) for j in range(N))
    assert W[N] @ np.array([p.v0, 1.0]) == pytest.approx(expected, rel=1e-12)


def test_regime_aware_implied_vol_is_higher_in_stress():
    calm = generate_market(MODEL, BOOK, S0, T, N, 10, start_regime=0, seed=0)
    stress = generate_market(MODEL, BOOK, S0, T, N, 10, start_regime=1, seed=0)
    assert stress.implied_vol[0, 0] > 1.5 * calm.implied_vol[0, 0]


def test_cost_free_delta_hedge_replicates_in_black_scholes_world():
    m = generate_market(_near_bs_model(), BOOK, S0, T, 252, 20_000, seed=1)
    out = hedge_outcome(m, black_delta(m), CostModel(0.0, 0.0))
    bs_price = float(gk.price(F0, F0, T, 0.0, 0.0, 0.1, True))
    assert out.loss.mean() == pytest.approx(bs_price, abs=3e-5)
    assert out.loss.std() < 0.1 * hedge_outcome(m, no_hedge(m), CostModel(0, 0)).loss.std()


def test_hedging_error_shrinks_with_rebalancing_frequency():
    stds = []
    for n_steps in (13, 52, 208):
        m = generate_market(_near_bs_model(), BOOK, S0, T, n_steps, 20_000, seed=2)
        stds.append(hedge_outcome(m, black_delta(m), CostModel(0, 0)).loss.std())
    # Discrete hedging error scales like 1 / sqrt(n_steps).
    assert stds[0] / stds[1] == pytest.approx(2.0, rel=0.15)
    assert stds[1] / stds[2] == pytest.approx(2.0, rel=0.15)


def test_costs_are_charged_on_every_trade():
    m = generate_market(MODEL, BOOK, S0, T, 3, 5, seed=0)
    pos = np.zeros((5, 3, 2))
    pos[:, :, 0] = [1.0, 0.5, 0.5]
    out = hedge_outcome(m, pos, CostModel(forward_bps=10.0, varswap_vol_pts=0.0))
    expected = 1e-3 * (1.0 * m.forward[:, 0] + 0.5 * m.forward[:, 1])
    np.testing.assert_allclose(out.costs, expected, rtol=1e-12)
    np.testing.assert_allclose(out.turnover[:, 0], 1.5)


def test_whalley_wilmott_collapses_to_delta_hedge(market):
    costs = CostModel()
    np.testing.assert_allclose(whalley_wilmott(market, costs, 1e30), black_delta(market),
                               atol=1e-8)


def test_whalley_wilmott_trades_less(market):
    costs = CostModel()
    band = hedge_outcome(market, whalley_wilmott(market, costs, 1e4), costs)
    delta = hedge_outcome(market, black_delta(market), costs)
    assert band.turnover[:, 0].mean() < delta.turnover[:, 0].mean()
    assert band.costs.mean() < delta.costs.mean()


def test_vega_hedge_reduces_risk_without_costs(market):
    free = CostModel(0.0, 0.0)
    delta = hedge_outcome(market, black_delta(market), free).loss
    dv = hedge_outcome(market, delta_vega(market), free).loss
    assert dv.std() < 0.7 * delta.std()
    assert cvar(dv) < cvar(delta)


def test_cvar_matches_tail_mean():
    x = np.arange(1, 101, dtype=float)
    assert cvar(x, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert cvar(x, 0.0) == pytest.approx(x.mean())


def test_summary_is_in_pips(market):
    s = summarize(hedge_outcome(market, no_hedge(market), CostModel()), market.fair_price)
    assert s["mean"] == pytest.approx(market.fair_price / 1e-4)
    assert s["risk_premium95"] > 0


def test_position_shape_validated(market):
    with pytest.raises(ValueError):
        hedge_outcome(market, np.zeros((market.n_paths, market.n_steps, 1)), CostModel())
