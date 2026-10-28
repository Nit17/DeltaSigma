import numpy as np
import pytest

pytest.importorskip("sklearn")

from deltasigma.mm import (  # noqa: E402
    FixedSpreadQuoter,
    MarketConfig,
    ModelQuoter,
    RFQMarket,
    WinProbabilityModel,
    band_hedge,
    calibration_report,
    collect_rfq_history,
    estimate_flow_markout,
    estimate_markout,
    evaluate,
    tune,
)

CFG = MarketConfig()


@pytest.fixture(scope="module")
def history():
    return collect_rfq_history(CFG, 300, seed=1)


@pytest.fixture(scope="module")
def win_model(history):
    return WinProbabilityModel().fit(history)


# ------------------------------------------------------------------ simulator
def test_episode_randomness_is_shared_across_policies():
    a = evaluate(FixedSpreadQuoter(0.1), CFG, 50, seed=3)
    b = evaluate(FixedSpreadQuoter(0.4), CFG, 50, seed=3)
    env1, env2 = RFQMarket(CFG, 50, seed=3), RFQMarket(CFG, 50, seed=3)
    env1.reset()
    env2.reset()
    np.testing.assert_array_equal(env1.vol, env2.vol)
    np.testing.assert_array_equal(env1.rfq_side, env2.rfq_side)
    assert a.episodes["fill_rate"].mean() > b.episodes["fill_rate"].mean()


def test_pnl_accounting_identity():
    env = RFQMarket(CFG, 64, seed=5)
    env.reset()
    for _ in range(CFG.n_steps):
        edge, hedge = FixedSpreadQuoter(0.2, band=4)(env)
        env.step(edge, hedge)
    s = env.stats
    np.testing.assert_allclose(env.pnl, s["spread"] + s["mtm"] - s["hedge_cost"], atol=1e-9)
    np.testing.assert_array_equal(env.q, 0.0)  # flat after end-of-day liquidation


def test_fill_frequency_matches_true_win_probability():
    env = RFQMarket(CFG, 4000, seed=6)
    env.reset()
    p_true = CFG.win_probability(np.full(4000, 0.2), env.rfq_tier[:, 0], env.regime[:, 0])
    _, _, _, info = env.step(np.full(4000, 0.2), np.zeros(4000))
    assert info["filled"].mean() == pytest.approx(p_true.mean(), abs=0.025)


def test_informed_flow_predicts_vol_moves():
    env = RFQMarket(CFG, 3000, seed=7)
    env.reset()
    H = CFG.informed_horizon
    move = np.sign(env.vol[:, H : H + CFG.n_steps] - env.vol[:, : CFG.n_steps])
    informed_t1 = env.rfq_informed & (env.rfq_tier == 1)
    assert np.all(env.rfq_side[informed_t1] == move[informed_t1])
    assert not env.rfq_informed[env.rfq_tier == 0].any()


def test_stress_start_and_jumps_raise_vol_risk():
    calm = RFQMarket(CFG, 2000, seed=8, start_regime=0)
    stress = RFQMarket(CFG, 2000, seed=8, start_regime=1)
    calm.reset()
    stress.reset()
    # Regimes can switch mid-episode, so the ratio is below the 2.5x per-regime ratio.
    assert np.diff(stress.vol, axis=1).std() > 1.3 * np.diff(calm.vol, axis=1).std()


def test_band_hedge():
    np.testing.assert_allclose(band_hedge(np.array([-7.0, 2.0, 9.0]), 5.0), [2.0, 0.0, -4.0])


# ------------------------------------------------------------------ win model & markout
def test_win_model_is_calibrated_to_ground_truth(win_model):
    holdout = collect_rfq_history(CFG, 150, seed=2)
    scores, table = calibration_report(win_model, holdout)
    assert scores["Mean |p_hat - p_true|"] < 0.03
    assert scores["Brier"] < scores["Brier (true probabilities)"] + 0.005
    assert np.allclose(table["predicted"], table["observed"], atol=0.05)


def test_optimal_edge_matches_closed_form_tradeoff(win_model):
    # The grid optimum must satisfy the first-order condition of (edge - c) * P(edge).
    edge = win_model.optimal_edge(np.array([0, 1]), 0.0, 0.0, cost=0.05)
    grid = np.linspace(-0.1, 1.0, 2001)
    for tier, e in zip((0, 1), edge, strict=True):
        p = win_model.predict_proba(grid, tier, 0.0, 0.0)
        assert e == pytest.approx(grid[np.argmax((grid - 0.05) * p)], abs=0.006)
    assert edge[1] < edge[0]  # price-sensitive tier gets a tighter quote


def test_markout_identifies_informed_tier(history):
    m = estimate_markout(history)
    assert abs(m[0]) < 0.03 and m[1] > 0.08
    base, coef = estimate_flow_markout(history)
    assert coef > 0


# ------------------------------------------------------------------ policies
def test_inventory_skew_direction(win_model):
    env = RFQMarket(CFG, 2, seed=9)
    env.reset()
    env.q = np.array([-10.0, -10.0])
    env.rfq_tier[:, 0] = 0
    env.rfq_side[:, 0] = [1.0, -1.0]  # client buys vol (dealer sells more) / client sells vol
    edge = ModelQuoter(win_model, inventory_aversion=0.05).quote(env)
    assert edge[0] > edge[1]  # widen when the fill adds to a short, tighten when it reduces it


def test_tuning_beats_untuned_on_training_objective(win_model):
    best, score = tune(lambda **p: ModelQuoter(win_model, **p),
                       {"inventory_aversion": [0.0, 0.05], "band": [3.0, np.inf]}, CFG, 100, 11)
    untuned = evaluate(ModelQuoter(win_model), CFG, 100, 11).risk_adjusted
    assert score >= untuned and np.isfinite(best["band"])


def test_ppo_residual_starts_at_baseline_and_trains(win_model):
    pytest.importorskip("torch")
    from deltasigma.mm import PPOConfig, PPOMarketMaker

    base = ModelQuoter(win_model, inventory_aversion=0.05, band=3.0)
    cfg = PPOConfig(n_envs=16, n_updates=2, eval_every=1, eval_episodes=32, seed=0)
    agent = PPOMarketMaker(base, CFG, cfg)
    env = RFQMarket(CFG, 32, seed=4)
    env.reset()
    edge_a, hedge_a = agent(env)
    edge_b, hedge_b = base(env)
    np.testing.assert_allclose(edge_a, edge_b, atol=0.01)
    np.testing.assert_allclose(hedge_a, hedge_b, atol=0.05)
    history = agent.train(train_seed=100, val_seed=21)
    assert len(history.val_score) == 2 and history.best_update in (0, 1)
    assert np.isfinite(evaluate(agent, CFG, 32, 5).risk_adjusted)
