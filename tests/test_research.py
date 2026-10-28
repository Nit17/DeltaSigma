import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from deltasigma.research import (  # noqa: E402
    HAR_COLUMNS,
    ColumnForecast,
    GaussianHMM,
    GBMForecast,
    LinearForecast,
    PurgedWalkForward,
    build_dataset,
    build_features,
    causal_regime_probability,
    check_no_lookahead,
    diebold_mariano,
    forecast_report,
    ladder_equity,
    purged_kfold,
    qlike,
    strategy_stats,
    synthetic_vol_dataset,
    timing_positions,
    vol_swap_pnl,
    walk_forward_predict,
)
from deltasigma.research.backtest import max_drawdown  # noqa: E402
from deltasigma.research.features import forward_realized_vol, log_returns  # noqa: E402

H = 21


@pytest.fixture(scope="module")
def raw():
    return synthetic_vol_dataset(n_days=2500, seed=3)


@pytest.fixture(scope="module")
def dataset(raw):
    return build_dataset(raw, H)


# --------------------------------------------------------------------------- data
def test_fred_loader_reads_cache_and_aligns_series(tmp_path):
    from deltasigma.research import load_eurusd_vol

    (tmp_path / "DEXUSEU.csv").write_text(
        "observation_date,DEXUSEU\n2020-01-02,1.1200\n2020-01-03,.\n2020-01-06,1.1190\n"
        "2020-01-07,1.1150\n"
    )
    (tmp_path / "EVZCLS.csv").write_text(
        "observation_date,EVZCLS\n2020-01-02,5.10\n2020-01-03,5.30\n2020-01-06,5.20\n"
        "2020-01-08,5.00\n"
    )
    df = load_eurusd_vol(cache_dir=tmp_path)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-06"]
    np.testing.assert_allclose(df["iv"], [0.051, 0.052])
    np.testing.assert_allclose(df["spot"], [1.12, 1.119])


# --------------------------------------------------------------------------- leakage
def test_features_have_no_lookahead(raw):
    assert check_no_lookahead(build_features, raw, n_checks=20) == []


def test_leakage_audit_catches_centred_window_and_full_sample_scaling(raw):
    def leaky(df):
        ret = log_returns(df["spot"])
        return pd.DataFrame({
            "centred_vol": ret.rolling(11, center=True).std(),
            "zscore_full_sample": (df["iv"] - df["iv"].mean()) / df["iv"].std(),
            "honest": ret.rolling(10).std(),
        })

    bad_cols = {col for _, col in check_no_lookahead(leaky, raw, n_checks=10)}
    assert bad_cols == {"centred_vol", "zscore_full_sample"}


def test_target_is_forward_realised_vol(raw):
    ret = log_returns(raw["spot"])
    rv = forward_realized_vol(ret, H)
    t = 500
    manual = np.sqrt(252 / H * np.sum(ret.iloc[t + 1 : t + 1 + H].to_numpy() ** 2))
    assert rv.iloc[t] == pytest.approx(manual, rel=1e-12)
    assert rv.iloc[-H:].isna().all()


def test_dataset_rows_are_complete(dataset):
    X, y, info = dataset
    assert X.notna().all().all() and y.notna().all()
    assert X.index.equals(y.index) and info.index.equals(y.index)


# --------------------------------------------------------------------------- CV
def test_walk_forward_purges_overlapping_labels():
    spl = PurgedWalkForward(horizon=H, min_train_size=200, test_size=50, embargo=3)
    seen_test = []
    for train, test in spl.split(600):
        # training label windows [i+1, i+H] must end before the test labels start
        assert train.max() + H <= test.min() - 3
        assert np.all(np.diff(test) == 1)
        seen_test.append(test)
    covered = np.concatenate(seen_test)
    np.testing.assert_array_equal(covered, np.arange(200, 600))


def test_rolling_window_limits_training_size():
    spl = PurgedWalkForward(horizon=5, min_train_size=100, test_size=20, max_train_size=60)
    assert all(len(tr) == 60 for tr, _ in spl.split(300))


def test_purged_kfold_removes_label_overlap_and_applies_embargo():
    n, h, emb = 400, 10, 5
    for train, test in purged_kfold(n, 4, h, emb):
        a, b = test.min(), test.max() + 1
        test_labels = set(range(a + 1, b + h))
        for i in train:
            assert test_labels.isdisjoint(range(i + 1, i + h + 1))
        assert not np.any((train >= b) & (train < b + h - 1 + emb))


# --------------------------------------------------------------------------- regime HMM
def test_hmm_recovers_regime_vols():
    rng = np.random.default_rng(0)
    states = np.repeat(np.tile([0, 1], 10), 150)
    x = rng.standard_normal(states.size) * np.where(states == 1, 0.02, 0.005)
    hmm = GaussianHMM(2).fit(x)
    np.testing.assert_allclose(hmm.sigma, [0.005, 0.02], rtol=0.1)
    accuracy = np.mean((hmm.filter(x)[:, 1] > 0.5) == states)
    assert accuracy > 0.9


def test_hmm_filter_is_causal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(800) * np.where(np.arange(800) % 300 < 150, 0.004, 0.015)
    hmm = GaussianHMM(2).fit(x)
    np.testing.assert_allclose(hmm.filter(x[:500]), hmm.filter(x)[:500], rtol=1e-12)


def test_causal_regime_probability_tracks_latent_vol(dataset):
    # The synthetic IV is the exact expected future vol, i.e. the latent vol level the
    # return-based HMM should be detecting.
    X, _, _ = dataset
    p = causal_regime_probability(X["ret_1"], min_history=504)
    valid = p.notna()
    assert p[:504].isna().all()
    assert np.corrcoef(p[valid], X["log_iv"][valid])[0, 1] > 0.5


def test_regime_probability_has_no_lookahead(dataset):
    X, _, _ = dataset

    def feature(frame):
        return causal_regime_probability(frame["ret_1"], min_history=504).to_frame()

    assert check_no_lookahead(feature, X, n_checks=3, min_rows=600) == []


# --------------------------------------------------------------------------- models & stats
def test_linear_forecast_recovers_ols():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((500, 3)), columns=["a", "b", "c"])
    y = pd.Series(0.3 + X @ np.array([1.0, -2.0, 0.5]))
    model = LinearForecast(["a", "b", "c"], "ols").fit(X, y)
    np.testing.assert_allclose(model.coef_, [0.3, 1.0, -2.0, 0.5], atol=1e-10)


def test_walk_forward_only_predicts_test_rows(dataset):
    X, y, _ = dataset
    spl = PurgedWalkForward(horizon=H, min_train_size=600, test_size=250)
    p = walk_forward_predict(lambda: LinearForecast(HAR_COLUMNS, "HAR"), X, y, spl)
    assert p.iloc[:600].isna().all() and p.iloc[600:].notna().all()


def test_implied_vol_information_beats_har_on_synthetic_truth(dataset):
    # The synthetic IV is the exact expected future vol (plus premium/noise), so a
    # HAR model that adds it must forecast better than HAR alone.
    X, y, _ = dataset
    spl = PurgedWalkForward(horizon=H, min_train_size=600, test_size=250)
    preds = {
        "HAR-RV": walk_forward_predict(lambda: LinearForecast(HAR_COLUMNS, "HAR-RV"), X, y, spl),
        "HAR-RV-IV": walk_forward_predict(
            lambda: LinearForecast([*HAR_COLUMNS, "log_iv"], "HAR-RV-IV"), X, y, spl),
        "RW": walk_forward_predict(lambda: ColumnForecast("log_rv_22", "RW"), X, y, spl),
    }
    rep = forecast_report(y, preds, "HAR-RV", H)
    assert rep.loc["HAR-RV-IV", "QLIKE"] < rep.loc["HAR-RV", "QLIKE"]
    assert rep.loc["HAR-RV-IV", "R2 oos vs HAR-RV"] > 0
    assert rep.loc["HAR-RV-IV", "DM stat (QLIKE)"] < 0


def test_gbm_selects_from_grid_and_predicts(dataset):
    X, y, _ = dataset
    model = GBMForecast(H, grid={"learning_rate": (0.05, 0.1), "max_leaf_nodes": (7,)},
                        inner_splits=3).fit(X.iloc[:800], y.iloc[:800])
    assert model.best_params_["learning_rate"] in (0.05, 0.1)
    assert np.isfinite(model.predict(X.iloc[800:900])).all()


def test_gru_forecast_is_finite_and_causal(dataset):
    pytest.importorskip("torch")
    from deltasigma.research import GRUForecast

    X, y, _ = dataset
    X = X.assign(const_then_moves=np.r_[np.zeros(700), np.ones(len(X) - 700)])
    model = GRUForecast(H, epochs=15).fit(X.iloc[:700], y.iloc[:700])
    full = model.predict(X.iloc[:900])
    prefix = model.predict(X.iloc[:850])
    assert np.isfinite(full[21:]).all() and np.abs(full[21:]).max() < 5
    np.testing.assert_allclose(prefix[21:], full[21:850], rtol=1e-5, atol=1e-6)


def test_qlike_is_minimised_at_the_truth():
    y = np.log(np.full(5, 0.08))
    assert qlike(y, y).sum() == pytest.approx(0.0)
    assert (qlike(y, y + 0.1) > 0).all() and (qlike(y, y - 0.1) > 0).all()


def test_diebold_mariano():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(1000) ** 2
    assert diebold_mariano(a, a, 5) == (0.0, 1.0)
    stat, p = diebold_mariano(a, a + 0.5, 5)
    assert stat < 0 and p < 1e-6


# --------------------------------------------------------------------------- backtest
def test_vol_swap_pnl_and_costs():
    idx = pd.RangeIndex(3)
    iv = pd.Series([0.08, 0.08, 0.10], index=idx)
    rv = pd.Series([0.06, 0.09, 0.10], index=idx)
    pos = pd.Series([-1.0, 1.0, 0.0], index=idx)
    pnl = vol_swap_pnl(iv, rv, pos, cost_vol_pts=0.25)
    np.testing.assert_allclose(pnl, [2.0 - 0.25, 1.0 - 0.25, 0.0])


def test_timing_positions():
    vrp = pd.Series([1.0, 0.2, -1.0, np.nan])
    np.testing.assert_array_equal(timing_positions(vrp, 0.5), [-1, 0, 0, 0])
    np.testing.assert_array_equal(timing_positions(vrp, 0.5, allow_long=True), [-1, 0, 1, 0])


def test_strategy_stats_and_drawdown():
    pnl = pd.Series(np.tile([1.0, -0.5], 50))
    pos = pd.Series(-1.0, index=pnl.index)
    s = strategy_stats(pnl, pos, horizon=2)
    assert s["Active %"] == 100.0 and s["Hit rate %"] == 50.0
    assert max_drawdown(pd.Series([1.0, -3.0, 1.0])) == pytest.approx(3.0)
    assert ladder_equity(pnl, 2).iloc[-1] == pytest.approx(pnl.iloc[:-2].sum() / 2)
