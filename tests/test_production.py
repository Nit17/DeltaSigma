import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from deltasigma.mm import FixedSpreadQuoter, MarketConfig, RFQMarket  # noqa: E402
from deltasigma.pricing import garman_kohlhagen as gk  # noqa: E402
from deltasigma.production import (  # noqa: E402
    CUSUM,
    TERMS,
    MarketState,
    ModelRegistry,
    OptionPosition,
    PageHinkley,
    attribute_path,
    book_greeks,
    book_value,
    detection_delay,
    explain,
    explain_ratio,
    ks_drift,
    out_of_range_rate,
    psi,
    rolling_feature_drift,
    run_detector,
)
from deltasigma.production.gating import (  # noqa: E402
    OnlineRegimeFilter,
    RegimeGatedPolicy,
    collect_vol_changes,
    fit_regime_hmm,
)

BOOK = [
    OptionPosition(1.03, 0.5, False, -3e6, vol_offset=0.008),
    OptionPosition(1.15, 1.0, True, -2e6, vol_offset=-0.002),
    OptionPosition(1.085, 0.75, True, 2.5e6),
]
S0 = MarketState(0.0, 1.085, 0.072, 0.043, 0.025)


# ------------------------------------------------------------------ attribution
def _moved(h: float, with_time: bool, with_rates: bool = True) -> MarketState:
    dr = 1e-4 * h if with_rates else 0.0
    return MarketState(h / 252 if with_time else 0.0, 1.085 * (1 + 0.01 * h), 0.072 + 0.005 * h,
                       0.043 + dr, 0.025 - dr)


def test_instantaneous_spot_vol_explain_is_third_order():
    r1 = abs(explain(BOOK, S0, _moved(0.1, False, False))["unexplained"])
    r2 = abs(explain(BOOK, S0, _moved(0.01, False, False))["unexplained"])
    assert r1 / r2 > 300  # only speed / zomma / ultima remain: ratio ~1000


def test_rate_cross_terms_make_the_residual_second_order():
    r1 = abs(explain(BOOK, S0, _moved(0.1, False))["unexplained"])
    r2 = abs(explain(BOOK, S0, _moved(0.01, False))["unexplained"])
    assert 50 < r1 / r2 < 300


def test_explain_with_time_decay_is_second_order():
    r1 = abs(explain(BOOK, S0, _moved(0.1, True))["unexplained"])
    r2 = abs(explain(BOOK, S0, _moved(0.01, True))["unexplained"])
    assert 50 < r1 / r2 < 300  # cross terms like charm (dS * dt) are second order


def test_explain_terms_add_up():
    e = explain(BOOK, S0, _moved(1.0, True))
    assert e["explained"] == pytest.approx(sum(e[k] for k in TERMS))
    assert e["actual"] == pytest.approx(book_value(BOOK, _moved(1.0, True)) - book_value(BOOK, S0))
    assert e["unexplained"] == pytest.approx(e["actual"] - e["explained"])


def test_book_greeks_match_finite_differences():
    g = book_greeks(BOOK, S0)
    h = 1e-5

    def v(**kw):
        return book_value(BOOK, MarketState(**{**S0.__dict__, **kw}))

    assert g["delta"] == pytest.approx((v(spot=S0.spot + h) - v(spot=S0.spot - h)) / (2 * h),
                                       rel=1e-6)
    assert g["vega"] == pytest.approx((v(atm_vol=0.072 + h) - v(atm_vol=0.072 - h)) / (2 * h),
                                      rel=1e-6)
    single = gk.price(1.085, 1.085, 0.75, 0.043, 0.025, 0.072, True) * 2.5e6
    assert book_value(BOOK[2:], S0) == pytest.approx(float(single))


def test_daily_path_is_well_explained_and_expiry_is_checked():
    rng = np.random.default_rng(0)
    states = [S0]
    for i in range(1, 60):
        prev = states[-1]
        states.append(MarketState(i / 252, prev.spot * np.exp(0.004 * rng.standard_normal()),
                                  prev.atm_vol + 0.002 * rng.standard_normal(), 0.043, 0.025))
    att = attribute_path(BOOK, states, pd.RangeIndex(60))
    assert len(att) == 59 and explain_ratio(att) > 0.99
    with pytest.raises(ValueError):
        book_value(BOOK, MarketState(0.6, 1.085, 0.072, 0.043, 0.025))


# ------------------------------------------------------------------ monitoring
def test_psi_and_ks():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal(20_000), rng.standard_normal(20_000)
    assert psi(a, b) < 0.01
    assert psi(a, b + 1.0) > 0.25
    assert ks_drift(a, b + 0.2)[1] < 1e-6


def test_cusum_quiet_in_control_and_fast_after_shift():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(500)
    stream = pd.Series(np.r_[rng.standard_normal(400), 1.5 + rng.standard_normal(100)])
    alerts = run_detector(CUSUM.fit(ref, k=0.5, h=8.0), stream, "cusum")
    assert alerts and min(a.time for a in alerts) >= 400
    assert detection_delay(alerts, 400, stream.index) < 15
    assert all(a.value > a.threshold for a in alerts)


def test_page_hinkley_detects_mean_increase():
    rng = np.random.default_rng(3)
    stream = pd.Series(np.r_[rng.normal(0, 1, 300), rng.normal(2, 1, 200)])
    alerts = run_detector(PageHinkley(delta=0.1, threshold=25.0), stream, "ph")
    assert alerts and alerts[0].time >= 300
    assert alerts[0].value > alerts[0].threshold


def test_cooldown_gives_one_alert_per_episode():
    rng = np.random.default_rng(5)
    ref = rng.standard_normal(500)
    stream = pd.Series(np.r_[rng.standard_normal(200), 2.0 + rng.standard_normal(150)])
    many = run_detector(CUSUM.fit(ref, h=4.0), stream, "c")
    one = run_detector(CUSUM.fit(ref, h=4.0), stream, "c", cooldown=1000)
    assert len(many) > 3 and len(one) == 1


def test_out_of_range_rate():
    ref = np.linspace(0, 1, 10_001)
    assert out_of_range_rate(ref, np.array([0.5, 0.6])) == 0.0
    assert out_of_range_rate(ref, np.array([0.5, 2.0, -1.0, 0.1])) == 0.5


def test_cusum_rejects_degenerate_reference():
    with pytest.raises(ValueError):
        CUSUM.fit(np.ones(10))


def test_rolling_feature_drift_flags_only_shifted_windows():
    rng = np.random.default_rng(4)
    idx = pd.RangeIndex(1000)
    ref = pd.DataFrame({"x": rng.standard_normal(500)})
    live = pd.DataFrame({"x": np.r_[rng.standard_normal(500), 3 + rng.standard_normal(500)]},
                        index=idx)
    frame, alerts = rolling_feature_drift(ref, live, window=100, step=50)
    assert alerts and min(a.time for a in alerts) >= 500
    _, alerts_exp = rolling_feature_drift(ref, live, window=100, step=50, expanding=True)
    assert len(alerts_exp) <= len(alerts)
    _, ood = rolling_feature_drift(ref, live, window=100, step=50, metric="ood")
    assert ood and min(a.time for a in ood) >= 500
    with pytest.raises(ValueError):
        rolling_feature_drift(ref, live, metric="kl")


# ------------------------------------------------------------------ registry
def test_registry_roundtrip_and_cache_lookup(tmp_path):
    reg = ModelRegistry(tmp_path)
    cfg = {"lr": 0.01, "layers": [64, 64]}
    reg.save("model", {"w": np.arange(3)}, cfg, {"val": 1.5})
    reg.save("model", {"w": np.arange(4)}, {**cfg, "lr": 0.02})
    assert reg.versions("model") == [1, 2]
    assert reg.find("model", cfg) == 1 and reg.find("model", {"lr": 9}) is None
    np.testing.assert_array_equal(reg.load("model", 1)["w"], np.arange(3))
    meta = reg.metadata("model", 1)
    assert meta["metrics"] == {"val": 1.5} and len(meta["config_hash"]) == 12
    with pytest.raises(FileNotFoundError):
        reg.load("missing")


# ------------------------------------------------------------------ gating
@pytest.fixture(scope="module")
def hmm():
    return fit_regime_hmm(collect_vol_changes(MarketConfig(), 60, seed=300))


def test_hmm_separates_calm_and_stress_vol_changes(hmm):
    assert hmm.sigma[1] > 1.5 * hmm.sigma[0]


def test_online_filter_matches_batch_filter(hmm):
    x = collect_vol_changes(MarketConfig(), 1, seed=5)[0]
    f = OnlineRegimeFilter(hmm, 1)
    f.p = hmm.initial[None, :] @ np.linalg.inv(hmm.transition)  # so first predict = initial
    online = np.array([f.update(np.array([xi]))[0] for xi in x])
    np.testing.assert_allclose(online, hmm.filter(x)[:, 1], atol=1e-8)


def test_gated_policy_routes_by_threshold(hmm):
    calm, stress = FixedSpreadQuoter(0.1, band=3), FixedSpreadQuoter(0.4, band=3)
    for threshold, expected in [(-1.0, 0.4), (2.0, 0.1)]:
        policy = RegimeGatedPolicy(calm, stress, hmm, threshold=threshold)
        env = RFQMarket(MarketConfig(), 8, seed=1)
        env.reset()
        for _ in range(5):
            edge, hedge = policy(env)
            assert np.all(edge == expected)
            env.step(edge, hedge)
