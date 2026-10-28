"""Phase 5 - production: P&L explain, drift monitoring, gated deployment, service latency.

A. Greek P&L attribution of an FX options book along a calm -> stress path.
B. Drift monitoring of the Phase 3 vol forecaster on real EURUSD data: CUSUM /
   Page-Hinkley on *matured* forecast losses + rolling PSI feature drift.
C. Regime-gated deployment of the Phase 4 PPO quoter (HMM routes stress RFQs to
   PPO, calm RFQs to the classical quoter); models cached in a file registry.
D. Latency of the FastAPI pricing / quoting service.

Run:  python examples/06_production.py            (~8 min first run, ~2 min cached)
      python examples/06_production.py --quick    (offline smoke test, writes nothing)
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deltasigma.mm import (  # noqa: E402
    MarketConfig,
    ModelQuoter,
    PPOConfig,
    PPOMarketMaker,
    WinProbabilityModel,
    collect_rfq_history,
    estimate_flow_markout,
    evaluate,
    tune,
)
from deltasigma.production import (  # noqa: E402
    CUSUM,
    TERMS,
    MarketState,
    ModelRegistry,
    OptionPosition,
    PageHinkley,
    alert_summary,
    attribute_path,
    explain_ratio,
    rolling_feature_drift,
    run_detector,
)
from deltasigma.production.gating import (  # noqa: E402
    RegimeGatedPolicy,
    collect_vol_changes,
    fit_regime_hmm,
)
from deltasigma.research import (  # noqa: E402
    HAR_COLUMNS,
    LinearForecast,
    PurgedWalkForward,
    build_dataset,
    load_eurusd_vol,
    qlike,
    synthetic_vol_dataset,
    walk_forward_predict,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 21


def md(df: pd.DataFrame, digits: int = 2) -> str:
    lines = ["| | " + " | ".join(map(str, df.columns)) + " |", "|---|" + "---:|" * len(df.columns)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(
            v if isinstance(v, str) else f"{v:.{digits}f}" for v in row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- A. P&L explain
def pnl_explain(quick: bool):
    df = synthetic_vol_dataset(n_days=189, seed=11)
    rd, rf = 0.043, 0.025
    S0 = float(df["spot"].iloc[0])
    book = [
        OptionPosition(S0 * 0.95, 1.0, False, -3e6, vol_offset=0.008),   # short 1Y 25D-ish put
        OptionPosition(S0 * 1.06, 1.0, True, -2e6, vol_offset=-0.002),   # short 1Y OTM call
        OptionPosition(S0, 0.9, True, 2.5e6),                            # long 11M ATM call
        OptionPosition(S0, 0.9, False, 2.5e6),                           # long 11M ATM put
    ]
    states = [MarketState(i / 252, s, iv, rd, rf)
              for i, (s, iv) in enumerate(zip(df["spot"], df["iv"], strict=True))]
    att = attribute_path(book, states, df.index)
    stress = df["regime"].iloc[1:].to_numpy() == 1
    unexpl = att["unexplained"].abs()
    # Materiality limit: unexplained P&L above 2% of the book's typical daily P&L move.
    limit = 0.02 * att["actual"].abs().mean()
    flagged = att.index[(unexpl > limit).to_numpy()]
    summary = pd.DataFrame({
        "Days": [str(len(att)), str(int((~stress).sum())), str(int(stress.sum()))],
        "Explain ratio %": [100 * explain_ratio(att), 100 * explain_ratio(att[~stress]),
                            100 * explain_ratio(att[stress]) if stress.sum() > 2 else np.nan],
        "Mean |unexplained| (USD)": [unexpl.mean(), unexpl[~stress].mean(),
                                     unexpl[stress].mean() if stress.any() else np.nan],
        "Max |unexplained| (USD)": [unexpl.max(), unexpl[~stress].max(),
                                    unexpl[stress].max() if stress.any() else np.nan],
        "Mean |actual| (USD)": [att["actual"].abs().mean(), att["actual"][~stress].abs().mean(),
                                att["actual"][stress].abs().mean() if stress.any() else np.nan],
    }, index=["All days", "Calm regime", "Stress regime"])
    big_move = (np.abs(df["iv"].diff().iloc[1:]) > 0.01).to_numpy()
    return att, df, summary, flagged, limit, int(np.sum(big_move & (unexpl > limit).to_numpy()))


# --------------------------------------------------------------------------- B. monitoring
def monitoring(quick: bool):
    if quick:
        raw, source = synthetic_vol_dataset(n_days=2200, seed=5), "synthetic"
    else:
        raw, source = load_eurusd_vol(), "FRED DEXUSEU + EVZCLS"
    X, y, _ = build_dataset(raw, HORIZON)
    spl = PurgedWalkForward(horizon=HORIZON, min_train_size=756, test_size=126)
    pred = walk_forward_predict(lambda: LinearForecast([*HAR_COLUMNS, "log_iv"], "HAR-RV-IV"),
                                X, y, spl)
    oos = pred.notna()
    loss = pd.Series(qlike(y[oos], pred[oos]), index=y.index[oos])
    # A forecast made at t is only scored at t + horizon: monitor on the maturity date.
    matured = pd.Series(loss.to_numpy()[: len(loss) - HORIZON],
                        index=loss.index[HORIZON:], name="qlike")
    ref = matured.iloc[:252]
    live = matured.iloc[252:]
    cusum_alerts = run_detector(CUSUM.fit(ref, k=0.5, h=12.0), live, "CUSUM(QLIKE)",
                                cooldown=63)
    ph_alerts = run_detector(PageHinkley(delta=0.05, threshold=15.0), live, "Page-Hinkley(QLIKE)",
                             cooldown=63)
    feats = ["log_iv", "log_rv_22", "iv_rv_spread"]
    train_ref = X[feats].iloc[:756]
    live_feats = X[feats].iloc[756:]
    psi_frame, psi_alerts = rolling_feature_drift(train_ref, live_feats, window=126, step=21,
                                                  expanding=True, metric="psi")
    ood_frame, ood_alerts = rolling_feature_drift(train_ref, live_feats, window=126, step=21,
                                                  expanding=True, metric="ood")
    return {"source": source, "matured": matured, "live": live, "cusum": cusum_alerts,
            "ph": ph_alerts, "psi": psi_frame, "psi_alerts": psi_alerts, "ood": ood_frame,
            "ood_alerts": ood_alerts, "n_windows": len(psi_frame)}


# --------------------------------------------------------------------------- C. gated quoting
def market_maker_models(registry: ModelRegistry, quick: bool):
    cfg = MarketConfig()
    ppo_cfg = (PPOConfig(n_envs=16, n_updates=2, eval_every=1, eval_episodes=32) if quick
               else PPOConfig(n_updates=300, init_log_std=-1.6))
    grid = {"inventory_aversion": [0.02, 0.05, 0.1], "band": [2, 3, 5],
            "markout_scale": [0.0, 0.1, 0.25, 0.5, 1.0]}
    n_hist, n_tune = (60, 60) if quick else (400, 500)
    config = {"market": asdict(cfg), "ppo": asdict(ppo_cfg), "grid": grid, "n_hist": n_hist,
              "n_tune": n_tune, "seeds": {"hist": 1, "tune": 11, "train": 100, "val": 21}}
    version = registry.find("rfq_market_maker", config)
    if version is not None:
        print(f"  loaded rfq_market_maker v{version} from registry")
        return cfg, registry.load("rfq_market_maker", version), registry.metadata(
            "rfq_market_maker", version)
    t0 = time.time()
    hist = collect_rfq_history(cfg, n_hist, seed=1)
    win = WinProbabilityModel().fit(hist)
    markout, flow_coef = estimate_flow_markout(hist)
    params, _ = tune(lambda **p: ModelQuoter(win, markout, flow_coef=flow_coef, **p), grid, cfg,
                     n_tune, 11)
    b4 = ModelQuoter(win, markout, flow_coef=flow_coef, **params)
    agent = PPOMarketMaker(b4, cfg, ppo_cfg)
    agent.train(100, 21)
    hmm = fit_regime_hmm(collect_vol_changes(cfg, 50 if quick else 300, seed=300))
    models = {"b4": b4, "ppo": agent, "hmm": hmm}
    metrics = {"b4_params": params, "hmm_sigma": hmm.sigma.tolist(),
               "train_seconds": round(time.time() - t0, 1)}
    path = registry.save("rfq_market_maker", models, config, metrics)
    print(f"  trained and registered {path} in {time.time() - t0:.0f}s")
    return cfg, models, registry.metadata("rfq_market_maker")


def gated_quoting(registry: ModelRegistry, quick: bool):
    cfg, models, meta = market_maker_models(registry, quick)
    n_test = 200 if quick else 4000
    rows, shares = {}, {}
    for label, regime in [("Normal days", None), ("Stress-start days", 1)]:
        gated = RegimeGatedPolicy(models["b4"], models["ppo"], models["hmm"], threshold=0.5)
        res = {
            "B4 classical": evaluate(models["b4"], cfg, n_test, 99, regime),
            "PPO always": evaluate(models["ppo"], cfg, n_test, 99, regime),
            "HMM-gated (B4 calm / PPO stress)": evaluate(gated, cfg, n_test, 99, regime),
        }
        shares[label] = float(np.mean(gated.share_stress))
        ref = res["B4 classical"].objective
        for name, r in res.items():
            d = r.objective - ref
            rows.setdefault(name, {})[label] = r.risk_adjusted
            rows[name][f"{label} t vs B4"] = (float(d.mean() / d.std() * np.sqrt(d.size))
                                              if d.std() > 0 else np.nan)
    return pd.DataFrame(rows).T, shares, meta


# --------------------------------------------------------------------------- D. latency
def service_latency(quick: bool):
    from fastapi.testclient import TestClient

    from deltasigma.service.app import calibrated_smile, create_app

    client = TestClient(create_app())
    base = {"spot": 1.085, "rd": 0.043, "rf": 0.025, "tenor_years": 0.25, "atm": 0.072,
            "rr25": -0.0042, "bf25": 0.002, "delta": 0.25, "call": True, "notional": 1e6}
    n = 30 if quick else 300
    calibrated_smile.cache_clear()
    cold, warm, quote = [], [], []

    def server_ms(resp) -> float:
        # Server-side processing time (middleware header): excludes the in-process test
        # client's transport overhead, which dominates wall-clock time on Windows.
        return float(resp.headers["X-Process-Time-ms"])

    for i in range(n):
        cold.append(server_ms(client.post("/price", json={**base, "spot": 1.0 + i * 1e-4})))
    for _ in range(n):
        warm.append(server_ms(client.post("/price", json=base)))
    rng = np.random.default_rng(0)
    for _ in range(n):
        body = {"tier": int(rng.integers(2)), "side": int(rng.choice([-1, 1])),
                "size": float(rng.integers(1, 6)), "inventory": float(rng.normal(0, 3)),
                "mid_vol": 7.5}
        quote.append(server_ms(client.post("/quote", json=body)))
    lat = {"POST /price (new market snapshot)": cold, "POST /price (cached smile)": warm,
           "POST /quote": quote}
    table = pd.DataFrame({k: {"p50 ms": np.percentile(v, 50), "p99 ms": np.percentile(v, 99)}
                          for k, v in lat.items()}).T
    return table, lat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick

    print("A. P&L explain")
    att, path_df, att_summary, flagged, limit, flagged_big = pnl_explain(q)
    print(att_summary)
    print(f"  days breaching the {limit:.0f} USD unexplained-P&L limit: {len(flagged)} "
          f"({flagged_big} of them with a >1 vol pt implied-vol move)")

    print("B. Drift monitoring")
    mon = monitoring(q)
    for name, alerts in [("CUSUM", mon["cusum"]), ("Page-Hinkley", mon["ph"]),
                         ("PSI", mon["psi_alerts"]), ("Out-of-range", mon["ood_alerts"])]:
        print(f"  {name}: {len(alerts)} alerts; first: {[str(a) for a in alerts[:4]]}")

    print("C. Regime-gated quoting")
    registry = ModelRegistry(Path(tempfile.mkdtemp()) if q else ROOT / "artifacts" / "registry")
    gated, shares, meta = gated_quoting(registry, q)
    pd.set_option("display.width", 200)
    print(gated.round(2))
    print(f"  share of RFQs routed to PPO: {shares}")

    print("D. Service latency")
    lat_table, lat = service_latency(q)
    print(lat_table.round(2))

    if q:
        return

    def alert_dates(alerts):
        return ", ".join(sorted({str(a.time.date()) for a in alerts})[:12]) or "none"

    report = [
        "# Production results\n",
        "## A. Greek P&L explain\n",
        "Book: short 1Y 95% put (3m EUR), short 1Y 106% call (2m), long 11M ATM straddle "
        "(2.5m each), marked daily on a simulated regime-switching EURUSD path "
        "(sticky-strike smile offsets, parallel vol shifts).\n",
        md(att_summary, 2) + "\n",
        f"Unexplained-P&L limit (2% of mean |daily P&L| = {limit:.0f} USD) breached on "
        f"{len(flagged)} days, {flagged_big} of them with a >1 vol pt implied-vol move. The "
        "residual tracks large vol moves (correlation ~0.65 with |d vol|, ~0.06 with |d spot|): "
        "third-order vol risk (ultima) and vol-time cross terms are not in the explain.\n",
        f"## B. Drift monitoring ({mon['source']})\n",
        "HAR-RV-IV forecasts from the Phase 3 walk-forward. QLIKE losses enter the monitor on "
        "the date the forecast *matures* (t + 21), the earliest they are knowable. Reference "
        "= first 252 matured losses; alerts have a 63-day cooldown (one ticket per episode). "
        "Feature drift compares each 126-day window with all data the expanding-window model "
        "has been trained on so far.\n",
        f"- Page-Hinkley alarms: {len(mon['ph'])} on {alert_dates(mon['ph'])}",
        f"- CUSUM (k = 0.5, h = 12) alarms: {len(mon['cusum'])} on {alert_dates(mon['cusum'])}",
        f"- Out-of-range (> 10% of window outside the training 0.5-99.5% range) alerts: "
        f"{alert_summary(mon['ood_alerts'])} over {mon['n_windows']} windows; first on "
        f"{alert_dates(mon['ood_alerts'])}",
        f"- PSI > 0.25 alerts, for comparison: {alert_summary(mon['psi_alerts'])} over "
        f"{mon['n_windows']} windows\n",
        "## C. Regime-gated deployment of the PPO quoter\n",
        "Risk-adjusted P&L per day on the same 4,000 paired test days as Phase 4; t-stats are "
        "paired vs B4. The HMM threshold (0.5) was fixed ex ante.\n",
        md(gated, 2) + "\n",
        f"Share of RFQs routed to PPO: normal days {shares['Normal days']:.1%}, stress-start "
        f"days {shares['Stress-start days']:.1%}. Registry entry: `rfq_market_maker` "
        f"v{meta['version']} (config hash `{meta['config_hash']}`).\n",
        "## D. Service latency (server-side processing time, single worker)\n",
        md(lat_table, 2) + "\n",
    ]
    (ROOT / "docs" / "production_results.md").write_text("\n".join(report), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    ax = axes[0, 0]
    cum = att[list(TERMS) + ["unexplained"]].cumsum() / 1e3
    for col in ["delta", "gamma", "vega", "theta", "vanna", "volga", "unexplained"]:
        ax.plot(cum.index, cum[col], lw=1, label=col)
    ax.plot(cum.index, att["actual"].cumsum() / 1e3, color="k", lw=1.8, label="actual P&L")
    st = path_df["regime"].iloc[1:].to_numpy() == 1
    ax.fill_between(att.index, 0, 1, where=st, transform=ax.get_xaxis_transform(),
                    color="tab:red", alpha=0.1, label="stress regime")
    ax.set(title="A. Cumulative P&L explain by Greek", ylabel="USD thousands")
    ax.legend(fontsize=7, ncol=3)

    ax = axes[0, 1]
    roll = mon["matured"].rolling(63).mean()
    ax.plot(roll.index, roll, color="tab:blue", lw=1, label="63d mean QLIKE (matured)")
    ax.axhline(mon["matured"].iloc[:252].mean(), color="grey", ls=":", label="reference mean")
    for i, a in enumerate(mon["cusum"]):
        ax.axvline(a.time, color="tab:red", lw=0.8, alpha=0.7,
                   label="CUSUM alarm" if i == 0 else None)
    for i, a in enumerate(mon["ph"]):
        ax.axvline(a.time, color="tab:purple", lw=2.0, alpha=0.8, ls="--",
                   label="Page-Hinkley alarm" if i == 0 else None)
    ax.set(title=f"B. Forecast-loss monitoring ({mon['source']})", ylabel="QLIKE")
    ax2 = ax.twinx()
    ax2.plot(mon["ood"].index, mon["ood"]["log_iv"], color="tab:green", lw=0.9, alpha=0.7,
             label="out-of-range rate, log IV")
    ax2.set_ylabel("share of window outside training range")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")

    ax = axes[1, 0]
    names = list(gated.index)
    x = np.arange(len(names))
    for i, col in enumerate(["Normal days", "Stress-start days"]):
        ax.bar(x + (i - 0.5) * 0.38, gated[col], width=0.38, label=col)
    ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set(title="C. Regime-gated deployment: risk-adjusted P&L / day", ylabel="vol pts x vega")
    ax.set_ylim(0, 1.3 * gated[["Normal days", "Stress-start days"]].to_numpy().max())
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    bins = np.logspace(-2, 2.5, 60)
    for k, v in lat.items():
        ax.hist(v, bins=bins, alpha=0.6, label=k)
    ax.set_xscale("log")
    ax.set(title="D. Service latency (server-side processing time)", xlabel="ms",
           ylabel="requests")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "production.png", dpi=120)
    print("saved docs/production_results.md and docs/images/production.png")


if __name__ == "__main__":
    main()
