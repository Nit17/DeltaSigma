"""EURUSD volatility forecasting and variance-risk-premium research on real data.

Data: FRED DEXUSEU (EURUSD) + EVZCLS (CBOE EuroCurrency vol index), downloaded and
cached in data/raw on first run. Pipeline:

1. leakage-audited features + causal HMM regime probability
2. purged expanding-window walk-forward (refit every 6 months, 3y minimum history)
3. forecast comparison vs HAR-RV with Diebold-Mariano tests, overall and by regime
4. VRP vol-selling backtest: always-short vs forecast-timed vs sized variants

Run:  python examples/04_vol_research.py
      python examples/04_vol_research.py --synthetic --quick   (offline smoke test, writes nothing)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deltasigma.research import (  # noqa: E402
    HAR_COLUMNS,
    ColumnForecast,
    GBMForecast,
    GRUForecast,
    LinearForecast,
    PurgedWalkForward,
    build_dataset,
    build_features,
    causal_regime_probability,
    check_no_lookahead,
    forecast_report,
    ladder_equity,
    load_eurusd_vol,
    predicted_vrp,
    strategy_stats,
    synthetic_vol_dataset,
    timing_positions,
    vol_swap_pnl,
    walk_forward_predict,
)

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 21
COST_VOL_PTS = 0.25
THRESHOLD_VOL_PTS = 0.5  # fixed ex ante: sell only if predicted VRP > 2x half-spread


def md(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ["" if pd.isna(v) else f"{v:.{digits}f}" for v in row]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use simulated data (offline)")
    parser.add_argument("--quick", action="store_true", help="small, fast configuration")
    args = parser.parse_args()

    if args.synthetic:
        df = synthetic_vol_dataset(n_days=1600 if args.quick else 4300, seed=7)
        source = "synthetic regime-switching Heston"
    else:
        df = load_eurusd_vol()
        source = "FRED DEXUSEU + EVZCLS"
    print(f"data: {source}, {len(df)} days {df.index[0].date()} .. {df.index[-1].date()}")

    violations = check_no_lookahead(build_features, df, n_checks=10 if args.quick else 40)
    print(f"look-ahead audit (features): {len(violations)} violations")
    X, y, info = build_dataset(df, HORIZON)

    def regime_feature(frame: pd.DataFrame) -> pd.DataFrame:
        return causal_regime_probability(frame["ret_1"]).to_frame()

    hmm_violations = check_no_lookahead(regime_feature, X, n_checks=2 if args.quick else 5,
                                        min_rows=600)
    print(f"look-ahead audit (HMM regime filter): {len(hmm_violations)} violations")
    X = X.assign(p_stress=causal_regime_probability(X["ret_1"]).fillna(0.5))

    splitter = PurgedWalkForward(horizon=HORIZON, min_train_size=504 if args.quick else 756,
                                 test_size=252 if args.quick else 126)
    print(f"walk-forward folds: {splitter.n_splits(len(y))}")

    gbm_kwargs = {"grid": {"learning_rate": (0.05,), "max_leaf_nodes": (7,)}} if args.quick else {}
    factories = {
        "Random walk (RV22)": lambda: ColumnForecast("log_rv_22", "Random walk (RV22)"),
        "Implied vol": lambda: ColumnForecast("log_iv", "Implied vol"),
        "HAR-RV": lambda: LinearForecast(HAR_COLUMNS, "HAR-RV"),
        "HAR-RV-IV": lambda: LinearForecast([*HAR_COLUMNS, "log_iv"], "HAR-RV-IV"),
        "Gradient boosting": lambda: GBMForecast(HORIZON, **gbm_kwargs),
        "GRU + linear highway": lambda: GRUForecast(HORIZON, epochs=20 if args.quick else 300),
    }
    preds = {}
    for name, factory in factories.items():
        t0 = time.time()
        preds[name] = walk_forward_predict(factory, X, y, splitter)
        print(f"  {name}: {time.time() - t0:.0f}s")

    test = pd.DataFrame(preds).notna().all(axis=1)
    stress = X["p_stress"] > 0.5
    reports = {
        "All test dates": forecast_report(y, preds, "HAR-RV", HORIZON),
        "Calm regime (HMM p_stress <= 0.5)": forecast_report(y, preds, "HAR-RV", HORIZON, ~stress),
        "Stress regime (HMM p_stress > 0.5)": forecast_report(y, preds, "HAR-RV", HORIZON, stress),
    }
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 10)
    for title, rep in reports.items():
        print(f"\n## {title} (n={rep.attrs['n_obs']})\n{rep.round(4)}")

    iv, rv = info["iv"][test], info["rv_fwd"][test]

    def norm(size: pd.Series) -> pd.Series:
        return size / size.expanding().mean().shift(1).bfill()

    vol_pts = np.exp(preds["HAR-RV-IV"][test]) * 100.0
    vrp_har = predicted_vrp(iv, preds["HAR-RV-IV"][test])
    positions = {
        "Always short vol": pd.Series(-1.0, index=iv.index),
        "HAR-RV-IV timed": timing_positions(vrp_har, THRESHOLD_VOL_PTS),
        "GRU timed": timing_positions(
            predicted_vrp(iv, preds["GRU + linear highway"][test]), THRESHOLD_VOL_PTS),
        "Gradient boosting timed": timing_positions(
            predicted_vrp(iv, preds["Gradient boosting"][test]), THRESHOLD_VOL_PTS),
        "Vol-managed (1/vol)": -norm(1.0 / vol_pts),
        "VRP/vol sized (HAR-RV-IV)": -norm(vrp_har.clip(lower=0.0) / vol_pts),
    }
    pnls = {k: vol_swap_pnl(iv, rv, pos, COST_VOL_PTS) for k, pos in positions.items()}
    bt = pd.DataFrame({k: strategy_stats(pnls[k], positions[k], HORIZON) for k in positions}).T
    print(f"\n## VRP backtest (vol pts per unit vega notional)\n{bt.round(2)}")

    if args.synthetic:
        return

    report = [
        "# Volatility forecasting & VRP research results\n",
        f"Data: {source}, {len(df):,} days ({df.index[0].date()} to {df.index[-1].date()}). "
        f"Target: log realised vol over the next {HORIZON} trading days. "
        f"Purged expanding walk-forward, {splitter.n_splits(len(y))} folds, refit every 6 months; "
        f"out-of-sample period {y[test].index[0].date()} to {y[test].index[-1].date()}.\n",
        f"Look-ahead audit: {len(violations)} feature violations, "
        f"{len(hmm_violations)} HMM-filter violations.\n",
        "DM = Diebold-Mariano test on QLIKE vs HAR-RV (HAC, HLN-corrected); negative = better.\n",
    ]
    for title, rep in reports.items():
        report.append(f"## {title} (n = {rep.attrs['n_obs']:,})\n\n{md(rep, 4)}\n")
    report.append(
        f"## VRP backtest\n\nSell (or size) a 1M vol swap struck at the vol index each day, "
        f"{COST_VOL_PTS} vol pt half-spread; timed strategies sell only when predicted VRP > "
        f"{THRESHOLD_VOL_PTS} vol pts (threshold fixed ex ante). Sharpe and drawdown use "
        f"non-overlapping trades averaged over all {HORIZON} roll offsets. All six variants "
        f"tested are reported.\n\n{md(bt, 2)}\n"
    )
    (ROOT / "docs" / "vol_research_results.md").write_text("\n".join(report), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 0.9]})
    idx = y[test].index
    ax = axes[0]
    ax.fill_between(idx, 0, 1, where=stress[test].to_numpy(), transform=ax.get_xaxis_transform(),
                    color="tab:red", alpha=0.12, label="HMM stress regime")
    ax.plot(idx, np.exp(y[test]) * 100, color="k", lw=0.8, label="realised vol (next 21d)")
    ax.plot(idx, info["iv"][test] * 100, color="tab:orange", lw=0.8, label="implied vol (EVZ)")
    ax.plot(idx, np.exp(preds["HAR-RV-IV"][test]) * 100, color="tab:blue", lw=0.9,
            label="HAR-RV-IV forecast")
    ax.set(title="EURUSD 1M vol: out-of-sample forecasts", ylabel="vol (%)")
    ax.legend(fontsize=7, loc="upper right")

    for k in ["Always short vol", "HAR-RV-IV timed", "GRU timed", "Vol-managed (1/vol)"]:
        axes[1].plot(ladder_equity(pnls[k], HORIZON), lw=1, label=k)
    axes[1].set(title="VRP ladder: cumulative P&L", ylabel="vol pts (unit vega notional)")
    axes[1].legend(fontsize=7)

    models = list(preds)
    x = np.arange(len(models))
    for i, (title, rep) in enumerate(list(reports.items())[1:]):
        axes[2].bar(x + (i - 0.5) * 0.4, rep.loc[models, "QLIKE"], width=0.4,
                    label=title.split(" (")[0])
    axes[2].set_xticks(x, [m.replace(" ", "\n") for m in models], fontsize=7)
    axes[2].set(title="Out-of-sample QLIKE by regime (lower = better)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "vol_research.png", dpi=130)
    print("saved docs/vol_research_results.md and docs/images/vol_research.png")


if __name__ == "__main__":
    main()
