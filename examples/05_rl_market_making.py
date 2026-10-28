"""RL market-making for FX options RFQs: win-probability model, classical quoters and PPO.

Pipeline:
1. RFQ history with randomised quotes -> logistic win-probability model (calibration
   checked against the simulator's true acceptance curve) + markout estimates.
2. Classical quoters, tuned on training seeds: fixed spread; model-optimal with inventory
   skew; + full markout; + tuned fraction of markout and informed-flow signal.
3. A PPO residual agent on top of the strongest classical quoter.
4. Paired out-of-sample comparison (common random numbers) on normal and stress-start days.

Run:  python examples/05_rl_market_making.py          (~9 min on a laptop CPU)
      python examples/05_rl_market_making.py --quick  (smoke test, writes nothing)
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

from deltasigma.mm import (  # noqa: E402
    FixedSpreadQuoter,
    MarketConfig,
    ModelQuoter,
    PPOConfig,
    PPOMarketMaker,
    RFQMarket,
    WinProbabilityModel,
    calibration_report,
    collect_rfq_history,
    estimate_flow_markout,
    estimate_markout,
    evaluate,
    tune,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SEED, VAL_SEED, TEST_SEED, TUNE_SEED = 100, 21, 99, 11


def md(df: pd.DataFrame, digits: int = 2) -> str:
    lines = ["| | " + " | ".join(df.columns) + " |", "|---|" + "---:|" * len(df.columns)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(
            "" if pd.isna(v) else f"{v:.{digits}f}" for v in row) + " |")
    return "\n".join(lines)


def inventory_profile(policy, config: MarketConfig, n: int, seed: int) -> np.ndarray:
    """Mean absolute inventory held at each step of the day."""
    env = RFQMarket(config, n, seed=seed)
    env.reset()
    prof = np.empty(config.n_steps)
    for t in range(config.n_steps):
        edge, hedge = policy(env)
        env.step(edge, hedge)
        prof[t] = np.abs(env.q).mean() if t < config.n_steps - 1 else 0.0
    return prof


def residual_by_signal(agent: PPOMarketMaker, config: MarketConfig, n: int, seed: int) -> float:
    """Slope of the PPO quote adjustment (vs its base quoter) on ``side * flow_tier1``."""
    env = RFQMarket(config, n, seed=seed)
    env.reset()
    sig, adj = [], []
    for _ in range(config.n_steps):
        base_edge = agent.base.quote(env)
        edge, hedge = agent(env)
        sig.append(env.current_rfq()["side"] * env.flow1)
        adj.append(edge - base_edge)
        env.step(edge, hedge)
    return float(np.polyfit(np.ravel(sig), np.ravel(adj), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    q = args.quick
    n_hist, n_tune, n_test = (60, 60, 200) if q else (400, 500, 4000)
    ppo_cfg = PPOConfig(n_envs=16, n_updates=2, eval_every=1, eval_episodes=32) if q else \
        PPOConfig(n_updates=300, init_log_std=-1.6)
    cfg = MarketConfig()

    hist = collect_rfq_history(cfg, n_hist, seed=1)
    holdout = collect_rfq_history(cfg, n_hist // 2, seed=2)
    win = WinProbabilityModel().fit(hist)
    scores, reliability = calibration_report(win, holdout)
    markout = estimate_markout(hist)
    flow_markout, flow_coef = estimate_flow_markout(hist)
    print("win model:", {k: round(v, 4) for k, v in scores.items()})
    print(f"markout cost/unit by tier {markout.round(3)}; flow coefficient {flow_coef:.3f}")

    inv_grid = {"inventory_aversion": [0.0, 0.02, 0.05, 0.1, 0.2], "band": [1, 2, 3, 5, 8]}
    tuned = {}
    t0 = time.time()
    tuned["fixed"], _ = tune(lambda **p: FixedSpreadQuoter(**p),
                             {"edge": [0.1, 0.15, 0.2, 0.25, 0.3], "band": [1, 2, 3, 5, 8]},
                             cfg, n_tune, TUNE_SEED)
    tuned["model"], _ = tune(lambda **p: ModelQuoter(win, **p), inv_grid, cfg, n_tune, TUNE_SEED)
    tuned["markout"], _ = tune(lambda **p: ModelQuoter(win, markout, **p), inv_grid, cfg, n_tune,
                               TUNE_SEED)
    tuned["flow"], _ = tune(
        lambda **p: ModelQuoter(win, flow_markout, flow_coef=flow_coef, **p),
        {"inventory_aversion": [0.02, 0.05, 0.1], "band": [2, 3, 5],
         "markout_scale": [0.0, 0.1, 0.25, 0.5, 1.0]},
        cfg, n_tune, TUNE_SEED,
    )
    print(f"tuned baselines in {time.time() - t0:.0f}s: {tuned}")

    best_name = "B4 + tuned markout & flow signal"
    policies = {
        "B1 Fixed spread + band hedge": FixedSpreadQuoter(**tuned["fixed"]),
        "B2 Win-model optimal + skew": ModelQuoter(win, **tuned["model"]),
        "B3 B2 + full 30-step markout": ModelQuoter(win, markout, **tuned["markout"]),
        best_name: ModelQuoter(win, flow_markout, flow_coef=flow_coef, **tuned["flow"]),
    }
    t0 = time.time()
    agent = PPOMarketMaker(policies[best_name], cfg, ppo_cfg)
    hist_ppo = agent.train(TRAIN_SEED, VAL_SEED)
    base_val = evaluate(policies[best_name], cfg, ppo_cfg.eval_episodes, VAL_SEED).risk_adjusted
    best_val = max(s for _, s in hist_ppo.val_score)
    deploy = best_val > base_val
    print(f"PPO: trained in {time.time() - t0:.0f}s, best update {hist_ppo.best_update}; "
          f"validation {best_val:.2f} vs base {base_val:.2f} -> deployment gate: "
          f"{'PASS' if deploy else 'FAIL (keep classical quoter)'}")
    policies["PPO residual on B4"] = agent

    reports, results = {}, {}
    for label, regime in [("Normal days (stationary regime mix)", None),
                          ("Stress-start days", 1)]:
        res = {k: evaluate(p, cfg, n_test, TEST_SEED, regime) for k, p in policies.items()}
        results[label] = res
        ref = res[best_name].objective
        rows = {}
        for k, r in res.items():
            d = r.objective - ref
            row = r.summary()
            row["Paired diff vs B4"] = float(d.mean())
            row["t-stat"] = float(d.mean() / d.std() * np.sqrt(d.size)) if d.std() > 0 else np.nan
            rows[k] = row
        reports[label] = pd.DataFrame(rows).T
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 12)
        print(f"\n## {label}\n{reports[label].round(2)}")

    slope = residual_by_signal(agent, cfg, 32 if q else 512, 7)
    print(f"\nPPO quote adjustment slope vs side*flow_tier1: {slope:+.3f} vol pts")

    if q:
        return

    report = [
        "# RL market-making results\n",
        "Synthetic FX options RFQ market (`deltasigma.mm.env`), 200 RFQs per day. Units: vol "
        "points x vega units. Objective = P&L - running inventory penalty "
        f"(gamma = {cfg.risk_aversion}). {n_test:,} paired test days per scenario (common random "
        "numbers); baselines tuned and PPO trained/selected on separate seeds.\n",
        "## Win-probability model (held-out RFQs)\n",
        md(pd.DataFrame(scores, index=["score"]).T, 4) + "\n",
        f"Estimated 30-step markout cost per vega unit: tier 0 = {markout[0]:.3f}, tier 1 = "
        f"{markout[1]:.3f} vol pts; informed-flow coefficient = {flow_coef:.3f}.\n",
        f"Tuned baseline parameters: `{tuned}`\n",
        f"PPO ({ppo_cfg.n_updates} updates): best validation objective {best_val:.2f} vs "
        f"B4 {base_val:.2f} on {ppo_cfg.eval_episodes} validation days -> deployment gate "
        f"**{'PASS' if deploy else 'FAIL'}**.\n",
    ]
    for label, rep in reports.items():
        report.append(f"## {label}\n\n{md(rep)}\n")
    report.append(f"PPO quote adjustment per unit of `side * flow_tier1` (beyond B4): "
                  f"{slope:+.3f} vol pts.\n")
    (ROOT / "docs" / "rl_market_making_results.md").write_text("\n".join(report), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    edges = np.linspace(-0.05, 0.6, 50)
    for tier, color in [(0, "tab:blue"), (1, "tab:red")]:
        sub = holdout[holdout["tier"] == tier]
        b = pd.cut(sub["edge"], np.linspace(-0.05, 0.6, 14))
        emp = sub.groupby(b, observed=True)["accepted"].mean()
        mids = [iv.mid for iv in emp.index]
        axes[0].scatter(mids, emp.to_numpy(), color=color, s=14)
        axes[0].plot(edges, cfg.win_probability(edges, np.full(50, tier), np.zeros(50)),
                     color=color, ls=":", lw=1)
        axes[0].plot(edges, win.predict_proba(edges, tier, 0.0, 0.0), color=color,
                     label=f"tier {tier} {'(real money)' if tier == 0 else '(fast money)'}")
    axes[0].set(title="Win probability: fitted (line), true (dotted), observed (dots)",
                xlabel="quoted edge (vol pts)", ylabel="P(client trades)")
    axes[0].legend(fontsize=8)

    names = list(policies)
    x = np.arange(len(names))
    for i, (label, res) in enumerate(results.items()):
        axes[1].bar(x + (i - 0.5) * 0.4, [res[n].risk_adjusted for n in names], width=0.4,
                    label=label.split(" (")[0])
    axes[1].set_xticks(x, [n.split(" ", 1)[0] for n in names], fontsize=8)
    axes[1].set(title="Risk-adjusted P&L per day (test)", ylabel="vol pts x vega units")
    top = max(r.risk_adjusted for res in results.values() for r in res.values())
    axes[1].set_ylim(0, 1.3 * top)
    axes[1].legend(fontsize=8, loc="upper left")

    for name in ["B2 Win-model optimal + skew", best_name, "PPO residual on B4"]:
        axes[2].plot(inventory_profile(policies[name], cfg, 1000, 5)[:-1], lw=1.2,
                     label=name.split(" ", 1)[0] if name.startswith("B") else "PPO")
    axes[2].set(title="Average |vega inventory| through the day",
                xlabel="RFQ number", ylabel="|inventory| (vega units)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "rl_market_making.png", dpi=130)
    print("saved docs/rl_market_making_results.md and docs/images/rl_market_making.png")


if __name__ == "__main__":
    main()
