"""Simulate a regime-switching EURUSD market (calm <-> stress Heston regimes).

Run:  python examples/02_regime_simulation.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deltasigma.sim import eurusd_calm_stress_model  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> None:
    model = eurusd_calm_stress_model()
    res = model.simulate(S0=1.085, T=3.0, n_steps=3 * 252, n_paths=5_000, seed=42)

    stress = res.regime == 1
    print(f"time in stress regime: {stress.mean():.1%}")
    rv = res.realized_vol
    print(f"realised vol  mean {rv.mean():.2%}  5%-95%: {np.quantile(rv, 0.05):.2%} - "
          f"{np.quantile(rv, 0.95):.2%}")

    path = int(np.argmax(stress.mean(axis=1) > 0.2))
    t = res.times
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax in axes:
        ax.fill_between(t, 0, 1, where=stress[path], transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.15, label="stress regime")
    axes[0].plot(t, res.spot[path], color="tab:blue")
    axes[0].set(title="Regime-switching Heston EURUSD path", ylabel="spot")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(t, np.sqrt(res.variance[path]) * 100, color="tab:purple")
    axes[1].set(ylabel="instantaneous vol (%)", xlabel="years")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "regime_simulation.png", dpi=130)
    print("saved", OUT / "regime_simulation.png")


if __name__ == "__main__":
    main()
