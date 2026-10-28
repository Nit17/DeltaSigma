"""Build a EURUSD vol surface from ATM / RR / BF quotes and plot smiles + densities.

Run:  python examples/01_vol_surface.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deltasigma.vol import sample_eurusd_surface  # noqa: E402
from deltasigma.vol.sample_market import EURUSD_QUOTES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> None:
    sabr, vv = sample_eurusd_surface("sabr"), sample_eurusd_surface("vanna_volga")
    labels = list(EURUSD_QUOTES)

    print(f"{'tenor':>5} {'K25P':>8} {'KATM':>8} {'K25C':>8} {'vol25P':>7} {'ATM':>6} "
          f"{'vol25C':>7} {'mktBF':>6} {'smileBF':>7}  arb-free")
    for label, s in zip(labels, sabr.smiles, strict=True):
        k, v = s.pillar_strikes, s.pillar_vols * 100
        arb_free = s.is_butterfly_arbitrage_free()
        print(f"{label:>5} {k[0]:8.4f} {k[1]:8.4f} {k[2]:8.4f} {v[0]:7.3f} {v[1]:6.3f} {v[2]:7.3f} "
              f"{s.quote.bf25 * 100:6.3f} {s.smile_bf25 * 100:7.3f}  {arb_free}")
    print("calendar-arbitrage violations:", len(sabr.calendar_arbitrage_violations()))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(labels)))
    for label, s, c in zip(labels, sabr.smiles, colors, strict=True):
        k = s.strike_grid(n=200, n_std=3.0)
        axes[0].plot(np.log(k / s.forward), s.vol(k) * 100, color=c, label=label)
        axes[0].scatter(np.log(s.pillar_strikes / s.forward), s.pillar_vols * 100, color=c, s=18)
    axes[0].set(title="EURUSD smiles (SABR, beta=1)", xlabel="log-moneyness ln(K/F)",
                ylabel="implied vol (%)")
    axes[0].legend(fontsize=8)

    s_sabr, s_vv = sabr.smiles[2], vv.smiles[2]
    k = s_sabr.strike_grid(n=300, n_std=4.5)
    axes[1].plot(k, s_sabr.vol(k) * 100, label="SABR")
    axes[1].plot(k, s_vv.vol(k) * 100, "--", label="Vanna-Volga")
    axes[1].scatter(s_sabr.pillar_strikes, s_sabr.pillar_vols * 100, color="k", zorder=3,
                    label="25P / ATM(DNS) / 25C")
    axes[1].set(title="3M smile: SABR vs Vanna-Volga", xlabel="strike", ylabel="implied vol (%)")
    axes[1].legend(fontsize=8)

    axes[2].plot(k, s_sabr.risk_neutral_density(k), label="SABR")
    axes[2].plot(k, s_vv.risk_neutral_density(k), "--", label="Vanna-Volga")
    axes[2].axvline(s_sabr.forward, color="grey", lw=0.8, ls=":")
    axes[2].set(title="3M risk-neutral density (Breeden-Litzenberger)", xlabel="EURUSD at expiry")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "eurusd_vol_surface.png", dpi=130)
    print("saved", OUT / "eurusd_vol_surface.png")


if __name__ == "__main__":
    main()
