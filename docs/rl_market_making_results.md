# RL market-making results

Synthetic FX options RFQ market (`deltasigma.mm.env`), 200 RFQs per day. Units: vol points x vega units. Objective = P&L - running inventory penalty (gamma = 2.0). 4,000 paired test days per scenario (common random numbers); baselines tuned and PPO trained/selected on separate seeds.

## Win-probability model (held-out RFQs)

| | score |
|---|---:|
| Brier | 0.1184 |
| Brier (true probabilities) | 0.1181 |
| Log loss | 0.3703 |
| AUC | 0.8984 |
| Mean |p_hat - p_true| | 0.0093 |

Estimated 30-step markout cost per vega unit: tier 0 = 0.006, tier 1 = 0.183 vol pts; informed-flow coefficient = 0.524.

Tuned baseline parameters: `{'fixed': {'edge': 0.25, 'band': 2}, 'model': {'inventory_aversion': 0.05, 'band': 3}, 'markout': {'inventory_aversion': 0.05, 'band': 3}, 'flow': {'inventory_aversion': 0.05, 'band': 5, 'markout_scale': 0.25}}`

PPO (300 updates): best validation objective 20.40 vs B4 20.51 on 512 validation days -> deployment gate **FAIL**.

## Normal days (stationary regime mix)

| | Risk-adj P&L | Mean P&L | Std P&L | CVaR5 (worst days) | Spread captured | Inventory MTM | Hedge cost | Fill rate % | Avg |inventory| | Paired diff vs B4 | t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Fixed spread + band hedge | 12.69 | 20.81 | 4.88 | 9.60 | 33.93 | -0.36 | 12.76 | 22.64 | 1.94 | -8.07 | -98.35 |
| B2 Win-model optimal + skew | 19.82 | 27.90 | 4.78 | 17.95 | 34.54 | -1.07 | 5.57 | 33.30 | 1.92 | -0.93 | -16.05 |
| B3 B2 + full 30-step markout | 18.06 | 25.43 | 4.74 | 16.12 | 30.23 | -0.29 | 4.51 | 25.64 | 1.83 | -2.70 | -39.82 |
| B4 + tuned markout & flow signal | 20.76 | 31.42 | 5.81 | 20.80 | 32.39 | 0.70 | 1.68 | 31.86 | 2.16 | 0.00 |  |
| PPO residual on B4 | 20.59 | 27.78 | 4.85 | 18.01 | 32.91 | 0.89 | 6.02 | 31.17 | 1.83 | -0.16 | -2.74 |

## Stress-start days

| | Risk-adj P&L | Mean P&L | Std P&L | CVaR5 (worst days) | Spread captured | Inventory MTM | Hedge cost | Fill rate % | Avg |inventory| | Paired diff vs B4 | t-stat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Fixed spread + band hedge | 2.62 | 19.07 | 5.98 | 5.60 | 38.26 | -0.76 | 18.43 | 25.51 | 1.99 | -10.55 | -99.33 |
| B2 Win-model optimal + skew | 12.03 | 28.20 | 5.80 | 16.06 | 38.06 | -2.00 | 7.86 | 34.91 | 1.97 | -1.15 | -14.23 |
| B3 B2 + full 30-step markout | 11.75 | 26.40 | 5.74 | 14.50 | 33.48 | -0.70 | 6.37 | 27.19 | 1.87 | -1.42 | -15.30 |
| B4 + tuned markout & flow signal | 13.17 | 34.41 | 6.76 | 21.18 | 35.62 | 1.07 | 2.28 | 33.27 | 2.21 | 0.00 |  |
| PPO residual on B4 | 14.74 | 27.83 | 5.62 | 15.96 | 35.32 | 1.63 | 9.12 | 29.93 | 1.77 | 1.57 | 19.24 |

PPO quote adjustment per unit of `side * flow_tier1` (beyond B4): -0.000 vol pts.
