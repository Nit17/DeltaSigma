# Volatility forecasting & VRP research results

Data: FRED DEXUSEU + EVZCLS, 4,314 days (2007-11-01 to 2025-03-11). Target: log realised vol over the next 21 trading days. Purged expanding walk-forward, 27 folds, refit every 6 months; out-of-sample period 2011-11-25 to 2025-02-07.

Look-ahead audit: 0 feature violations, 0 HMM-filter violations.

DM = Diebold-Mariano test on QLIKE vs HAR-RV (HAC, HLN-corrected); negative = better.

## All test dates (n = 3,286)

| | RMSE (log vol) | QLIKE | R2 oos vs HAR-RV | DM stat (QLIKE) | DM p-value |
|---|---:|---:|---:|---:|---:|
| Random walk (RV22) | 0.2918 | 0.2010 | -0.1097 | 3.2357 | 0.0012 |
| Implied vol | 0.2873 | 0.1415 | -0.0756 | -1.1914 | 0.2336 |
| HAR-RV | 0.2770 | 0.1581 | 0.0000 |  |  |
| HAR-RV-IV | 0.2476 | 0.1416 | 0.2009 | -1.8870 | 0.0593 |
| Gradient boosting | 0.2729 | 0.1597 | 0.0294 | 0.1589 | 0.8738 |
| GRU + linear highway | 0.2658 | 0.1479 | 0.0793 | -1.2237 | 0.2212 |

## Calm regime (HMM p_stress <= 0.5) (n = 2,608)

| | RMSE (log vol) | QLIKE | R2 oos vs HAR-RV | DM stat (QLIKE) | DM p-value |
|---|---:|---:|---:|---:|---:|
| Random walk (RV22) | 0.2862 | 0.2111 | -0.0708 | 2.9395 | 0.0033 |
| Implied vol | 0.2981 | 0.1521 | -0.1620 | -0.6371 | 0.5241 |
| HAR-RV | 0.2766 | 0.1627 | 0.0000 |  |  |
| HAR-RV-IV | 0.2522 | 0.1474 | 0.1685 | -1.5915 | 0.1116 |
| Gradient boosting | 0.2804 | 0.1672 | -0.0277 | 0.4160 | 0.6775 |
| GRU + linear highway | 0.2744 | 0.1591 | 0.0155 | -0.3758 | 0.7071 |

## Stress regime (HMM p_stress > 0.5) (n = 678)

| | RMSE (log vol) | QLIKE | R2 oos vs HAR-RV | DM stat (QLIKE) | DM p-value |
|---|---:|---:|---:|---:|---:|
| Random walk (RV22) | 0.3124 | 0.1623 | -0.2567 | 2.4632 | 0.0140 |
| Implied vol | 0.2411 | 0.1006 | 0.2517 | -3.0648 | 0.0023 |
| HAR-RV | 0.2787 | 0.1405 | 0.0000 |  |  |
| HAR-RV-IV | 0.2292 | 0.1193 | 0.3234 | -1.2684 | 0.2051 |
| Gradient boosting | 0.2420 | 0.1309 | 0.2457 | -0.4207 | 0.6741 |
| GRU + linear highway | 0.2296 | 0.1045 | 0.3209 | -2.3423 | 0.0195 |

## VRP backtest

Sell (or size) a 1M vol swap struck at the vol index each day, 0.25 vol pt half-spread; timed strategies sell only when predicted VRP > 0.5 vol pts (threshold fixed ex ante). Sharpe and drawdown use non-overlapping trades averaged over all 21 roll offsets. All six variants tested are reported.

| | Active % | Avg P&L / trade | Hit rate % | Sharpe | Max drawdown | Worst 5% trades (avg) | Total P&L / yr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Always short vol | 100.00 | 0.74 | 68.35 | 1.34 | 9.34 | -3.69 | 8.89 |
| HAR-RV-IV timed | 97.96 | 0.75 | 68.47 | 1.34 | 9.32 | -3.68 | 8.80 |
| GRU timed | 63.60 | 0.84 | 70.05 | 1.16 | 7.15 | -3.52 | 6.42 |
| Gradient boosting timed | 67.86 | 0.84 | 70.36 | 1.19 | 8.86 | -3.53 | 6.82 |
| Vol-managed (1/vol) | 100.00 | 0.75 | 68.35 | 1.27 | 12.18 | -4.27 | 9.01 |
| VRP/vol sized (HAR-RV-IV) | 100.00 | 0.76 | 68.35 | 1.19 | 10.96 | -4.04 | 9.15 |
