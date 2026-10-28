# Production results

## A. Greek P&L explain

Book: short 1Y 95% put (3m EUR), short 1Y 106% call (2m), long 11M ATM straddle (2.5m each), marked daily on a simulated regime-switching EURUSD path (sticky-strike smile offsets, parallel vol shifts).

| | Days | Explain ratio % | Mean |unexplained| (USD) | Max |unexplained| (USD) | Mean |actual| (USD) |
|---|---:|---:|---:|---:|---:|
| All days | 189 | 99.99 | 17.45 | 270.05 | 2762.42 |
| Calm regime | 140 | 99.99 | 19.33 | 270.05 | 2912.58 |
| Stress regime | 49 | 99.99 | 12.09 | 107.11 | 2333.40 |

Unexplained-P&L limit (2% of mean |daily P&L| = 55 USD) breached on 10 days, 6 of them with a >1 vol pt implied-vol move. The residual tracks large vol moves (correlation ~0.65 with |d vol|, ~0.06 with |d spot|): third-order vol risk (ultima) and vol-time cross terms are not in the explain.

## B. Drift monitoring (FRED DEXUSEU + EVZCLS)

HAR-RV-IV forecasts from the Phase 3 walk-forward. QLIKE losses enter the monitor on the date the forecast *matures* (t + 21), the earliest they are knowable. Reference = first 252 matured losses; alerts have a 63-day cooldown (one ticket per episode). Feature drift compares each 126-day window with all data the expanding-window model has been trained on so far.

- Page-Hinkley alarms: 2 on 2020-03-12, 2022-03-15
- CUSUM (k = 0.5, h = 12) alarms: 26 on 2013-08-15, 2014-04-21, 2014-09-08, 2014-12-22, 2015-03-31, 2015-08-27, 2015-12-15, 2016-06-28, 2017-01-05, 2017-07-26, 2018-01-24, 2019-06-03
- Out-of-range (> 10% of window outside the training 0.5-99.5% range) alerts: {'OOD[log_rv_22]': 21, 'OOD[log_iv]': 28, 'OOD[iv_rv_spread]': 6} over 151 windows; first on 2012-05-25, 2012-06-26, 2012-07-26, 2012-08-24, 2012-09-25, 2012-10-25, 2012-11-29, 2012-12-31, 2013-01-31, 2013-03-04, 2013-04-03, 2013-05-02
- PSI > 0.25 alerts, for comparison: {'PSI[log_iv]': 151, 'PSI[log_rv_22]': 151, 'PSI[iv_rv_spread]': 149} over 151 windows

## C. Regime-gated deployment of the PPO quoter

Risk-adjusted P&L per day on the same 4,000 paired test days as Phase 4; t-stats are paired vs B4. The HMM threshold (0.5) was fixed ex ante.

| | Normal days | Normal days t vs B4 | Stress-start days | Stress-start days t vs B4 |
|---|---:|---:|---:|---:|
| B4 classical | 20.76 | nan | 13.17 | nan |
| PPO always | 20.59 | -2.74 | 14.74 | 19.24 |
| HMM-gated (B4 calm / PPO stress) | 21.27 | 11.95 | 14.69 | 21.99 |

Share of RFQs routed to PPO: normal days 11.4%, stress-start days 32.1%. Registry entry: `rfq_market_maker` v1 (config hash `7184f3c8364b`).

## D. Service latency (server-side processing time, single worker)

| | p50 ms | p99 ms |
|---|---:|---:|
| POST /price (new market snapshot) | 59.59 | 158.06 |
| POST /price (cached smile) | 8.77 | 30.55 |
| POST /quote | 4.73 | 11.06 |
