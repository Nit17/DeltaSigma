"""Market data: real EURUSD spot + implied vol from FRED, and a synthetic twin.

Real data
    * ``DEXUSEU``: EURUSD noon buying rate (Federal Reserve H.10), daily.
    * ``EVZCLS``: CBOE EuroCurrency ETF Volatility Index, a 30-day EURUSD
      implied-vol index (2007-11 to 2025-03, now discontinued).

The raw files are downloaded on demand and cached locally (``data/raw``, git-ignored);
they are not redistributed with the repository.

Synthetic data
    The regime-switching Heston simulator plus an "implied vol" equal to the exact
    expected future realised vol, scaled by a variance risk premium and observed
    with noise. Used for tests, CI, and as a sanity check where the truth is known.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from deltasigma.hedging.market import remaining_variance_weights
from deltasigma.sim.regime import RegimeSwitchingHeston, eurusd_calm_stress_model

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DEFAULT_CACHE = Path("data") / "raw"


def fetch_fred(
    series_id: str, cache_dir: str | Path = DEFAULT_CACHE, refresh: bool = False, timeout: int = 60
) -> pd.Series:
    """Download (or read from cache) a daily FRED series; missing values are dropped."""
    path = Path(cache_dir) / f"{series_id}.csv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FRED_CSV.format(series=series_id), timeout=timeout) as resp:
            path.write_bytes(resp.read())
    raw = pd.read_csv(path, na_values=["."])
    values = pd.to_numeric(raw[series_id], errors="coerce").to_numpy(dtype=float)
    series = pd.Series(values, index=pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0])),
                       name=series_id)
    return series.dropna()


def load_eurusd_vol(cache_dir: str | Path = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Daily EURUSD spot and 30-day implied vol (decimal) on common dates."""
    spot = fetch_fred("DEXUSEU", cache_dir, refresh)
    iv = fetch_fred("EVZCLS", cache_dir, refresh) / 100.0
    df = pd.concat({"spot": spot, "iv": iv}, axis=1, join="inner").dropna()
    return df[(df["spot"] > 0) & (df["iv"] > 0)]


def synthetic_vol_dataset(
    n_days: int = 3000,
    seed: int = 0,
    horizon: int = 21,
    vrp: float = 0.08,
    iv_noise: float = 0.002,
    model: RegimeSwitchingHeston | None = None,
) -> pd.DataFrame:
    """Daily spot / implied vol with known dynamics; includes the true ``regime``."""
    model = model or eurusd_calm_stress_model()
    dt = 1.0 / 252.0
    sim = model.simulate(1.10, n_days * dt, n_days, 1, seed=seed)
    n = len(model.regimes)
    W = remaining_variance_weights(model, dt, horizon)
    v, r = sim.variance[0], sim.regime[0]
    fair_vol = np.sqrt((W[horizon, r] * v + W[horizon, n + r]) / (horizon * dt))
    rng = np.random.default_rng(seed + 1)
    iv = fair_vol * (1.0 + vrp) + iv_noise * rng.standard_normal(n_days + 1)
    index = pd.bdate_range("2005-01-03", periods=n_days + 1, name="date")
    return pd.DataFrame(
        {"spot": sim.spot[0], "iv": np.maximum(iv, 1e-3), "regime": r.astype(int)}, index=index
    )
