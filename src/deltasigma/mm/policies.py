"""Classical quoting and hedging policies, evaluation and tuning.

A policy is a callable ``policy(env) -> (edge, hedge)`` evaluated on the current
RFQ of every environment.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np

from deltasigma.mm.env import MarketConfig, RFQMarket, episode_summary
from deltasigma.mm.winprob import WinProbabilityModel

Policy = Callable[[RFQMarket], tuple[np.ndarray, np.ndarray]]


def band_hedge(q: np.ndarray, band: float) -> np.ndarray:
    """Interbank vega trade that brings inventory back inside ``[-band, band]``."""
    return np.clip(q, -band, band) - q


@dataclass
class FixedSpreadQuoter:
    edge: float = 0.25
    band: float = np.inf

    def __call__(self, env: RFQMarket) -> tuple[np.ndarray, np.ndarray]:
        return np.full(env.n_envs, self.edge), band_hedge(env.q, self.band)


@dataclass
class ModelQuoter:
    """Myopic expected-profit quote from the estimated win-probability model.

    The quote maximises ``(edge - cost) * P(win | edge)`` where the per-unit cost is

    * ``markout_scale * (markout[tier] + flow_coef * side * flow_tier1)``: expected
      adverse selection, optionally conditioned on recent informed (tier-1) flow.
      Markouts are measured over a fixed horizon, but band hedging cuts the actual
      holding period, so the fraction actually priced in is a tuned parameter;
    * ``inventory_aversion * (size / 2 - side * q)``: the marginal mean-variance
      cost of the inventory change (Avellaneda-Stoikov / Gueant-style skew:
      quotes tighten when a fill reduces risk and widen when it adds risk).

    Inventory outside ``[-band, band]`` is hedged interbank.
    """

    win_model: WinProbabilityModel
    markout: np.ndarray = field(default_factory=lambda: np.zeros(2))
    inventory_aversion: float = 0.0
    band: float = np.inf
    flow_coef: float = 0.0
    markout_scale: float = 1.0

    def quote(self, env: RFQMarket) -> np.ndarray:
        rfq = env.current_rfq()
        obs = env.observe()
        adverse = np.asarray(self.markout)[rfq["tier"]] + self.flow_coef * rfq["side"] * env.flow1
        cost = self.markout_scale * adverse + self.inventory_aversion * (
            0.5 * rfq["size"] - rfq["side"] * env.q
        )
        return self.win_model.optimal_edge(rfq["tier"], obs[:, 1], obs[:, 2], cost)

    def __call__(self, env: RFQMarket) -> tuple[np.ndarray, np.ndarray]:
        return self.quote(env), band_hedge(env.q, self.band)


@dataclass(frozen=True)
class EvalResult:
    episodes: dict[str, np.ndarray]

    @property
    def pnl(self) -> np.ndarray:
        return self.episodes["pnl"]

    @property
    def objective(self) -> np.ndarray:
        """Per-episode P&L minus the running inventory-risk penalty
        ``sum_t (gamma / 2) q_t^2 Var(d vol)`` (Cartea-Jaimungal style)."""
        return self.pnl - self.episodes["inventory_penalty"]

    @property
    def risk_adjusted(self) -> float:
        return float(self.objective.mean())

    def summary(self) -> dict[str, float]:
        e, pnl = self.episodes, self.pnl
        n_tail = max(1, int(np.ceil(0.05 * pnl.size)))
        return {
            "Risk-adj P&L": self.risk_adjusted,
            "Mean P&L": float(pnl.mean()),
            "Std P&L": float(pnl.std()),
            "CVaR5 (worst days)": float(np.sort(pnl)[:n_tail].mean()),
            "Spread captured": float(e["spread"].mean()),
            "Inventory MTM": float(e["mtm"].mean()),
            "Hedge cost": float(e["hedge_cost"].mean()),
            "Fill rate %": 100.0 * float(e["fill_rate"].mean()),
            "Avg |inventory|": float(e["avg_abs_inventory"].mean()),
        }


def evaluate(
    policy: Policy,
    config: MarketConfig,
    n_episodes: int,
    seed: int,
    start_regime: int | None = None,
) -> EvalResult:
    env = RFQMarket(config, n_episodes, seed=seed, start_regime=start_regime)
    env.reset()
    for _ in range(config.n_steps):
        edge, hedge = policy(env)
        env.step(edge, hedge)
    return EvalResult(episode_summary(env))


def tune(
    make_policy: Callable[..., Policy],
    grid: dict[str, Iterable],
    config: MarketConfig,
    n_episodes: int,
    seed: int,
) -> tuple[dict, float]:
    """Grid search on *training* seeds maximising the risk-adjusted objective."""
    keys = list(grid)
    best, best_score = {}, -np.inf
    for values in product(*grid.values()):
        params = dict(zip(keys, values, strict=True))
        score = evaluate(make_policy(**params), config, n_episodes, seed).risk_adjusted
        if score > best_score:
            best, best_score = params, score
    return best, best_score


def with_risk_aversion(config: MarketConfig, gamma: float) -> MarketConfig:
    return replace(config, risk_aversion=gamma)
