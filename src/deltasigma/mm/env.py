"""Vectorised FX options RFQ market-making simulator.

Each step one client RFQ arrives for ATM EURUSD vol (a vega-notional option
trade). The dealer quotes an *edge* ``delta`` in vol points away from mid on the
client's side and may hedge vega in the interbank market. Units: vol points
(1 = 1% vol) and vega units (P&L of 1 per vol point).

Market features:

* **Regime-switching vol**: mean-reverting ATM vol with calm/stress regimes
  (stress: higher level, higher vol-of-vol, wider interbank spreads, clients more
  willing to pay).
* **Client tiers**: tier 0 (real money) is uninformed and price-insensitive; tier 1
  (fast money) is price-sensitive and partly **informed**: with probability
  ``informed_prob`` its side matches the sign of the vol move over the next
  ``informed_horizon`` steps, creating adverse selection for the dealer.
* **Win probability**: ``P(accept) = sigmoid(alpha_tier + alpha_stress - beta_tier * delta)``.

All randomness for an episode is drawn at ``reset()``, so different policies run
with the same seed see identical vol paths, RFQs and acceptance draws (common
random numbers), which makes strategy comparisons paired and low-noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBS_NAMES = (
    "inventory",
    "vol_level",
    "vol_of_vol",
    "flow_tier1",
    "flow_all",
    "rfq_side",
    "rfq_size",
    "rfq_tier",
    "time_left",
)


@dataclass(frozen=True)
class MarketConfig:
    n_steps: int = 200
    vol_level: tuple[float, float] = (7.0, 11.0)        # long-run ATM vol by regime
    vol_of_vol: tuple[float, float] = (0.04, 0.10)      # per-step std of vol moves
    jump_prob: tuple[float, float] = (0.003, 0.006)     # per-step probability of a vol jump
    jump_size: tuple[float, float] = (0.8, 1.5)         # jump std (vol pts): event risk
    mean_reversion: float = 0.01
    p_to_stress: float = 0.002
    p_to_calm: float = 0.01
    tier_probs: tuple[float, float] = (0.7, 0.3)
    win_alpha: tuple[float, float] = (1.5, 2.0)
    win_beta: tuple[float, float] = (10.0, 18.0)
    stress_alpha_shift: float = 0.5
    informed_prob: tuple[float, float] = (0.0, 0.5)
    informed_horizon: int = 30
    max_size: int = 5
    hedge_half_spread: tuple[float, float] = (0.15, 0.30)
    risk_aversion: float = 2.0

    def win_probability(self, delta, tier, stress):
        tier = np.asarray(tier, dtype=int)
        alpha = np.asarray(self.win_alpha)[tier] + self.stress_alpha_shift * np.asarray(stress)
        return 1.0 / (1.0 + np.exp(-(alpha - np.asarray(self.win_beta)[tier] * delta)))


class RFQMarket:
    """``n_envs`` synchronous episodes; call ``reset()`` then ``step()`` ``n_steps`` times."""

    def __init__(
        self,
        config: MarketConfig,
        n_envs: int,
        seed: int | None = None,
        start_regime: int | None = None,
    ) -> None:
        self.cfg, self.n_envs, self.start_regime = config, n_envs, start_regime
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ episode generation
    def _generate(self) -> None:
        c, N, rng = self.cfg, self.n_envs, self.rng
        T, H = c.n_steps, c.informed_horizon
        L = T + H + 1
        regime = np.empty((N, L), dtype=int)
        if self.start_regime is None:
            pi_stress = c.p_to_stress / (c.p_to_stress + c.p_to_calm)
            regime[:, 0] = rng.random(N) < pi_stress
        else:
            regime[:, 0] = self.start_regime
        u = rng.random((N, L))
        for t in range(1, L):
            flip_p = np.where(regime[:, t - 1] == 1, c.p_to_calm, c.p_to_stress)
            regime[:, t] = np.where(u[:, t] < flip_p, 1 - regime[:, t - 1], regime[:, t - 1])

        level, vov = np.asarray(c.vol_level), np.asarray(c.vol_of_vol)
        jp, js = np.asarray(c.jump_prob), np.asarray(c.jump_size)
        vol = np.empty((N, L))
        vol[:, 0] = level[regime[:, 0]] + rng.normal(0.0, 0.5, N)
        eps = rng.standard_normal((N, L))
        jump_u, jump_z = rng.random((N, L)), rng.standard_normal((N, L))
        for t in range(1, L):
            r = regime[:, t - 1]
            jump = np.where(jump_u[:, t] < jp[r], js[r] * jump_z[:, t], 0.0)
            vol[:, t] = np.maximum(
                vol[:, t - 1] + c.mean_reversion * (level[r] - vol[:, t - 1]) + vov[r] * eps[:, t]
                + jump,
                1.0,
            )

        tier = (rng.random((N, T)) < c.tier_probs[1]).astype(int)
        informed = rng.random((N, T)) < np.asarray(c.informed_prob)[tier]
        future_move = np.sign(vol[:, H : H + T] - vol[:, :T])
        future_move[future_move == 0] = 1.0
        random_side = np.where(rng.random((N, T)) < 0.5, 1.0, -1.0)
        self.vol, self.regime = vol, regime
        self.rfq_tier, self.rfq_informed = tier, informed
        self.rfq_side = np.where(informed, future_move, random_side)
        self.rfq_size = rng.integers(1, c.max_size + 1, (N, T)).astype(float)
        self.u_accept = rng.random((N, T))

    def reset(self) -> np.ndarray:
        self._generate()
        N = self.n_envs
        self.t = 0
        self.q = np.zeros(N)
        self.pnl = np.zeros(N)
        self.vov_ewma = np.full(N, np.mean(self.cfg.vol_of_vol))
        self.flow1 = np.zeros(N)
        self.flow_all = np.zeros(N)
        self.stats = {k: np.zeros(N) for k in
                      ("spread", "mtm", "hedge_cost", "fills", "rfqs", "abs_inventory",
                       "markout", "variance_penalty")}
        return self.observe()

    # ------------------------------------------------------------------ observation
    @property
    def stress(self) -> np.ndarray:
        return self.regime[:, self.t]

    def current_rfq(self) -> dict[str, np.ndarray]:
        t = self.t
        return {"side": self.rfq_side[:, t], "size": self.rfq_size[:, t],
                "tier": self.rfq_tier[:, t]}

    def observe(self) -> np.ndarray:
        t = min(self.t, self.cfg.n_steps - 1)
        return np.column_stack([
            self.q / 20.0,
            (self.vol[:, self.t] - 8.0) / 3.0,
            self.vov_ewma / 0.05 - 1.0,
            self.flow1,
            self.flow_all,
            self.rfq_side[:, t],
            self.rfq_size[:, t] / self.cfg.max_size,
            self.rfq_tier[:, t].astype(float),
            np.full(self.n_envs, 1.0 - t / self.cfg.n_steps),
        ])

    # ------------------------------------------------------------------ dynamics
    def _step_variance(self, regime: np.ndarray) -> np.ndarray:
        """Per-step variance of vol moves (diffusion + jumps) in each regime."""
        c = self.cfg
        var = np.asarray(c.vol_of_vol) ** 2 + np.asarray(c.jump_prob) * np.asarray(c.jump_size) ** 2
        return var[regime]

    def step(self, edge: np.ndarray, hedge: np.ndarray):
        """Quote ``edge`` (vol pts from mid on the client's side) and trade ``hedge``
        vega units interbank. Returns ``(obs, reward, done, info)``.
        """
        c, t = self.cfg, self.t
        side, size, tier = self.rfq_side[:, t], self.rfq_size[:, t], self.rfq_tier[:, t]
        stress = self.regime[:, t]
        edge = np.asarray(edge, dtype=float)
        hedge = np.asarray(hedge, dtype=float)

        p_win = c.win_probability(edge, tier, stress)
        filled = self.u_accept[:, t] < p_win
        dq_client = np.where(filled, -side * size, 0.0)
        spread = np.where(filled, size * edge, 0.0)
        H = c.informed_horizon
        markout = dq_client * (self.vol[:, t + H] - self.vol[:, t])

        hedge_spread = np.asarray(c.hedge_half_spread)[stress]
        self.q = self.q + dq_client + hedge
        hedge_cost = hedge_spread * np.abs(hedge)

        dvol = self.vol[:, t + 1] - self.vol[:, t]
        mtm = self.q * dvol
        var_penalty = 0.5 * c.risk_aversion * self.q**2 * self._step_variance(self.regime[:, t])

        step_pnl = spread + mtm - hedge_cost
        done = t == c.n_steps - 1
        if done:
            liq = np.asarray(c.hedge_half_spread)[self.regime[:, t + 1]] * np.abs(self.q)
            step_pnl = step_pnl - liq
            hedge_cost = hedge_cost + liq
            self.q = np.zeros_like(self.q)
        self.pnl += step_pnl

        s = self.stats
        s["spread"] += spread
        s["mtm"] += mtm
        s["hedge_cost"] += hedge_cost
        s["fills"] += filled
        s["rfqs"] += 1.0
        s["abs_inventory"] += np.abs(self.q)
        s["markout"] += markout
        s["variance_penalty"] += var_penalty

        self.vov_ewma = 0.9 * self.vov_ewma + 0.1 * np.abs(dvol)
        is_t1 = tier == 1
        self.flow1 = np.where(is_t1, 0.8 * self.flow1 + 0.2 * side, self.flow1)
        self.flow_all = 0.9 * self.flow_all + 0.1 * side
        self.t += 1
        obs = self.observe() if not done else np.zeros((self.n_envs, len(OBS_NAMES)))
        return obs, step_pnl - var_penalty, done, {"pnl": step_pnl, "filled": filled}


def episode_summary(env: RFQMarket) -> dict[str, np.ndarray]:
    """Per-episode statistics after an episode has finished."""
    s = env.stats
    return {
        "pnl": env.pnl.copy(),
        "spread": s["spread"].copy(),
        "mtm": s["mtm"].copy(),
        "hedge_cost": s["hedge_cost"].copy(),
        "fill_rate": s["fills"] / s["rfqs"],
        "avg_abs_inventory": s["abs_inventory"] / s["rfqs"],
        "markout": s["markout"].copy(),
        "inventory_penalty": s["variance_penalty"].copy(),
    }
