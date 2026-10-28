"""PPO market-maker (Schulman et al., 2017), written from scratch in PyTorch.

The agent acts as a **residual** on the tuned model-based quoter: it outputs a
correction to the quoted edge and to the interbank hedge. With a zero action it
reproduces the baseline exactly, so the policy only has to learn where state
information (inventory, regime proxies, informed-flow imbalance, time of day)
justifies deviating. Training maximises the same mean-variance reward used to
tune the baselines, with model selection on separate validation episodes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from deltasigma.mm.env import OBS_NAMES, MarketConfig, RFQMarket
from deltasigma.mm.policies import ModelQuoter, band_hedge, evaluate


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 128
    n_updates: int = 150
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64
    init_log_std: float = -1.2
    edge_scale: float = 0.1
    reward_scale: float = 0.2
    eval_every: int = 10
    eval_episodes: int = 512
    seed: int = 0


class ActorCritic(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden: int, init_log_std: float) -> None:
        super().__init__()

        def mlp(n_out: int, gain: float) -> nn.Sequential:
            net = nn.Sequential(
                nn.Linear(n_obs, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, n_out),
            )
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.zeros_(m.bias)
            nn.init.orthogonal_(net[-1].weight, gain)
            return net

        self.pi = mlp(n_act, 0.01)
        self.v = mlp(1, 1.0)
        self.log_std = nn.Parameter(torch.full((n_act,), init_log_std))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.pi(obs), self.log_std.exp().expand(obs.shape[0], -1))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


@dataclass
class PPOHistory:
    mean_reward: list[float] = field(default_factory=list)
    val_score: list[tuple[int, float]] = field(default_factory=list)
    best_update: int = -1


class PPOMarketMaker:
    N_ACT = 2

    def __init__(self, base: ModelQuoter, market: MarketConfig, config: PPOConfig) -> None:
        self.base, self.market, self.cfg = base, market, config
        torch.manual_seed(config.seed)
        self.net = ActorCritic(len(OBS_NAMES) + 2, self.N_ACT, config.hidden, config.init_log_std)
        self.history = PPOHistory()

    # ------------------------------------------------------------------ env interface
    def _features(self, env: RFQMarket) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base_edge = self.base.quote(env)
        base_hedge = band_hedge(env.q, self.base.band)
        x = np.column_stack([env.observe(), (base_edge - 0.2) / 0.1,
                             (env.q + base_hedge) / 20.0])
        return x.astype(np.float32), base_edge, base_hedge

    def _to_env(self, env: RFQMarket, action: np.ndarray, base_edge, base_hedge):
        edge = base_edge + self.cfg.edge_scale * action[:, 0]
        frac = 0.5 * np.tanh(action[:, 1])
        hedge = base_hedge - frac * (env.q + base_hedge)
        return edge, hedge

    def __call__(self, env: RFQMarket) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic (mean-action) policy for evaluation."""
        x, base_edge, base_hedge = self._features(env)
        with torch.no_grad():
            action = self.net.pi(torch.as_tensor(x)).numpy()
        return self._to_env(env, action, base_edge, base_hedge)

    # ------------------------------------------------------------------ training
    def train(self, train_seed: int, val_seed: int) -> PPOHistory:
        cfg, T = self.cfg, self.market.n_steps
        env = RFQMarket(self.market, cfg.n_envs, seed=train_seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        n_obs = len(OBS_NAMES) + 2
        best_score, best_state = -np.inf, None
        self.history = PPOHistory()

        for update in range(cfg.n_updates):
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1.0 - update / cfg.n_updates)
            obs_buf = np.zeros((T, cfg.n_envs, n_obs), np.float32)
            act_buf = np.zeros((T, cfg.n_envs, self.N_ACT), np.float32)
            logp_buf = np.zeros((T, cfg.n_envs), np.float32)
            val_buf = np.zeros((T + 1, cfg.n_envs), np.float32)
            rew_buf = np.zeros((T, cfg.n_envs), np.float32)

            env.reset()
            for t in range(T):
                x, base_edge, base_hedge = self._features(env)
                with torch.no_grad():
                    xt = torch.as_tensor(x)
                    dist = self.net.dist(xt)
                    a = dist.sample()
                    logp_buf[t] = dist.log_prob(a).sum(-1).numpy()
                    val_buf[t] = self.net.value(xt).numpy()
                obs_buf[t], act_buf[t] = x, a.numpy()
                edge, hedge = self._to_env(env, a.numpy(), base_edge, base_hedge)
                _, reward, _, _ = env.step(edge, hedge)
                rew_buf[t] = reward * cfg.reward_scale

            adv = np.zeros_like(rew_buf)
            last = np.zeros(cfg.n_envs, np.float32)
            for t in reversed(range(T)):
                delta = rew_buf[t] + cfg.gamma * val_buf[t + 1] - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * last
                adv[t] = last
            ret = adv + val_buf[:T]

            b_obs = torch.as_tensor(obs_buf.reshape(-1, n_obs))
            b_act = torch.as_tensor(act_buf.reshape(-1, self.N_ACT))
            b_logp = torch.as_tensor(logp_buf.reshape(-1))
            b_adv = torch.as_tensor(adv.reshape(-1))
            b_ret = torch.as_tensor(ret.reshape(-1))
            n = b_obs.shape[0]
            mb = n // cfg.minibatches
            for _ in range(cfg.epochs):
                perm = torch.randperm(n)
                for i in range(cfg.minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    dist = self.net.dist(b_obs[idx])
                    ratio = (dist.log_prob(b_act[idx]).sum(-1) - b_logp[idx]).exp()
                    a_n = (b_adv[idx] - b_adv[idx].mean()) / (b_adv[idx].std() + 1e-8)
                    pg = -torch.min(ratio * a_n,
                                    ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_n).mean()
                    v_loss = ((self.net.value(b_obs[idx]) - b_ret[idx]) ** 2).mean()
                    loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * dist.entropy().sum(-1).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                    opt.step()

            self.history.mean_reward.append(float(rew_buf.sum(0).mean() / cfg.reward_scale))
            if (update + 1) % cfg.eval_every == 0 or update == cfg.n_updates - 1:
                score = evaluate(self, self.market, cfg.eval_episodes, val_seed).risk_adjusted
                self.history.val_score.append((update, score))
                if score > best_score:
                    best_score, best_state = score, copy.deepcopy(self.net.state_dict())
                    self.history.best_update = update
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self.history
