"""Deep hedging (Buehler, Gonon, Teichmann & Wood, 2019) for an FX options book.

A recurrent policy maps observable market state plus its current position to
new hedge positions at every rebalancing date, and is trained end to end
through the simulated terminal P&L (including transaction costs) to minimise
CVaR via the Rockafellar-Uryasev representation

    CVaR_a(L) = min_w  w + E[(L - w)^+] / (1 - a).

The policy is parameterised as a *residual* on the classical Black delta
(-vega) hedge: it starts as the textbook hedge and learns where deviating
pays (no-trade regions, skew/vol-dynamics corrections) instead of
rediscovering option theory from scratch. Model selection uses a held-out
validation market with early stopping.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from deltasigma.hedging.market import HedgingMarket
from deltasigma.hedging.pnl import PIP, CostModel, cost_rates, cvar
from deltasigma.hedging.strategies import vega_hedge_ratio


@dataclass(frozen=True)
class DeepHedgerConfig:
    instruments: tuple[str, ...] = ("forward",)
    hidden: int = 64
    depth: int = 3
    alpha: float = 0.95
    epochs: int = 40
    batch_size: int = 4096
    lr: float = 2e-3
    patience: int = 8
    residual: bool = True
    seed: int = 0


@dataclass(frozen=True)
class FeatureScaler:
    """Normalisation constants fitted on the training market only (no test leakage)."""

    F_ref: float
    sigma_ref: float
    maturity: float
    vega_scale: float

    @classmethod
    def fit(cls, market: HedgingMarket) -> FeatureScaler:
        return cls(
            F_ref=float(market.forward[:, 0].mean()),
            sigma_ref=float(market.implied_vol[:, 0].mean()),
            maturity=market.maturity,
            vega_scale=float(np.abs(vega_hedge_ratio(market)[:, 0]).mean()) or 1.0,
        )


def build_features(
    market: HedgingMarket, scaler: FeatureScaler, instruments: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features (P,N,f), baseline (P,N,n), position_scale (n,))``."""
    F = market.forward[:, :-1]
    tau = np.broadcast_to(market.tau[None, :], F.shape)
    _, delta, gamma, _ = market.book_greeks()
    vega_ratio = vega_hedge_ratio(market)
    sd_ref = scaler.sigma_ref * np.sqrt(scaler.maturity)
    feats = [
        np.log(F / scaler.F_ref) / sd_ref,
        tau / scaler.maturity,
        market.implied_vol / scaler.sigma_ref - 1.0,
        delta,
        gamma * F * sd_ref,
    ]
    base_cols = {"forward": delta, "variance_swap": vega_ratio}
    scale_cols = {"forward": 1.0, "variance_swap": scaler.vega_scale}
    if "variance_swap" in instruments:
        feats.append(vega_ratio / scaler.vega_scale)
    baseline = np.stack([base_cols[i] for i in instruments], axis=-1)
    scale = np.array([scale_cols[i] for i in instruments])
    return np.stack(feats, axis=-1), baseline, scale


class HedgeNet(nn.Module):
    def __init__(self, n_features: int, n_out: int, hidden: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = n_features
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


@dataclass
class TrainingHistory:
    train_cvar: list[float] = field(default_factory=list)
    val_cvar: list[float] = field(default_factory=list)
    best_epoch: int = -1


class DeepHedger:
    def __init__(self, config: DeepHedgerConfig, costs: CostModel) -> None:
        self.config, self.costs = config, costs
        self.scaler: FeatureScaler | None = None
        self.net: HedgeNet | None = None
        self.history = TrainingHistory()

    @property
    def instruments(self) -> tuple[str, ...]:
        return self.config.instruments

    def _tensors(self, market: HedgingMarket) -> dict[str, torch.Tensor]:
        assert self.scaler is not None
        feats, base, scale = build_features(market, self.scaler, self.instruments)

        def t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)

        return {
            "feats": t(feats),
            "base": t(base),
            "scale": t(scale),
            "dX": t(np.diff(market.prices(self.instruments), axis=1) / PIP),
            "cost": t(cost_rates(market, self.costs, self.instruments) / PIP),
            "payoff": t(market.payoff / PIP),
        }

    def _rollout(
        self, feats: torch.Tensor, base: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        assert self.net is not None
        n_paths, n_steps, _ = feats.shape
        h_prev = torch.zeros(n_paths, base.shape[-1])
        out = []
        for k in range(n_steps):
            x = torch.cat([feats[:, k], (h_prev - base[:, k]) / scale], dim=-1)
            h = self.net(x) * scale
            if self.config.residual:
                h = h + base[:, k]
            out.append(h)
            h_prev = h
        return torch.stack(out, dim=1)

    @staticmethod
    def _loss(pos: torch.Tensor, d: dict[str, torch.Tensor]) -> torch.Tensor:
        gains = (pos * d["dX"]).sum(dim=(1, 2))
        trades = torch.diff(pos, dim=1, prepend=torch.zeros_like(pos[:, :1]))
        cost = (d["cost"] * trades.abs()).sum(dim=(1, 2))
        return d["payoff"] - gains + cost

    def fit(self, train: HedgingMarket, val: HedgingMarket) -> TrainingHistory:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        self.scaler = FeatureScaler.fit(train)
        tr, va = self._tensors(train), self._tensors(val)
        n_feat = tr["feats"].shape[-1] + len(self.instruments)
        self.net = HedgeNet(n_feat, len(self.instruments), cfg.hidden, cfg.depth)

        with torch.no_grad():
            init_loss = self._loss(self._rollout(tr["feats"], tr["base"], tr["scale"]), tr)
        w = nn.Parameter(torch.quantile(init_loss, cfg.alpha).reshape(()))
        opt = torch.optim.Adam([*self.net.parameters(), w], lr=cfg.lr)
        n = tr["feats"].shape[0]
        steps_per_epoch = max(1, n // cfg.batch_size)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * steps_per_epoch)
        )
        gen = torch.Generator().manual_seed(cfg.seed)

        best_val, best_state, bad_epochs = np.inf, None, 0
        self.history = TrainingHistory()
        for epoch in range(cfg.epochs):
            perm = torch.randperm(n, generator=gen)
            batch_scores = []
            for b in range(steps_per_epoch):
                idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
                batch = {k: v[idx] if v.dim() and v.shape[0] == n else v for k, v in tr.items()}
                pos = self._rollout(batch["feats"], batch["base"], batch["scale"])
                loss = self._loss(pos, batch)
                objective = w + torch.relu(loss - w).mean() / (1.0 - cfg.alpha)
                opt.zero_grad()
                objective.backward()
                opt.step()
                sched.step()
                batch_scores.append(float(objective.detach()))
            val_score = cvar(self._evaluate(va), cfg.alpha)
            self.history.train_cvar.append(float(np.mean(batch_scores)))
            self.history.val_cvar.append(val_score)
            if val_score < best_val - 1e-6:
                best_val, bad_epochs = val_score, 0
                best_state = copy.deepcopy(self.net.state_dict())
                self.history.best_epoch = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self.history

    def _evaluate(self, d: dict[str, torch.Tensor]) -> np.ndarray:
        with torch.no_grad():
            return self._loss(self._rollout(d["feats"], d["base"], d["scale"]), d).numpy()

    def positions(self, market: HedgingMarket) -> np.ndarray:
        """Positions in ``(forward, variance_swap)`` units, zero for unused instruments."""
        if self.net is None:
            raise RuntimeError("call fit() first")
        d = self._tensors(market)
        with torch.no_grad():
            pos = self._rollout(d["feats"], d["base"], d["scale"]).double().numpy()
        out = np.zeros((market.n_paths, market.n_steps, 2))
        for j, name in enumerate(self.instruments):
            out[..., ("forward", "variance_swap").index(name)] = pos[..., j]
        return out

    def trade_response(
        self, market: HedgingMarket, step: int, deviations: np.ndarray, path: int = 0
    ) -> np.ndarray:
        """Forward trade chosen when the inherited position deviates from Black delta
        by ``deviations`` at ``(path, step)``. A flat region would indicate a no-trade
        band; a slope shallower than -1 indicates partial adjustment towards delta.
        """
        if self.net is None:
            raise RuntimeError("call fit() first")
        d = self._tensors(market)
        f = d["feats"][path, step].expand(len(deviations), -1)
        base = d["base"][path, step].expand(len(deviations), -1)
        prev = base.clone()
        prev[:, 0] = prev[:, 0] + torch.as_tensor(deviations, dtype=torch.float32)
        with torch.no_grad():
            x = torch.cat([f, (prev - base) / d["scale"]], dim=-1)
            h = self.net(x) * d["scale"] + (base if self.config.residual else 0.0)
        return (h - prev)[:, 0].numpy()
