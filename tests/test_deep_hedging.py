import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deltasigma.hedging import (  # noqa: E402
    CostModel,
    OptionBook,
    VanillaOption,
    black_delta,
    cvar,
    generate_market,
    hedge_outcome,
)
from deltasigma.hedging.deep import DeepHedger, DeepHedgerConfig  # noqa: E402
from deltasigma.sim import eurusd_calm_stress_model  # noqa: E402

S0, T, N = 1.085, 0.25, 21
MODEL = eurusd_calm_stress_model()
F0 = S0 * np.exp((MODEL.rd - MODEL.rf) * T)
BOOK = OptionBook.of([VanillaOption(F0, call=True)])
COSTS = CostModel(forward_bps=5.0, varswap_vol_pts=0.2)


@pytest.fixture(scope="module")
def markets():
    return tuple(generate_market(MODEL, BOOK, S0, T, N, n, seed=s)
                 for n, s in ((6_000, 1), (3_000, 2), (6_000, 3)))


def _config(**kw):
    base = dict(epochs=6, batch_size=1024, hidden=32, depth=2, patience=10, seed=0)
    return DeepHedgerConfig(**{**base, **kw})


def test_untrained_residual_policy_is_black_delta(markets):
    train, val, test = markets
    hedger = DeepHedger(_config(epochs=0), COSTS)
    hedger.fit(train, val)
    np.testing.assert_allclose(hedger.positions(test), black_delta(test), atol=1e-6)


def test_torch_loss_matches_numpy_pnl(markets):
    train, val, test = markets
    hedger = DeepHedger(_config(epochs=1, instruments=("forward", "variance_swap")), COSTS)
    hedger.fit(train, val)
    d = hedger._tensors(test)
    with torch.no_grad():
        pos = hedger._rollout(d["feats"], d["base"], d["scale"])
        torch_loss = hedger._loss(pos, d).double().numpy() * 1e-4
    np_loss = hedge_outcome(test, hedger.positions(test), COSTS).loss
    np.testing.assert_allclose(torch_loss, np_loss, atol=2e-6)


def test_training_improves_out_of_sample_cvar_over_delta(markets):
    train, val, test = markets
    hedger = DeepHedger(_config(), COSTS)
    history = hedger.fit(train, val)
    assert history.val_cvar[history.best_epoch] <= history.val_cvar[0]
    deep = cvar(hedge_outcome(test, hedger.positions(test), COSTS).loss)
    delta = cvar(hedge_outcome(test, black_delta(test), COSTS).loss)
    assert deep < delta


def test_training_is_deterministic(markets):
    train, val, test = markets
    a = DeepHedger(_config(epochs=2), COSTS)
    b = DeepHedger(_config(epochs=2), COSTS)
    a.fit(train, val)
    b.fit(train, val)
    np.testing.assert_array_equal(a.positions(test), b.positions(test))


def test_positions_require_fit(markets):
    with pytest.raises(RuntimeError):
        DeepHedger(_config(), COSTS).positions(markets[2])
