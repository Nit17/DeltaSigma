import numpy as np
import pytest

from deltasigma.vol import sample_eurusd_surface
from deltasigma.vol.sample_market import EURUSD_QUOTES, EURUSD_SPOT


@pytest.fixture(scope="module")
def surface():
    return sample_eurusd_surface("sabr")


def test_surface_reprices_pillar_smiles(surface):
    for smile in surface.smiles:
        k = smile.strike_grid(n=11)
        np.testing.assert_allclose(surface.vol(k, smile.T), smile.vol(k), atol=1e-12)
        assert surface.forward(smile.T) == pytest.approx(smile.forward, rel=1e-14)


def test_atm_quotes_recovered(surface):
    for (_, atm, _, _), smile in zip(EURUSD_QUOTES.values(), surface.smiles, strict=True):
        assert float(smile.vol(smile.pillar_strikes[1])) == pytest.approx(atm, abs=1e-8)


def test_total_variance_interpolation_is_monotone_in_time(surface):
    times = np.linspace(surface.tenors[0], surface.tenors[-1], 60)
    w = [float(surface.vol(surface.forward(t), t)) ** 2 * t for t in times]
    assert np.all(np.diff(w) > 0)


def test_sample_surface_has_no_calendar_arbitrage(surface):
    assert surface.calendar_arbitrage_violations() == []


def test_calendar_arbitrage_detected_when_vol_collapses():
    from deltasigma.pricing import CONVENTIONS
    from deltasigma.vol import SmileQuote, build_surface

    pillars = [
        (0.25, 0.043, 0.025, SmileQuote(0.12, -0.01, 0.003)),
        (0.50, 0.043, 0.025, SmileQuote(0.06, -0.01, 0.003)),
    ]
    surf = build_surface(EURUSD_SPOT, pillars, CONVENTIONS["EURUSD"])
    assert surf.calendar_arbitrage_violations()


def test_forward_interpolation_starts_at_spot(surface):
    assert surface.forward(1e-12) == pytest.approx(EURUSD_SPOT, rel=1e-10)
