"""Tests for lake.simulate: settling and slow sweeps of a."""

import numpy as np
import pytest

from lake.config import load_params
from lake.equilibria import equilibria, folds
from lake.simulate import jump_points, settle, sweep_a

PARAMS = load_params()
NUM, INTEG, SW = PARAMS["numerics"], PARAMS["integration"], PARAMS["sweep"]
Q = PARAMS["model"]["q"]
A0, B0 = PARAMS["model"]["a"], PARAMS["model"]["b"]
SETTLE_ARGS = (INTEG["rtol"], INTEG["atol"], INTEG["settle_tol"], INTEG["settle_t_max"])
EQ_TOL = 1e-6  # settled state vs equilibrium from the root finder


def _equilibria(a, b):
    return equilibria(a, b, Q, NUM["n_grid"], NUM["xtol"], NUM["tangency_tol"])


@pytest.mark.parametrize("x0, which", [(0.3, 0), (0.6, 0), (0.7, 2), (1.8, 2)])
def test_settle_ends_in_the_basin_it_started_in(x0, which):
    """Below x_u (~0.658) the lake settles clear, above it turbid."""
    x_end, settled = settle(x0, A0, B0, Q, *SETTLE_ARGS)
    assert settled
    assert x_end == pytest.approx(_equilibria(A0, B0)[which].x, abs=EQ_TOL)


def test_settle_above_upper_fold_goes_turbid():
    """'Treat the lake chemically' above the upper fold: it rolls back to turbid."""
    upper, _ = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
    a = upper.a + 0.01
    x_end, settled = settle(0.0, a, B0, Q, *SETTLE_ARGS)
    assert settled
    assert x_end == pytest.approx(_equilibria(a, B0)[-1].x, abs=EQ_TOL)


def _sweep(b, rate):
    return sweep_a(0.0, b, Q, SW["a_min"], SW["a_max"], rate, SW["max_step"],
                   SW["n_points"], INTEG["rtol"], INTEG["atol"])


def test_slow_sweep_shows_hysteresis_between_folds():
    """Up-sweep jumps just after the upper fold, down-sweep just before the
    lower fold; in between the two paths differ (clear vs turbid)."""
    upper, lower = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
    halves = _sweep(B0, SW["rate"])
    a_up, a_down = jump_points(halves, 0.5 * (upper.x + lower.x))
    assert upper.a < a_up < upper.a + 0.1 * (upper.a - lower.a)
    assert lower.a - 0.1 * (upper.a - lower.a) < a_down < lower.a

    a_mid = 0.5 * (upper.a + lower.a)
    x_up = np.interp(a_mid, halves["up"]["a"], halves["up"]["x"])
    x_down = np.interp(a_mid, halves["down"]["a"][::-1], halves["down"]["x"][::-1])
    assert x_up < upper.x < lower.x < x_down


def test_sweep_without_bistability_has_no_loop():
    """For b above max f' (reversible) up and down paths coincide closely."""
    halves = _sweep(0.8, SW["rate"])
    x_down_on_up_grid = np.interp(halves["up"]["a"], halves["down"]["a"][::-1],
                                  halves["down"]["x"][::-1])
    assert np.max(np.abs(halves["up"]["x"] - x_down_on_up_grid)) < 1e-3
