"""Unit tests for lake.equilibria that support the quantitative checks."""

import numpy as np
import pytest

from lake.config import load_params
from lake.equilibria import (
    STABLE,
    TANGENT,
    UNSTABLE,
    equilibria,
    fold_curve,
    folds,
)
from lake.model import F

NUM = load_params()["numerics"]
N_GRID, XTOL, TAN_TOL = NUM["n_grid"], NUM["xtol"], NUM["tangency_tol"]
Q = 2.0
ROOT_TOL = 1e-10  # |F(x*)| at a refined root


def _eq(a, b):
    return equilibria(a, b, Q, N_GRID, XTOL, TAN_TOL)


@pytest.mark.parametrize("a, b", [(0.04, 0.52), (0.3, 0.8), (0.1, 0.45), (0.0, 0.45)])
def test_equilibria_are_roots(a, b):
    for e in _eq(a, b):
        assert abs(F(e.x, a, b, Q)) < ROOT_TOL


def test_bistable_case_has_stable_unstable_stable():
    """Baseline a = 0.04, b = 0.52 lies between the folds: three equilibria."""
    eqs = _eq(0.04, 0.52)
    assert [e.stability for e in eqs] == [STABLE, UNSTABLE, STABLE]


def test_reversible_case_has_one_equilibrium():
    eqs = _eq(0.1, 0.8)
    assert len(eqs) == 1 and eqs[0].stability == STABLE


def test_a_zero_gives_equilibrium_on_boundary():
    """At a = 0, x = 0 is an equilibrium (F(0) = 0) and it is stable."""
    eqs = _eq(0.0, 0.45)
    assert eqs[0].x == 0.0 and eqs[0].stability == STABLE
    assert len(eqs) == 3


def test_closely_spaced_roots_near_fold_are_resolved():
    """Just below the upper fold the clear state and the threshold are very
    close (closer than one grid cell) and must both be found."""
    upper, _ = folds(0.52, Q, N_GRID, XTOL)
    eqs = _eq(upper.a - 1e-10, 0.52)
    assert [e.stability for e in eqs] == [STABLE, UNSTABLE, STABLE]
    assert eqs[1].x - eqs[0].x < 1e-3


def test_exactly_at_fold_gives_tangent_root():
    upper, _ = folds(0.52, Q, N_GRID, XTOL)
    eqs = _eq(upper.a, 0.52)
    assert [e.stability for e in eqs] == [TANGENT, STABLE]
    assert eqs[0].x == pytest.approx(upper.x, abs=1e-8)


def test_folds_lie_on_fold_curve():
    """The fold points for given b satisfy the parametric fold curve."""
    for fold in folds(0.52, Q, N_GRID, XTOL):
        a, b = fold_curve(fold.x, Q)
        assert b == pytest.approx(0.52, abs=1e-10)
        assert a == pytest.approx(fold.a, abs=1e-12)


def test_no_folds_above_max_f_prime():
    assert folds(0.7, Q, N_GRID, XTOL) is None
