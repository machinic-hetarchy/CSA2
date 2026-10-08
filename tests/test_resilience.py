"""Unit tests for lake.resilience."""

import math

import pytest

from lake.config import load_params
from lake.equilibria import equilibria, folds
from lake.model import V
from lake.resilience import clear_basin_measures

PARAMS = load_params()
NUM = PARAMS["numerics"]
ARGS = (NUM["n_grid"], NUM["xtol"], NUM["tangency_tol"])
Q = PARAMS["model"]["q"]
A0, B0 = PARAMS["model"]["a"], PARAMS["model"]["b"]
X_CURRENT = PARAMS["baseline"]["x_current"]


def _measures(x_current, a, b):
    return clear_basin_measures(x_current, a, b, Q, *ARGS)


def test_baseline_measures_follow_definitions():
    """At the baseline, each measure matches its definition in CLAUDE.md."""
    m = _measures(X_CURRENT, A0, B0)
    x_s, x_u, _ = (e.x for e in equilibria(A0, B0, Q, *ARGS))
    assert m.x_s == x_s and m.x_u == x_u
    assert m.L == x_u
    assert m.R == pytest.approx(V(x_u, A0, B0, Q) - V(x_s, A0, B0, Q), abs=1e-15)
    assert m.R > 0
    assert m.R_over_L == pytest.approx(m.R / m.L)
    assert m.Pr == pytest.approx(x_u - X_CURRENT)
    assert m.Pr_over_L == pytest.approx(m.Pr / m.L)
    assert math.isnan(m.L_turbid)


def test_resistance_shrinks_as_a_increases():
    """Raising a makes the clear basin shallower and narrower (Walker Fig. 1b)."""
    upper, _ = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
    a_values = [0.03, 0.04, 0.05, 0.06, 0.07]
    assert a_values[-1] < upper.a
    ms = [_measures(X_CURRENT, a, B0) for a in a_values]
    assert all(m1.R > m2.R for m1, m2 in zip(ms, ms[1:]))
    assert all(m1.L > m2.L for m1, m2 in zip(ms, ms[1:]))


def test_above_upper_fold_clear_basin_is_nan():
    upper, _ = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
    m = _measures(X_CURRENT, upper.a + 0.01, B0)
    assert all(math.isnan(v) for v in (m.x_s, m.x_u, m.L, m.R, m.R_over_L, m.Pr))
    assert "clear basin has vanished" in m.reason


def test_below_lower_fold_has_no_threshold():
    _, lower = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
    m = _measures(X_CURRENT, lower.a / 2, B0)
    assert math.isnan(m.L) and math.isnan(m.R)
    assert "below the lower fold" in m.reason


def test_reversible_regime_has_no_threshold():
    m = _measures(X_CURRENT, 0.1, 0.8)
    assert math.isnan(m.L)
    assert "reversible" in m.reason


def test_state_beyond_threshold_gives_negative_precariousness():
    m = _measures(1.0, A0, B0)
    assert m.Pr < 0 and m.Pr_over_L < 0
    assert "beyond the threshold" in m.reason


def test_exactly_at_lower_fold_threshold_is_tangent_point():
    """At the lower fold the threshold is the tangent point; the clear basin is
    still well defined and has positive depth."""
    _, lower = folds(B0, Q, NUM["n_grid"], NUM["xtol"])
    m = _measures(X_CURRENT, lower.a, B0)
    assert m.x_u == pytest.approx(lower.x, abs=1e-8)
    assert m.R > 0
