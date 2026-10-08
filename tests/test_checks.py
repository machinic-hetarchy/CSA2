"""Quantitative checks from the project brief (CLAUDE.md, "Quantitative checks").

Each test compares a numerically computed value with the expected value in
the brief. Checks are added step by step as the corresponding code is built.
"""

import numpy as np
import pytest

from lake.checks import compute_checks
from lake.config import load_params
from lake.model import F, V

PARAMS = load_params()
Q = PARAMS["model"]["q"]
A0 = PARAMS["model"]["a"]
B0 = PARAMS["model"]["b"]
CHECKS = compute_checks(PARAMS)

# Grid of states on which the potential is tested.
X_GRID = np.linspace(0.0, 3.0, 301)

# Step for the central finite difference. Truncation error ~ h^2 * V''' and
# round-off ~ eps / h are both far below the tolerance with this choice.
FD_STEP = 1e-5
GRADIENT_TOL = 1e-8

# Baseline (a, b) plus a few others covering all three regimes for q = 2.
AB_CASES = [(A0, B0), (0.0, 0.45), (0.1, 0.62), (0.3, 0.8)]


@pytest.mark.parametrize("a, b", AB_CASES)
def test_potential_gradient_equals_rate(a, b):
    """Check: -dV/dx equals F(x) (numerical gradient test)."""
    dVdx = (V(X_GRID + FD_STEP, a, b, Q) - V(X_GRID - FD_STEP, a, b, Q)) / (2 * FD_STEP)
    # At x = 0 the left point x - h is negative; fine for integer q = 2.
    np.testing.assert_allclose(-dVdx, F(X_GRID, a, b, Q), rtol=0, atol=GRADIENT_TOL)


@pytest.mark.parametrize(
    "name", ["max_f_prime", "irreversibility_boundary", "upper_fold", "lower_fold"]
)
def test_check_matches_expectation(name):
    """Checks with an expected value: computed and expected agree within tolerance.

    Covers: max of f', the reversible/irreversible boundary at a = 0, and the
    upper and lower folds at b = 0.52.
    """
    entry = CHECKS[name]
    assert entry["expected"] is not None, f"no expectation for {name} at q = {Q}"
    for key, diff in entry["abs_diff"].items():
        assert diff <= entry["tolerance"], (
            f"{name}.{key}: computed {entry['computed'][key]}, "
            f"expected {entry['expected'][key]}, diff {diff}"
        )


def test_resistance_vanishes_at_upper_fold():
    """Check: R -> 0 and x_s -> x_u as a -> upper fold (b = 0.52).

    R and the gap x_u - x_s shrink monotonically as a approaches the fold
    from below, and both are exactly zero at the fold.
    """
    c = CHECKS["resistance_at_fold"]["computed"]
    assert c["R_decreasing"] and c["gap_decreasing"]
    assert c["R_at_fold"] == 0.0
    assert c["gap_at_fold"] == 0.0


def test_bistable_window_at_b062_exists():
    """b = 0.62 lies below max f', so the window must be open (width > 0)."""
    window = CHECKS["bistable_window"]["computed"]
    assert window["regime"] != "reversible"
    assert window["width"] > 0.0
