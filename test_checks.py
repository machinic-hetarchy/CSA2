"""Quantitative checks (README.md, "Quantitative checks"). Run with: pytest"""

import numpy as np
import pytest

import lake

CHECKS = lake.quantitative_checks()


@pytest.mark.parametrize("name", ["max_f_prime", "irreversibility_boundary",
                                  "upper_fold_b052", "lower_fold_b052"])
def test_matches_expected_value(name):
    """Max of f', the reversible/irreversible boundary, and both folds at b = 0.52."""
    check = CHECKS[name]
    for key, expected in check["expected"].items():
        assert abs(check["computed"][key] - expected) <= check["tol"], key


@pytest.mark.parametrize("a, b", [(0.04, 0.52), (0.0, 0.45), (0.1, 0.62), (0.3, 0.8)])
def test_potential_gives_the_model(a, b):
    """-dV/dx equals F(x), by central finite differences."""
    x, h = np.linspace(0.0, 3.0, 301), 1e-5
    dV = (lake.V(x + h, a, b) - lake.V(x - h, a, b)) / (2 * h)
    np.testing.assert_allclose(-dV, lake.F(x, a, b), atol=1e-8)


def test_resistance_vanishes_at_upper_fold():
    """R and x_u - x_s shrink as a approaches the upper fold, and are 0 at it."""
    R = CHECKS["resistance_at_upper_fold"]["computed"]["R"]
    gap = CHECKS["resistance_at_upper_fold"]["computed"]["gap"]
    assert all(np.diff(R) < 0) and all(np.diff(gap) < 0)
    assert R[-1] == 0.0 and gap[-1] == 0.0


def test_b062_has_a_bistable_window():
    """Carpenter et al.'s 'reversible' b = 0.62 lies below max f' = 0.6495."""
    window = CHECKS["bistable_window_b062"]["computed"]
    assert window["regime"] == "hysteretic" and window["width"] > 0
