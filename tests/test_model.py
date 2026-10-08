"""Unit tests for lake.model that support the quantitative checks."""

import numpy as np
import pytest

from lake.model import V_closed_q2, V_numerical, f, f_prime

X_GRID = np.linspace(0.0, 3.0, 301)
FD_STEP = 1e-6
DERIV_TOL = 1e-8
# quad's default absolute tolerance is ~1.5e-8, so the two forms of V
# should agree to about that level.
POTENTIAL_TOL = 1e-8


@pytest.mark.parametrize("q", [2.0, 3.0, 4.0])
def test_f_prime_matches_finite_difference(q):
    """f'(x) agrees with a central finite difference of f(x)."""
    x = X_GRID[1:]  # keep x - h >= 0 so non-integer powers stay real
    fd = (f(x + FD_STEP, q) - f(x - FD_STEP, q)) / (2 * FD_STEP)
    np.testing.assert_allclose(f_prime(x, q), fd, rtol=0, atol=DERIV_TOL)


@pytest.mark.parametrize("a, b", [(0.04, 0.52), (0.0, 0.45), (0.1, 0.62)])
def test_closed_form_potential_matches_numerical(a, b):
    """For q = 2 the closed-form V equals -int_0^x F(s) ds computed by quad."""
    np.testing.assert_allclose(
        V_closed_q2(X_GRID, a, b), V_numerical(X_GRID, a, b, 2.0), rtol=0, atol=POTENTIAL_TOL
    )


def test_potential_is_zero_at_origin():
    """V(0) = 0 by construction."""
    assert V_closed_q2(0.0, 0.04, 0.52) == 0.0
    assert V_numerical(0.0, 0.04, 0.52, 2.0) == 0.0
