"""Equilibria, their stability, the fold (saddle-node) curve and the regime.

Equilibria solve F(x) = a - b*x + f(x) = 0. This is the same condition as
Carpenter et al.'s discrete-time model (1999, Eq. A.8 with a = A/C, b = B/C),
so the equilibria, folds and regimes here can be compared directly with their
Appendix A and Fig. A1.

Root finding (see `equilibria`): F is split into pieces on which it is
monotone, and each piece holds at most one root, which brentq refines.
Splitting at the turning points means two equilibria close together near a
fold are never missed, even when they lie in the same grid cell.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from lake.model import F, F_prime, f, f_prime

STABLE = "stable"
UNSTABLE = "unstable"
TANGENT = "tangent"  # double root exactly at a fold, F'(x*) = 0

REVERSIBLE = "reversible"
HYSTERETIC = "hysteretic"
IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class Equilibrium:
    """An equilibrium x* of dx/dt = F(x) and its stability."""

    x: float
    stability: str


@dataclass(frozen=True)
class Fold:
    """A fold (saddle-node) point: the value of a, and the x where it occurs."""

    a: float
    x: float


def sign_change_roots(g, lo, hi, n_grid, xtol):
    """Roots of g on [lo, hi]: scan a grid for sign changes, refine with brentq.

    A grid point where g is exactly zero is also returned. Two roots inside
    the same grid cell cancel and are missed, so callers must choose n_grid
    with that in mind (or split the interval first, as `equilibria` does).
    """
    xs = np.linspace(lo, hi, n_grid)
    gs = g(xs)
    roots = []
    for i in range(n_grid):
        if gs[i] == 0.0:
            roots.append(float(xs[i]))
        elif i < n_grid - 1 and gs[i] * gs[i + 1] < 0.0:
            roots.append(brentq(g, xs[i], xs[i + 1], xtol=xtol))
    return roots


def search_upper_bound(a, b):
    """Upper end of the x interval that contains every equilibrium.

    Since f(x) < 1, F(x) < a + 1 - b*x, which is negative for x >= (a + 1)/b.
    So no equilibrium lies beyond (a + 1)/b. Requires b > 0.
    """
    return (a + 1.0) / b


def equilibria(a, b, q, n_grid, xtol, tangency_tol):
    """All equilibria of F(x) = a - b*x + f(x) on x >= 0, sorted by x.

    Method:
    1. Find the turning points of F (roots of F' = -b + f') by a grid scan
       and brentq. Between consecutive turning points F is monotone, so each
       such piece holds at most one root.
    2. A turning point where |F| <= tangency_tol is a double root (we are
       exactly at a fold); it is returned with stability TANGENT. The same
       test catches x = 0 when a = 0, which is an ordinary equilibrium.
    3. On each piece where F changes sign, refine the single root with brentq.

    Stability (from F'(x*)): stable if F' < 0, unstable if F' > 0.

    Limitation: if b is so close to max f' that both turning points of F
    fall in one grid cell, they are missed and the very narrow bistable
    window is not resolved. This only matters right at the cusp.
    """
    x_max = search_upper_bound(a, b)
    turning = sign_change_roots(lambda x: F_prime(x, a, b, q), 0.0, x_max, n_grid, xtol)

    breakpoints = [0.0] + turning + [x_max]
    F_at = [F(x, a, b, q) for x in breakpoints]

    found = []  # (x, stability)
    for i, x in enumerate(breakpoints):
        if abs(F_at[i]) <= tangency_tol:
            stability = TANGENT if x in turning else _stability(x, a, b, q)
            found.append((x, stability))
            F_at[i] = 0.0  # so the neighbouring pieces do not find it again

    for i in range(len(breakpoints) - 1):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        if F_at[i] * F_at[i + 1] < 0.0:
            x = brentq(F, lo, hi, args=(a, b, q), xtol=xtol)
            found.append((x, _stability(x, a, b, q)))

    found.sort()
    return [Equilibrium(x, s) for x, s in found]


def _stability(x, a, b, q):
    """STABLE if F'(x) < 0, UNSTABLE if F'(x) > 0 (linear stability)."""
    return STABLE if F_prime(x, a, b, q) < 0.0 else UNSTABLE


# ---------------------------------------------------------------------------
# Fold curve and regimes
# ---------------------------------------------------------------------------


def fold_curve(x, q):
    """Fold (saddle-node) curve in the (a, b) plane, parametrised by x.

    At a fold the line b*x - a is tangent to f(x), so both F(x) = 0 and
    F'(x) = 0 hold. Solving these for (a, b):

        b = f'(x),     a = x*f'(x) - f(x)

    Returns the arrays (a, b). Compare Carpenter et al. (1999), Fig. A1.
    """
    b = f_prime(x, q)
    a = x * b - f(x, q)
    return a, b


def max_f_prime(q, xtol):
    """Maximum of f'(x) and where it occurs, found numerically.

    For b >= max f' the line b*x - a can never cut f three times, so the
    lake has a single equilibrium for every a (the reversible regime).

    f' rises from f'(0) = 0 to one maximum and then decays (for q > 1).
    The maximum lies in [0, 1]: the sign of f'' is the sign of
    (q - 1) - (q + 1)*x^q, which is negative at x = 1. So a bounded scalar
    minimisation of -f' on [0, 1] finds it. Returns (x_at_max, max_value).
    """
    res = minimize_scalar(lambda x: -f_prime(x, q), bounds=(0.0, 1.0),
                          method="bounded", options={"xatol": xtol})
    return float(res.x), float(-res.fun)


def folds(b, q, n_grid, xtol):
    """The upper and lower fold for a given b, or None if there are none.

    Solves b = f'(x) for x; each solution gives a fold with
    a = x*f'(x) - f(x) (see `fold_curve`). For 0 < b < max f' there are two:

    - upper fold (smaller x): the clear state and the threshold merge, so
      above this a the clear state no longer exists;
    - lower fold (larger x): the turbid state and the threshold merge, so
      below this a the turbid state no longer exists.

    Search interval: f'(x) < q*x^(-q-1), which is <= b once
    x >= (q/b)^(1/(q+1)), so every solution lies below that bound.

    Returns (upper, lower) as Fold objects, or None for b >= max f'.
    """
    x_hi = (q / b) ** (1.0 / (q + 1.0))
    xs = sign_change_roots(lambda x: f_prime(x, q) - b, 0.0, x_hi, n_grid, xtol)
    if len(xs) != 2:
        return None
    x_up, x_low = sorted(xs)
    a_up, _ = fold_curve(x_up, q)
    a_low, _ = fold_curve(x_low, q)
    return Fold(float(a_up), x_up), Fold(float(a_low), x_low)


def irreversibility_boundary(q, xtol):
    """The b at which the lower fold reaches a = 0, and the tangency x.

    For b below this value the lower fold lies at a < 0, so the turbid state
    persists at a = 0 and no non-negative input restores the lake
    (irreversible regime, Carpenter et al. 1999, Appendix A).

    Solves a(x) = x*f'(x) - f(x) = 0 on the lower-fold branch, i.e. for x
    beyond the maximum of f'. On that branch da/dx = x*f''(x) < 0, so a
    decreases monotonically and there is exactly one root. Bracket:
    - at x = argmax f', a > 0 (the cusp lies at positive a);
    - for x >= (2q)^(1/q), x*f'(x) < q*x^(-q) <= 1/2 <= f(x), so a < 0.
    Returns (b, x).
    """
    x_cusp, _ = max_f_prime(q, xtol)
    x_hi = (2.0 * q) ** (1.0 / q)
    x = brentq(lambda s: fold_curve(s, q)[0], x_cusp, x_hi, xtol=xtol)
    return float(f_prime(x, q)), x


def classify_regime(b, q, n_grid, xtol):
    """Regime of the lake for given b and q (Carpenter et al. 1999, App. A).

    - reversible:   b >= max f'; one equilibrium for every a >= 0.
    - irreversible: the lower fold lies at a < 0, so bistability persists
                    at a = 0 and a turbid lake cannot be restored by
                    reducing the input.
    - hysteretic:   everything in between.

    Ambiguity: at the boundary b where the lower fold sits exactly at
    a = 0, we call the lake hysteretic (the turbid state just vanishes at
    a = 0). This boundary case has zero width in b.
    """
    _, f_prime_max = max_f_prime(q, xtol)
    if b >= f_prime_max:
        return REVERSIBLE
    fold_pair = folds(b, q, n_grid, xtol)
    if fold_pair is None:
        raise ValueError(
            f"b = {b} is below max f' = {f_prime_max} but no folds were "
            "resolved; increase n_grid."
        )
    _, lower = fold_pair
    return IRREVERSIBLE if lower.a < 0.0 else HYSTERETIC
