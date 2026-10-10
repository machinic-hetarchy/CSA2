"""Walker et al.'s (2004) resilience measures for the clear-water basin.

Walker et al. (2004, Fig. 1) describe a basin of attraction by:
- latitude L:        the width of the basin;
- resistance R:      the depth of the basin;
- precariousness Pr: how close the current state is to the threshold.

They give no formulas. Our operational choices for the 1D lake model
(see README.md, "Definitions"):

- clear basin:  the interval [0, x_u), where x_u is the unstable
                equilibrium (the threshold). x = 0 is the domain boundary.
- L  = x_u - 0 = x_u
- R  = V(x_u) - V(x_s), with x_s the clear stable equilibrium
- R/L, which Walker et al. call the more accurate measure
- Pr = x_u - x_current, also reported as Pr/L

Ambiguities (for the critical review):
- L is measured from the domain boundary x = 0, not from x_s. The lower
  edge of the clear basin is a physical limit (no negative phosphorus),
  not a threshold, so "width" mixes two kinds of edge.
- The turbid basin is [x_u, infinity): its latitude is undefined. We
  report NaN rather than inventing a cutoff.
- R and L depend on the choice of state variable: rescaling x changes V
  and the widths, so the numbers are not coordinate-free.
- When the current state lies beyond the threshold, Pr is negative. We
  report it as is, with a note, rather than clipping it.
"""

import math
from dataclasses import dataclass

from lake.equilibria import STABLE, equilibria, folds
from lake.model import V


@dataclass(frozen=True)
class ClearBasinMeasures:
    """Resilience measures of the clear basin for one (a, b, q, x_current).

    Undefined measures are NaN; `reason` says why. `L_turbid` is always NaN
    because the turbid basin is unbounded above.
    """

    a: float
    b: float
    q: float
    x_current: float
    x_s: float
    x_u: float
    L: float
    R: float
    R_over_L: float
    Pr: float
    Pr_over_L: float
    L_turbid: float
    reason: str


TURBID_NOTE = "turbid latitude undefined: turbid basin [x_u, inf) is unbounded above"


def clear_basin_measures(x_current, a, b, q, n_grid, xtol, tangency_tol):
    """Compute L, R, R/L, Pr and Pr/L of the clear basin (definitions above).

    x_current is the current state. Pass None to place the state at the
    clear equilibrium x_s (the lake at rest), so that Pr = x_u - x_s.

    The threshold x_u is the lowest equilibrium that is not stable (unstable,
    or tangent exactly at a fold). The clear state x_s is the lowest
    equilibrium. Cases:

    - bistable (stable, unstable, stable): the usual case.
    - exactly at the upper fold: x_s = x_u (tangent), so R = 0. This is the
      limit the clear basin reaches just before it disappears.
    - exactly at the lower fold: the threshold is the tangent point where
      the turbid state is born; the clear basin is still well defined.
    - one stable equilibrium only: there is no threshold, so every measure
      is NaN and `reason` says whether the clear basin has vanished (a above
      the upper fold), the turbid basin has vanished (a below the lower
      fold), or the regime is reversible (no folds at all).
    """
    eqs = equilibria(a, b, q, n_grid, xtol, tangency_tol)
    thresholds = [e for e in eqs if e.stability != STABLE]

    if not thresholds:
        # x_s is NaN too: the single equilibrium may be the turbid state, and
        # without a threshold there is no clear *basin* to measure.
        nan = math.nan
        reason = _single_equilibrium_reason(a, b, q, n_grid, xtol)
        return ClearBasinMeasures(a, b, q, x_current, nan, nan,
                                  nan, nan, nan, nan, nan, nan, reason)

    x_s = eqs[0].x
    x_u = thresholds[0].x
    if x_current is None:
        x_current = x_s
    L = x_u - 0.0
    R = V(x_u, a, b, q) - V(x_s, a, b, q)
    Pr = x_u - x_current

    notes = [TURBID_NOTE]
    if x_s == x_u:
        notes.insert(0, "exactly at the upper fold: clear basin has zero depth")
    if Pr < 0:
        notes.insert(0, "current state is beyond the threshold (in the turbid basin): Pr < 0")
    # L = 0 only if x_u = 0, which cannot happen for a >= 0, so R/L and
    # Pr/L are always defined here.
    return ClearBasinMeasures(a, b, q, x_current, x_s, x_u, L, float(R), float(R / L),
                              Pr, Pr / L, math.nan, "; ".join(notes))


def _single_equilibrium_reason(a, b, q, n_grid, xtol):
    """Explain why there is only one equilibrium for these parameters."""
    fold_pair = folds(b, q, n_grid, xtol)
    if fold_pair is None:
        return "single equilibrium (reversible regime, no folds): no threshold, basin unbounded"
    upper, lower = fold_pair
    if a > upper.a:
        return (f"clear basin has vanished: a = {a:.6g} is above the upper fold "
                f"a = {upper.a:.6g}")
    if a < lower.a:
        return (f"only the clear state exists: a = {a:.6g} is below the lower fold "
                f"a = {lower.a:.6g}, so there is no threshold and the basin is unbounded")
    # Not expected: between the folds there should be three equilibria.
    return "single equilibrium inside the bistable window (root finding did not resolve the threshold)"
