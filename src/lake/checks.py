"""Compute the quantitative checks listed in CLAUDE.md.

`compute_checks` returns one entry per check. Each entry holds the computed
values, the expected values (where the brief gives them), the absolute
differences, the tolerance and where the expectation comes from. The tests
assert on these entries and scripts/make_figures.py writes them to
results/checks.json, so both use exactly the same numbers.
"""

import numpy as np

from lake.equilibria import classify_regime, folds, irreversibility_boundary, max_f_prime
from lake.resilience import clear_basin_measures
from lake.simulate import jump_points, sweep_a


def _entry(computed, expected, tolerance, source, note=""):
    """Bundle computed and expected values with their absolute differences."""
    entry = {"computed": computed, "expected": expected, "tolerance": tolerance, "source": source}
    if expected is not None:
        entry["abs_diff"] = {k: abs(computed[k] - expected[k]) for k in expected}
        entry["passed"] = all(d <= tolerance for d in entry["abs_diff"].values())
    if note:
        entry["note"] = note
    return entry


def compute_checks(params):
    """Run every check for the q in params and return them as a dict."""
    q = params["model"]["q"]
    num = params["numerics"]
    chk = params["checks"]
    n_grid, xtol = num["n_grid"], num["xtol"]
    checks = {"q": q}

    # Max of f' (boundary for any bistability). Analytic for q = 2:
    # 3*sqrt(3)/8 at x = 1/sqrt(3).
    x_max, value_max = max_f_prime(q, xtol)
    checks["max_f_prime"] = _entry(
        {"value": value_max, "x": x_max},
        {"value": 3 * np.sqrt(3) / 8, "x": 1 / np.sqrt(3)} if q == 2 else None,
        chk["analytic_tol"], "analytic (q = 2)",
    )

    # Reversible/irreversible boundary: lower fold at a = 0.
    # Analytic for q = 2: b = 0.5 with tangency at x = 1.
    b_irr, x_irr = irreversibility_boundary(q, xtol)
    checks["irreversibility_boundary"] = _entry(
        {"b": b_irr, "x": x_irr},
        {"b": 0.5, "x": 1.0} if q == 2 else None,
        chk["analytic_tol"], "analytic (q = 2)",
    )

    # Upper and lower folds at b = b_folds (hand-calculated expectations).
    b_f = chk["b_folds"]
    upper, lower = folds(b_f, q, n_grid, xtol)
    expected = chk["expected_folds_b052"] if q == 2 and b_f == 0.52 else None
    checks["upper_fold"] = _entry(
        {"b": b_f, "a": upper.a, "x": upper.x},
        {"a": expected["upper"]["a"], "x": expected["upper"]["x"]} if expected else None,
        chk["hand_tol"], "hand calculation in CLAUDE.md (approximate)",
        note="clear state disappears for a above this value",
    )
    checks["lower_fold"] = _entry(
        {"b": b_f, "a": lower.a, "x": lower.x},
        {"a": expected["lower"]["a"], "x": expected["lower"]["x"]} if expected else None,
        chk["hand_tol"], "hand calculation in CLAUDE.md (approximate)",
        note="turbid state disappears for a below this value",
    )

    # Resistance vanishes at the upper fold: R -> 0 and x_s -> x_u.
    checks["resistance_at_fold"] = _resistance_at_fold(params, upper)

    # Hysteresis loop from slow sweeps: how far the jumps lag the folds.
    checks["sweep_lag"] = _sweep_lag(params, upper, lower)

    # Bistable window at b = b_window (no expected value; for the report).
    b_w = chk["b_window"]
    regime = classify_regime(b_w, q, n_grid, xtol)
    window = folds(b_w, q, n_grid, xtol)
    computed = {"b": b_w, "regime": regime, "max_f_prime": value_max}
    if window is not None:
        up_w, low_w = window
        computed.update({"a_lower_fold": low_w.a, "a_upper_fold": up_w.a,
                         "width": up_w.a - low_w.a})
    checks["bistable_window"] = _entry(
        computed, None, None, "no expectation; reported for discussion",
        note=("Carpenter et al. (1999) use b = 0.62 as a 'reversible' example "
              "(Fig. 11), but 0.62 < max f', so a narrow bistable window exists."),
    )
    return checks


def _sweep_lag(params, upper, lower):
    """Jump points of up-and-down sweeps of a at several ramp rates.

    No expected value in the brief. A finite ramp rate delays each jump past
    its fold, so the simulated loop is wider than the bistable window.
    Theory for a slowly swept saddle-node predicts lag ~ rate^(2/3); the
    fitted exponent is reported for comparison.
    """
    q = params["model"]["q"]
    b = params["checks"]["b_folds"]
    sw = params["sweep"]
    integ = params["integration"]
    x_level = 0.5 * (upper.x + lower.x)

    rates = np.array(sw["lag_rates"])
    lag_up, lag_down = [], []
    for rate in rates:
        halves = sweep_a(0.0, b, q, sw["a_min"], sw["a_max"], rate, sw["max_step"],
                         sw["n_points"], integ["rtol"], integ["atol"])
        a_up, a_down = jump_points(halves, x_level)
        lag_up.append(a_up - upper.a)
        lag_down.append(lower.a - a_down)

    computed = {
        "b": b,
        "x_level_for_jump": x_level,
        "rates": rates.tolist(),
        "lag_up_beyond_upper_fold": lag_up,
        "lag_down_beyond_lower_fold": lag_down,
        "bistable_window_width": upper.a - lower.a,
        "fitted_exponent_lag_up": _fit_exponent(rates, lag_up),
        "fitted_exponent_lag_down": _fit_exponent(rates, lag_down),
    }
    return _entry(computed, None, None, "no expectation; reported for discussion",
                  note=("slow-passage theory predicts lag ~ rate^(2/3). A NaN lag on the "
                        "down-sweep means the lake never returned to clear before a "
                        "reached a_min: at that rate a hysteretic lake looks irreversible. "
                        "Exponents are fitted on the finite lags only."))


def _fit_exponent(rates, lags):
    """Slope of log(lag) against log(rate), using only finite lags."""
    lags = np.asarray(lags)
    ok = np.isfinite(lags)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(rates[ok]), np.log(lags[ok]), 1)[0])


def _resistance_at_fold(params, upper):
    """R and the gap x_u - x_s for a = a_fold - delta, delta -> 0.

    Expectation from the brief: both go to zero at the fold, and exactly at
    the fold R = 0 and x_s = x_u.

    Extra (not in the brief): near a saddle-node the gap scales as
    delta^(1/2) and R as delta^(3/2) (normal form dx/dt = delta - c*y^2).
    We fit these exponents on a log-log scale and report them.
    """
    q = params["model"]["q"]
    b = params["checks"]["b_folds"]
    x_current = params["baseline"]["x_current"]
    num = params["numerics"]
    args = (b, q, num["n_grid"], num["xtol"], num["tangency_tol"])

    deltas = np.array(params["checks"]["fold_approach_deltas"])
    measures = [clear_basin_measures(x_current, upper.a - d, *args) for d in deltas]
    R = np.array([m.R for m in measures])
    gap = np.array([m.x_u - m.x_s for m in measures])
    at_fold = clear_basin_measures(x_current, upper.a, *args)

    R_exponent = np.polyfit(np.log(deltas), np.log(R), 1)[0]
    gap_exponent = np.polyfit(np.log(deltas), np.log(gap), 1)[0]

    computed = {
        "b": b,
        "a_upper_fold": upper.a,
        "deltas": deltas.tolist(),
        "R": R.tolist(),
        "gap_x_u_minus_x_s": gap.tolist(),
        "R_at_fold": at_fold.R,
        "gap_at_fold": at_fold.x_u - at_fold.x_s,
        "R_decreasing": bool(np.all(np.diff(R) < 0)),
        "gap_decreasing": bool(np.all(np.diff(gap) < 0)),
        "fitted_exponent_R": float(R_exponent),
        "fitted_exponent_gap": float(gap_exponent),
    }
    entry = _entry(
        computed, None, None, "CLAUDE.md: R -> 0 and x_s -> x_u as a -> upper fold",
        note=("exponents are an extra: saddle-node normal form predicts "
              "R ~ delta^1.5 and gap ~ delta^0.5"),
    )
    entry["passed"] = bool(
        computed["R_decreasing"] and computed["gap_decreasing"]
        and computed["R_at_fold"] == 0.0 and computed["gap_at_fold"] == 0.0
    )
    return entry
