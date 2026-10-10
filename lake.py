"""Lake model of Carpenter, Ludwig & Brock (1999) and Walker et al.'s (2004)
resilience measures for its clear-water basin.

    dx/dt = F(x) = a - b*x + f(x),     f(x) = x^q / (1 + x^q)

x: phosphorus in the lake water, a: phosphorus input rate, b: loss rate,
q: steepness of the recycling term. The measures L, R and Pr are defined in
README.md, "Definitions".
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

# Baseline parameters. At b = 0.52 the lake is bistable for 0.0196 < a < 0.0735.
A0, B0, Q = 0.04, 0.52, 2.0
X_CURRENT = 0.3   # current state in Fig. 1, away from the clear equilibrium (~0.094)
SEED = 20041      # seed for the optional noise in the demo

# Numerics
N_GRID = 2000         # grid points when scanning a function for sign changes
XTOL = 1e-12          # tolerance on x for root finding
TANGENCY_TOL = 1e-12  # |F| at a turning point of F below this counts as a fold

STABLE, UNSTABLE, TANGENT = "stable", "unstable", "tangent"

# Colours shared by the figures and the demo (from the polder.mplstyle palette).
COLORS = {"clear": "#0A4D9E", "turbid": "#E6A700", "threshold": "#C8417B",
          "L": "#00A39B", "R": "#6A4C93", "Pr": "#E8633A",
          "ink": "#222222", "neutral": "#5F6B7A"}


# ---------------------------------------------------------------------------
# Model and potential
# ---------------------------------------------------------------------------

def f(x, q=Q):
    """Recycling term f(x) = x^q / (1 + x^q)."""
    return x**q / (1.0 + x**q)


def f_prime(x, q=Q):
    """f'(x) = q x^(q-1) / (1 + x^q)^2."""
    return q * x ** (q - 1) / (1.0 + x**q) ** 2


def F(x, a, b, q=Q):
    """dx/dt = a - b*x + f(x) (Carpenter et al. 1999, Eq. 1 rescaled)."""
    return a - b * x + f(x, q)


def F_prime(x, a, b, q=Q):
    """dF/dx = -b + f'(x). An equilibrium is stable where this is negative."""
    return -b + f_prime(x, q)


def V(x, a, b):
    """Potential for q = 2, defined by dx/dt = -dV/dx and V(0) = 0.

    Integrating -F(x) = -a + b*x - 1 + 1/(1 + x^2) from 0 gives
    V(x) = -a*x + b*x^2/2 - x + arctan(x).
    """
    return -a * x + 0.5 * b * x**2 - x + np.arctan(x)


# ---------------------------------------------------------------------------
# Equilibria, folds and regimes
# ---------------------------------------------------------------------------

def sign_change_roots(g, lo, hi):
    """Roots of g on [lo, hi]: scan a grid for sign changes, refine with brentq."""
    xs = np.linspace(lo, hi, N_GRID)
    gs = g(xs)
    roots = [float(x) for x, gx in zip(xs, gs) if gx == 0.0]
    for i in np.nonzero(gs[:-1] * gs[1:] < 0)[0]:
        roots.append(brentq(g, xs[i], xs[i + 1], xtol=XTOL))
    return sorted(roots)


def equilibria(a, b, q=Q):
    """All equilibria on x >= 0 as a sorted list of (x, kind).

    kind is STABLE (F' < 0), UNSTABLE (F' > 0) or TANGENT (a double root,
    exactly at a fold). Every equilibrium lies below (a + 1)/b, because
    f < 1 makes F negative beyond that.

    F is first split at its turning points (roots of F'). Between turning
    points F is monotone, so each piece holds at most one root. This finds
    the two equilibria next to a fold even when they are closer together
    than one grid cell.
    """
    turning = sign_change_roots(lambda x: F_prime(x, a, b, q), 0.0, (a + 1.0) / b)
    points = [0.0] + turning + [(a + 1.0) / b]
    values = [F(x, a, b, q) for x in points]

    found = []
    for i, x in enumerate(points):
        if abs(values[i]) <= TANGENCY_TOL:  # x = 0 when a = 0, or a fold
            found.append((x, TANGENT if x in turning else STABLE))
            values[i] = 0.0                 # so the pieces next to it skip it
    for i in range(len(points) - 1):
        if values[i] * values[i + 1] < 0:
            x = brentq(F, points[i], points[i + 1], args=(a, b, q), xtol=XTOL)
            found.append((x, STABLE if F_prime(x, a, b, q) < 0 else UNSTABLE))
    return sorted(found)


def fold_curve(x, q=Q):
    """Fold (saddle-node) curve in the (a, b) plane, parametrised by x.

    At a fold both F = 0 and F' = 0, which gives b = f'(x), a = x f'(x) - f(x).
    """
    b = f_prime(x, q)
    return x * b - f(x, q), b


def max_f_prime(q=Q):
    """(x, value) at the maximum of f'. For b at or above it, no folds exist.

    The maximum lies in [0, 1]: f'' has the sign of (q-1) - (q+1) x^q,
    which is negative at x = 1.
    """
    res = minimize_scalar(lambda x: -f_prime(x, q), bounds=(0.0, 1.0),
                          method="bounded", options={"xatol": XTOL})
    return float(res.x), float(-res.fun)


def folds(b, q=Q):
    """((a_upper, x_upper), (a_lower, x_lower)) for this b, or None if none.

    Upper fold: the clear state and the threshold merge (smaller x).
    Lower fold: the turbid state and the threshold merge (larger x).
    Solves b = f'(x); since f'(x) < q x^(-q-1), all solutions lie below
    (q/b)^(1/(q+1)).
    """
    xs = sign_change_roots(lambda x: f_prime(x, q) - b, 0.0, (q / b) ** (1 / (q + 1)))
    if len(xs) != 2:
        return None
    return tuple((float(fold_curve(x, q)[0]), x) for x in xs)


def irreversibility_boundary(q=Q):
    """(b, x) where the lower fold reaches a = 0; below this b, irreversible.

    Solves a(x) = x f'(x) - f(x) = 0 on the lower-fold branch (x beyond the
    maximum of f'), where a decreases monotonically. At x = (2q)^(1/q) we
    already have a < 0, which closes the bracket.
    """
    x_cusp, _ = max_f_prime(q)
    x = brentq(lambda s: fold_curve(s, q)[0], x_cusp, (2 * q) ** (1 / q), xtol=XTOL)
    return float(f_prime(x, q)), x


def regime(b, q=Q):
    """'reversible', 'hysteretic' or 'irreversible' (Carpenter et al. 1999, App. A).

    The boundary case where the lower fold sits exactly at a = 0 counts as
    hysteretic. Just below max f' the bistable window can be too narrow for
    the grid to resolve (width ~1e-10 in a); that case counts as reversible.
    """
    fold_pair = folds(b, q)
    if b >= max_f_prime(q)[1] or fold_pair is None:
        return "reversible"
    (_, _), (a_lower, _) = fold_pair
    return "irreversible" if a_lower < 0 else "hysteretic"


# ---------------------------------------------------------------------------
# Resilience measures (q = 2, because they use V)
# ---------------------------------------------------------------------------

def clear_basin(a, b, x=None):
    """Walker et al.'s measures for the clear basin [0, x_u), as a dict.

    L = x_u, R = V(x_u) - V(x_s), Pr = x_u - x, plus R/L and Pr/L.
    x is the current state; None puts the lake at rest at x_s. Without a
    threshold every measure is NaN and 'reason' says why.
    """
    eqs = equilibria(a, b)
    thresholds = [xe for xe, kind in eqs if kind != STABLE]
    if not thresholds:
        nan = float("nan")
        return dict(x_s=nan, x_u=nan, L=nan, R=nan, R_over_L=nan, Pr=nan,
                    Pr_over_L=nan, reason=_no_threshold_reason(a, b))

    x_s, x_u = eqs[0][0], thresholds[0]
    x = x_s if x is None else x
    L, R, Pr = x_u, float(V(x_u, a, b) - V(x_s, a, b)), x_u - x
    return dict(x_s=x_s, x_u=x_u, L=L, R=R, R_over_L=R / L, Pr=Pr, Pr_over_L=Pr / L,
                reason="past the threshold: Pr < 0" if Pr < 0 else "")


def _no_threshold_reason(a, b):
    fold_pair = folds(b)
    if fold_pair is None:
        return "reversible regime: one equilibrium, no threshold"
    (a_upper, _), _ = fold_pair
    if a > a_upper:
        return "clear basin has vanished (a above the upper fold)"
    return "only the clear state exists (a below the lower fold)"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def relax(x0, a, b, q=Q, settle_tol=1e-9, t_max=1e4):
    """Integrate from x0 with a, b fixed until |dx/dt| < settle_tol.

    Returns the solver's (t, x). In 1D the state ends at the stable
    equilibrium of the basin it started in.
    """
    if abs(F(x0, a, b, q)) < settle_tol:
        return np.array([0.0]), np.array([float(x0)])

    def settled(t, y):
        return abs(F(y[0], a, b, q)) - settle_tol
    settled.terminal = True

    sol = solve_ivp(lambda t, y: F(y, a, b, q), (0.0, t_max), [x0],
                    events=settled, rtol=1e-8, atol=1e-10)
    return sol.t, sol.y[0]


def sweep(b, a_min, a_max, rate, q=Q, n_points=4000):
    """Ramp a slowly from a_min to a_max and back at speed `rate`, starting
    from the equilibrium at a_min. Returns {"up": (a, x), "down": (a, x)}.

    The jumps between branches lag the folds a little; the lag shrinks as
    the rate decreases.
    """
    T = (a_max - a_min) / rate

    def a_of_t(t):
        return a_min + rate * t if t <= T else a_max - rate * (t - T)

    halves, x_start = {}, equilibria(a_min, b, q)[0][0]
    for name, (t0, t1) in (("up", (0.0, T)), ("down", (T, 2 * T))):
        t = np.linspace(t0, t1, n_points)
        sol = solve_ivp(lambda s, y: F(y, a_of_t(s), b, q), (t0, t1), [x_start],
                        t_eval=t, max_step=1.0, rtol=1e-8, atol=1e-10)
        halves[name] = (np.array([a_of_t(s) for s in sol.t]), sol.y[0])
        x_start = sol.y[0, -1]
    return halves


def euler_maruyama(x0, a, b, sigma, dt, n_steps, rng, q=Q):
    """Noisy path of dx = F(x) dt + sigma dW. Negative x is set to 0."""
    path = np.empty(n_steps + 1)
    path[0] = x0
    kicks = sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
    for n in range(n_steps):
        path[n + 1] = max(path[n] + F(path[n], a, b, q) * dt + kicks[n], 0.0)
    return path


# ---------------------------------------------------------------------------
# Quantitative checks (README.md, "Quantitative checks")
# ---------------------------------------------------------------------------

def quantitative_checks():
    """Computed values next to their expected values, for q = 2.

    Used by test_checks.py and written to results/checks.json by
    make_figures.py. The fold values at b = 0.52 were calculated by hand.
    """
    x_max, f_max = max_f_prime()
    b_irr, x_irr = irreversibility_boundary()
    (a_up, x_up), (a_low, x_low) = folds(B0)

    # R and the gap x_u - x_s as a approaches the upper fold from below.
    deltas = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
    near = [clear_basin(a_up - d, B0) for d in deltas]

    (a_up62, _), (a_low62, _) = folds(0.62)
    return {
        "max_f_prime": dict(computed=dict(value=f_max, x=x_max),
                            expected=dict(value=3 * np.sqrt(3) / 8, x=1 / np.sqrt(3)), tol=1e-7),
        "irreversibility_boundary": dict(computed=dict(b=b_irr, x=x_irr),
                                         expected=dict(b=0.5, x=1.0), tol=1e-7),
        "upper_fold_b052": dict(computed=dict(a=a_up, x=x_up),
                                expected=dict(a=0.0735, x=0.314), tol=1e-3),
        "lower_fold_b052": dict(computed=dict(a=a_low, x=x_low),
                                expected=dict(a=0.0196, x=0.960), tol=1e-3),
        "resistance_at_upper_fold": dict(computed=dict(
            deltas=deltas, R=[m["R"] for m in near],
            gap=[m["x_u"] - m["x_s"] for m in near])),
        "bistable_window_b062": dict(computed=dict(
            regime=regime(0.62), a_lower=a_low62, a_upper=a_up62, width=a_up62 - a_low62)),
    }
