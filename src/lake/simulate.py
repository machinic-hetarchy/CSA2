"""Time integration of the lake model.

Deterministic runs use scipy.integrate.solve_ivp on
dx/dt = F(x) = a - b*x + f(x) (Carpenter et al. 1999, Eq. 1 rescaled).
"""

import numpy as np
from scipy.integrate import solve_ivp

from lake.model import F


def relax(x0, a, b, q, rtol, atol, settle_tol, t_max):
    """Integrate from x0 with a, b fixed until the state stops moving.

    Stops when |dx/dt| < settle_tol (a terminal solve_ivp event) or at t_max.
    Because the model is one-dimensional and the state only rolls downhill
    on V, it ends at the stable equilibrium of the basin x0 started in.
    This is what the demo does after every slider move, so hysteresis shows.

    Returns (t, x, settled): the solver's time points and states, and False
    if t_max was reached before settling.
    """
    if abs(F(x0, a, b, q)) < settle_tol:
        return np.array([0.0]), np.array([float(x0)]), True

    def still(t, y):
        return abs(F(y[0], a, b, q)) - settle_tol

    still.terminal = True
    still.direction = -1

    sol = solve_ivp(lambda t, y: F(y, a, b, q), (0.0, t_max), [x0],
                    events=still, rtol=rtol, atol=atol)
    return sol.t, sol.y[0], sol.status == 1


def settle(x0, a, b, q, rtol, atol, settle_tol, t_max):
    """Final state of `relax`. Returns (x_final, settled)."""
    _, x, settled = relax(x0, a, b, q, rtol, atol, settle_tol, t_max)
    return float(x[-1]), settled


def euler_maruyama(x0, a, b, q, sigma, dt, n_steps, rng):
    """Noisy path of dx = F(x) dt + sigma dW (Euler-Maruyama, additive noise).

    x_{n+1} = x_n + F(x_n) dt + sigma * sqrt(dt) * N(0, 1).
    Phosphorus cannot be negative, so a step that would go below zero is
    set to zero (a modelling choice; it only matters next to x = 0).
    rng is a numpy Generator created from the seed in config/params.yaml.
    Returns the path as an array of length n_steps + 1.
    """
    path = np.empty(n_steps + 1)
    path[0] = x0
    kicks = sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
    for n in range(n_steps):
        path[n + 1] = max(path[n] + F(path[n], a, b, q) * dt + kicks[n], 0.0)
    return path


def sweep_a(x0, b, q, a_min, a_max, rate, max_step, n_points, rtol, atol):
    """Slowly ramp a up from a_min to a_max and back down, integrating x.

    a(t) rises linearly at speed `rate` for T = (a_max - a_min)/rate time
    units, then falls back at the same speed. With the ramp slow compared
    to the relaxation time of the lake, x tracks a stable equilibrium until
    that equilibrium disappears at a fold, then jumps to the other branch.
    Up and down paths differ inside the bistable window: the hysteresis loop.

    Returns two dicts, "up" and "down", each with arrays t, a and x.
    """
    T = (a_max - a_min) / rate

    def a_of_t(t):
        return a_min + rate * t if t <= T else a_max - rate * (t - T)

    def rhs(t, y):
        return F(y, a_of_t(t), b, q)

    halves = {}
    x_start = x0
    for name, (t0, t1) in {"up": (0.0, T), "down": (T, 2 * T)}.items():
        t_eval = np.linspace(t0, t1, n_points)
        sol = solve_ivp(rhs, (t0, t1), [x_start], t_eval=t_eval,
                        max_step=max_step, rtol=rtol, atol=atol)
        a_vals = np.array([a_of_t(t) for t in sol.t])
        halves[name] = {"t": sol.t, "a": a_vals, "x": sol.y[0]}
        x_start = sol.y[0, -1]
    return halves


def jump_points(halves, x_level):
    """Values of a at which a sweep jumps between branches.

    The jump is taken as the first crossing of the level x_level: upwards
    on the up-sweep (clear -> turbid), downwards on the down-sweep
    (turbid -> clear). We use x_level halfway between the x of the two
    folds, which lies between the branches. Returns (a_up, a_down), NaN if
    no jump happened.
    """
    up, down = halves["up"], halves["down"]
    crossed_up = np.nonzero(up["x"] > x_level)[0]
    crossed_down = np.nonzero(down["x"] < x_level)[0]
    a_up = up["a"][crossed_up[0]] if crossed_up.size else np.nan
    a_down = down["a"][crossed_down[0]] if crossed_down.size else np.nan
    return float(a_up), float(a_down)
