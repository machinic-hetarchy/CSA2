"""The lake model and its potential (stability landscape).

Dimensionless continuous-time model, Carpenter et al. (1999), Eq. 1 rescaled:

    dx/dt = F(x) = a - b*x + f(x),     f(x) = x**q / (1 + x**q)

x is phosphorus in the lake water (x >= 0), a is the phosphorus input rate,
b is the loss rate and q sets the steepness of the recycling term f.

The potential V is defined by dx/dt = -dV/dx, with V(0) = 0. The state rolls
downhill on V, so V is the quantitative version of the landscape drawn in
Walker et al. (2004), Fig. 1.

Function names follow the maths (f, F, V) so the code can be read next to
the equations.
"""

import numpy as np
from scipy.integrate import quad


def f(x, q):
    """Sigmoid phosphorus recycling, f(x) = x^q / (1 + x^q).

    Carpenter et al. (1999), recycling term of Eq. 1. Defined for x >= 0.
    """
    xq = np.power(x, q)
    return xq / (1.0 + xq)


def f_prime(x, q):
    """Derivative of the recycling term, f'(x) = q x^(q-1) / (1 + x^q)^2.

    Used for stability (F' = -b + f') and for the fold condition b = f'(x).
    Assumes q >= 1 so that f' is finite at x = 0.
    """
    xq = np.power(x, q)
    return q * np.power(x, q - 1.0) / (1.0 + xq) ** 2


def F(x, a, b, q):
    """Rate of change dx/dt = a - b*x + f(x) (Carpenter et al. 1999, Eq. 1).

    Equilibria satisfy F(x) = 0, the same condition as Carpenter et al.'s
    discrete-time model (their Eq. A.8 with a = A/C, b = B/C).
    """
    return a - b * x + f(x, q)


def F_prime(x, a, b, q):
    """Derivative dF/dx = -b + f'(x).

    An equilibrium x* is stable if F'(x*) < 0 and unstable if F'(x*) > 0.
    The parameter a does not appear but is kept so the signature matches F.
    """
    return -b + f_prime(x, q)


def V_closed_q2(x, a, b):
    """Closed-form potential for q = 2: V(x) = -a x + b x^2/2 - x + arctan(x).

    Derivation: -dV/dx must equal F(x) = a - b x + x^2/(1+x^2)
    = a - b x + 1 - 1/(1+x^2). Integrating -F from 0 to x gives the
    expression above, with V(0) = 0.
    """
    return -a * x + 0.5 * b * x**2 - x + np.arctan(x)


def V_numerical(x, a, b, q):
    """Potential for any q, by numerical integration: V(x) = -int_0^x F(s) ds.

    Uses scipy.integrate.quad with its default tolerances for each point, so
    it is slower than the closed form; fine for plotting grids of a few
    thousand points. Accepts a scalar or an array of x values (x >= 0).
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    V = np.empty_like(x_arr)
    for i, xi in enumerate(x_arr):
        integral, _abserr = quad(F, 0.0, xi, args=(a, b, q))
        V[i] = -integral
    return V[0] if np.ndim(x) == 0 else V


def V(x, a, b, q):
    """Potential V(x) with dx/dt = -dV/dx and V(0) = 0.

    Uses the closed form when q = 2 and numerical integration otherwise.
    The two are checked against each other in the tests.
    """
    if q == 2:
        return V_closed_q2(x, a, b)
    return V_numerical(x, a, b, q)
