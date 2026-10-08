"""Regenerate every figure and results/checks.json from scratch.

Run from the repository root:

    python scripts/make_figures.py

Each fig_* function draws one figure; its docstring is the figure caption.
Every figure is saved as PDF and PNG in figures/. All parameter values come
from config/params.yaml.
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402

# Make the `lake` package in src/ importable without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lake.checks import compute_checks  # noqa: E402
from lake.config import load_params  # noqa: E402
from lake.equilibria import (  # noqa: E402
    STABLE,
    equilibria,
    fold_curve,
    folds,
    irreversibility_boundary,
    max_f_prime,
)
from lake.model import V, f, f_prime  # noqa: E402
from lake.resilience import clear_basin_measures  # noqa: E402
from lake.simulate import sweep_a  # noqa: E402
from lake.style import COLORS, TINT_ALPHA, use_style  # noqa: E402

FIG_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# Layout constants (fractions of the axis range), used only for placing
# annotations so they do not overlap the curves.
ANNOT_PAD = 0.12
BALL_SIZE = 12      # marker size of the "ball" (current state), points
EQ_SIZE = 7         # marker size of equilibria, points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save(fig, name, formats):
    """Save a figure in every requested format and close it."""
    FIG_DIR.mkdir(exist_ok=True)
    for ext in formats:
        # No creation date in PDFs, so regenerating gives identical files.
        metadata = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(FIG_DIR / f"{name}.{ext}", metadata=metadata)
    plt.close(fig)
    print(f"wrote figures/{name}.{{{','.join(formats)}}}")


def root_args(params):
    """(n_grid, xtol, tangency_tol) for the root finder."""
    num = params["numerics"]
    return num["n_grid"], num["xtol"], num["tangency_tol"]


def branch_color(x, fold_pair):
    """Colour of a stable equilibrium: clear if it lies on the clear branch
    (x at most the x of the upper fold), turbid otherwise. Without folds
    (reversible regime) there is no clear/turbid distinction."""
    if fold_pair is None:
        return COLORS["neutral"]
    upper, _ = fold_pair
    return COLORS["clear"] if x <= upper.x else COLORS["turbid"]


def draw_landscape(ax, a, b, q, x, params, ball_x=None):
    """Draw V(x) with shaded basins, equilibria and (optionally) the ball.

    Shared by Fig. 1a and Fig. 1b so both use the same visual language.
    Returns the clear-basin measures for (a, b, q).
    """
    n_grid, xtol, tan_tol = root_args(params)
    eqs = equilibria(a, b, q, n_grid, xtol, tan_tol)
    fold_pair = folds(b, q, n_grid, xtol)
    m = clear_basin_measures(ball_x, a, b, q, n_grid, xtol, tan_tol)

    if not math.isnan(m.x_u):
        ax.axvspan(x[0], m.x_u, color=COLORS["clear"], alpha=TINT_ALPHA, lw=0)
        ax.axvspan(m.x_u, x[-1], color=COLORS["turbid"], alpha=TINT_ALPHA, lw=0)
    ax.plot(x, V(x, a, b, q), color=COLORS["ink"], lw=2)

    for e in eqs:
        y = V(e.x, a, b, q)
        if e.stability == STABLE:
            ax.plot(e.x, y, "o", ms=EQ_SIZE, color=branch_color(e.x, fold_pair), zorder=4)
        else:  # threshold, or tangent point at a fold
            ax.plot(e.x, y, "o", ms=EQ_SIZE, mfc="white", mec=COLORS["threshold"],
                    mew=2, zorder=4)

    if ball_x is not None:
        # Draw the ball resting on the curve: shift it up by its radius.
        on_curve = mtransforms.offset_copy(ax.transData, fig=ax.figure,
                                           y=BALL_SIZE / 2, units="points")
        ax.plot(ball_x, V(ball_x, a, b, q), "o", ms=BALL_SIZE, color=COLORS["ink"],
                mec="white", mew=1.5, transform=on_curve, zorder=5)
    ax.set_xlim(x[0], x[-1])
    return m


def double_arrow(ax, xy_from, xy_to, color):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.6,
                                shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_landscape(params):
    """Fig. 1a analogue: stability landscape with latitude, resistance and
    precariousness.

    Caption: Stability landscape V(x) of the lake model, defined by
    dx/dt = -dV/dx, at the baseline parameters (config/params.yaml;
    a = 0.04, b = 0.52, q = 2). Blue shading: clear basin [0, x_u);
    amber shading: turbid basin [x_u, inf). Filled circles are stable
    equilibria (blue clear, amber turbid); the open circle is the threshold
    x_u. The black ball is the current state. Latitude L = x_u is the width
    of the clear basin, resistance R = V(x_u) - V(x_s) its depth, and
    precariousness Pr = x_u - x the distance from the current state to the
    threshold. Quantitative version of Walker et al. (2004), Fig. 1a.
    """
    a, b, q = (params["model"][k] for k in ("a", "b", "q"))
    xc = params["baseline"]["x_current"]
    fp = params["figures"]
    x = np.linspace(0.0, fp["x_max"], fp["n_x"])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    m = draw_landscape(ax, a, b, q, x, params, ball_x=xc)

    Vx = V(x, a, b, q)
    span = Vx.max() - Vx.min()
    V_s, V_u = V(m.x_s, a, b, q), V(m.x_u, a, b, q)

    # Latitude: width of the clear basin, drawn below the curve.
    y_L = Vx.min() - ANNOT_PAD * span
    double_arrow(ax, (0.0, y_L), (m.x_u, y_L), COLORS["L"])
    ax.text(m.x_u / 2, y_L, f"latitude L = {m.L:.3f}\n", ha="center", va="center",
            color=COLORS["ink"], fontsize=10)

    # Resistance: depth of the clear basin, from the level of the clear
    # equilibrium up to the threshold, drawn at the threshold (barrier height).
    ax.plot([m.x_s, m.x_u], [V_s, V_s], ls=":", color=COLORS["R"], lw=1.2)
    double_arrow(ax, (m.x_u, V_s), (m.x_u, V_u), COLORS["R"])
    ax.text(m.x_u + 0.03, (V_s + V_u) / 2, f"resistance\nR = {m.R:.4f}", ha="left",
            va="center", color=COLORS["ink"], fontsize=10)

    # Precariousness: from the current state to the threshold, above the curve.
    y_Pr = V_u + ANNOT_PAD * span
    double_arrow(ax, (xc, y_Pr), (m.x_u, y_Pr), COLORS["Pr"])
    ax.plot([xc, xc], [V(xc, a, b, q), y_Pr], ls=":", color=COLORS["Pr"], lw=1.2)
    ax.plot([m.x_u, m.x_u], [V_u, y_Pr], ls=":", color=COLORS["Pr"], lw=1.2)
    ax.text((xc + m.x_u) / 2, y_Pr, f"precariousness Pr = {m.Pr:.3f}\n",
            ha="center", va="center", color=COLORS["ink"], fontsize=10)

    ax.text(m.x_u, V_u, "  threshold $x_u$", ha="left", va="bottom", fontsize=10,
            color=COLORS["ink"])
    ax.text(0.02, 0.97, "clear basin", transform=ax.transAxes, va="top",
            color=COLORS["clear"], fontweight="bold")
    ax.text(0.98, 0.97, "turbid basin", transform=ax.transAxes, va="top", ha="right",
            color=COLORS["ink"], fontweight="bold")

    ax.set_ylim(y_L - ANNOT_PAD * span, Vx.max() + ANNOT_PAD * span)
    ax.set_xlabel("phosphorus in lake water, $x$ (dimensionless)")
    ax.set_ylabel("potential $V(x)$")
    ax.set_title(f"Stability landscape  (a = {a}, b = {b}, q = {q:g})")
    save(fig, "fig1_landscape", fp["formats"])


def fig_landscape_sequence(params):
    """Fig. 1b analogue: the clear basin shrinks and vanishes as a rises.

    Caption: Stability landscapes V(x) for increasing phosphorus input a at
    b = 0.52, q = 2 (small multiples with a shared vertical axis, so depths
    are directly comparable). The ball sits at the clear equilibrium while it
    exists. As a rises, the clear basin becomes narrower and shallower; at
    the upper fold (a = a_fold, marked) the clear equilibrium and the
    threshold merge (R = 0), and beyond it the ball rolls to the turbid
    state. Quantitative version of Walker et al. (2004), Fig. 1b.
    """
    b, q = params["model"]["b"], params["model"]["q"]
    fp = params["figures"]
    n_grid, xtol, tan_tol = root_args(params)
    upper, _ = folds(b, q, n_grid, xtol)

    a_values = list(fp["sequence_a"])
    if fp["include_upper_fold"]:
        a_values.append(upper.a)
    a_values.sort()
    x = np.linspace(0.0, fp["x_max"], fp["n_x"])

    fig, axes = plt.subplots(1, len(a_values), figsize=(2.15 * len(a_values), 3.4),
                             sharey=True)
    for ax, a in zip(axes, a_values):
        eqs = equilibria(a, b, q, n_grid, xtol, tan_tol)
        # The lake follows the clear state while it exists, else it is turbid.
        ball = eqs[0].x if a <= upper.a else eqs[-1].x
        m = draw_landscape(ax, a, b, q, x, params, ball_x=ball)

        at_fold = a == upper.a
        ax.set_title(f"a = {a:.4f}" + ("\n(upper fold)" if at_fold else ""), fontsize=11)
        if math.isnan(m.R):
            label = "no clear basin"
        else:
            label = f"R = {m.R:.4f}\nL = {m.L:.2f}"
        ax.text(0.05, 0.95, label, transform=ax.transAxes, ha="left", va="top",
                fontsize=9, color=COLORS["ink"])
    axes[0].set_ylabel("potential $V(x)$")
    fig.supxlabel("phosphorus in lake water, $x$ (dimensionless)", fontsize=11)
    fig.suptitle(f"Clear basin shrinking as phosphorus input rises  (b = {b}, q = {q:g})",
                 fontweight="bold", color=plt.rcParams["axes.titlecolor"])
    fig.tight_layout()
    save(fig, "fig2_landscape_sequence", fp["formats"])


def fig_bifurcation(params):
    """Bifurcation diagram with a simulated slow sweep (hysteresis loop).

    Caption: Equilibria x* of the lake model against phosphorus input a for
    b = 0.52, q = 2. Solid lines: stable equilibria (blue clear branch, amber
    turbid branch); dashed magenta: the unstable threshold. Open circles:
    the upper and lower folds. Grey: simulated trajectory when a is ramped
    slowly up from 0 to 0.1 and back (da/dt = 1e-5). The lake stays clear
    until just past the upper fold, jumps to turbid, and returns to clear only
    below the lower fold: hysteresis. The jumps lag the folds slightly
    because the ramp has a finite speed (see results/checks.json, sweep_lag).
    """
    b, q = params["model"]["b"], params["model"]["q"]
    fp, sw, integ = params["figures"], params["sweep"], params["integration"]
    n_grid, xtol, _ = root_args(params)
    upper, lower = folds(b, q, n_grid, xtol)

    # Equilibrium curve: F(x) = 0 solved for a gives a = b*x - f(x), exactly.
    x = np.linspace(0.0, fp["x_max"], fp["fold_curve_n"])
    a_eq = b * x - f(x, q)
    in_range = (a_eq >= sw["a_min"]) & (a_eq <= sw["a_max"])
    clear = in_range & (x <= upper.x)
    unstable = in_range & (x >= upper.x) & (x <= lower.x)
    turbid = in_range & (x >= lower.x)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    halves = sweep_a(0.0, b, q, sw["a_min"], sw["a_max"], sw["rate"], sw["max_step"],
                     sw["n_points"], integ["rtol"], integ["atol"])
    for name, h in halves.items():
        ax.plot(h["a"], h["x"], color=COLORS["neutral"], lw=1.2, zorder=1,
                label="slow sweep of a (simulated)" if name == "up" else None)

    ax.plot(a_eq[clear], x[clear], color=COLORS["clear"], lw=2.5, label="stable equilibrium")
    ax.plot(a_eq[turbid], x[turbid], color=COLORS["turbid"], lw=2.5)
    ax.plot(a_eq[unstable], x[unstable], color=COLORS["threshold"], lw=2, ls="--",
            label="unstable equilibrium (threshold)")

    for fold, name, va in ((upper, "upper fold", "top"), (lower, "lower fold", "bottom")):
        ax.plot(fold.a, fold.x, "o", ms=EQ_SIZE, mfc="white", mec=COLORS["ink"], mew=1.5,
                zorder=4)
        ax.annotate(f"{name}\na = {fold.a:.4f}", (fold.a, fold.x),
                    xytext=(8, -8 if va == "top" else 8), textcoords="offset points",
                    ha="left", va=va, fontsize=9)

    # Direction arrows on the jumps of the sweep.
    x_mid = 0.5 * (upper.x + lower.x)
    for h, sign in ((halves["up"], 1), (halves["down"], -1)):
        i = np.nonzero(h["x"] > x_mid)[0][0] if sign > 0 else np.nonzero(h["x"] < x_mid)[0][0]
        ax.annotate("", xy=(h["a"][i], x_mid + sign * 0.1), xytext=(h["a"][i], x_mid - sign * 0.1),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["neutral"], lw=1.2))

    ax.text(0.003, 0.15, "clear", color=COLORS["clear"], fontweight="bold")
    ax.text(0.003, 1.62, "turbid", color=COLORS["ink"], fontweight="bold")
    ax.set_xlim(sw["a_min"], sw["a_max"])
    ax.set_ylim(0.0, fp["x_max"])
    ax.set_xlabel("phosphorus input rate, $a$")
    ax.set_ylabel("equilibrium phosphorus, $x^*$")
    ax.set_title(f"Bifurcation diagram and hysteresis  (b = {b}, q = {q:g})")
    ax.legend(loc="center right")
    save(fig, "fig3_bifurcation", fp["formats"])


def fig_measures(params):
    """Resilience measures of the clear basin against phosphorus input a.

    Caption: Walker et al.'s measures for the clear basin as a function of
    phosphorus input a (b = 0.52, q = 2), with the lake at rest at the clear
    equilibrium x_s. Latitude L = x_u, resistance R = V(x_u) - V(x_s), the
    ratio R/L, and precariousness Pr = x_u - x_s. Dashed line: upper fold,
    where the clear basin vanishes. Left grey band: below the lower fold
    there is no threshold, so the measures are undefined (the clear basin
    is unbounded). Right amber band: clear basin vanished. Note that R and
    Pr go to zero at the fold but L does not, because L is measured from
    the domain boundary x = 0.
    """
    b, q = params["model"]["b"], params["model"]["q"]
    fp = params["figures"]
    n_grid, xtol, tan_tol = root_args(params)
    upper, lower = folds(b, q, n_grid, xtol)

    a_max = params["sweep"]["a_max"]
    a_grid = np.union1d(np.linspace(0.0, a_max, fp["measures_n_a"]), [upper.a])
    ms = [clear_basin_measures(None, a, b, q, n_grid, xtol, tan_tol) for a in a_grid]

    panels = [
        ("L", "latitude  $L = x_u$", [m.L for m in ms], COLORS["L"]),
        ("R", "resistance  $R = V(x_u) - V(x_s)$", [m.R for m in ms], COLORS["R"]),
        ("R/L", "ratio  $R / L$", [m.R_over_L for m in ms], COLORS["R"]),
        ("Pr", "precariousness  $Pr = x_u - x_s$", [m.Pr for m in ms], COLORS["Pr"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True)
    for ax, (key, title, values, color) in zip(axes.flat, panels):
        ax.axvspan(0.0, lower.a, color=COLORS["neutral"], alpha=TINT_ALPHA, lw=0)
        ax.axvspan(upper.a, a_max, color=COLORS["turbid"], alpha=TINT_ALPHA, lw=0)
        ax.axvline(upper.a, color=COLORS["ink"], ls="--", lw=1)
        ax.plot(a_grid, values, color=color, lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0.0, a_max)
        ax.set_ylim(bottom=0.0)
    for ax in axes[1]:
        ax.set_xlabel("phosphorus input rate, $a$")
    axes[0, 0].text(lower.a / 2, 0.5, "no\nthreshold", transform=axes[0, 0].get_xaxis_transform(),
                    ha="center", va="center", fontsize=9, color=COLORS["ink"])
    axes[0, 0].text(upper.a, 0.95, " upper fold", transform=axes[0, 0].get_xaxis_transform(),
                    ha="left", va="top", fontsize=9, color=COLORS["ink"])
    axes[0, 1].text((upper.a + a_max) / 2, 0.5, "clear basin\nvanished",
                    transform=axes[0, 1].get_xaxis_transform(), ha="center", va="center",
                    fontsize=9, color=COLORS["ink"])
    fig.suptitle(f"Resilience of the clear state as input rises  (b = {b}, q = {q:g})",
                 fontweight="bold", color=plt.rcParams["axes.titlecolor"])
    fig.tight_layout()
    save(fig, "fig4_measures", fp["formats"])


def _fold_branches(q, params):
    """Parametric fold curve split at the cusp into the two fold branches.

    Returns ((a_up, b_up), (a_low, b_low)): the upper-fold branch (x below
    the maximum of f') and the lower-fold branch (x above it).
    """
    fp = params["figures"]
    x_cusp, _ = max_f_prime(q, params["numerics"]["xtol"])
    x = np.linspace(0.0, fp["fold_curve_x_max"], fp["fold_curve_n"])[1:]
    a, b = fold_curve(x, q)
    up, low = x <= x_cusp, x >= x_cusp
    return (a[up], b[up]), (a[low], b[low])


def _draw_fold_curve(ax, q, params):
    """Draw the fold curve and return it as a closed matplotlib Polygon
    (not added to the axes) enclosing the bistable region."""
    (a_up, b_up), (a_low, b_low) = _fold_branches(q, params)
    ax.plot(a_up, b_up, color=COLORS["ink"], lw=2)
    ax.plot(a_low, b_low, color=COLORS["ink"], lw=2)
    # Upper-fold branch runs from the origin to the cusp, lower-fold branch
    # from the cusp onwards: together they outline the bistable region.
    xy = np.column_stack([np.concatenate([a_up, a_low]), np.concatenate([b_up, b_low])])
    return Polygon(xy, closed=True, transform=ax.transData)


def _label(ax, a, b, text, **kwargs):
    """Text with a white backing so it stays legible over lines and hatching."""
    ax.text(a, b, text, fontsize=9, color=COLORS["ink"],
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), **kwargs)


def _pointer(ax, text, xy_target, xy_text, **kwargs):
    """Label placed in empty space with an arrow into the region it names."""
    ax.annotate(text, xy=xy_target, xytext=xy_text, fontsize=9, color=COLORS["ink"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1), **kwargs)


def fig_regimes(params):
    """Regime diagram in the (a, b) plane for q = 2.

    Caption: Fold (saddle-node) curve of the lake model in the (a, b) plane
    for q = 2, traced from b = f'(x), a = x f'(x) - f(x). Hatched: bistable
    region (two stable equilibria and a threshold). Horizontal bands give
    the regime as defined for this project (a property of b alone):
    reversible for b >= max f' = 0.6495, irreversible for b < 0.5 (lower
    fold at a < 0), hysteretic in between. Carpenter et al. (1999, Fig. A1B)
    instead label regions of the (a, b) plane (see Fig. 6B). Marked: our
    baseline (a = 0.04, b = 0.52) and b = 0.62, which Carpenter et al. use
    as a 'reversible' example (their Fig. 11) but which lies in the
    hysteretic band, with a bistable window of width about 0.0057 in a.
    """
    q = params["model"]["q"]
    fp = params["figures"]
    xtol = params["numerics"]["xtol"]
    a_max, b_max = fp["regime_a_max"], fp["regime_b_max"]
    _, b_rev = max_f_prime(q, xtol)
    b_irr, _ = irreversibility_boundary(q, xtol)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axhspan(0.0, b_irr, color=COLORS["turbid"], alpha=TINT_ALPHA, lw=0)
    ax.axhspan(b_irr, b_rev, color=COLORS["neutral"], alpha=TINT_ALPHA, lw=0)
    ax.axhspan(b_rev, b_max, color=COLORS["clear"], alpha=TINT_ALPHA, lw=0)
    for b_line in (b_irr, b_rev):
        ax.axhline(b_line, color=COLORS["ink"], lw=0.8, ls=":")

    cusp = _draw_fold_curve(ax, q, params)
    cusp.set(facecolor="none", hatch="///", edgecolor=COLORS["neutral"], lw=0)
    ax.add_patch(cusp)

    for name, b_txt in (("reversible", 0.72), ("hysteretic", 0.535), ("irreversible", 0.25)):
        ax.text(0.17, b_txt, name, fontsize=10, color=COLORS["ink"], fontweight="bold")
    _label(ax, a_max, b_irr, f"b = {b_irr:.3f} ", ha="right", va="bottom")
    _label(ax, a_max, b_rev, f"b = max f' = {b_rev:.4f} ", ha="right", va="bottom")

    b_w = params["checks"]["b_window"]
    ax.axhline(b_w, color=COLORS["Pr"], lw=1.2, ls="--")
    _label(ax, a_max, b_w, f"b = {b_w}: Carpenter et al.'s 'reversible' example ",
           ha="right", va="top")

    a0, b0 = params["model"]["a"], params["model"]["b"]
    ax.plot(a0, b0, "o", ms=EQ_SIZE + 1, color=COLORS["ink"], zorder=5)
    _label(ax, a0 + 0.005, b0, "baseline", ha="left", va="center")
    _pointer(ax, "bistable (inside fold curve)", (0.012, 0.3), (0.06, 0.3))

    ax.set_xlim(0.0, a_max)
    ax.set_ylim(0.0, b_max)
    ax.set_xlabel("input parameter $a$")
    ax.set_ylabel("loss parameter $b$")
    ax.set_title(f"Regimes of the lake model  (q = {q:g})")
    save(fig, "fig5_regimes", fp["formats"])


def fig_carpenter_comparison(params):
    """Side-by-side reproduction of Carpenter et al. (1999), Fig. A1.

    Caption: Our fold curves drawn on the same axes as Carpenter et al.
    (1999), Fig. A1, for visual comparison. (A) Fold curves for q = 2 and
    q = 8. (B) Regimes for q = 8, labelled as in their Fig. A1B: outside the
    fold curve the lake is reversible (one equilibrium); inside it, the
    horizontal line at the b where the lower fold reaches a = 0 separates
    hysteretic (above) from irreversible (below). Note this labels regions
    of the (a, b) plane, whereas Fig. 5 labels values of b. The
    continuous-time model used here has the same equilibria and folds as
    their discrete-time model (Eq. A.8 with a = A/C, b = B/C), so the curves
    should coincide. Their dotted 'minimum input' lines are not reproduced.
    """
    fp = params["figures"]
    xtol = params["numerics"]["xtol"]
    a_max, b_max = fp["carpenter_a_max"], fp["carpenter_b_max"]
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.0, 8.0), sharex=True)

    for q, color in zip(fp["carpenter_q"], (COLORS["clear"], COLORS["ink"])):
        (a_up, b_up), (a_low, b_low) = _fold_branches(q, params)
        ax_a.plot(a_up, b_up, color=color, lw=2, label=f"q = {q:g}")
        ax_a.plot(a_low, b_low, color=color, lw=2)
    ax_a.set_xlim(0.0, a_max)
    ax_a.set_ylim(0.0, b_max)
    ax_a.set_ylabel("loss parameter $b$")
    ax_a.legend(loc="upper left")
    ax_a.set_title("(A) fold curves", loc="left")

    q_b = max(fp["carpenter_q"])
    b_irr, _ = irreversibility_boundary(q_b, xtol)
    cusp = _draw_fold_curve(ax_b, q_b, params)
    # Shade the inside of the fold curve only, split at b_irr.
    for lo, hi, color in ((0.0, b_irr, COLORS["turbid"]), (b_irr, b_max, COLORS["neutral"])):
        band = ax_b.axhspan(lo, hi, color=color, alpha=3 * TINT_ALPHA, lw=0)
        band.set_clip_path(cusp)
    line = ax_b.axhline(b_irr, color=COLORS["ink"], lw=1.5)
    line.set_clip_path(cusp)
    ax_b.text(0.1, 1.75, "reversible", fontsize=10, fontweight="bold", color=COLORS["ink"])
    _pointer(ax_b, "hysteretic", (0.6, 1.05), (0.95, 0.95), fontweight="bold")
    _pointer(ax_b, "irreversible", (0.15, 0.45), (0.6, 0.3), fontweight="bold")
    _label(ax_b, a_max, b_irr, f"b = {b_irr:.3f} ", ha="right", va="bottom")
    ax_b.set_xlim(0.0, a_max)
    ax_b.set_ylim(0.0, b_max)
    ax_b.set_xlabel("input parameter $a$")
    ax_b.set_ylabel("loss parameter $b$")
    ax_b.set_title(f"(B) regimes for q = {q_b:g}", loc="left")
    fig.suptitle("Comparison with Carpenter et al. (1999), Fig. A1",
                 fontweight="bold", color=plt.rcParams["axes.titlecolor"])
    fig.tight_layout()
    save(fig, "fig6_carpenter_A1_comparison", fp["formats"])


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def write_checks(params):
    """Compute all checks and write them, with the seed, to results/checks.json."""
    checks = compute_checks(params)
    checks["metadata"] = {"seed": params["random"]["seed"], "config": "config/params.yaml"}
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "checks.json"
    with open(out, "w") as fh:
        json.dump(_nan_to_none(checks), fh, indent=2, allow_nan=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def _nan_to_none(obj):
    """Replace NaN by None (JSON null): bare NaN is not valid JSON."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def main():
    params = load_params()
    use_style(params["figures"]["style"])
    write_checks(params)
    fig_landscape(params)
    fig_landscape_sequence(params)
    fig_bifurcation(params)
    fig_measures(params)
    fig_regimes(params)
    fig_carpenter_comparison(params)


if __name__ == "__main__":
    main()
