"""Regenerate every figure (figures/*.pdf) and results/checks.json.

    python make_figures.py

Each fig_* function draws one figure; its docstring is the caption.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import lake  # noqa: E402
from lake import A0, B0, COLORS, Q, X_CURRENT  # noqa: E402

HERE = Path(__file__).parent
FIG_DIR, RESULTS_DIR = HERE / "figures", HERE / "results"

X_MAX = 1.8                                  # right edge of the landscape plots
SEQUENCE_A = [0.02, 0.04, 0.06, 0.07, 0.08]  # Fig. 2 (the exact upper fold is added)
SWEEP = dict(a_min=0.0, a_max=0.1, rate=1e-5)  # Fig. 3, slow ramp of a
TINT = 0.12                                  # transparency of shaded regions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    # No creation date, so regenerating gives identical files.
    fig.savefig(FIG_DIR / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def draw_landscape(ax, a, b, x, ball_x=None):
    """V(x) with shaded basins, equilibria (filled stable, open threshold)
    and optionally the ball. Returns the clear-basin measures."""
    m = lake.clear_basin(a, b, ball_x)
    fold_pair = lake.folds(b)
    if not math.isnan(m["x_u"]):
        ax.axvspan(x[0], m["x_u"], color=COLORS["clear"], alpha=TINT, lw=0)
        ax.axvspan(m["x_u"], x[-1], color=COLORS["turbid"], alpha=TINT, lw=0)
    ax.plot(x, lake.V(x, a, b), color=COLORS["ink"], lw=2)

    for xe, kind in lake.equilibria(a, b):
        if kind == lake.STABLE:
            on_clear_branch = fold_pair is None or xe <= fold_pair[0][1]
            ax.plot(xe, lake.V(xe, a, b), "o", ms=7, zorder=4,
                    color=COLORS["clear" if on_clear_branch else "turbid"])
        else:
            ax.plot(xe, lake.V(xe, a, b), "o", ms=7, mfc="white",
                    mec=COLORS["threshold"], mew=2, zorder=4)

    if ball_x is not None:
        # Shift the ball up by its radius so it rests on the curve.
        on_curve = mtransforms.offset_copy(ax.transData, fig=ax.figure, y=6, units="points")
        ax.plot(ball_x, lake.V(ball_x, a, b), "o", ms=12, color=COLORS["ink"],
                mec="white", mew=1.5, transform=on_curve, zorder=5)
    ax.set_xlim(x[0], x[-1])
    return m


def double_arrow(ax, start, end, color):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.6, shrinkA=0, shrinkB=0))


def label(ax, a, b, text, **kwargs):
    """Text on a white backing, legible over lines and hatching."""
    ax.text(a, b, text, fontsize=9, color=COLORS["ink"],
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5), **kwargs)


def pointer(ax, text, target, text_pos, **kwargs):
    """Label in empty space with an arrow into the region it names."""
    ax.annotate(text, xy=target, xytext=text_pos, fontsize=9, color=COLORS["ink"],
                va="center", arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1),
                **kwargs)


def fold_branches(q):
    """Fold curve split at the cusp: (a, b) of the upper- and lower-fold branches."""
    x_cusp, _ = lake.max_f_prime(q)
    x = np.linspace(0.0, 6.0, 6000)[1:]
    a, b = lake.fold_curve(x, q)
    return (a[x <= x_cusp], b[x <= x_cusp]), (a[x >= x_cusp], b[x >= x_cusp])


def draw_fold_curve(ax, q):
    """Draw the fold curve; return the bistable region as a Polygon (not added)."""
    (a_up, b_up), (a_low, b_low) = fold_branches(q)
    ax.plot(a_up, b_up, color=COLORS["ink"], lw=2)
    ax.plot(a_low, b_low, color=COLORS["ink"], lw=2)
    xy = np.column_stack([np.concatenate([a_up, a_low]), np.concatenate([b_up, b_low])])
    return Polygon(xy, closed=True, transform=ax.transData)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_landscape():
    """Fig. 1 (Walker et al. Fig. 1a, made quantitative).

    Stability landscape V(x) at a = 0.04, b = 0.52, q = 2. Blue shading: clear
    basin [0, x_u); amber: turbid basin. Filled circles: stable equilibria;
    open circle: the threshold x_u. The ball is the current state x = 0.3.
    Latitude L = x_u is the width of the clear basin, resistance
    R = V(x_u) - V(x_s) its depth, precariousness Pr = x_u - x the distance
    from the current state to the threshold.
    """
    a, b, xc = A0, B0, X_CURRENT
    x = np.linspace(0.0, X_MAX, 800)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    m = draw_landscape(ax, a, b, x, ball_x=xc)

    Vx = lake.V(x, a, b)
    pad = 0.12 * (Vx.max() - Vx.min())
    V_s, V_u = lake.V(m["x_s"], a, b), lake.V(m["x_u"], a, b)

    y_L = Vx.min() - pad
    double_arrow(ax, (0.0, y_L), (m["x_u"], y_L), COLORS["L"])
    ax.text(m["x_u"] / 2, y_L, f"latitude L = {m['L']:.3f}\n", ha="center", va="center")

    ax.plot([m["x_s"], m["x_u"]], [V_s, V_s], ls=":", color=COLORS["R"], lw=1.2)
    double_arrow(ax, (m["x_u"], V_s), (m["x_u"], V_u), COLORS["R"])
    ax.text(m["x_u"] + 0.03, (V_s + V_u) / 2, f"resistance\nR = {m['R']:.4f}", va="center")

    y_Pr = V_u + pad
    double_arrow(ax, (xc, y_Pr), (m["x_u"], y_Pr), COLORS["Pr"])
    ax.plot([xc, xc], [lake.V(xc, a, b), y_Pr], ls=":", color=COLORS["Pr"], lw=1.2)
    ax.plot([m["x_u"], m["x_u"]], [V_u, y_Pr], ls=":", color=COLORS["Pr"], lw=1.2)
    ax.text((xc + m["x_u"]) / 2, y_Pr, f"precariousness Pr = {m['Pr']:.3f}\n",
            ha="center", va="center")

    ax.text(m["x_u"], V_u, "  threshold $x_u$", va="bottom")
    ax.text(0.02, 0.97, "clear basin", transform=ax.transAxes, va="top",
            color=COLORS["clear"], fontweight="bold")
    ax.text(0.98, 0.97, "turbid basin", transform=ax.transAxes, va="top", ha="right",
            fontweight="bold")
    ax.set_ylim(y_L - pad, Vx.max() + pad)
    ax.set_xlabel("phosphorus in lake water, $x$ (dimensionless)")
    ax.set_ylabel("potential $V(x)$")
    ax.set_title(f"Stability landscape  (a = {a}, b = {b}, q = {Q:g})")
    save(fig, "fig1_landscape")


def fig_landscape_sequence():
    """Fig. 2 (Walker et al. Fig. 1b, made quantitative).

    Landscapes for increasing input a at b = 0.52, q = 2, on a shared vertical
    axis. The ball sits at the clear equilibrium while it exists. The clear
    basin narrows and flattens; at the upper fold the clear state and the
    threshold merge (R = 0), and beyond it the lake is turbid.
    """
    (a_upper, _), _ = lake.folds(B0)
    a_values = sorted(SEQUENCE_A + [a_upper])
    x = np.linspace(0.0, X_MAX, 800)
    fig, axes = plt.subplots(1, len(a_values), figsize=(2.15 * len(a_values), 3.4), sharey=True)
    for ax, a in zip(axes, a_values):
        eqs = lake.equilibria(a, B0)
        ball = eqs[0][0] if a <= a_upper else eqs[-1][0]
        m = draw_landscape(ax, a, B0, x, ball_x=ball)
        ax.set_title(f"a = {a:.4f}" + ("\n(upper fold)" if a == a_upper else ""), fontsize=11)
        text = "no clear basin" if math.isnan(m["R"]) else f"R = {m['R']:.4f}\nL = {m['L']:.2f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=9)
    axes[0].set_ylabel("potential $V(x)$")
    fig.supxlabel("phosphorus in lake water, $x$ (dimensionless)", fontsize=11)
    fig.suptitle(f"Clear basin shrinking as phosphorus input rises  (b = {B0}, q = {Q:g})",
                 fontweight="bold", color=plt.rcParams["axes.titlecolor"])
    fig.tight_layout()
    save(fig, "fig2_landscape_sequence")


def fig_bifurcation():
    """Fig. 3: bifurcation diagram with a simulated slow sweep.

    Equilibria x* against a for b = 0.52, q = 2: stable (solid; blue clear,
    amber turbid) and the unstable threshold (dashed). Open circles: folds.
    Grey: the lake when a is ramped from 0 to 0.1 and back at da/dt = 1e-5.
    It stays clear until just past the upper fold and recovers only below the
    lower fold: hysteresis. The jumps lag the folds slightly because the ramp
    has a finite speed.
    """
    (a_up, x_up), (a_low, x_low) = lake.folds(B0)
    # F(x) = 0 solved for a gives a = b*x - f(x): the equilibrium curve, exactly.
    x = np.linspace(0.0, X_MAX, 6000)
    a_eq = B0 * x - lake.f(x)
    shown = (a_eq >= SWEEP["a_min"]) & (a_eq <= SWEEP["a_max"])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    halves = lake.sweep(B0, **SWEEP)
    for name, (a_path, x_path) in halves.items():
        ax.plot(a_path, x_path, color=COLORS["neutral"], lw=1.2, zorder=1,
                label="slow sweep of a (simulated)" if name == "up" else None)
    clear, turbid = shown & (x <= x_up), shown & (x >= x_low)
    middle = shown & (x >= x_up) & (x <= x_low)
    ax.plot(a_eq[clear], x[clear], color=COLORS["clear"], lw=2.5, label="stable equilibrium")
    ax.plot(a_eq[turbid], x[turbid], color=COLORS["turbid"], lw=2.5)
    ax.plot(a_eq[middle], x[middle], color=COLORS["threshold"], lw=2, ls="--",
            label="unstable equilibrium (threshold)")

    for a_f, x_f, name, dy in ((a_up, x_up, "upper fold", -8), (a_low, x_low, "lower fold", 8)):
        ax.plot(a_f, x_f, "o", ms=7, mfc="white", mec=COLORS["ink"], mew=1.5, zorder=4)
        ax.annotate(f"{name}\na = {a_f:.4f}", (a_f, x_f), xytext=(8, dy),
                    textcoords="offset points", va="top" if dy < 0 else "bottom", fontsize=9)

    # Direction arrows on the two jumps.
    x_mid = 0.5 * (x_up + x_low)
    for (a_path, x_path), up in ((halves["up"], True), (halves["down"], False)):
        i = np.nonzero(x_path > x_mid if up else x_path < x_mid)[0][0]
        sign = 1 if up else -1
        ax.annotate("", xy=(a_path[i], x_mid + 0.1 * sign), xytext=(a_path[i], x_mid - 0.1 * sign),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["neutral"], lw=1.2))

    ax.text(0.003, 0.15, "clear", color=COLORS["clear"], fontweight="bold")
    ax.text(0.097, 1.6, "turbid", ha="right", fontweight="bold")
    ax.set_xlim(SWEEP["a_min"], SWEEP["a_max"])
    ax.set_ylim(0.0, X_MAX)
    ax.set_xlabel("phosphorus input rate, $a$")
    ax.set_ylabel("equilibrium phosphorus, $x^*$")
    ax.set_title(f"Bifurcation diagram and hysteresis  (b = {B0}, q = {Q:g})")
    ax.legend(loc="upper left")  # empty corner above the turbid branch
    save(fig, "fig3_bifurcation")


def fig_measures():
    """Fig. 4: resilience measures of the clear basin against a.

    b = 0.52, q = 2, with the lake at rest at x_s (so Pr = x_u - x_s).
    Dashed line: upper fold. Grey band: below the lower fold there is no
    threshold, so the measures are undefined. Amber band: clear basin gone.
    R and Pr go to zero at the fold but L does not, because L is measured
    from x = 0.
    """
    (a_up, _), (a_low, _) = lake.folds(B0)
    a_max = SWEEP["a_max"]
    a_grid = np.union1d(np.linspace(0.0, a_max, 600), [a_up])
    ms = [lake.clear_basin(a, B0) for a in a_grid]

    panels = [("latitude  $L = x_u$", "L", COLORS["L"]),
              ("resistance  $R = V(x_u) - V(x_s)$", "R", COLORS["R"]),
              ("ratio  $R / L$", "R_over_L", COLORS["R"]),
              ("precariousness  $Pr = x_u - x_s$", "Pr", COLORS["Pr"])]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.6), sharex=True)
    for ax, (title, key, color) in zip(axes.flat, panels):
        ax.axvspan(0.0, a_low, color=COLORS["neutral"], alpha=TINT, lw=0)
        ax.axvspan(a_up, a_max, color=COLORS["turbid"], alpha=TINT, lw=0)
        ax.axvline(a_up, color=COLORS["ink"], ls="--", lw=1)
        ax.plot(a_grid, [m[key] for m in ms], color=color, lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0.0, a_max)
        ax.set_ylim(bottom=0.0)
    for ax in axes[1]:
        ax.set_xlabel("phosphorus input rate, $a$")
    top_left, top_right = axes[0]
    top_left.text(a_low / 2, 0.5, "no\nthreshold", transform=top_left.get_xaxis_transform(),
                  ha="center", va="center", fontsize=9)
    top_left.text(a_up, 0.95, " upper fold", transform=top_left.get_xaxis_transform(),
                  va="top", fontsize=9)
    top_right.text((a_up + a_max) / 2, 0.5, "clear basin\nvanished",
                   transform=top_right.get_xaxis_transform(), ha="center", va="center", fontsize=9)
    fig.suptitle(f"Resilience of the clear state as input rises  (b = {B0}, q = {Q:g})",
                 fontweight="bold", color=plt.rcParams["axes.titlecolor"])
    fig.tight_layout()
    save(fig, "fig4_measures")


def fig_regimes():
    """Fig. 5: regimes in the (a, b) plane for q = 2.

    Fold curve b = f'(x), a = x f'(x) - f(x); hatched: bistable region.
    Bands give the regime as a property of b: reversible for b >= 0.6495,
    irreversible for b < 0.5, hysteretic in between. Marked: the baseline and
    b = 0.62, which Carpenter et al. (1999, Fig. 11) use as a 'reversible'
    example but which has a bistable window of width 0.0057 in a.
    """
    a_max, b_max = 0.25, 0.8
    _, b_rev = lake.max_f_prime()
    b_irr, _ = lake.irreversibility_boundary()

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for lo, hi, color in ((0.0, b_irr, "turbid"), (b_irr, b_rev, "neutral"), (b_rev, b_max, "clear")):
        ax.axhspan(lo, hi, color=COLORS[color], alpha=TINT, lw=0)
    for b_line in (b_irr, b_rev):
        ax.axhline(b_line, color=COLORS["ink"], lw=0.8, ls=":")
    cusp = draw_fold_curve(ax, Q)
    cusp.set(facecolor="none", hatch="///", edgecolor=COLORS["neutral"], lw=0)
    ax.add_patch(cusp)

    for name, b_txt in (("reversible", 0.72), ("hysteretic", 0.535), ("irreversible", 0.25)):
        ax.text(0.17, b_txt, name, fontsize=10, fontweight="bold")
    label(ax, a_max, b_irr, f"b = {b_irr:.3f} ", ha="right", va="bottom")
    label(ax, a_max, b_rev, f"b = max f' = {b_rev:.4f} ", ha="right", va="bottom")
    ax.axhline(0.62, color=COLORS["Pr"], lw=1.2, ls="--")
    label(ax, a_max, 0.62, "b = 0.62: Carpenter et al.'s 'reversible' example ",
          ha="right", va="top")
    ax.plot(A0, B0, "o", ms=8, color=COLORS["ink"], zorder=5)
    label(ax, A0 + 0.005, B0, "baseline", ha="left", va="center")
    pointer(ax, "bistable (inside fold curve)", (0.012, 0.3), (0.06, 0.3))

    ax.set_xlim(0.0, a_max)
    ax.set_ylim(0.0, b_max)
    ax.set_xlabel("input parameter $a$")
    ax.set_ylabel("loss parameter $b$")
    ax.set_title(f"Regimes of the lake model  (q = {Q:g})")
    save(fig, "fig5_regimes")


def fig_carpenter_comparison():
    """Fig. 6: our curves on the axes of Carpenter et al. (1999), Fig. A1.

    (A) Fold curves for q = 2 and q = 8. (B) Regimes for q = 8, labelled as in
    their Fig. A1B: outside the fold curve reversible; inside it, hysteretic
    above and irreversible below the b where the lower fold reaches a = 0.
    This labels regions of the (a, b) plane, unlike Fig. 5. Their dotted
    'minimum input' lines are not reproduced.
    """
    a_max, b_max = 1.6, 2.0
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(5.0, 8.0), sharex=True)
    for q, color in ((2.0, COLORS["clear"]), (8.0, COLORS["ink"])):
        (a_up, b_up), (a_low, b_low) = fold_branches(q)
        ax_a.plot(a_up, b_up, color=color, lw=2, label=f"q = {q:g}")
        ax_a.plot(a_low, b_low, color=color, lw=2)
    ax_a.set_ylabel("loss parameter $b$")
    ax_a.legend(loc="upper left")
    ax_a.set_title("(A) fold curves", loc="left")

    b_irr, _ = lake.irreversibility_boundary(8.0)
    cusp = draw_fold_curve(ax_b, 8.0)
    for lo, hi, color in ((0.0, b_irr, "turbid"), (b_irr, b_max, "neutral")):
        band = ax_b.axhspan(lo, hi, color=COLORS[color], alpha=3 * TINT, lw=0)
        band.set_clip_path(cusp)  # shade only inside the fold curve
    ax_b.axhline(b_irr, color=COLORS["ink"], lw=1.5).set_clip_path(cusp)
    ax_b.text(0.1, 1.75, "reversible", fontsize=10, fontweight="bold")
    pointer(ax_b, "hysteretic", (0.6, 1.05), (0.95, 0.95), fontweight="bold")
    pointer(ax_b, "irreversible", (0.15, 0.45), (0.6, 0.3), fontweight="bold")
    label(ax_b, a_max, b_irr, f"b = {b_irr:.3f} ", ha="right", va="bottom")
    ax_b.set_xlabel("input parameter $a$")
    ax_b.set_ylabel("loss parameter $b$")
    ax_b.set_title("(B) regimes for q = 8", loc="left")
    for ax in (ax_a, ax_b):
        ax.set_xlim(0.0, a_max)
        ax.set_ylim(0.0, b_max)
    fig.suptitle("Comparison with Carpenter et al. (1999), Fig. A1",
                 fontweight="bold", color=plt.rcParams["axes.titlecolor"])
    fig.tight_layout()
    save(fig, "fig6_carpenter_A1_comparison")


def write_checks():
    """Write the quantitative checks to results/checks.json."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "checks.json", "w") as fh:
        json.dump(lake.quantitative_checks(), fh, indent=2)
    print("wrote results/checks.json")


if __name__ == "__main__":
    plt.style.use(HERE / "polder.mplstyle")
    write_checks()
    fig_landscape()
    fig_landscape_sequence()
    fig_bifurcation()
    fig_measures()
    fig_regimes()
    fig_carpenter_comparison()
