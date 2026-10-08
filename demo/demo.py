"""Live demo: the lake's stability landscape with sliders and a rolling ball.

Run from the repository root (no Jupyter needed):

    python demo/demo.py

Top panel: the landscape V(x) for the current a and b, with the lake's state
as a ball. Bottom panel: the bifurcation diagram for the current b, with the
current (a, x) and a grey trail of where the lake has been.

- Sliders for a (phosphorus input) and b (loss rate). The state persists:
  after every slider move the ball rolls from where it is to the bottom of
  its current basin, so hysteresis is visible live.
- "Treat lake chemically" moves the state to a low phosphorus value without
  changing a or b: Walker et al.'s "move the state, keep the landscape". If a
  is above the upper fold there is no clear basin and the ball rolls back to
  turbid.
- "Reset" returns a, b, the state, the trail and the noise to the baseline.
- "noise" (off by default) adds Euler-Maruyama noise with the seed from
  config/params.yaml, to show tipping before the fold.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider

# Make the `lake` package in src/ importable without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lake.config import load_params  # noqa: E402
from lake.equilibria import STABLE, classify_regime, equilibria, folds  # noqa: E402
from lake.model import V, f, f_prime  # noqa: E402
from lake.resilience import clear_basin_measures  # noqa: E402
from lake.simulate import euler_maruyama, relax  # noqa: E402
from lake.style import COLORS, TINT_ALPHA, use_style  # noqa: E402

BALL_SIZE = 14   # points
EQ_SIZE = 8      # points
N_CURVE = 600    # points on the landscape and bifurcation curves
Y_PAD = 0.1      # fraction of the V range added above and below


class LakeDemo:
    """Holds the state (a, b, x) and the figure, and reacts to the widgets."""

    def __init__(self, params, interactive=True):
        self.params = params
        self.q = params["model"]["q"]
        self.dp = params["demo"]
        num = params["numerics"]
        self.root_args = (num["n_grid"], num["xtol"], num["tangency_tol"])
        integ = params["integration"]
        self.relax_args = (integ["rtol"], integ["atol"], integ["settle_tol"],
                           integ["settle_t_max"])
        self.x_grid = np.linspace(0.0, self.dp["x_max"], N_CURVE)

        self.frames = []     # pending ball positions of the current relaxation
        self.history = []    # (a, x) trail for the bifurcation panel
        self.noise_on = False
        self._resetting = False

        self._build_figure()
        self.reset()
        if interactive:
            self.timer = self.fig.canvas.new_timer(interval=self.dp["frame_interval_ms"])
            self.timer.add_callback(self.tick)
            self.timer.start()

    # ------------------------------------------------------------------ layout

    def _build_figure(self):
        self.fig = plt.figure(figsize=(10, 8.5))
        if self.fig.canvas.manager is not None:  # None when drawing off-screen
            self.fig.canvas.manager.set_window_title("Lake stability landscape")
        self.ax_land = self.fig.add_axes([0.08, 0.60, 0.60, 0.34])
        self.ax_bif = self.fig.add_axes([0.08, 0.19, 0.60, 0.31])
        ax_info = self.fig.add_axes([0.72, 0.60, 0.26, 0.34])
        ax_info.axis("off")
        self.info = ax_info.text(0.0, 1.0, "", va="top", ha="left", fontsize=10,
                                 family="monospace", transform=ax_info.transAxes)

        # Landscape panel.
        ax = self.ax_land
        (self.curve,) = ax.plot([], [], color=COLORS["ink"], lw=2)
        (self.eq_clear,) = ax.plot([], [], "o", ms=EQ_SIZE, color=COLORS["clear"],
                                   ls="none", zorder=4)
        (self.eq_turbid,) = ax.plot([], [], "o", ms=EQ_SIZE, color=COLORS["turbid"],
                                    ls="none", zorder=4)
        (self.eq_unstable,) = ax.plot([], [], "o", ms=EQ_SIZE, mfc="white",
                                      mec=COLORS["threshold"], mew=2, ls="none", zorder=4)
        on_curve = mtransforms.offset_copy(ax.transData, fig=self.fig, y=BALL_SIZE / 2,
                                           units="points")
        (self.ball,) = ax.plot([], [], "o", ms=BALL_SIZE, color=COLORS["ink"], mec="white",
                               mew=1.5, transform=on_curve, zorder=5)
        self.basin_spans = []
        ax.set_xlim(0.0, self.dp["x_max"])
        ax.set_xlabel("phosphorus in lake water, $x$")
        ax.set_ylabel("potential $V(x)$")
        ax.set_title("Stability landscape (Walker et al. 2004, Fig. 1)", fontsize=12)

        # Bifurcation panel.
        ax = self.ax_bif
        (self.br_clear,) = ax.plot([], [], color=COLORS["clear"], lw=2.5)
        (self.br_turbid,) = ax.plot([], [], color=COLORS["turbid"], lw=2.5)
        (self.br_unstable,) = ax.plot([], [], color=COLORS["threshold"], lw=2, ls="--")
        (self.trail,) = ax.plot([], [], color=COLORS["neutral"], lw=1.2, zorder=1)
        self.a_line = ax.axvline(0.0, color=COLORS["ink"], lw=0.8, ls=":")
        (self.marker,) = ax.plot([], [], "o", ms=EQ_SIZE + 2, color=COLORS["ink"],
                                 mec="white", mew=1.5, zorder=5)
        ax.set_xlim(*self.dp["a_range"])
        ax.set_ylim(0.0, self.dp["x_max"])
        ax.set_xlabel("phosphorus input rate, $a$")
        ax.set_ylabel("equilibrium $x^*$")
        ax.set_title("Bifurcation diagram, with the lake's path in grey", fontsize=12)

        # Widgets.
        a0, b0 = self.params["model"]["a"], self.params["model"]["b"]
        self.slider_a = Slider(self.fig.add_axes([0.12, 0.08, 0.52, 0.03]), "input a",
                               *self.dp["a_range"], valinit=a0, color=COLORS["clear"])
        self.slider_b = Slider(self.fig.add_axes([0.12, 0.03, 0.52, 0.03]), "loss b",
                               *self.dp["b_range"], valinit=b0, color=COLORS["neutral"])
        self.btn_treat = Button(self.fig.add_axes([0.74, 0.42, 0.22, 0.07]),
                                "Treat lake chemically")
        self.btn_reset = Button(self.fig.add_axes([0.74, 0.32, 0.22, 0.07]), "Reset")
        self.chk_noise = CheckButtons(self.fig.add_axes([0.74, 0.20, 0.22, 0.08]),
                                      ["noise"], [False])
        self.slider_a.on_changed(self.on_slider)
        self.slider_b.on_changed(self.on_slider)
        self.btn_treat.on_clicked(self.treat)
        self.btn_reset.on_clicked(self.reset)
        self.chk_noise.on_clicked(self.toggle_noise)

    # --------------------------------------------------------------- callbacks

    def on_slider(self, _value):
        """a or b changed: redraw the landscape, then let the ball roll."""
        if self._resetting:
            return
        self.a, self.b = self.slider_a.val, self.slider_b.val
        self._redraw_parameters()
        self._start_relaxing()

    def treat(self, _event=None):
        """Move the state to low phosphorus, keep a and b (Walker et al.)."""
        self.x = self.dp["x_treat"]
        self._record()
        self._start_relaxing()
        self._redraw_state()

    def reset(self, _event=None):
        """Everything back to the baseline in config/params.yaml."""
        self._resetting = True
        self.slider_a.set_val(self.params["model"]["a"])
        self.slider_b.set_val(self.params["model"]["b"])
        if self.chk_noise.get_status()[0]:
            self.chk_noise.set_active(0)  # triggers toggle_noise -> off
        self._resetting = False

        self.a, self.b = self.slider_a.val, self.slider_b.val
        self.x = self.params["baseline"]["x_current"]
        self.rng = np.random.default_rng(self.params["random"]["seed"])
        self.history = []
        self._redraw_parameters()
        self._record()
        self._start_relaxing()

    def toggle_noise(self, _label=None):
        self.noise_on = self.chk_noise.get_status()[0]
        self.frames = []
        if not self.noise_on and not self._resetting:
            self._start_relaxing()

    def tick(self):
        """One animation frame: advance the ball, then redraw it."""
        if self.noise_on:
            dp = self.dp
            path = euler_maruyama(self.x, self.a, self.b, self.q, dp["noise_sigma"],
                                  dp["noise_dt"], dp["noise_steps_per_frame"], self.rng)
            self.x = float(path[-1])
        elif self.frames:
            self.x = self.frames.pop(0)
        else:
            return
        self._record()
        self._redraw_state()

    # ----------------------------------------------------------------- helpers

    def _start_relaxing(self):
        """Precompute the ball's path to rest as evenly timed frames.

        The trajectory comes from solve_ivp (lake.simulate.relax). Long
        relaxations (near a fold the ball moves very slowly) are shown with
        at most max_frames frames, so they play faster than real time.
        """
        if self.noise_on:
            return
        t, x, _ = relax(self.x, self.a, self.b, self.q, *self.relax_args)
        n = int(np.clip(np.ceil(t[-1] / self.dp["dt_per_frame"]), 1, self.dp["max_frames"]))
        self.frames = list(np.interp(np.linspace(t[0], t[-1], n + 1)[1:], t, x))

    def _record(self):
        self.history.append((self.a, self.x))
        del self.history[:-self.dp["history_max"]]

    def _redraw_parameters(self):
        """Landscape, equilibria and bifurcation branches for the current a, b."""
        a, b, q = self.a, self.b, self.q
        eqs = equilibria(a, b, q, *self.root_args)
        self.fold_pair = folds(b, q, *self.root_args[:2])
        self.regime = classify_regime(b, q, *self.root_args[:2])

        # Landscape and equilibria.
        Vx = V(self.x_grid, a, b, q)
        self.curve.set_data(self.x_grid, Vx)
        stable = [e.x for e in eqs if e.stability == STABLE]
        unstable = [e.x for e in eqs if e.stability != STABLE]
        # Colour stable equilibria by branch, as in the figures: clear if on
        # the clear branch (x up to the upper fold), else turbid. Without folds
        # (reversible) there is one state; it is drawn as clear.
        x_split = self.fold_pair[0].x if self.fold_pair else np.inf
        clear_eq = np.array([s for s in stable if s <= x_split])
        turbid_eq = np.array([s for s in stable if s > x_split])
        self.eq_clear.set_data(clear_eq, V(clear_eq, a, b, q))
        self.eq_turbid.set_data(turbid_eq, V(turbid_eq, a, b, q))
        self.eq_unstable.set_data(unstable, V(np.array(unstable), a, b, q))
        x_top = min(self.dp["x_max"], 1.25 * max(e.x for e in eqs))
        shown = Vx[self.x_grid <= x_top]
        lo, hi = shown.min(), shown.max()
        self.ax_land.set_ylim(lo - Y_PAD * (hi - lo), hi + Y_PAD * (hi - lo))

        # Basin shading (clear left of the threshold, turbid right of it).
        for span in self.basin_spans:
            span.remove()
        self.basin_spans = []
        if unstable:
            x_u = unstable[0]
            self.basin_spans = [
                self.ax_land.axvspan(0.0, x_u, color=COLORS["clear"], alpha=TINT_ALPHA, lw=0),
                self.ax_land.axvspan(x_u, self.dp["x_max"], color=COLORS["turbid"],
                                     alpha=TINT_ALPHA, lw=0),
            ]

        # Bifurcation branches: F(x) = 0 solved for a gives a = b*x - f(x).
        xs = self.x_grid
        a_eq = b * xs - f(xs, q)
        is_stable = f_prime(xs, q) < b
        if self.fold_pair is None:
            clear_mask = is_stable          # one branch only (reversible)
            turbid_mask = np.zeros_like(xs, dtype=bool)
        else:
            upper, _ = self.fold_pair
            clear_mask = is_stable & (xs <= upper.x)
            turbid_mask = is_stable & (xs > upper.x)
        self.br_clear.set_data(a_eq[clear_mask], xs[clear_mask])
        self.br_turbid.set_data(a_eq[turbid_mask], xs[turbid_mask])
        self.br_unstable.set_data(np.where(is_stable, np.nan, a_eq), xs)
        self.a_line.set_xdata([a, a])
        self._redraw_state()

    def _state_label(self, m):
        """'clear' or 'turbid' for the current state, given the measures m."""
        if not np.isnan(m.x_u):          # bistable: which side of the threshold
            return "clear" if self.x < m.x_u else "turbid"
        if self.fold_pair is None:       # reversible: no clear/turbid distinction
            return "single state"
        _, lower = self.fold_pair        # monostable: below the lower fold only
        return "clear" if self.a < lower.a else "turbid"  # clear exists, else turbid

    def _redraw_state(self):
        """Ball, marker, trail and the info text for the current x."""
        a, b, q, x = self.a, self.b, self.q, self.x
        self.ball.set_data([x], [V(x, a, b, q)])
        self.marker.set_data([a], [x])
        if self.history:
            ha, hx = zip(*self.history)
            self.trail.set_data(ha, hx)

        m = clear_basin_measures(x, a, b, q, *self.root_args)
        lines = [
            f"a      = {a:.4f}",
            f"b      = {b:.4f}",
            f"regime : {self.regime}",
            f"x      = {x:.3f}  ({self._state_label(m)})",
            "",
            "clear basin",
            f"  L    = {m.L:.3f}",
            f"  R    = {m.R:.5f}",
            f"  R/L  = {m.R_over_L:.4f}",
            f"  Pr   = {m.Pr:.3f}" + ("  (past threshold)" if m.Pr < 0 else ""),
        ]
        if self.fold_pair is not None:
            upper, lower = self.fold_pair
            lines += ["", f"upper fold a = {upper.a:.4f}", f"lower fold a = {lower.a:.4f}"]
        if self.noise_on:
            lines += ["", f"noise on (sigma = {self.dp['noise_sigma']},",
                      f"  seed = {self.params['random']['seed']})"]
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


def main():
    params = load_params()
    use_style(params["figures"]["style"])
    demo = LakeDemo(params)  # noqa: F841  (keep a reference so widgets stay alive)
    plt.show()


if __name__ == "__main__":
    main()
