"""Live demo: the lake's stability landscape with sliders and a rolling ball.

    python demo.py

Left: the landscape V(x) with the lake's state as a ball (top) and the
bifurcation diagram for the current b, with the lake's path in grey (bottom).
Right: phosphorus x over model time, and the two phosphorus inputs over time:
external loading a and internal loading f(x) (recycling from the sediment).

- Sliders for a and b. The state persists: after every move the ball rolls
  from where it is to the bottom of its basin, so hysteresis shows live.
- "Treat lake chemically" sets x to a low value and keeps a and b (Walker et
  al.'s "move the state, keep the landscape"). Above the upper fold there is
  no clear basin, so the ball rolls back to turbid.
- "Reset" returns everything to the baseline.
- "noise" (off by default) adds seeded Euler-Maruyama noise, to show tipping
  before the fold (try a = 0.065).
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider

import lake
from lake import A0, B0, COLORS, SEED, X_CURRENT

A_RANGE, B_RANGE = (0.0, 0.12), (0.45, 0.8)  # slider ranges; b covers all regimes
X_MAX = 2.6             # right edge of the landscape, above every equilibrium in range
X_VIEW_MIN = 1.5        # smallest x range used to fit the landscape's y-axis
X_TREAT = 0.05          # state after "Treat lake chemically"
FRAME_MS = 30           # animation timer interval
DT_PER_FRAME = 1.0      # model time per frame
MAX_FRAMES = 90         # longer relaxations play faster
NOISE_SIGMA, NOISE_DT, NOISE_STEPS = 0.06, 0.05, 10   # steps per frame
HISTORY_MAX = 5000      # points kept for the trail and the time series


class LakeDemo:
    def __init__(self):
        self.x_grid = np.linspace(0.0, X_MAX, 600)
        self.frames, self.history, self.noise_on = [], [], False
        self._build_figure()
        self.reset()
        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self.tick)
        self.timer.start()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 9))
        self.ax_land = self.fig.add_axes([0.07, 0.64, 0.52, 0.31])
        self.ax_bif = self.fig.add_axes([0.07, 0.24, 0.52, 0.30])
        self.ax_P = self.fig.add_axes([0.68, 0.40, 0.29, 0.14])
        self.ax_load = self.fig.add_axes([0.68, 0.17, 0.29, 0.14], sharex=self.ax_P)
        self.fig.text(0.825, 0.90, r"$\frac{dx}{dt} = a - b\,x + \frac{x^q}{1 + x^q}$",
                      ha="center", va="center", fontsize=26)
        ax_info = self.fig.add_axes([0.68, 0.58, 0.29, 0.23])
        ax_info.axis("off")
        self.info = ax_info.text(0.0, 1.0, "", va="top", family="monospace", fontsize=9.5)

        ax = self.ax_land
        (self.curve,) = ax.plot([], [], color=COLORS["ink"], lw=2)
        (self.eq_clear,) = ax.plot([], [], "o", ms=8, color=COLORS["clear"], zorder=4)
        (self.eq_turbid,) = ax.plot([], [], "o", ms=8, color=COLORS["turbid"], zorder=4)
        (self.eq_unstable,) = ax.plot([], [], "o", ms=8, mfc="white",
                                      mec=COLORS["threshold"], mew=2, zorder=4)
        on_curve = mtransforms.offset_copy(ax.transData, fig=self.fig, y=7, units="points")
        (self.ball,) = ax.plot([], [], "o", ms=14, color=COLORS["ink"], mec="white",
                               mew=1.5, transform=on_curve, zorder=5)
        self.spans = []
        ax.set_xlim(0.0, X_MAX)
        ax.set_xlabel("phosphorus in lake water, $x$")
        ax.set_ylabel("potential $V(x)$")
        ax.set_title("Stability landscape (Walker et al. 2004, Fig. 1)", fontsize=12)

        ax = self.ax_bif
        (self.br_clear,) = ax.plot([], [], color=COLORS["clear"], lw=2.5)
        (self.br_turbid,) = ax.plot([], [], color=COLORS["turbid"], lw=2.5)
        (self.br_unstable,) = ax.plot([], [], color=COLORS["threshold"], lw=2, ls="--")
        (self.trail,) = ax.plot([], [], color=COLORS["neutral"], lw=1.2, zorder=1)
        self.a_line = ax.axvline(A0, color=COLORS["ink"], lw=0.8, ls=":")
        (self.marker,) = ax.plot([], [], "o", ms=10, color=COLORS["ink"], mec="white", zorder=5)
        ax.set_xlim(*A_RANGE)
        ax.set_ylim(0.0, X_MAX)
        ax.set_xlabel("phosphorus input rate, $a$")
        ax.set_ylabel("equilibrium $x^*$")
        ax.set_title("Bifurcation diagram, with the lake's path in grey", fontsize=12)

        # Time series: phosphorus (a stock) and the inputs (rates) on separate axes.
        (self.ts_P,) = self.ax_P.plot([], [], color=COLORS["ink"], lw=1.5)
        self.ax_P.set_title("phosphorus in the water, $x$", fontsize=10)
        self.ax_P.tick_params(labelbottom=False)
        (self.ts_internal,) = self.ax_load.plot([], [], color=COLORS["Pr"], lw=1.5,
                                                label="internal, $f(x)$")
        (self.ts_external,) = self.ax_load.plot([], [], color=COLORS["clear"], lw=1.5,
                                                label="external, $a$")
        self.ax_load.set_title("phosphorus loading (input per unit time)", fontsize=10)
        self.ax_load.set_xlabel("model time")
        self.ax_load.legend(loc="upper left", fontsize=8)
        for ax in (self.ax_P, self.ax_load):
            ax.tick_params(labelsize=8)

        self.slider_a = Slider(self.fig.add_axes([0.11, 0.09, 0.44, 0.03]), "input a",
                               *A_RANGE, valinit=A0, color=COLORS["clear"])
        self.slider_b = Slider(self.fig.add_axes([0.11, 0.04, 0.44, 0.03]), "loss b",
                               *B_RANGE, valinit=B0, color=COLORS["neutral"])
        # Controls: one compact row under the time series.
        self.btn_treat = Button(self.fig.add_axes([0.68, 0.035, 0.135, 0.045]),
                                "Treat lake chemically")
        self.btn_reset = Button(self.fig.add_axes([0.825, 0.035, 0.065, 0.045]), "Reset")
        self.chk_noise = CheckButtons(self.fig.add_axes([0.90, 0.035, 0.07, 0.045]),
                                      ["noise"], [False])
        self.btn_treat.label.set_fontsize(9)
        self.btn_reset.label.set_fontsize(9)
        self.chk_noise.labels[0].set_fontsize(9)
        self.slider_a.on_changed(self.on_slider)
        self.slider_b.on_changed(self.on_slider)
        self.btn_treat.on_clicked(self.treat)
        self.btn_reset.on_clicked(self.reset)
        self.chk_noise.on_clicked(self.toggle_noise)

    # Callbacks -------------------------------------------------------------

    def on_slider(self, _value):
        self.a, self.b = self.slider_a.val, self.slider_b.val
        self.record()
        self.redraw_landscape()
        self.start_rolling()

    def treat(self, _event=None):
        self.x = X_TREAT
        self.record()
        self.start_rolling()
        self.redraw_state()

    def reset(self, _event=None):
        self.slider_a.eventson = self.slider_b.eventson = False  # one redraw, not two
        self.slider_a.set_val(A0)
        self.slider_b.set_val(B0)
        self.slider_a.eventson = self.slider_b.eventson = True
        if self.chk_noise.get_status()[0]:
            self.chk_noise.set_active(0)
        self.a, self.b, self.x, self.t = A0, B0, X_CURRENT, 0.0
        self.rng = np.random.default_rng(SEED)
        self.history = []
        self.record()
        self.redraw_landscape()
        self.start_rolling()

    def toggle_noise(self, _label):
        self.noise_on = self.chk_noise.get_status()[0]
        self.start_rolling()

    def tick(self):
        """One animation frame: advance model time and the state."""
        if self.noise_on:
            path = lake.euler_maruyama(self.x, self.a, self.b, NOISE_SIGMA, NOISE_DT,
                                       NOISE_STEPS, self.rng)
            self.t += NOISE_STEPS * NOISE_DT
            self.x = float(path[-1])
        elif self.frames:
            self.t, self.x = self.frames.pop(0)
        else:
            return  # at rest: model time stands still until something changes
        self.record()
        self.redraw_state()

    # Drawing ---------------------------------------------------------------

    def record(self):
        """Store (t, a, x) for the trail and the time series."""
        self.history.append((self.t, self.a, self.x))
        del self.history[:-HISTORY_MAX]

    def start_rolling(self):
        """Precompute the ball's path to rest (lake.relax) as (t, x) frames."""
        self.frames = []
        if self.noise_on:
            return
        t, x = lake.relax(self.x, self.a, self.b)
        n = int(np.clip(np.ceil(t[-1] / DT_PER_FRAME), 1, MAX_FRAMES))
        t_frames = np.linspace(t[0], t[-1], n + 1)[1:]
        self.frames = list(zip(self.t + t_frames, np.interp(t_frames, t, x)))

    def redraw_landscape(self):
        """Landscape, equilibria and bifurcation branches for the current a, b."""
        a, b = self.a, self.b
        eqs = lake.equilibria(a, b)
        self.fold_pair = lake.folds(b)
        self.regime = lake.regime(b)
        x_split = self.fold_pair[0][1] if self.fold_pair else np.inf  # clear-branch end

        Vx = lake.V(self.x_grid, a, b)
        self.curve.set_data(self.x_grid, Vx)
        stable = np.array([xe for xe, kind in eqs if kind == lake.STABLE])
        unstable = np.array([xe for xe, kind in eqs if kind != lake.STABLE])
        for line, xs in ((self.eq_clear, stable[stable <= x_split]),
                         (self.eq_turbid, stable[stable > x_split]),
                         (self.eq_unstable, unstable)):
            line.set_data(xs, lake.V(xs, a, b))
        # Fit the y-axis to the part of the landscape around the equilibria,
        # but never narrower than X_VIEW_MIN (a lone clear state sits near 0).
        shown = Vx[self.x_grid <= min(X_MAX, max(X_VIEW_MIN, 1.25 * eqs[-1][0]))]
        pad = 0.1 * (shown.max() - shown.min())
        self.ax_land.set_ylim(shown.min() - pad, shown.max() + pad)

        for span in self.spans:
            span.remove()
        self.spans = []
        if len(unstable):
            self.spans = [
                self.ax_land.axvspan(0.0, unstable[0], color=COLORS["clear"], alpha=0.12, lw=0),
                self.ax_land.axvspan(unstable[0], X_MAX, color=COLORS["turbid"], alpha=0.12, lw=0),
            ]
        elif self.fold_pair is not None:  # one state only: the whole domain is its basin
            only_turbid = a > self.fold_pair[0][0]
            self.spans = [self.ax_land.axvspan(0.0, X_MAX, alpha=0.12, lw=0,
                                               color=COLORS["turbid" if only_turbid else "clear"])]

        # Bifurcation branches: F(x) = 0 solved for a gives a = b*x - f(x).
        xs = self.x_grid
        a_eq = b * xs - lake.f(xs)
        stable_pt = lake.f_prime(xs) < b
        self.br_clear.set_data(a_eq[stable_pt & (xs <= x_split)], xs[stable_pt & (xs <= x_split)])
        self.br_turbid.set_data(a_eq[stable_pt & (xs > x_split)], xs[stable_pt & (xs > x_split)])
        self.br_unstable.set_data(np.where(stable_pt, np.nan, a_eq), xs)
        self.a_line.set_xdata([a, a])
        self.redraw_state()

    def redraw_state(self):
        """Ball, marker, trail, time series and info text for the current x."""
        a, b, x = self.a, self.b, self.x
        self.ball.set_data([x], [lake.V(x, a, b)])
        self.marker.set_data([a], [x])

        t_hist, a_hist, x_hist = (np.array(v) for v in zip(*self.history))
        self.trail.set_data(a_hist, x_hist)
        internal = lake.f(x_hist)
        self.ts_P.set_data(t_hist, x_hist)
        self.ts_internal.set_data(t_hist, internal)
        self.ts_external.set_data(t_hist, a_hist)
        self.ax_P.set_xlim(0.0, max(t_hist[-1], 1.0))
        self.ax_P.set_ylim(0.0, 1.1 * max(x_hist.max(), 0.1))
        self.ax_load.set_ylim(0.0, 1.1 * max(internal.max(), a_hist.max(), 0.05))

        m = lake.clear_basin(a, b, x)
        if not np.isnan(m["x_u"]):
            state = "clear" if x < m["x_u"] else "turbid"
        elif self.fold_pair is None:
            state = "single state"
        else:
            state = "clear" if a < self.fold_pair[1][0] else "turbid"
        lines = [f"a = {a:.4f}   b = {b:.4f}   q = {lake.Q:g}", f"regime: {self.regime}",
                 f"x = {x:.3f}  ({state})", "", "clear basin"]
        if np.isnan(m["L"]):  # no threshold: say why instead of printing NaN
            lines += textwrap.wrap(m["reason"], 36, initial_indent="  ", subsequent_indent="  ")
        else:
            lines += [f"  L    = {m['L']:.3f}", f"  R    = {m['R']:.5f}",
                      f"  R/L  = {m['R_over_L']:.4f}",
                      f"  Pr   = {m['Pr']:.3f}" + ("  (past threshold)" if m["Pr"] < 0 else "")]
        if self.fold_pair:
            (a_up, _), (a_low, _) = self.fold_pair
            lines += ["", f"folds at a = {a_low:.4f}, {a_up:.4f}"]
        if self.noise_on:
            lines += [f"noise on (sigma = {NOISE_SIGMA}, seed = {SEED})"]
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    plt.style.use(Path(__file__).parent / "polder.mplstyle")
    demo = LakeDemo()  # keep a reference so the widgets stay alive
    plt.show()
