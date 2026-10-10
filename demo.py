"""Live demo: the lake's stability landscape with sliders and a rolling ball.

    python demo.py

Top: the landscape V(x) with the lake's state as a ball. Bottom: the
bifurcation diagram for the current b, with the lake's path in grey.

- Sliders for a and b. The state persists: after every move the ball rolls
  from where it is to the bottom of its basin, so hysteresis shows live.
- "Treat lake chemically" sets x to a low value and keeps a and b (Walker et
  al.'s "move the state, keep the landscape"). Above the upper fold there is
  no clear basin, so the ball rolls back to turbid.
- "Reset" returns everything to the baseline.
- "noise" (off by default) adds seeded Euler-Maruyama noise, to show tipping
  before the fold (try a = 0.065).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider

import lake
from lake import A0, B0, COLORS, SEED, X_CURRENT

A_RANGE, B_RANGE = (0.0, 0.12), (0.45, 0.8)  # slider ranges; b covers all regimes
X_MAX = 2.6             # right edge of the landscape, above every equilibrium in range
X_TREAT = 0.05          # state after "Treat lake chemically"
FRAME_MS = 30           # animation timer interval
DT_PER_FRAME = 1.0      # model time per frame
MAX_FRAMES = 90         # longer relaxations play faster
NOISE_SIGMA, NOISE_DT, NOISE_STEPS = 0.06, 0.05, 10   # steps per frame


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
        self.fig = plt.figure(figsize=(10, 8.5))
        self.ax_land = self.fig.add_axes([0.08, 0.60, 0.60, 0.34])
        self.ax_bif = self.fig.add_axes([0.08, 0.19, 0.60, 0.31])
        ax_info = self.fig.add_axes([0.72, 0.60, 0.26, 0.34])
        ax_info.axis("off")
        self.info = ax_info.text(0.0, 1.0, "", va="top", family="monospace", fontsize=10)

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

        self.slider_a = Slider(self.fig.add_axes([0.12, 0.08, 0.52, 0.03]), "input a",
                               *A_RANGE, valinit=A0, color=COLORS["clear"])
        self.slider_b = Slider(self.fig.add_axes([0.12, 0.03, 0.52, 0.03]), "loss b",
                               *B_RANGE, valinit=B0, color=COLORS["neutral"])
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

    # Callbacks -------------------------------------------------------------

    def on_slider(self, _value):
        self.a, self.b = self.slider_a.val, self.slider_b.val
        self.redraw_landscape()
        self.start_rolling()

    def treat(self, _event=None):
        self.x = X_TREAT
        self.history.append((self.a, self.x))
        self.start_rolling()
        self.redraw_state()

    def reset(self, _event=None):
        self.slider_a.eventson = self.slider_b.eventson = False  # one redraw, not two
        self.slider_a.set_val(A0)
        self.slider_b.set_val(B0)
        self.slider_a.eventson = self.slider_b.eventson = True
        if self.chk_noise.get_status()[0]:
            self.chk_noise.set_active(0)
        self.a, self.b, self.x = A0, B0, X_CURRENT
        self.rng = np.random.default_rng(SEED)
        self.history = [(self.a, self.x)]
        self.redraw_landscape()
        self.start_rolling()

    def toggle_noise(self, _label):
        self.noise_on = self.chk_noise.get_status()[0]
        self.start_rolling()

    def tick(self):
        """One animation frame."""
        if self.noise_on:
            path = lake.euler_maruyama(self.x, self.a, self.b, NOISE_SIGMA, NOISE_DT,
                                       NOISE_STEPS, self.rng)
            self.x = float(path[-1])
        elif self.frames:
            self.x = self.frames.pop(0)
        else:
            return
        self.history.append((self.a, self.x))
        self.redraw_state()

    # Drawing ---------------------------------------------------------------

    def start_rolling(self):
        """Precompute the ball's path to rest (lake.relax) as animation frames."""
        self.frames = []
        if self.noise_on:
            return
        t, x = lake.relax(self.x, self.a, self.b)
        n = int(np.clip(np.ceil(t[-1] / DT_PER_FRAME), 1, MAX_FRAMES))
        self.frames = list(np.interp(np.linspace(t[0], t[-1], n + 1)[1:], t, x))

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
        shown = Vx[self.x_grid <= min(X_MAX, 1.25 * eqs[-1][0])]
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
        """Ball, marker, trail and info text for the current x."""
        a, b, x = self.a, self.b, self.x
        self.ball.set_data([x], [lake.V(x, a, b)])
        self.marker.set_data([a], [x])
        self.trail.set_data(*zip(*self.history[-3000:]))

        m = lake.clear_basin(a, b, x)
        if not np.isnan(m["x_u"]):
            state = "clear" if x < m["x_u"] else "turbid"
        elif self.fold_pair is None:
            state = "single state"
        else:
            state = "clear" if a < self.fold_pair[1][0] else "turbid"
        lines = [f"a      = {a:.4f}", f"b      = {b:.4f}", f"regime : {self.regime}",
                 f"x      = {x:.3f}  ({state})", "", "clear basin",
                 f"  L    = {m['L']:.3f}", f"  R    = {m['R']:.5f}",
                 f"  R/L  = {m['R_over_L']:.4f}",
                 f"  Pr   = {m['Pr']:.3f}" + ("  (past threshold)" if m["Pr"] < 0 else "")]
        if self.fold_pair:
            (a_up, _), (a_low, _) = self.fold_pair
            lines += ["", f"upper fold a = {a_up:.4f}", f"lower fold a = {a_low:.4f}"]
        if self.noise_on:
            lines += ["", f"noise on (sigma = {NOISE_SIGMA}, seed = {SEED})"]
        self.info.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    plt.style.use(Path(__file__).parent / "polder.mplstyle")
    demo = LakeDemo()  # keep a reference so the widgets stay alive
    plt.show()
