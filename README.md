# A quantitative stability landscape for Walker et al. (2004)

MSc Complex Systems, Assignment 2: Classic-Model Replication and Critical Review.

Walker et al. (2004) describe resilience with a drawn stability landscape (their Fig. 1) and three measures: **latitude** L (basin width), **resistance** R (basin depth) and **precariousness** Pr (distance to the threshold). The paper has no equations. This project makes Fig. 1a and 1b quantitative using the one-dimensional lake model of Carpenter, Ludwig and Brock (1999). It then tests whether the three measures can be computed from a real dynamical model, and where they break down.

## Quick start (RUNME)

Run everything from the repository root.

```bash
# 1. Environment (Python 3.11; versions pinned)
conda env create -f environment.yml
conda activate CSA2

# 2. Quantitative checks (< 1 s)
pytest

# 3. Regenerate every figure and results/checks.json (~5 s)
python make_figures.py

# 4. Live demo (opens a window; no Jupyter needed)
python demo.py
```

## Repository layout

```
lake.py            the model, potential, equilibria, folds, regimes, resilience
                   measures, simulation and quantitative checks; baseline
                   parameters and the random seed are constants at the top
make_figures.py    writes figures/*.pdf and results/checks.json
demo.py            interactive demo (matplotlib widgets)
test_checks.py     the quantitative checks (pytest)
polder.mplstyle    figure style
figures/, results/ generated output
```

## The model

Dimensionless continuous-time lake model (Carpenter et al. 1999, Eq. 1 rescaled):

$$\frac{dx}{dt} = F(x) = a - b x + f(x), \qquad f(x) = \frac{x^q}{1 + x^q}$$

| Symbol | Meaning |
|---|---|
| x | phosphorus in the lake water (x ≥ 0) |
| a | phosphorus input rate (the driver) |
| b | loss rate (sedimentation, outflow) |
| q | steepness of the recycling term (q = 2 throughout, q = 8 for one comparison) |

**Potential.** The potential is defined by dx/dt = −dV/dx and V(0) = 0. For q = 2 it has a closed form, V(x) = −a x + b x²/2 − x + arctan x. For other q it is computed by numerical integration. The tests check that the two forms agree and that −dV/dx = F.

**Equilibria.** The equilibrium condition a − b x + f(x) = 0 is the same as in Carpenter et al.'s discrete-time model (their Eq. A.8 with a = A/C, b = B/C). Equilibria, folds and regimes can therefore be compared directly with their Appendix A. An equilibrium is stable if F′(x\*) = −b + f′(x\*) < 0.

**Folds.** A fold (saddle-node) occurs where the line b x − a is tangent to f(x). This gives the parametric fold curve b = f′(x), a = x f′(x) − f(x).

## Definitions

Walker et al. (2004) define their resilience measures in words only. This section gives the formulas we use for the 1D lake model. The code in [lake.py](lake.py) implements exactly these definitions. They are our choices, not Walker et al.'s; the notes say where another reading is possible.

### States, basins and folds

When the lake is bistable it has three equilibria, x_s < x_u < x_t:

| Symbol | Name | Meaning |
|---|---|---|
| x_s | clear state | the lower stable equilibrium (low phosphorus) |
| x_u | threshold | the unstable equilibrium between the two stable ones; the top of the barrier in V |
| x_t | turbid state | the upper stable equilibrium (high phosphorus) |
| x | current state | where the lake is now (the ball in the figures and the demo) |

- **Clear basin**: the interval [0, x_u). Every state in it rolls to x_s. Its lower edge, x = 0, is the edge of the domain (phosphorus cannot be negative), not a threshold.
- **Turbid basin**: the interval (x_u, ∞). Every state in it rolls to x_t. It has no upper edge.
- **Upper fold, a_upper**: the input at which the clear state and the threshold merge (x_s = x_u). For a > a_upper only the turbid state exists.
- **Lower fold, a_lower**: the input at which the turbid state and the threshold merge (x_t = x_u). For a < a_lower only the clear state exists.
- **Bistable window**: a_lower < a < a_upper, where both states exist. At the baseline b = 0.52, q = 2 this is 0.0196 < a < 0.0735.

### Resilience measures

All measures describe the **clear basin**.

| Measure | Walker et al. (2004), in their words | Our definition |
|---|---|---|
| Latitude L | "the maximum amount the system can be changed before losing its ability to recover; basically the width of the basin of attraction" | L = x_u − 0 = x_u, the width of the clear basin [0, x_u) |
| Resistance R | "the ease or difficulty of changing the system; related to the topology of the basin—deep basins of attraction (R, or more accurately, higher ratios of R:L) indicate that greater forces or perturbations are required" | R = V(x_u) − V(x_s), the depth of the clear basin: the height of the barrier between the clear state and the threshold. We also report **R/L** |
| Precariousness Pr | "the current trajectory of the system, and how close it currently is to a limit or 'threshold'" | Pr = x_u − x, the distance from the current state to the threshold. We also report **Pr/L**, the fraction of the basin width still left |

Notes on these choices:

- **L is measured from x = 0, the edge of the domain.** Walker's first phrase ("the maximum amount the system can be changed before losing its ability to recover") could instead be read as the distance from the clear state to the threshold, x_u − x_s. We follow the second phrase, "the width of the basin". With the lake at rest (x = x_s), that alternative reading equals Pr.
- **R uses the potential V**, with dx/dt = −dV/dx and V(0) = 0. Only differences in V matter, so the choice V(0) = 0 does not affect R.
- **Pr uses position only.** Walker et al. also mention the system's "trajectory"; a single distance does not capture where the state is heading.
- **The turbid basin's latitude is undefined**, because that basin has no upper edge. It is reported as NaN rather than cut off at an arbitrary value.

### Which current state x

Pr depends on the current state, so we fix it per output:

| Output | Current state |
|---|---|
| Fig. 1 (landscape) | x = 0.3 (`X_CURRENT` in `lake.py`): a lake pushed away from its clear state |
| Fig. 4 and the fold check in `checks.json` | the lake at rest, x = x_s, so Pr = x_u − x_s |
| Live demo | wherever the ball is |

### Edge cases

| Situation | What is reported |
|---|---|
| Bistable (a_lower < a < a_upper) | all measures defined |
| Exactly at the upper fold | x_s = x_u, so R = 0. L = x_u stays finite (0.314 at b = 0.52) |
| Exactly at the lower fold | the threshold is the point where the turbid state appears; L and R are defined and positive |
| a > a_upper | no clear basin: every measure is NaN, with reason "clear basin has vanished" |
| a < a_lower | no threshold: every measure is NaN, with reason "only the clear state exists" |
| Reversible regime | one equilibrium and no threshold: every measure is NaN |
| Current state past the threshold (x > x_u) | Pr < 0, reported as is rather than clipped |

### Regimes

Following Carpenter et al. (1999), Appendix A. The regime is a property of the loss rate b (and q), not of the input a:

| Regime | Condition | For q = 2 | What it means |
|---|---|---|---|
| Reversible | b ≥ max f′ | b ≥ 3√3/8 ≈ 0.6495 | one equilibrium for every a ≥ 0; no tipping and no hysteresis |
| Hysteretic | max f′ > b and a_lower ≥ 0 | 0.5 ≤ b < 0.6495 | bistable for some a; a turbid lake recovers if a is lowered below a_lower |
| Irreversible | a_lower < 0 | b < 0.5 | once turbid, the lake stays turbid for every a ≥ 0 |

The boundary case b = 0.5, where a_lower = 0 exactly, is counted as hysteretic. Carpenter et al.'s Fig. A1B instead labels regions of the (a, b) plane; Fig. 6 reproduces their version.

## Results

### Quantitative checks

All values are computed numerically for q = 2 and written to [results/checks.json](results/checks.json). The expected values are exact for the analytic checks and hand-calculated (approximate) for the two folds. All checks pass.

| Check | Expected | Computed | Abs. difference |
|---|---|---|---|
| Max of f′ | 3√3/8 at x = 1/√3 | 0.6495190528 at x = 0.5773502606 | 0 and 8.5e-9 |
| Reversible/irreversible boundary at a = 0 | b = 0.5, x = 1 | b = 0.5, x = 1.0 | 0 |
| Upper fold, b = 0.52 | a ≈ 0.0735, x ≈ 0.314 | a = 0.0735328, x = 0.313684 | 3.3e-5 and 3.2e-4 |
| Lower fold, b = 0.52 | a ≈ 0.0196, x ≈ 0.960 | a = 0.0195997, x = 0.959932 | 3.3e-7 and 6.8e-5 |
| −dV/dx = F(x) | equal | max error < 1e-8 | — |
| R → 0 and x_s → x_u at the upper fold | both → 0 | both exactly 0 at the fold; strictly decreasing on approach | — |

The tolerance is 1e-7 for the analytic checks. It is no tighter because f′ is flat at its maximum, so the location of the maximum is only determined to about 1e-8. The tolerance for the hand-calculated fold values is 1e-3.

### Additional computed values (no expected value)

- **Bistable window at b = 0.62.** The lake is hysteretic, with folds at a = 0.104956 and a = 0.110607, a window **0.00565** wide. Carpenter et al. use b = 0.62 as a "reversible" example (their Fig. 11), but 0.62 < max f′ = 0.6495.
- **Comparison with Carpenter et al. Fig. A1.**
  - For q = 2, our fold curve has its cusp at (a, b) ≈ (0.125, 0.6495).
  - For q = 8, the irreversibility boundary is b = 0.686; their line reads about 0.7 by eye.
  - The curve shapes match.

## Figures

The caption of each figure is the docstring of its function in [make_figures.py](make_figures.py). Figures are saved as PDF in [figures/](figures/).

| File | Content |
|---|---|
| `fig1_landscape` | Walker Fig. 1a analogue: V(x) at the baseline (a = 0.04, b = 0.52) with L, R, Pr and the current state (ball at x = 0.3) |
| `fig2_landscape_sequence` | Walker Fig. 1b analogue: V(x) for a = 0.02 … 0.08, including the exact upper fold; shared vertical axis |
| `fig3_bifurcation` | Equilibria against a (stable solid, unstable dashed), folds marked, simulated slow sweep showing the hysteresis loop |
| `fig4_measures` | L, R, R/L and Pr against a, with the lake at rest at x_s, as a approaches the upper fold |
| `fig5_regimes` | Regime diagram in the (a, b) plane for q = 2 (our b-based definition), with the baseline and b = 0.62 marked |
| `fig6_carpenter_A1_comparison` | Our curves on the axes of Carpenter et al. Fig. A1: fold curves for q = 2 and 8, and regimes for q = 8 labelled as they do |
| `fig7_treatment` | Policy reading: the same chemical treatment (x cut to 0.05, a unchanged) in a turbid lake above the upper fold (a = 0.08: returns to turbid after lingering for ~40 time units) and inside the bistable window (a = 0.04: stays clear); landscapes and P over time |

## Live demo

`python demo.py` opens the landscape with a ball and the bifurcation diagram with the lake's path in grey. Two small time series show phosphorus in the water, x(t), and the two inputs over model time: external loading a and internal loading f(x), the recycling from the sediment. After the lake tips, internal loading is roughly ten times the external input, which is why cutting a does not bring the lake back until a falls below the lower fold. An info panel shows a, b, the regime, the state and the four measures.

- **Sliders for a and b.** The state persists between moves: after each change the ball rolls from where it is to the bottom of its current basin (integrated with `solve_ivp`). This makes hysteresis visible.
- **Treat lake chemically.** Sets x to a low value (0.05) without changing a or b: Walker et al.'s "move the state, keep the landscape". Above the upper fold, the ball rolls straight back to turbid.
- **Reset.** Returns a, b, the state, the trail and the noise to the baseline.
- **Noise** (off by default). Additive Euler–Maruyama noise (σ = 0.06, `SEED` in `lake.py`) shows tipping before the fold. The median time to tip on screen is about 5 s at a = 0.065 (fold at 0.0735), and it does not tip at the baseline a = 0.04.

<!-- Suggested sequence for the presentation:
1. Start at the baseline; the ball rolls into the clear basin.
2. Drag a past 0.0735: the lake turns turbid.
3. Treat: the ball rolls back to turbid.
4. Drag a back to about 0.045: the lake stays turbid (hysteresis).
5. Treat: the lake now stays clear.
6. Optionally, set b < 0.5 (irreversible) and repeat.

## Ambiguities and discrepancies (for the critical review)

1. **Latitude does not vanish when the basin does.** L is measured from the domain boundary x = 0, so at the upper fold L → 0.314 while R → 0 and Pr → 0. In our definition, latitude reports a wide basin at the moment the basin disappears. The lower edge of the clear basin is a physical limit (no negative phosphorus), not a threshold, so "width" mixes two kinds of edge. Measuring L from x_s instead would make it vanish.
2. **Undefined versus maximally resilient.** Below the lower fold there is no threshold, so the measures are NaN, although the clear state is then at its most resilient (its basin is the whole domain). Just above the lower fold L ≈ 0.96; just below it, L is undefined.
3. **The turbid basin has no latitude**, because it is unbounded above. Walker's measures are not symmetric between the two states in this model.
4. **The measures depend on the coordinates.** Rescaling x changes V and all widths, so R, L and R/L are not invariant. R/L also mixes units (potential over phosphorus).
5. **Negative precariousness.** When the state lies past the threshold, Pr = x_u − x < 0. We report it as is rather than clipping it.
6. **Carpenter et al.'s b = 0.62 is not reversible.** It lies below max f′ = 0.6495 and has a bistable window 0.00565 wide in a.
7. **Two different regime definitions.**
   - Carpenter et al. Fig. A1B labels regions of the (a, b) plane: outside the fold curve is reversible; inside it, hysteretic above the a = 0 line and irreversible below.
   - Our definition (Fig. 5) assigns the regime by b alone.
   - Fig. A1B is also drawn for q = 8, not q = 2. Fig. 6 reproduces their labelling for q = 8.
8. **Hysteresis depends on how fast a changes.** Simulated jumps lag the folds by an amount that grows with the ramp rate (≈ rate^(2/3)). At da/dt = 1e-3 a hysteretic lake never returns to clear, so it looks irreversible. Our Fig. 3 uses da/dt = 1e-5.
9. **Boundary case b = 0.5.** The lower fold sits exactly at a = 0; we classify it as hysteretic (a set of zero width).
10. **Model-time scale.** The continuous-time model has the same equilibria and folds as Carpenter et al.'s discrete-time model, but different transient dynamics. Sweep lags, relaxation times and noise-induced tipping times are therefore properties of our model, not of theirs.

## Implementation notes

- **Root finding.**
  - F is split at its turning points (roots of F′, found by a grid scan and `brentq`), so each piece is monotone and holds at most one root, which `brentq` then refines.
  - This resolves the two equilibria next to a fold even when they are closer than one grid cell (3e-5 apart, 1e-10 below the fold).
  - At exactly a fold, the double root is reported as "tangent".
  - It fails only right at the cusp (b ≈ max f′), where both turning points can fall in one grid cell.
- **Search bounds are derived, not chosen.**
  - Equilibria lie below (a + 1)/b because f < 1.
  - Fold points lie below (q/b)^(1/(q+1)).
- **Reproducibility.**
  - The only random component, the demo's optional noise, uses `SEED` in `lake.py`. Figures and checks are deterministic. -->

<!-- On the dev branch only: unit tests, sweep-lag and scaling-exponent analyses (lag ~ rate^0.63; R ~ delta^1.5). -->

## References

- Walker, B., C. S. Holling, S. R. Carpenter, and A. Kinzig. 2004. Resilience, adaptability and transformability in social–ecological systems. *Ecology and Society* 9(2): 5.
- Carpenter, S. R., D. Ludwig, and W. A. Brock. 1999. Management of eutrophication for lakes subject to potentially irreversible change. *Ecological Applications* 9: 751–771.
