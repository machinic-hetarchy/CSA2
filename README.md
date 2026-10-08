# A quantitative stability landscape for Walker et al. (2004)

MSc Complex Systems, Assignment 2: Classic-Model Replication and Critical Review.

Walker et al. (2004) describe resilience with a drawn stability landscape (their Fig. 1) and three measures: **latitude** L (basin width), **resistance** R (basin depth) and **precariousness** Pr (distance to the threshold). The paper has no equations. This project makes Fig. 1a and 1b quantitative using the one-dimensional lake model of Carpenter, Ludwig and Brock (1999). It then tests whether the three measures can be computed from a real dynamical model, and where they break down.

## Quick start (RUNME)

Run everything from the repository root.

```bash
# 1. Environment (Python 3.11; versions pinned)
conda env create -f environment.yml      # or: pip install -r requirements.txt
conda activate CSA2

# 2. Tests: the quantitative checks plus unit tests (~5 s)
pytest

# 3. Regenerate every figure and results/checks.json from scratch (~5 s)
python scripts/make_figures.py

# 4. Live demo (opens a window; no Jupyter needed)
python demo/demo.py
```

All parameter values are in [config/params.yaml](config/params.yaml). Figures use the group style [polder.mplstyle](polder.mplstyle).

## Repository layout

```
config/params.yaml       all parameters, tolerances, figure and demo settings, random seed
src/lake/
    model.py             f(x), f'(x), F(x), F'(x), potential V(x)
    equilibria.py        root finding, stability, fold curve, regime classification
    resilience.py        L, R, R/L, Pr for the clear basin
    simulate.py          relaxation to rest, slow sweeps of a, Euler-Maruyama noise
    checks.py            computes every quantitative check (used by tests and by the script)
    style.py             figure style and the fixed colour for each entity
    config.py            loads the YAML config
scripts/make_figures.py  writes figures/*.pdf|png and results/checks.json
demo/demo.py             interactive demo (matplotlib widgets)
tests/                   pytest suite: checks, model, equilibria, resilience, simulation, demo
figures/, results/       generated output
docs/                    assignment brief and the two papers
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

## Definitions we chose

Walker et al. give no formulas, so these are our operational choices for a 1D landscape. They are implemented in [resilience.py](src/lake/resilience.py) and [equilibria.py](src/lake/equilibria.py).

| Measure | Definition |
|---|---|
| Clear basin | the interval [0, x_u), with x_u the unstable equilibrium (threshold) and x = 0 the domain boundary |
| Latitude L | width of the clear basin, L = x_u − 0 = x_u |
| Resistance R | depth of the clear basin, R = V(x_u) − V(x_s), with x_s the clear stable equilibrium. R/L is also reported, which Walker et al. call the more accurate measure |
| Precariousness Pr | distance from the current state to the threshold, Pr = x_u − x. Also reported as Pr/L |
| Turbid basin latitude | undefined: the turbid basin [x_u, ∞) has no upper edge. Reported as NaN, not cut off |

When there is no threshold (only one equilibrium), every clear-basin measure is NaN, with a stated reason:
- the clear basin has vanished (a above the upper fold);
- only the clear state exists (a below the lower fold);
- the regime is reversible.

**Regimes**, following Carpenter et al. (1999), Appendix A:

| Regime | Condition (for given b, q) |
|---|---|
| Reversible | b ≥ max f′; one equilibrium for every a ≥ 0 |
| Irreversible | the lower fold lies at a < 0: once turbid, no non-negative input restores the lake |
| Hysteretic | everything in between |

## Results

### Quantitative checks (q = 2)

All values are computed numerically and written to [results/checks.json](results/checks.json). All checks pass.

| Check | Expected | Computed | Abs. difference |
|---|---|---|---|
| Max of f′ | 3√3/8 at x = 1/√3 | 0.6495190528 at x = 0.5773502606 | 0 and 8.5e-9 |
| Reversible/irreversible boundary at a = 0 | b = 0.5, x = 1 | b = 0.5, x = 1.0 | 0 |
| Upper fold, b = 0.52 | a ≈ 0.0735, x ≈ 0.314 | a = 0.0735328, x = 0.313684 | 3.3e-5 and 3.2e-4 |
| Lower fold, b = 0.52 | a ≈ 0.0196, x ≈ 0.960 | a = 0.0195997, x = 0.959932 | 3.3e-7 and 6.8e-5 |
| −dV/dx = F(x) | equal | max error < 1e-8 | — |
| R → 0 and x_s → x_u at the upper fold | both → 0 | both exactly 0 at the fold; strictly decreasing on approach | — |

The tolerance is 1e-7 for the analytic checks. It is no tighter because f′ is flat at its maximum, so the location of the maximum is only determined to about 1e-8. The tolerance for the hand-calculated fold values is 1e-3.

### Additional computed values (no expected value in the brief)

- **Bistable window at b = 0.62.** The lake is hysteretic, with folds at a = 0.104956 and a = 0.110607, a window **0.00565** wide. Carpenter et al. use b = 0.62 as a "reversible" example (their Fig. 11), but 0.62 < max f′ = 0.6495.
- **Scaling near the fold.** Approaching the upper fold with δ = a_fold − a, R ∝ δ^1.500 and x_u − x_s ∝ δ^0.500 (fitted exponents). This is the saddle-node normal form: resistance collapses much faster than the basin narrows.
- **Sweep lag.** In a slow up-and-down sweep of a at b = 0.52, the jumps occur *after* the folds:

  | Ramp rate da/dt | Lag past upper fold | Lag past lower fold |
  |---|---|---|
  | 1e-3 | 0.0243 | never returns to clear before a = 0 |
  | 1e-4 | 0.0059 | 0.0069 |
  | 1e-5 | 0.0013 | 0.0016 |

  The fitted exponents are 0.63 (up and down). Slow-passage theory predicts 2/3.
- **Comparison with Carpenter et al. Fig. A1.**
  - For q = 2, our fold curve has its cusp at (a, b) ≈ (0.125, 0.6495).
  - For q = 8, the irreversibility boundary is b = 0.686; their line reads about 0.7 by eye.
  - The curve shapes match.

## Figures

The caption of each figure is the docstring of its function in [scripts/make_figures.py](scripts/make_figures.py). Every figure is saved as PDF and PNG in [figures/](figures/).

| File | Content |
|---|---|
| `fig1_landscape` | Walker Fig. 1a analogue: V(x) at the baseline (a = 0.04, b = 0.52) with L, R, Pr and the current state (ball at x = 0.3) |
| `fig2_landscape_sequence` | Walker Fig. 1b analogue: V(x) for a = 0.02 … 0.08, including the exact upper fold; shared vertical axis |
| `fig3_bifurcation` | Equilibria against a (stable solid, unstable dashed), folds marked, simulated slow sweep showing the hysteresis loop |
| `fig4_measures` | L, R, R/L and Pr against a, with the lake at rest at x_s, as a approaches the upper fold |
| `fig5_regimes` | Regime diagram in the (a, b) plane for q = 2 (our b-based definition), with the baseline and b = 0.62 marked |
| `fig6_carpenter_A1_comparison` | Our curves on the axes of Carpenter et al. Fig. A1: fold curves for q = 2 and 8, and regimes for q = 8 labelled as they do |

## Live demo

`python demo/demo.py` opens two panels: the landscape with a ball (top) and the bifurcation diagram with the lake's path in grey (bottom). An info panel shows a, b, the regime, the state and the four measures.

- **Sliders for a and b.** The state persists between moves: after each change the ball rolls from where it is to the bottom of its current basin (integrated with `solve_ivp`). This makes hysteresis visible.
- **Treat lake chemically.** Sets x to a low value (0.05) without changing a or b: Walker et al.'s "move the state, keep the landscape". Above the upper fold, the ball rolls straight back to turbid.
- **Reset.** Returns a, b, the state, the trail and the noise to the baseline.
- **Noise** (off by default). Additive Euler–Maruyama noise (σ = 0.06, seed from the config) shows tipping before the fold. The median time to tip on screen is about 5 s at a = 0.065 (fold at 0.0735), and it does not tip at the baseline a = 0.04.

Suggested sequence for the presentation:
1. Start at the baseline; the ball rolls into the clear basin.
2. Drag a past 0.0735: the lake turns turbid.
3. Treat: the ball rolls back to turbid.
4. Drag a back to about 0.045: the lake stays turbid (hysteresis).
5. Treat: the lake now stays clear.
6. Optionally, set b < 0.5 (irreversible) and repeat.

## Ambiguities and discrepancies (for the critical review)

These are flagged in code comments where they arise.

1. **Latitude does not vanish when the basin does.** L is measured from the domain boundary x = 0, so at the upper fold L → 0.314 while R → 0 and Pr → 0. In our definition, latitude reports a wide basin at the moment the basin disappears. The lower edge of the clear basin is a physical limit (no negative phosphorus), not a threshold, so "width" mixes two kinds of edge. Measuring L from x_s instead would make it vanish.
2. **Undefined versus maximally resilient.** Below the lower fold there is no threshold, so the measures are NaN, although the clear state is then at its most resilient (its basin is the whole domain). Just above the lower fold L ≈ 0.96; just below it, L is undefined.
3. **The turbid basin has no latitude**, because it is unbounded above. Walker's measures are not symmetric between the two states in this model.
4. **The measures depend on the coordinates.** Rescaling x changes V and all widths, so R, L and R/L are not invariant. R/L also mixes units (potential over phosphorus).
5. **Negative precariousness.** When the state lies past the threshold, Pr = x_u − x < 0. We report it as is rather than clipping it.
6. **Carpenter et al.'s b = 0.62 is not reversible.** It lies below max f′ = 0.6495 and has a bistable window 0.00565 wide in a.
7. **Two different regime definitions.**
   - Carpenter et al. Fig. A1B labels regions of the (a, b) plane: outside the fold curve is reversible; inside it, hysteretic above the a = 0 line and irreversible below.
   - The brief (and our Fig. 5) assigns the regime by b alone.
   - Fig. A1B is also drawn for q = 8, not q = 2. Fig. 6 reproduces their labelling for q = 8.
8. **Hysteresis depends on how fast a changes.** Simulated jumps lag the folds by an amount that grows with the ramp rate (≈ rate^(2/3)). At da/dt = 1e-3 a hysteretic lake never returns to clear, so it looks irreversible. Our Fig. 3 uses da/dt = 1e-5.
9. **Boundary case b = 0.5.** The lower fold sits exactly at a = 0; we classify it as hysteretic (a set of zero width).
10. **Model-time scale.** The continuous-time model has the same equilibria and folds as Carpenter et al.'s discrete-time model, but different transient dynamics. Sweep lags, relaxation times and noise-induced tipping times are therefore properties of our model, not of theirs.

## Implementation notes

- **Root finding.**
  - F is split at its turning points (roots of F′, found by a grid scan and `brentq`), so each piece is monotone and holds at most one root, which `brentq` then refines.
  - This resolves the two equilibria next to a fold even when they are closer than one grid cell (3e-5 apart, 1e-10 below the fold, is tested).
  - At exactly a fold, the double root is reported as "tangent".
  - It fails only right at the cusp (b ≈ max f′), where both turning points can fall in one grid cell.
- **Search bounds are derived, not chosen.**
  - Equilibria lie below (a + 1)/b because f < 1.
  - Fold points lie below (q/b)^(1/(q+1)).
- **Reproducibility.**
  - Every random component takes its seed from `random.seed` in the config, and the seed is recorded in `results/checks.json`.
  - NaN values are written to the JSON as `null`.

## References

- Walker, B., C. S. Holling, S. R. Carpenter, and A. Kinzig. 2004. Resilience, adaptability and transformability in social–ecological systems. *Ecology and Society* 9(2): 5.
- Carpenter, S. R., D. Ludwig, and W. A. Brock. 1999. Management of eutrophication for lakes subject to potentially irreversible change. *Ecological Applications* 9: 751–771.
