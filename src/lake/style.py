"""Shared figure style for the figures and the demo.

Base style is the group's polder.mplstyle (repo root). Colours are taken from
its palette and assigned to *entities*, so the same thing has the same colour
in every figure and in the demo:

- clear state / clear basin       blue
- turbid state / turbid basin     amber
- threshold (unstable equilibrium) magenta
- latitude L, resistance R, precariousness Pr: teal, purple, orange

The clear/turbid/threshold set and the L/R/Pr set were checked for
colour-blind separation. Amber has low contrast on white, so it is never the
only cue: turbid elements are always labelled in text as well.
"""

import matplotlib.pyplot as plt

from lake.config import REPO_ROOT

COLORS = {
    "clear": "#0A4D9E",
    "turbid": "#E6A700",
    "threshold": "#C8417B",
    "L": "#00A39B",
    "R": "#6A4C93",
    "Pr": "#E8633A",
    "ink": "#222222",
    "neutral": "#5F6B7A",
}

TINT_ALPHA = 0.12  # transparency of basin / regime shading


def use_style(style_file):
    """Apply the group style; style_file is relative to the repo root."""
    plt.style.use(REPO_ROOT / style_file)
