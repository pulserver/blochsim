"""Figure style shared by the documentation gallery and explanation figures.

The canvas is transparent and the foreground is a mid grey that remains
readable on both the light and dark Sphinx themes. Individual examples should
set only figure-specific geometry, colormaps and annotations.
"""

from __future__ import annotations

#: Documentation-column width used by full-width figures.
PAGE_WIDTH = 8.6

#: Mid grey with useful contrast against both the light and dark page.
FOREGROUND = "#8a8a8a"

STYLE = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "figure.constrained_layout.use": True,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "legend.frameon": False,
    "text.color": FOREGROUND,
    "axes.labelcolor": FOREGROUND,
    "axes.titlecolor": FOREGROUND,
    "axes.edgecolor": FOREGROUND,
    "xtick.color": FOREGROUND,
    "ytick.color": FOREGROUND,
    "xtick.labelcolor": FOREGROUND,
    "ytick.labelcolor": FOREGROUND,
    "grid.color": FOREGROUND,
    "grid.alpha": 0.3,
}


def apply() -> None:
    """Apply the documentation figure style to Matplotlib."""
    import matplotlib

    matplotlib.rcParams.update(STYLE)


def reset(gallery_conf, fname) -> None:
    """Reset Matplotlib, then apply the documentation style for one example."""
    import matplotlib

    matplotlib.rcdefaults()
    apply()
