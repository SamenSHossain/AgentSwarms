"""Shared matplotlib styling: reference categorical palette, thin marks, recessive axes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b3"


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def pct_axis(ax, axis: str = "y") -> None:
    """A 0-1 share axis in percent: ticks on 1/2/2.5/5 steps, printed with the decimals the axis range needs, so a
    tick at 0.025 reads 2.5%, not 2%."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(MaxNLocator(steps=[1, 2, 2.5, 5, 10]))
    a.set_major_formatter(PercentFormatter(xmax=1, decimals=None))


def save(fig, path) -> str:
    fig.savefig(path)
    plt.close(fig)
    return str(path)
