"""Shared figure style. Object-oriented matplotlib only (no pyplot global state)."""

import textwrap

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots in fixed order (validated palette); fingers use slots 1-5.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FINGERS = ["Pulgar", "Índice", "Medio", "Anular", "Meñique"]
FINGER_COLORS = SLOTS[:5]
KEPT = SLOTS[0]
EXCLUDED = SLOTS[7]
COMPARATOR = SLOTS[1]

SEQUENTIAL = LinearSegmentedColormap.from_list("tt_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("tt_blue_red", [
    "#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#b8322f"])

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "lines.linewidth": 1.4, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "legend.fontsize": 8, "figure.titlesize": 11,
}


def figure(nrows: int = 1, ncols: int = 3, width: float = 4.6, height: float = 3.2,
           sharex: bool = False, sharey: bool = False):
    """Figure with a 2-D axes array; S1/S2/S3 usually occupy the columns."""
    import matplotlib
    with matplotlib.rc_context(RC):
        fig = Figure(figsize=(width * ncols, height * nrows), layout="constrained")
        axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey)
    for ax in axes.flat:
        style_axes(ax)
    return fig, axes


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def shade_band(ax, band, label: str | None = "Banda 40–200 Hz") -> None:
    ax.axvspan(band[0], band[1], color=SLOTS[0], alpha=0.08, lw=0, label=label)


def mark_lines(ax, lines, label: str | None = "Red 60/120/180 Hz") -> None:
    for k, line in enumerate(lines):
        ax.axvline(line, color=MUTED, lw=0.8, label=label if k == 0 else None)


def plot_step(n_samples: int, max_points: int) -> int:
    return max(1, int(np.ceil(n_samples / max_points)))


def robust_limits(values: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [low, high])
    return (float(lo), float(hi)) if hi > lo else (float(lo) - 1, float(lo) + 1)


def subject_title(subject: str, extra: str = "") -> str:
    return f"{subject}{' · ' + extra if extra else ''}"


def wrap(text: str, width: int = 120) -> str:
    return "\n".join(textwrap.wrap(text, width))


def adaptive_log(ax, values) -> None:
    """Log y-scale only when the data span more than about one decade."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size and finite.max() / finite.min() > 20:
        ax.set_yscale("log")


def plain_colorbar_ticks(colorbar) -> None:
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda value, _: f"{value:g}")
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.ax.yaxis.set_minor_formatter(formatter)
