"""Helpers shared by EDA sections."""

import numpy as np

from .. import plotting as pl


def pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of x (A, T) and of y (B, T) -> (A, B)."""
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    x = x / np.maximum(np.sqrt((x ** 2).sum(axis=1, keepdims=True)), 1e-300)
    y = y / np.maximum(np.sqrt((y ** 2).sum(axis=1, keepdims=True)), 1e-300)
    return x @ y.T


def empty_panel(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color=pl.INK_SECONDARY, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def kept_original(bundle) -> np.ndarray:
    return np.array(bundle.cmap.kept)


def finger_legend(ax) -> None:
    ax.legend(loc="upper right", ncols=1)
