"""Alternatives shown ONLY for contrast in EDA figures.

Nothing here is a pipeline step: notebook 02 and ``tt_ecog.preprocessing`` must
never import this module. Keeping the alternatives here preserves the single
implementation of the real transforms.
"""

import numpy as np


def common_average(x: np.ndarray) -> np.ndarray:
    """Mean across channels per instant (CAR reference signal), shape (T,)."""
    return np.asarray(x, dtype=np.float64).mean(axis=0)


def naive_slice(x: np.ndarray, factor: int) -> np.ndarray:
    """Decimation without antialiasing (what MNE's ``decim`` would do)."""
    return np.asarray(x)[..., ::factor]


def zero_order_hold(t_source: np.ndarray, y_source: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    index = np.clip(np.searchsorted(t_source, t_target, side="right") - 1, 0, t_source.size - 1)
    return np.asarray(y_source)[..., index]


def linear(t_source: np.ndarray, y_source: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    y = np.atleast_2d(y_source)
    return np.vstack([np.interp(t_target, t_source, row) for row in y])
