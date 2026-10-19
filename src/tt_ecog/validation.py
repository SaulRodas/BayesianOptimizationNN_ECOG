"""Trajectory reconstruction by overlap-add and fold scoring (protocol §8.3).

Window predictions are averaged at every column they cover (uniform weights),
so the result keeps the block's length and timeline. Every valid column must be
covered: incomplete coverage is an error, never a silent loss.
"""

from typing import Any

import numpy as np

from .datasets import FoldData
from .metrics import trajectory_metrics


class IncompleteCoverage(ValueError):
    """A valid column has no window prediction."""


def overlap_add(predictions: np.ndarray, starts: np.ndarray, length: int,
                valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Average (n, K, L) window predictions placed at ``starts`` on a (K, length) timeline.

    Returns (trajectory with NaN where uncovered, coverage count per column).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    if predictions.ndim != 3 or starts.shape != (predictions.shape[0],):
        raise ValueError(f"Expected (n, K, L) predictions and (n,) starts; got {predictions.shape}, {starts.shape}")
    window = predictions.shape[-1]
    if starts.size and (starts.min() < 0 or starts.max() + window > length):
        raise ValueError("A window extends beyond the timeline")
    total = np.zeros((predictions.shape[1], length))
    coverage = np.zeros(length, dtype=np.int64)
    for start, prediction in zip(starts, predictions):
        total[:, start:start + window] += prediction
        coverage[start:start + window] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        trajectory = np.where(coverage > 0, total / np.maximum(coverage, 1), np.nan)
    if valid is not None:
        missing = np.flatnonzero(np.asarray(valid, dtype=bool) & (coverage == 0))
        if missing.size:
            raise IncompleteCoverage(f"{missing.size} valid columns have no prediction (first {missing[0]})")
    return trajectory, coverage


def score_validation(fold: FoldData, window_predictions: np.ndarray, degenerate_std_relative: float) -> dict[str, Any]:
    """val_pearson_full and secondary metrics of one fold from its validation windows.

    ``window_predictions`` (n_val_windows, 5, L) are in the scaled target space;
    Pearson uses the reconstructed trajectory, errors are reported in original
    glove units after the fold's inverse MinMax transform.
    """
    block = fold.blocks[fold.val_block]
    trajectory, coverage = overlap_add(window_predictions, fold.val_windows, block.valid.size, block.valid)
    original = np.full_like(trajectory, np.nan)
    original[:, block.valid] = fold.target_scaler.inverse_transform(trajectory[:, block.valid].T).T
    result = trajectory_metrics(block.y_original, original, block.valid,
                                degenerate_std_relative=degenerate_std_relative)
    result["val_pearson_full"] = result["pearson_macro"]
    result["coverage_min_valid"] = int(coverage[block.valid].min())
    return result
