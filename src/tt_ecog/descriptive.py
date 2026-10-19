"""Descriptive signal statistics for the EDA. No thresholds, gates or decisions."""

import numpy as np
from scipy.signal import welch


def robust_scores(values: np.ndarray, epsilon: float) -> tuple[np.ndarray, float, float]:
    """Absolute robust z, using 1.4826*MAD; finite floor for constant signals."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    denominator = max(1.4826 * mad, epsilon * max(1.0, abs(median)))
    return np.abs(values - median) / denominator, median, mad


def grouped_intervals(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Half-open intervals of True values; gaps <= max_gap are merged."""
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > max_gap + 1)
    starts = np.r_[indices[0], indices[boundaries + 1]]
    stops = np.r_[indices[boundaries] + 1, indices[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def welch_psd(values: np.ndarray, fs: float, window_seconds: float,
              overlap_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Welch/Hann PSD along the last axis; constant detrend for the estimator only."""
    nperseg = min(values.shape[-1], int(round(window_seconds * fs)))
    overlap = min(nperseg - 1, int(nperseg * overlap_fraction))
    return welch(values, fs=fs, window="hann", nperseg=nperseg, noverlap=overlap,
                 detrend="constant", scaling="density", axis=-1)


def band_line_fraction(freq: np.ndarray, psd: np.ndarray, band: list, lines: list,
                       half_width: float, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Band power and fraction of it within +/- half_width of each line frequency."""
    in_band = (freq >= band[0]) & (freq <= band[1])
    line_bins = np.zeros(freq.size, dtype=bool)
    for line in lines:
        line_bins |= np.abs(freq - line) <= half_width
    df = freq[1] - freq[0]
    power = psd[..., in_band].sum(axis=-1) * df
    line_power = psd[..., line_bins & in_band].sum(axis=-1) * df
    return power, line_power / np.maximum(power, epsilon)


def windowed_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard deviation over consecutive non-overlapping windows.

    Returns (std (C, n_windows), window starts). A trailing remainder shorter
    than two samples is ignored.
    """
    n = values.shape[-1]
    starts = np.array([s for s in range(0, n, window) if min(s + window, n) - s >= 2])
    out = np.empty((values.shape[0], starts.size))
    for k, start in enumerate(starts):
        out[:, k] = values[:, start:start + window].std(axis=-1)
    return out, starts
