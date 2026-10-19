"""Descriptive movement annotation from the 25 Hz development glove.

Used only for confirmatory EDA figures (band and delay). It never changes a
preprocessing parameter and never touches test labels.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MovementInfo:
    speed: np.ndarray  # (5, n25) absolute flexion velocity
    thresholds: np.ndarray  # (5,) per-finger speed quantile
    onsets_by_finger: list[np.ndarray]  # onset times (s) per finger
    onsets: np.ndarray  # merged onset times (s), min inter-onset enforced
    moving_target: np.ndarray  # (T_target,) any finger above its threshold


def _enforce_gap(times: np.ndarray, gap: float) -> np.ndarray:
    kept: list[float] = []
    for value in np.sort(times):
        if not kept or value - kept[-1] >= gap:
            kept.append(float(value))
    return np.array(kept)


def detect_movement(y25: np.ndarray, t25: np.ndarray, t_target: np.ndarray,
                    speed_quantile: float, min_interonset_s: float) -> MovementInfo:
    speed = np.abs(np.gradient(y25, t25, axis=1))
    thresholds = np.quantile(speed, speed_quantile, axis=1)
    above = speed > thresholds[:, None]
    onsets_by_finger = []
    for finger in range(y25.shape[0]):
        rising = np.flatnonzero(above[finger, 1:] & ~above[finger, :-1]) + 1
        onsets_by_finger.append(_enforce_gap(t25[rising], min_interonset_s))
    merged = _enforce_gap(np.concatenate(onsets_by_finger), min_interonset_s)
    index = np.clip(np.searchsorted(t25, t_target, side="right") - 1, 0, t25.size - 1)
    moving = above.any(axis=0)[index]
    return MovementInfo(speed, thresholds, onsets_by_finger, merged, moving)
