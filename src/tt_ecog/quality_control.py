"""Descriptive acquisition QC. No labels, transformations or cleaning decisions."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch


EVENT_COLUMNS = ["subject", "channel_0based", "channel_1based", "detector",
                 "start_sample", "end_sample_exclusive", "start_s", "end_s",
                 "duration_s", "peak_score", "review_required"]


@dataclass
class QCResult:
    channels: pd.DataFrame
    events: pd.DataFrame
    temporal: pd.DataFrame
    frequencies: np.ndarray
    psd: np.ndarray
    detector_status: dict[str, str]


def robust_scores(values: np.ndarray, epsilon: float) -> tuple[np.ndarray, float, float]:
    """Absolute robust z, using 1.4826*MAD; finite floor for constant signals."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    denominator = max(1.4826 * mad, epsilon * max(1.0, abs(median)))
    return np.abs(values - median) / denominator, median, mad


def grouped_intervals(mask: np.ndarray, max_gap_samples: int) -> list[tuple[int, int]]:
    """Half-open intervals; merge only gaps <= explicit descriptive tolerance."""
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > max_gap_samples + 1)
    starts = np.r_[indices[0], indices[boundaries + 1]]
    stops = np.r_[indices[boundaries] + 1, indices[-1] + 1]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _spectrum(values: np.ndarray, fs: float, analysis: dict) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(values.shape[-1], int(round(analysis["psd_window_seconds"] * fs)))
    overlap = min(nperseg - 1, int(nperseg * analysis["psd_overlap_fraction"]))
    return welch(values, fs=fs, window="hann", nperseg=nperseg,
                 noverlap=overlap, detrend="constant", scaling="density", axis=-1)


def _band_powers(freq: np.ndarray, psd: np.ndarray, band: list,
                 lines: list, half_width: float, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    in_band = (freq >= band[0]) & (freq <= band[1])
    line_bins = np.zeros(freq.size, dtype=bool)
    for line in lines:
        line_bins |= np.abs(freq - line) <= half_width
    df = freq[1] - freq[0]
    power = psd[..., in_band].sum(axis=-1) * df
    line_power = psd[..., line_bins & in_band].sum(axis=-1) * df
    return power, line_power / np.maximum(power, epsilon)


def analyze_ecog(ecog: np.ndarray, subject: str, fs: float,
                 qc_config: dict[str, Any], signal_config: dict[str, Any]) -> QCResult:
    """Analyze all raw channels with the same rules. Input is never modified."""
    if ecog.ndim != 2 or ecog.dtype.kind not in "fiu" or not np.isfinite(ecog).all():
        raise ValueError("QC requires a finite real array (channels,time)")
    if ecog.shape[0] < 1 or ecog.shape[1] < 2 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("QC requires channels, at least two samples and a positive sampling rate")
    analysis = qc_config["analysis"]
    thresholds = qc_config["thresholds"]
    epsilon = analysis["numerical_epsilon"]
    window = int(round(analysis["window_seconds"] * fs))
    merge_gap = int(round(analysis["event_merge_gap_seconds"] * fs))
    n_channels, n_samples = ecog.shape
    frequencies, psd = _spectrum(ecog, fs, analysis)
    band = signal_config["bandpass_hz"]
    lines = signal_config["notch_hz"]
    powers, line_fractions = _band_powers(frequencies, psd, band, lines,
                                         analysis["line_half_width_hz"], epsilon)
    channel_rows, temporal_rows, event_rows = [], [], []
    requirements = {"robust_amplitude": ["robust_z"], "discontinuity": ["jump_z"],
                    "flatline": ["flatline_fraction"],
                    "spectral": ["spectral_z", "spectral_min_duration_s"]}
    statuses = {name: ("PENDING: " + ", ".join(k for k in keys if thresholds[k] is None)
                       if any(thresholds[k] is None for k in keys) else "EXECUTED_CANDIDATE")
                for name, keys in requirements.items()}
    statuses["bad_fraction_review"] = ("PENDING: max_bad_fraction" if thresholds["max_bad_fraction"] is None
                                         else "EXECUTED_CANDIDATE")
    # Per-window PSD also describes persistence, without treating neighboring
    # acquisition indices as anatomical neighbors.
    temporal_spectra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, n_samples, window):
        end = min(start + window, n_samples)
        if end - start < 2:
            continue
        freq, power = _spectrum(ecog[:, start:end], fs, analysis)
        _, fractions = _band_powers(freq, power, band, lines, analysis["line_half_width_hz"], epsilon)
        scores, _, _ = robust_scores(np.log10(np.maximum(fractions, epsilon)), epsilon)
        # Only high line-power ratios are spectral candidates (not unusually low ratios).
        scores = np.where(fractions > np.median(fractions), scores, 0.0)
        temporal_spectra[start] = (fractions, scores)

    for channel in range(n_channels):
        values = ecog[channel].astype(np.float64)
        amplitude_scores, median, mad = robust_scores(values, epsilon)
        differences = np.diff(values)
        jump_scores, _, _ = robust_scores(differences, epsilon)
        global_std = float(np.std(values))
        masks: dict[str, np.ndarray] = {}
        score_arrays: dict[str, np.ndarray] = {}
        if thresholds["robust_z"] is not None:
            masks["robust_amplitude"] = amplitude_scores > thresholds["robust_z"]
            score_arrays["robust_amplitude"] = amplitude_scores
        if thresholds["jump_z"] is not None:
            masks["discontinuity"] = np.r_[False, jump_scores > thresholds["jump_z"]]
            score_arrays["discontinuity"] = np.r_[0.0, jump_scores]
        if thresholds["flatline_fraction"] is not None:
            masks["flatline"] = np.zeros(n_samples, dtype=bool)
            score_arrays["flatline"] = np.zeros(n_samples)
        if statuses["spectral"] == "EXECUTED_CANDIDATE":
            masks["spectral"] = np.zeros(n_samples, dtype=bool)
            score_arrays["spectral"] = np.zeros(n_samples)
        for start, (fractions, spectral_scores) in temporal_spectra.items():
            end = min(start + window, n_samples)
            local = values[start:end]
            local_std = float(np.std(local))
            flatness = local_std / max(global_std, epsilon)
            temporal_rows.append({"subject": subject, "channel_0based": channel,
                                  "start_sample": start, "end_sample_exclusive": end,
                                  "start_s": start / fs, "end_s": end / fs,
                                  "std": local_std, "median": float(np.median(local)),
                                  "p01": float(np.quantile(local, .01)),
                                  "p99": float(np.quantile(local, .99)),
                                  "flatness_fraction": flatness,
                                  "line_power_fraction": float(fractions[channel]),
                                  "spectral_robust_z": float(spectral_scores[channel])})
            if "flatline" in masks:
                masks["flatline"][start:end] = flatness <= thresholds["flatline_fraction"]
                score_arrays["flatline"][start:end] = 1 / max(flatness, epsilon)
            if "spectral" in masks:
                masks["spectral"][start:end] = spectral_scores[channel] > thresholds["spectral_z"]
                score_arrays["spectral"][start:end] = spectral_scores[channel]
        marked = np.zeros(n_samples, dtype=bool)
        for detector, mask in masks.items():
            for start, end in grouped_intervals(mask, merge_gap):
                # Require actual marked duration, not the gaps added by grouping.
                if (detector == "spectral" and
                        np.count_nonzero(mask[start:end]) / fs < thresholds["spectral_min_duration_s"]):
                    continue
                marked[start:end] = True
                event_rows.append({"subject": subject, "channel_0based": channel,
                                   "channel_1based": channel + 1, "detector": detector,
                                   "start_sample": start, "end_sample_exclusive": end,
                                   "start_s": start / fs, "end_s": end / fs,
                                   "duration_s": (end - start) / fs,
                                   "peak_score": float(score_arrays[detector][start:end].max()),
                                   "review_required": True})
        fraction = float(marked.mean())
        channel_rows.append({"subject": subject, "channel_0based": channel, "channel_1based": channel + 1,
                             "n_samples": n_samples, "median": median, "mad": mad,
                             "mad_floor_used": 1.4826 * mad < epsilon * max(1.0, abs(median)),
                             "std": global_std, "min": float(values.min()), "max": float(values.max()),
                             "p01": float(np.quantile(values, .01)), "p99": float(np.quantile(values, .99)),
                             "max_abs": float(np.abs(values).max()),
                             "max_abs_difference": float(np.abs(differences).max()),
                             "band_power": float(powers[channel]), "line_power_fraction": float(line_fractions[channel]),
                             "grouped_candidate_fraction": fraction,
                             "fraction_review_candidate": (None if thresholds["max_bad_fraction"] is None
                                                            else fraction > thresholds["max_bad_fraction"]),
                             "automatic_exclusion": False})
    channels = pd.DataFrame(channel_rows)
    temporal = pd.DataFrame(temporal_rows)
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    for frame in (channels, temporal, events):
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Nonfinite QC diagnostic; acquisition values exceed numerical range")
    if not np.isfinite(psd).all():
        raise ValueError("Nonfinite PSD")
    return QCResult(channels, events, temporal, frequencies, psd, statuses)
