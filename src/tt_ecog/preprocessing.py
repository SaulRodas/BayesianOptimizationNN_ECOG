"""Segment-aware ECoG and glove transforms: the single implementation.

The EDA calls these functions in memory to draw before/after figures and
notebook 02 applies them per chronological block. Every function receives ONE
contiguous segment, never mutates its input and reports its temporal
half-support so callers can mask edges. Fold logic lives in ``splits``.
Comparison-only alternatives (mean reference, naive slicing, ...) live in
``tt_ecog.eda.comparators`` and must never be imported here.
"""

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.signal import firwin, freqz, kaiserord, oaconvolve, resample_poly

from .config import ExperimentConfig
from .io import DataContractError


# --------------------------------------------------------------------------- channels

@dataclass(frozen=True)
class ChannelMap:
    """Original 0-based acquisition indices kept, in their effective order."""

    subject: str
    n_original: int
    excluded: tuple[int, ...]
    kept: tuple[int, ...]

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    def new_index(self, original: int) -> int:
        if original not in self.kept:
            raise KeyError(f"{self.subject}: channel {original} is excluded or out of range")
        return self.kept.index(original)

    def to_json(self) -> dict[str, Any]:
        return {"subject": self.subject, "index_convention": "zero_based",
                "n_original": self.n_original, "n_kept": self.n_kept,
                "excluded_original": list(self.excluded),
                "kept_original_in_effective_order": list(self.kept)}


def channel_map(subject: str, n_original: int, excluded: tuple[int, ...] | list[int]) -> ChannelMap:
    excluded = tuple(excluded)
    if any(type(c) is not int for c in excluded):
        raise ValueError("Channel indices must be plain integers")
    if len(set(excluded)) != len(excluded):
        raise ValueError("Duplicated excluded channel")
    if any(c < 0 or c >= n_original for c in excluded):
        raise ValueError(f"{subject}: excluded channel outside 0..{n_original - 1}")
    if len(excluded) >= n_original:
        raise ValueError(f"{subject}: cannot exclude every channel")
    kept = tuple(c for c in range(n_original) if c not in excluded)
    return ChannelMap(subject, n_original, tuple(sorted(excluded)), kept)


def exclude_channels(x: np.ndarray, cmap: ChannelMap) -> np.ndarray:
    """Return a float64 copy with only the kept channels (never a view of raw data)."""
    if x.ndim != 2 or x.shape[0] != cmap.n_original:
        raise ValueError(f"Expected ({cmap.n_original}, T); got {x.shape}")
    return np.array(x[list(cmap.kept)], dtype=np.float64)


# --------------------------------------------------------------------------- reference

def common_median(x: np.ndarray, chunk_samples: int = 50_000) -> np.ndarray:
    """Instantaneous median across channels, shape (T,). No temporal statistics."""
    out = np.empty(x.shape[1], dtype=np.float64)
    for start in range(0, x.shape[1], chunk_samples):
        out[start:start + chunk_samples] = np.median(x[:, start:start + chunk_samples], axis=0)
    return out


def common_median_reference(x: np.ndarray, chunk_samples: int = 50_000) -> np.ndarray:
    """x - median(x, axis=0) per instant; with an even channel count the median
    averages the two central values. Exact in chunks because it is instantaneous."""
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("CMR requires (channels>=2, T)")
    return np.asarray(x, dtype=np.float64) - common_median(x, chunk_samples)[None, :]


# --------------------------------------------------------------------------- FIR filters

@dataclass(frozen=True)
class FirDesign:
    """Odd-length linear-phase FIR applied once, centred (zero phase)."""

    name: str
    taps: np.ndarray
    sfreq: float
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        taps = np.array(self.taps, dtype=np.float64)
        if taps.ndim != 1 or taps.size % 2 == 0:
            raise ValueError("FIR taps must be a one-dimensional odd-length array")
        taps.setflags(write=False)
        object.__setattr__(self, "taps", taps)

    @property
    def n_taps(self) -> int:
        return int(self.taps.size)

    @property
    def half_support(self) -> int:
        return (self.n_taps - 1) // 2

    def response(self, n_points: int = 16384) -> tuple[np.ndarray, np.ndarray]:
        freqs, h = freqz(self.taps, worN=n_points, fs=self.sfreq)
        return freqs, h

    def gain_db(self, freqs_hz: np.ndarray | list[float]) -> np.ndarray:
        _, h = freqz(self.taps, worN=np.asarray(freqs_hz, dtype=float), fs=self.sfreq)
        return 20 * np.log10(np.maximum(np.abs(h), 1e-300))

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "sfreq": self.sfreq, "n_taps": self.n_taps,
                "half_support_samples": self.half_support,
                "half_support_seconds": self.half_support / self.sfreq,
                "phase": "zero (centred linear-phase kernel, single pass)",
                "library": "scipy.signal", "params": self.params,
                "taps_sha256": hashlib.sha256(self.taps.tobytes()).hexdigest()}


def _hamming_length(transition_hz: float, sfreq: float) -> int:
    # Hamming window rule of thumb (also used by MNE): N = 3.3 / (df / fs), odd.
    n = int(math.ceil(3.3 * sfreq / transition_hz))
    return n + 1 if n % 2 == 0 else n


def design_bandpass(sfreq: float, band_hz: list[float], l_trans_hz: float,
                    h_trans_hz: float, window: str = "hamming") -> FirDesign:
    low, high = band_hz
    cutoffs = [low - l_trans_hz / 2, high + h_trans_hz / 2]
    if cutoffs[0] <= 0 or cutoffs[1] >= sfreq / 2:
        raise ValueError("Bandpass transition bands exceed the valid frequency range")
    n_taps = _hamming_length(min(l_trans_hz, h_trans_hz), sfreq)
    taps = firwin(n_taps, cutoffs, pass_zero=False, window=window, fs=sfreq)
    return FirDesign("bandpass", taps, sfreq, {
        "type": "bandpass", "window": window, "passband_hz": [low, high],
        "cutoffs_minus6db_hz": cutoffs, "l_trans_hz": l_trans_hz, "h_trans_hz": h_trans_hz})


def design_notch(sfreq: float, lines_hz: list[float], width_hz: float, trans_hz: float,
                 window: str = "hamming") -> FirDesign:
    edges = []
    for line in lines_hz:
        half = width_hz / 2 + trans_hz / 2
        edges += [line - half, line + half]
    n_taps = _hamming_length(trans_hz, sfreq)
    taps = firwin(n_taps, edges, pass_zero=True, window=window, fs=sfreq)
    return FirDesign("notch", taps, sfreq, {
        "type": "bandstop", "window": window, "lines_hz": list(lines_hz),
        "stop_width_hz": width_hz, "trans_hz": trans_hz, "cutoffs_minus6db_hz": edges})


def combined_fir(*designs: FirDesign) -> FirDesign:
    """Single kernel equivalent to applying the designs in sequence."""
    sfreq = designs[0].sfreq
    if any(d.sfreq != sfreq for d in designs):
        raise ValueError("Cannot combine FIR designs with different sampling rates")
    taps = designs[0].taps
    for design in designs[1:]:
        taps = np.convolve(taps, design.taps)
    return FirDesign("+".join(d.name for d in designs), taps, sfreq,
                     {"components": [d.to_json() for d in designs]})


def apply_fir(x: np.ndarray, design: FirDesign) -> np.ndarray:
    """Zero-phase centred convolution on one segment; output length == input length.

    Edges are reflection padded to keep the shape, but the first and last
    ``design.half_support`` samples depend on padding and must be masked.
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    x2 = x[None, :] if squeeze else x
    half = design.half_support
    padded = np.pad(x2, ((0, 0), (half, half)), mode="reflect")
    out = oaconvolve(padded, design.taps[None, :], mode="valid", axes=-1)
    return out[0] if squeeze else out


def valid_slice(n_samples: int, half_support: int) -> slice:
    """Interior samples whose full kernel lies inside the segment."""
    return slice(half_support, max(half_support, n_samples - half_support))


# --------------------------------------------------------------------------- Morlet

def morlet_frequencies(freq_range_hz: list[float], n_frequencies: int, spacing: str = "log") -> np.ndarray:
    if spacing != "log":
        raise ValueError("Only logarithmic Morlet spacing is specified by the protocol")
    return np.geomspace(freq_range_hz[0], freq_range_hz[1], n_frequencies)


def morlet_half_support(freqs: np.ndarray, n_cycles: float, sfreq: float) -> np.ndarray:
    """Half-length in samples of each MNE Morlet wavelet (MNE uses +/- 5 sigma_t)."""
    from mne.time_frequency import morlet
    wavelets = morlet(sfreq, np.asarray(freqs, dtype=float), n_cycles=n_cycles, zero_mean=False)
    return np.array([(len(w) - 1) // 2 for w in wavelets], dtype=int)


def morlet_power(x: np.ndarray, sfreq: float, freqs: np.ndarray, n_cycles: float,
                 zero_mean: bool) -> np.ndarray:
    """|W|^2 at the input rate, shape (C, F, T). Every MNE parameter is explicit;
    MNE's ``decim`` is never used because it slices without antialiasing."""
    from mne.time_frequency import tfr_array_morlet
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("morlet_power expects (C, T)")
    power = tfr_array_morlet(x[None], sfreq=float(sfreq), freqs=np.asarray(freqs, dtype=float),
                             n_cycles=float(n_cycles), zero_mean=bool(zero_mean), use_fft=True,
                             decim=1, output="power", n_jobs=None, verbose=False)
    return power[0]


# --------------------------------------------------------------------------- decimation

def design_antialias(sfreq: float, factor: int, passband_hz: float, stopband_hz: float,
                     attenuation_db: float) -> FirDesign:
    if not passband_hz < stopband_hz <= sfreq / (2 * factor):
        raise ValueError("Antialias stopband must not exceed the output Nyquist frequency")
    width = (stopband_hz - passband_hz) / (sfreq / 2)
    n_taps, beta = kaiserord(attenuation_db, width)
    n_taps = n_taps + 1 if n_taps % 2 == 0 else n_taps
    cutoff = (passband_hz + stopband_hz) / 2
    taps = firwin(n_taps, cutoff, window=("kaiser", beta), fs=sfreq)
    return FirDesign("antialias", taps, sfreq, {
        "type": "lowpass", "window": "kaiser", "beta": float(beta), "factor": factor,
        "passband_hz": passband_hz, "stopband_hz": stopband_hz,
        "attenuation_db": attenuation_db, "cutoff_minus6db_hz": cutoff})


def decimate(x: np.ndarray, design: FirDesign, factor: int) -> np.ndarray:
    """Polyphase FIR decimation along the last axis; output k sits on input k*factor.

    Output length is ceil(T / factor). The first and last
    ``ceil(design.half_support / factor)`` output samples depend on padding.
    FIR ringing can produce small negative power values: they are reported by
    callers, never clipped.
    """
    return resample_poly(np.asarray(x, dtype=np.float64), 1, factor, axis=-1,
                         window=design.taps, padtype="line")


def output_times(n_input: int, factor: int, sfreq: float, t0: float = 0.0) -> np.ndarray:
    return t0 + np.arange(math.ceil(n_input / factor)) * factor / sfreq


# --------------------------------------------------------------------------- spec

@dataclass(frozen=True)
class PreprocessingSpec:
    subject: str
    channels: ChannelMap
    sfreq: float
    target_sfreq: float
    factor: int
    bandpass: FirDesign
    notch: FirDesign
    fir: FirDesign
    freqs: np.ndarray
    n_cycles: float
    zero_mean: bool
    antialias: FirDesign
    glove_factor: int
    delay_samples: int

    @classmethod
    def from_config(cls, config: ExperimentConfig, subject: str) -> "PreprocessingSpec":
        data = config.data
        signal = data["signal"]
        filtering = signal["filtering"]
        resample = signal["resample"]
        sfreq = float(data["raw_data"]["ecog_fs_hz"])
        target = float(signal["target_fs_hz"])
        factor = int(round(sfreq / target))
        n_original = data["raw_data"]["expected_channels_nominal"][subject]
        cmap = channel_map(subject, n_original, config.excluded_channels(subject))
        bandpass = design_bandpass(sfreq, signal["bandpass_hz"], filtering["bandpass_l_trans_hz"],
                                   filtering["bandpass_h_trans_hz"], filtering["fir_window"])
        notch = design_notch(sfreq, signal["notch_hz"], filtering["notch_width_hz"],
                             filtering["notch_trans_hz"], filtering["fir_window"])
        aa = resample["antialias_fir"]
        antialias = design_antialias(sfreq, factor, aa["passband_hz"], aa["stopband_hz"], aa["attenuation_db"])
        morlet = signal["morlet"]
        freqs = morlet_frequencies(morlet["freq_range_hz"], morlet["n_frequencies"], morlet["spacing"])
        freqs.setflags(write=False)
        return cls(subject, cmap, sfreq, target, factor, bandpass, notch, combined_fir(bandpass, notch),
                   freqs, float(morlet["n_cycles"]), bool(morlet["zero_mean"]), antialias,
                   int(signal["glove"]["storage_factor"]),
                   int(signal["temporal_alignment"]["delay_samples_at_target_fs"]))


def morlet_power_decimated(x: np.ndarray, spec: PreprocessingSpec, freqs: np.ndarray | None = None,
                           n_cycles: float | None = None,
                           out_dtype: type = np.float32) -> tuple[np.ndarray, dict[str, Any]]:
    """Morlet power at the source rate then antialiased decimation, one channel at a
    time (all channels at once would need ~16 GB for a 400 s subject).

    Returns (power (C, F, ceil(T/factor)), report with negative-value counts).
    """
    freqs = spec.freqs if freqs is None else np.asarray(freqs, dtype=float)
    n_cycles = spec.n_cycles if n_cycles is None else n_cycles
    n_out = math.ceil(x.shape[1] / spec.factor)
    out = np.empty((x.shape[0], freqs.size, n_out), dtype=out_dtype)
    negative = 0
    for channel in range(x.shape[0]):
        power = morlet_power(x[channel:channel + 1], spec.sfreq, freqs, n_cycles, spec.zero_mean)[0]
        decimated = decimate(power, spec.antialias, spec.factor)
        negative += int(np.count_nonzero(decimated < 0))
        out[channel] = decimated
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite Morlet power after decimation")
    return out, {"negative_values": negative, "total_values": int(out.size),
                 "dtype": np.dtype(out_dtype).name, "magnitude": "power |W|^2"}


def support_report(spec: PreprocessingSpec) -> dict[str, Any]:
    """Analytic half-support of the chain. Informational: purge stays pending."""
    morlet = morlet_half_support(spec.freqs, spec.n_cycles, spec.sfreq)
    total_source = spec.fir.half_support + int(morlet.max()) + spec.antialias.half_support
    total_target = math.ceil(total_source / spec.factor)
    return {"fir_half_support_samples_source": spec.fir.half_support,
            "morlet_half_support_samples_source_max": int(morlet.max()),
            "morlet_half_support_samples_source_by_frequency": morlet.tolist(),
            "antialias_half_support_samples_source": spec.antialias.half_support,
            "total_half_support_samples_source": total_source,
            "total_half_support_seconds": total_source / spec.sfreq,
            "total_half_support_samples_target": total_target,
            "delay_samples_target": spec.delay_samples,
            "note": "Analytic support of exclusion -> CMR -> FIR -> Morlet -> antialias decimation. "
                    "CMR is instantaneous. The approved purge remains pending (V04/V09)."}


def chain_half_support(spec: PreprocessingSpec) -> int:
    """Source samples on each side that one decimated feature depends on (exact bound)."""
    return int(support_report(spec)["total_half_support_samples_source"])


def feature_valid_mask(n_source: int, spec: PreprocessingSpec, half_support: int | None = None) -> np.ndarray:
    """Decimated columns of a segment of ``n_source`` samples whose whole chain
    context lies inside it; column k sits on source sample k*factor."""
    half = chain_half_support(spec) if half_support is None else half_support
    position = np.arange(math.ceil(n_source / spec.factor)) * spec.factor
    return (position - half >= 0) & (position + half <= n_source - 1)


def measured_support(spec: PreprocessingSpec, tolerance: float) -> dict[str, Any]:
    """Propagate a unit impulse through FIR -> Morlet -> antialias decimation."""
    analytic = support_report(spec)
    # Place the impulse on a decimated sample so lag 0 is exact.
    half = (math.ceil(analytic["total_half_support_samples_source"] / spec.factor) + 10) * spec.factor
    n = 2 * half + 1
    impulse = np.zeros((1, n))
    impulse[0, half] = 1.0
    power = morlet_power(apply_fir(impulse, spec.fir), spec.sfreq, spec.freqs, spec.n_cycles, spec.zero_mean)[0]
    decimated = decimate(power, spec.antialias, spec.factor)
    response = np.abs(decimated).max(axis=0)
    response = response / response.max()
    centre = half // spec.factor
    lags = np.arange(response.size) - centre
    above = np.flatnonzero(response > tolerance)
    measured = int(max(abs(lags[above[0]]), abs(lags[above[-1]])))
    return {"lags_target": lags, "response": response, "measured_half_support_target": measured,
            "analytic": analytic, "tolerance": tolerance}


def segment_features(raw: np.ndarray, spec: PreprocessingSpec,
                     out_dtype: type = np.float32) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Full ECoG chain on ONE raw segment (C_original, n): exclusion -> CMR -> FIR ->
    Morlet power -> antialiased decimation.

    Returns (x (C_kept, F, ceil(n/factor)), mask of columns whose context lies
    inside the segment, report). Negative power from FIR ringing is counted on
    valid columns and never clipped.
    """
    if raw.ndim != 2 or raw.shape[0] != spec.channels.n_original:
        raise DataContractError(f"{spec.subject}: expected ({spec.channels.n_original}, n) raw ECoG; got {raw.shape}")
    referenced = common_median_reference(exclude_channels(raw, spec.channels))
    power, report = morlet_power_decimated(apply_fir(referenced, spec.fir), spec, out_dtype=out_dtype)
    valid = feature_valid_mask(raw.shape[1], spec)
    report.update(n_source=int(raw.shape[1]), n_columns=int(valid.size), valid_columns=int(valid.sum()),
                  negative_values_valid=int(np.count_nonzero(power[:, :, valid] < 0)),
                  values_valid=int(power[:, :, valid].size))
    return power, valid, report


# --------------------------------------------------------------------------- glove

GLOVE_METHODS = ("zero_order_hold", "linear", "cubic_not_a_knot", "pchip")


def _reconstruct(knots: np.ndarray, values: np.ndarray, positions: np.ndarray, method: str) -> np.ndarray:
    if method == "zero_order_hold":
        index = np.searchsorted(knots, positions, side="right") - 1
        return values[:, index]
    if method == "linear":
        return np.vstack([np.interp(positions, knots, row) for row in values])
    if method == "cubic_not_a_knot":
        return CubicSpline(knots, values, axis=1, bc_type="not-a-knot")(positions)
    if method == "pchip":
        return PchipInterpolator(knots, values, axis=1)(positions)
    raise ValueError(method)


def inspect_glove_storage(dg: np.ndarray, factor: int, tolerance: float = 1e-6) -> dict[str, Any]:
    """Identify how 25 Hz glove samples were stored on the 1 kHz grid.

    For every phase and candidate upsampling method, rebuild the stored signal
    from the samples at ``phase::factor`` and measure the relative RMS error.
    The stored pattern is identified only when one (method, phase) is exact.
    """
    dg = np.asarray(dg, dtype=np.float64)
    if dg.ndim != 2 or dg.shape[1] < 3 * factor:
        raise DataContractError("Glove storage inspection needs (fingers, T) with T >= 3*factor")
    scale = np.maximum(dg.std(axis=1, keepdims=True), 1e-12)
    errors = np.full((len(GLOVE_METHODS), factor), np.inf)
    for phase in range(factor):
        knots = np.arange(phase, dg.shape[1], factor)
        if knots.size < 4:
            continue
        positions = np.arange(knots[0], knots[-1] + 1)
        stored = dg[:, positions]
        for m, method in enumerate(GLOVE_METHODS):
            rebuilt = _reconstruct(knots, dg[:, knots], positions, method)
            errors[m, phase] = float(np.sqrt(np.mean(((rebuilt - stored) / scale) ** 2)))
    m_best, phase_best = np.unravel_index(np.argmin(errors), errors.shape)
    best_error = float(errors[m_best, phase_best])
    # Run lengths of identical consecutive rows (meaningful for sample-and-hold storage).
    change = np.flatnonzero(np.any(np.diff(dg, axis=1) != 0, axis=0)) + 1
    boundaries = np.r_[0, change, dg.shape[1]]
    run_lengths = np.diff(boundaries)
    residues = change % factor
    phase_fraction = float(np.mean(residues == phase_best)) if change.size else float("nan")
    identified = best_error <= tolerance
    return {"identified": bool(identified),
            "method": GLOVE_METHODS[m_best] if identified else "unknown",
            "best_candidate_method": GLOVE_METHODS[m_best], "phase": int(phase_best),
            "relative_rms_error": best_error, "tolerance": tolerance, "factor": factor,
            "errors_by_method_phase": {method: errors[m].tolist() for m, method in enumerate(GLOVE_METHODS)},
            "run_lengths": run_lengths, "change_residues": residues,
            "change_fraction_at_phase": phase_fraction}


def recover_glove_samples(dg: np.ndarray, factor: int, phase: int, sfreq: float,
                          inspection: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Effective glove samples and their physical times (first row of each hold).

    Rows before ``phase`` belong to an unrecorded previous block and are dropped.
    """
    if inspection is not None and not inspection["identified"]:
        raise DataContractError(
            f"Glove storage pattern not identified (best {inspection['best_candidate_method']}, "
            f"phase {inspection['phase']}, relative RMS {inspection['relative_rms_error']:.3g}); "
            "a documented adapter is required")
    if not 0 <= phase < factor:
        raise ValueError("phase must lie in [0, factor)")
    index = np.arange(phase, dg.shape[1], factor)
    return np.array(dg[:, index], dtype=np.float64), index / sfreq


def interpolate_glove(t_source: np.ndarray, y_source: np.ndarray,
                      t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubic (not-a-knot) spline on physical time for ONE segment, no extrapolation.

    A spline has global support: call it per segment so that labels from another
    partition can never shape these values.
    """
    spline = CubicSpline(np.asarray(t_source, float), np.asarray(y_source, float), axis=1,
                         bc_type="not-a-knot", extrapolate=False)
    values = spline(np.asarray(t_target, float))
    valid = np.isfinite(values).all(axis=0)
    return values, valid


def glove_storage_inspection(values: np.ndarray, storage_fs: float, spec: PreprocessingSpec) -> dict[str, Any]:
    """How the stored train_dg (5, T) relates to the effective 25 Hz samples."""
    if storage_fs == spec.sfreq / spec.glove_factor:
        return {"identified": True, "method": "effective_rate_grid", "phase": 0,
                "best_candidate_method": "effective_rate_grid", "relative_rms_error": 0.0,
                "factor": 1, "tolerance": 0.0, "errors_by_method_phase": {},
                "run_lengths": np.array([]), "change_residues": np.array([]),
                "change_fraction_at_phase": float("nan")}
    if storage_fs != spec.sfreq:
        raise DataContractError(f"Glove storage rate {storage_fs} Hz is neither the ECoG nor the effective rate")
    return inspect_glove_storage(values, spec.glove_factor)


def effective_glove(values: np.ndarray, inspection: dict[str, Any],
                    spec: PreprocessingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Effective glove samples (5, K) float64 and their indices on the source grid.

    Sample j of a glove stored at its effective rate sits on source index
    j*glove_factor; a glove stored on the ECoG grid needs an exact inspection.
    """
    if inspection["method"] == "effective_rate_grid":
        y25 = np.array(values, dtype=np.float64)
        return y25, np.arange(y25.shape[1]) * spec.glove_factor
    y25, _ = recover_glove_samples(values, spec.glove_factor, inspection["phase"], spec.sfreq, inspection)
    return y25, np.arange(inspection["phase"], values.shape[1], spec.glove_factor)


def segment_glove(y25: np.ndarray, knot_source: np.ndarray, start_source: int, stop_source: int,
                  spec: PreprocessingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Glove on the decimated grid of ONE segment [start_source, stop_source), built
    only from the effective samples inside it.

    Knots and targets use integer source indices, so a target that shares a
    sample with a knot reproduces it exactly. Returns (y (5, ceil(n/factor))
    float64 with NaN where no interpolation exists, valid mask).
    """
    inside = (knot_source >= start_source) & (knot_source < stop_source)
    if int(inside.sum()) < 4:
        raise DataContractError(f"Segment [{start_source}, {stop_source}) holds {int(inside.sum())} glove samples; "
                                "the cubic spline needs at least 4")
    targets = start_source + np.arange(math.ceil((stop_source - start_source) / spec.factor)) * spec.factor
    return interpolate_glove(knot_source[inside].astype(np.float64), y25[:, inside], targets.astype(np.float64))


# --------------------------------------------------------------------------- alignment

def align_delay(x: np.ndarray, y: np.ndarray, delay: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair x(t) with y(t + delay) on a shared grid: X(t) -> Y(t + delay).

    Returns (x_aligned, y_aligned, x_index, y_index). ``delay`` is in samples.
    """
    if type(delay) is not int and not isinstance(delay, np.integer):
        raise TypeError("delay must be an integer number of samples")
    if delay < 0:
        raise ValueError("Negative delays would pair the target with future ECoG; not allowed")
    n = x.shape[-1]
    if y.shape[-1] != n:
        raise ValueError("x and y must share the time grid")
    keep = max(n - int(delay), 0)
    x_index = np.arange(keep)
    return x[..., :keep], y[..., int(delay):int(delay) + keep], x_index, x_index + int(delay)


# --------------------------------------------------------------------------- scaling

def features_to_matrix(features: np.ndarray) -> np.ndarray:
    """(C, F, T) -> (T, C*F); column index is c*F + f."""
    c, f, t = features.shape
    return features.reshape(c * f, t).T


def matrix_to_features(matrix: np.ndarray, n_channels: int, n_frequencies: int) -> np.ndarray:
    return matrix.T.reshape(n_channels, n_frequencies, matrix.shape[0])


def make_feature_scaler(config: ExperimentConfig):
    """Unfitted RobustScaler; fitting happens per fold on training data only."""
    from sklearn.preprocessing import RobustScaler
    ecog = config.data["scaling"]["ecog"]
    return RobustScaler(quantile_range=tuple(float(q) for q in ecog["quantile_range_percent"]),
                        unit_variance=bool(ecog["unit_variance"]))


def make_target_scaler(config: ExperimentConfig):
    from sklearn.preprocessing import MinMaxScaler
    targets = config.data["scaling"]["targets"]
    return MinMaxScaler(feature_range=tuple(float(v) for v in targets["feature_range"]))


def restore_feature_scaler(config: ExperimentConfig, center: np.ndarray, scale: np.ndarray):
    """RobustScaler with persisted per-column center and scale (fold-train fit)."""
    center, scale = np.asarray(center, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if center.ndim != 1 or center.shape != scale.shape:
        raise ValueError("Feature scaler parameters must be aligned one-dimensional arrays")
    scaler = make_feature_scaler(config)
    scaler.center_, scaler.scale_, scaler.n_features_in_ = center, scale, int(center.size)
    return scaler


def restore_target_scaler(config: ExperimentConfig, data_min: np.ndarray, data_max: np.ndarray):
    """MinMaxScaler from persisted per-finger minima and maxima: fitting on those two
    rows reproduces every learned parameter of the fold-train fit."""
    return make_target_scaler(config).fit(np.vstack([data_min, data_max]).astype(np.float64))
