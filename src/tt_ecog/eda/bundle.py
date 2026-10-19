"""Per-subject data and cached derivatives, always computed with ``preprocessing``.

The EDA processes the whole development record as ONE segment and only in
memory, to draw figures. Nothing here is persisted as processed data and
nothing feeds notebook 02, which recomputes from raw data per fold segment.
"""

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from .. import preprocessing as pp
from ..config import ExperimentConfig
from ..descriptive import band_line_fraction, robust_scores, welch_psd, windowed_std
from ..io import DevelopmentRecord
from ..splits import chronological_blocks
from .movement import MovementInfo, detect_movement


class SubjectBundle:
    def __init__(self, config: ExperimentConfig, ecog: DevelopmentRecord, glove: DevelopmentRecord):
        self.config = config
        self.subject = ecog.subject
        self.ecog_record = ecog
        self.glove_record = glove
        self.spec = pp.PreprocessingSpec.from_config(config, self.subject)
        self.eda = config.data["eda"]
        self.signal = config.data["signal"]
        self.fs = self.spec.sfreq
        self.raw = ecog.values
        self.n_samples = self.raw.shape[1]
        self.duration_s = self.n_samples / self.fs
        if self.raw.shape[0] != self.spec.channels.n_original:
            raise ValueError(f"{self.subject}: channel count differs from the nominal contract")

    # ------------------------------------------------------------- ECoG chain
    @property
    def cmap(self) -> pp.ChannelMap:
        return self.spec.channels

    @cached_property
    def kept(self) -> np.ndarray:
        return pp.exclude_channels(self.raw, self.cmap)

    @cached_property
    def cmr(self) -> np.ndarray:
        return pp.common_median_reference(self.kept)

    @cached_property
    def filtered(self) -> np.ndarray:
        return pp.apply_fir(self.cmr, self.spec.fir)

    @cached_property
    def _power(self) -> tuple[np.ndarray, dict[str, Any]]:
        return pp.morlet_power_decimated(self.filtered, self.spec)

    @property
    def power100(self) -> np.ndarray:
        return self._power[0]

    @property
    def power_report(self) -> dict[str, Any]:
        return self._power[1]

    @cached_property
    def t100(self) -> np.ndarray:
        return pp.output_times(self.n_samples, self.spec.factor, self.fs)

    @cached_property
    def edge100(self) -> int:
        """Target samples at each record edge that depend on padding."""
        return pp.support_report(self.spec)["total_half_support_samples_target"]

    @cached_property
    def hg100(self) -> np.ndarray:
        """log10 of the 40-200 Hz band-mean power, (C_kept, T100)."""
        return np.log10(np.maximum(self.power100.mean(axis=1, dtype=np.float64),
                                   self.eda["numerical_epsilon"]))

    # ------------------------------------------------------------- glove
    @cached_property
    def glove_inspection(self) -> dict[str, Any]:
        meta = self.glove_record.metadata
        if meta["sampling_hz_glove_storage"] == meta["sampling_hz_glove_effective"]:
            return {"identified": True, "method": "effective_rate_grid", "phase": 0,
                    "best_candidate_method": "effective_rate_grid", "relative_rms_error": 0.0,
                    "factor": 1, "tolerance": 0.0, "errors_by_method_phase": {},
                    "run_lengths": np.array([]), "change_residues": np.array([]),
                    "change_fraction_at_phase": float("nan")}
        return pp.inspect_glove_storage(self.glove_record.values, self.spec.glove_factor)

    @cached_property
    def glove25(self) -> tuple[np.ndarray, np.ndarray]:
        info = self.glove_inspection
        if info["method"] == "effective_rate_grid":
            values = np.array(self.glove_record.values, dtype=np.float64)
            return values, np.arange(values.shape[1]) / self.signal["glove"]["effective_fs_hz"]
        return pp.recover_glove_samples(self.glove_record.values, self.spec.glove_factor,
                                        info["phase"], self.fs, info)

    @cached_property
    def glove100(self) -> tuple[np.ndarray, np.ndarray]:
        y25, t25 = self.glove25
        return pp.interpolate_glove(t25, y25, self.t100)

    @cached_property
    def movement(self) -> MovementInfo:
        y25, t25 = self.glove25
        movement = self.eda["movement"]
        return detect_movement(y25, t25, self.t100, movement["speed_quantile"],
                               movement["min_interonset_seconds"])

    @cached_property
    def valid100(self) -> np.ndarray:
        """Target samples with interpolated glove and interior ECoG features."""
        valid = self.glove100[1].copy()
        valid[:self.edge100] = False
        valid[len(valid) - self.edge100:] = False
        return valid

    # ------------------------------------------------------------- descriptive
    @cached_property
    def local_std(self) -> tuple[np.ndarray, np.ndarray]:
        window = int(round(self.eda["window_seconds"] * self.fs))
        return windowed_std(self.raw, window)

    def _psd(self, values: np.ndarray, window_seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        return welch_psd(values, self.fs, window_seconds or self.eda["psd_window_seconds"],
                         self.eda["psd_overlap_fraction"])

    @cached_property
    def raw_psd(self) -> tuple[np.ndarray, np.ndarray]:
        return self._psd(self.raw)

    @cached_property
    def cmr_psd(self) -> tuple[np.ndarray, np.ndarray]:
        return self._psd(self.cmr)

    @cached_property
    def filtered_psd(self) -> tuple[np.ndarray, np.ndarray]:
        return self._psd(self.filtered)

    def line_fraction(self, freq: np.ndarray, psd: np.ndarray) -> np.ndarray:
        return band_line_fraction(freq, psd, self.signal["bandpass_hz"], self.signal["notch_hz"],
                                  self.eda["line_half_width_hz"], self.eda["numerical_epsilon"])[1]

    @cached_property
    def channel_summary(self) -> pd.DataFrame:
        stds, _ = self.local_std
        typical = np.median(stds, axis=1)
        ratio = stds / np.maximum(typical[:, None], self.eda["numerical_epsilon"])
        freq, psd = self.raw_psd
        excluded = set(self.cmap.excluded)
        rows = []
        for channel in range(self.raw.shape[0]):
            values = self.raw[channel].astype(np.float64)
            scores, median, mad = robust_scores(values, self.eda["numerical_epsilon"])
            row = {"subject": self.subject, "channel_0based": channel, "channel_1based": channel + 1,
                   "excluded": channel in excluded, "median": median, "mad": mad,
                   "max_amplitude_robust_z": float(scores.max()),
                   "max_abs_jump": float(np.abs(np.diff(values)).max()),
                   "typical_local_std": float(typical[channel]),
                   "peak_local_std": float(stds[channel].max()),
                   "peak_over_typical": float(ratio[channel].max())}
            for level in self.eda["std_ratio_levels"]:
                row[f"fraction_windows_above_{level}x"] = float(np.mean(ratio[channel] > level))
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame["line_fraction_raw"] = self.line_fraction(freq, psd)
        return frame

    @cached_property
    def std_ratio(self) -> np.ndarray:
        stds, _ = self.local_std
        typical = np.median(stds, axis=1, keepdims=True)
        return stds / np.maximum(typical, self.eda["numerical_epsilon"])

    @cached_property
    def blocks(self):
        validation = self.config.data["validation"]
        return chronological_blocks(self.n_samples, validation["block_names"], self.spec.factor)

    @cached_property
    def descriptive_channel(self) -> int:
        """Effective index of the kept channel whose high-gamma power correlates best
        with flexion at the protocol delay. Chosen only to illustrate figures; it
        drives no decision."""
        y100, _ = self.glove100
        x, y, _, _ = pp.align_delay(self.hg100, y100, self.spec.delay_samples)
        valid = self.valid100[:x.shape[-1]] & self.valid100[self.spec.delay_samples:]
        x, y = x[:, valid], y[:, valid]
        x = (x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-12)
        y = (y - y.mean(axis=1, keepdims=True)) / np.maximum(y.std(axis=1, keepdims=True), 1e-12)
        corr = np.abs(x @ y.T / x.shape[1])
        return int(np.argmax(corr.max(axis=1)))

    def excerpt(self, seconds: float, center_s: float | None = None) -> slice:
        n = min(self.n_samples, int(round(seconds * self.fs)))
        center = self.n_samples // 2 if center_s is None else int(round(center_s * self.fs))
        start = int(np.clip(center - n // 2, 0, self.n_samples - n))
        return slice(start, start + n)
