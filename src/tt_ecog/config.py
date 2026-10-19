"""Configuration loading and validation; pending values stay pending."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import math
import re

import yaml


class ConfigError(ValueError):
    """Invalid or unsafe stage configuration."""


RUN_ID = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{6}$")
PURGE_CANDIDATE_STATUS = "candidate_pending_V04_V09"


def relative_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}: expected a nonempty relative path")
    path = Path(value)
    # Reject Windows paths also when validating on Linux/Colab.
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ConfigError(f"{label}: must be a portable relative path without '..'")
    return path


def inside(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    value = relative.as_posix() if isinstance(relative, Path) else relative
    path = (root / relative_path(value, "path")).resolve()
    if not path.is_relative_to(root):
        raise ConfigError("Path escapes its configured root")
    return path


def positive(value: Any, name: str, *, zero: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ConfigError(f"{name}: expected a finite {'nonnegative' if zero else 'positive'} number")


def _positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ConfigError(f"{name}: expected a positive integer")


def _interval(value: Any, name: str, *, ordered: bool = True) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name}: expected [start, end]")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ConfigError(f"{name}: expected finite numbers")
    if ordered and not value[0] < value[1]:
        raise ConfigError(f"{name}: start must be lower than end")


RESOLVABLE_PATHS = {"raw_dir", "processed_dir", "manifests_dir", "results_dir", "eda_dir", "preprocessing_dir",
                    "verification_dir", "training_dir", "optimization_dir", "evaluation_dir"}
VERIFICATION_IDS = [f"V{k:02d}" for k in range(1, 15)]


@dataclass(frozen=True)
class ExperimentConfig:
    data: dict[str, Any]
    source: Path
    sha256: str

    def storage_root(self, drive_root: Path) -> Path:
        """drive_root is MyDrive (or an explicit local fixture root)."""
        return inside(drive_root, self.data["paths"]["persistent_storage_dir"])

    def path(self, drive_root: Path, name: str) -> Path:
        if name not in RESOLVABLE_PATHS:
            raise ConfigError(f"Path {name!r} cannot be resolved before its stage")
        return inside(self.storage_root(drive_root), self.data["paths"][name])

    def channel_exclusions(self, subject: str) -> list[dict[str, Any]]:
        return list(self.data["channels"]["excluded"][subject])

    def excluded_channels(self, subject: str) -> tuple[int, ...]:
        return tuple(sorted(entry["index"] for entry in self.channel_exclusions(subject)))

    def approved_purge(self) -> int:
        purge = self.data["validation"]["purge_samples_at_100hz"]
        if purge is None:
            raise ConfigError("validation.purge_samples_at_100hz is pending (null); approve it after V04/V09")
        return purge

    def purge_for_build(self) -> tuple[int, str]:
        """Purge used to build fold artifacts and its status. While the purge is null
        this is the verification candidate, always labelled as such: never an approval."""
        validation = self.data["validation"]
        if validation["purge_samples_at_100hz"] is not None:
            return validation["purge_samples_at_100hz"], "approved"
        return validation["purge_candidate_for_tests"], PURGE_CANDIDATE_STATUS


def _validate_channels(channels: dict, nominal: dict[str, int], subjects: list[str]) -> None:
    expected = {"index_convention": "zero_based",
                "application_stage": "preprocessing_before_common_reference",
                "selection_basis": "ecog_acquisition_evidence_only"}
    for key, value in expected.items():
        if channels[key] != value:
            raise ConfigError(f"channels.{key}: expected {value!r}")
    excluded = channels["excluded"]
    if not isinstance(excluded, dict) or list(excluded) != subjects:
        raise ConfigError("channels.excluded: expected exactly the keys S1, S2, S3 in order")
    for subject, entries in excluded.items():
        if not isinstance(entries, list):
            raise ConfigError(f"channels.excluded.{subject}: expected a list")
        indices = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"index", "reason", "evidence_figure"}:
                raise ConfigError(f"channels.excluded.{subject}: entries need index, reason, evidence_figure")
            index = entry["index"]
            if type(index) is not int or not 0 <= index < nominal[subject]:
                raise ConfigError(f"channels.excluded.{subject}: index outside 0..{nominal[subject] - 1}")
            if not isinstance(entry["reason"], str) or not entry["reason"].strip():
                raise ConfigError(f"channels.excluded.{subject}[{index}]: empty reason")
            relative_path(entry["evidence_figure"], f"channels.excluded.{subject}[{index}].evidence_figure")
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ConfigError(f"channels.excluded.{subject}: duplicated index")
        if indices != sorted(indices):
            raise ConfigError(f"channels.excluded.{subject}: indices must be sorted")
        if len(indices) >= nominal[subject]:
            raise ConfigError(f"channels.excluded.{subject}: cannot exclude every channel")


def _validate_signal(signal: dict, raw: dict) -> None:
    if signal["bandpass_hz"] != [40, 200] or signal["notch_hz"] != [60, 120, 180]:
        raise ConfigError("signal: protocol band 40-200 Hz and lines 60/120/180 Hz are fixed")
    if signal["common_reference"] != "common_median_instantaneous":
        raise ConfigError("signal.common_reference must be common_median_instantaneous")
    morlet = signal["morlet"]
    if (morlet["n_frequencies"] != 40 or morlet["freq_range_hz"] != [40, 200] or morlet["n_cycles"] != 7
            or morlet["zero_mean"] is not True or morlet["spacing"] != "log" or morlet["output"] != "power"):
        raise ConfigError("signal.morlet: 40 log frequencies 40-200 Hz, 7 cycles, zero_mean, power are fixed")
    if morlet["implementation"] != "mne_tfr_array_morlet":
        raise ConfigError("signal.morlet.implementation: expected mne_tfr_array_morlet")
    filtering = signal["filtering"]
    if filtering["phase"] != "zero" or filtering["segmentwise"] is not True:
        raise ConfigError("signal.filtering: zero-phase segmentwise filtering is required")
    if filtering["fir_window"] != "hamming" or filtering["library"] != "scipy_firwin":
        raise ConfigError("signal.filtering: expected scipy_firwin with a Hamming window")
    for key in ("bandpass_l_trans_hz", "bandpass_h_trans_hz", "notch_width_hz", "notch_trans_hz"):
        positive(filtering[key], f"signal.filtering.{key}")
    source = raw["ecog_fs_hz"]
    target = signal["target_fs_hz"]
    if signal["resample"]["source_fs_hz"] != source or signal["resample"]["target_fs_hz"] != target:
        raise ConfigError("signal.resample rates must match raw_data and target_fs_hz")
    if source % target:
        raise ConfigError("Source rate must be an integer multiple of the target rate")
    aa = signal["resample"]["antialias_fir"]
    if aa["window"] != "kaiser":
        raise ConfigError("signal.resample.antialias_fir.window: expected kaiser")
    for key in ("passband_hz", "stopband_hz", "attenuation_db"):
        positive(aa[key], f"antialias_fir.{key}")
    if not aa["passband_hz"] < aa["stopband_hz"] <= target / 2:
        raise ConfigError("antialias_fir: passband < stopband <= target Nyquist required")
    glove = signal["glove"]
    if glove["effective_fs_hz"] != raw["glove_fs_effective_hz"] or glove["target_fs_hz"] != target:
        raise ConfigError("signal.glove rates are inconsistent")
    if glove["storage_factor"] != source // raw["glove_fs_effective_hz"]:
        raise ConfigError("signal.glove.storage_factor must equal ecog_fs / glove_fs_effective")
    if glove["interpolation"] != "cubic" or glove["extrapolate"] is not False:
        raise ConfigError("signal.glove: cubic interpolation without extrapolation is required")
    alignment = signal["temporal_alignment"]
    if alignment["direction"] != "ecog_t_to_flexion_t_plus_delay":
        raise ConfigError("signal.temporal_alignment.direction is fixed: ECoG(t) -> flexion(t+delay)")
    if alignment["delay_samples_at_target_fs"] * 1000 != alignment["delay_ms"] * target:
        raise ConfigError("delay_samples_at_target_fs must equal delay_ms * target_fs / 1000")


def _validate_eda(eda: dict) -> None:
    if eda["development_labels_use"] != "descriptive_only":
        raise ConfigError("eda.development_labels_use must be descriptive_only")
    for key in ("window_seconds", "psd_window_seconds", "line_zoom_psd_window_seconds",
                "line_half_width_hz", "numerical_epsilon", "episode_context_seconds",
                "excerpt_seconds", "correlation_excerpt_seconds", "autocorr_max_lag_seconds"):
        positive(eda[key], f"eda.{key}")
    positive(eda["psd_overlap_fraction"], "eda.psd_overlap_fraction", zero=True)
    if eda["psd_overlap_fraction"] >= 1:
        raise ConfigError("eda.psd_overlap_fraction must be < 1")
    levels = eda["std_ratio_levels"]
    if not isinstance(levels, list) or not levels or levels != sorted(levels):
        raise ConfigError("eda.std_ratio_levels: expected a sorted nonempty list")
    for level in levels:
        positive(level, "eda.std_ratio_levels")
    _interval(eda["lag_range_seconds"], "eda.lag_range_seconds")
    if not eda["lag_range_seconds"][0] < 0 < eda["lag_range_seconds"][1]:
        raise ConfigError("eda.lag_range_seconds must include negative and positive lags")
    diag = eda["diagnostic_morlet"]
    _interval(diag["freq_range_hz"], "eda.diagnostic_morlet.freq_range_hz")
    _positive_int(diag["n_frequencies"], "eda.diagnostic_morlet.n_frequencies")
    positive(diag["n_cycles"], "eda.diagnostic_morlet.n_cycles")
    movement = eda["movement"]
    if not 0 < movement["speed_quantile"] < 1:
        raise ConfigError("eda.movement.speed_quantile must lie in (0, 1)")
    positive(movement["min_interonset_seconds"], "eda.movement.min_interonset_seconds")
    _interval(movement["ersp_window_seconds"], "eda.movement.ersp_window_seconds")
    _interval(movement["baseline_seconds"], "eda.movement.baseline_seconds")
    window, baseline = movement["ersp_window_seconds"], movement["baseline_seconds"]
    if not window[0] <= baseline[0] < baseline[1] <= min(0, window[1]):
        raise ConfigError("eda.movement.baseline_seconds must lie inside the pre-onset ERSP window")
    _positive_int(eda["figure_dpi"], "eda.figure_dpi")
    _positive_int(eda["plot_max_points"], "eda.plot_max_points")


def _validate_validation(validation: dict) -> None:
    if validation["design"] != "purged_blocked_cv":
        raise ConfigError("validation.design must be purged_blocked_cv")
    names = validation["block_names"]
    if validation["n_base_blocks"] != 5 or names != ["B1", "B2", "B3", "B4", "B5"]:
        raise ConfigError("validation: five chronological blocks B1-B5 are fixed")
    if validation["stage1_validation_blocks"] != ["B2", "B4"] or validation["stage2_validation_blocks"] != names:
        raise ConfigError("validation: stage 1 validates B2 and B4; stage 2 validates B1-B5")
    purge = validation["purge_samples_at_100hz"]
    if purge is not None and (type(purge) is not int or purge < 0):
        raise ConfigError("validation.purge_samples_at_100hz: expected null (pending) or an integer >= 0")
    candidate = validation["purge_candidate_for_tests"]
    if type(candidate) is not int or candidate < 0:
        raise ConfigError("validation.purge_candidate_for_tests: expected an integer >= 0")
    for key in ("train_window_samples", "train_stride_samples", "inference_window_samples",
                "inference_stride_samples"):
        _positive_int(validation[key], f"validation.{key}")
    if validation["train_window_samples"] != validation["inference_window_samples"]:
        raise ConfigError("validation: training and inference windows must have the same length")
    if validation["inference_stride_samples"] > validation["inference_window_samples"]:
        raise ConfigError("validation.inference_stride_samples cannot exceed the window (coverage gaps)")
    if validation["allow_windows_across_gaps"] is not False:
        raise ConfigError("validation.allow_windows_across_gaps must be false")
    if (validation["score_reconstruction"] != "overlap_add"
            or validation["coverage_policy"] != "full_valid_timestamps_required"):
        raise ConfigError("validation: overlap-add with full coverage of valid timestamps is fixed")


def _validate_scaling(scaling: dict) -> None:
    ecog, targets = scaling["ecog"], scaling["targets"]
    if (ecog["type"] != "robust" or ecog["quantile_range_percent"] != [10, 90] or ecog["unit_variance"] is not True
            or ecog["fit_only_on"] != "fold_train" or ecog["feature_axis"] != "channel_frequency"):
        raise ConfigError("scaling.ecog: RobustScaler(10, 90, unit_variance) per channel-frequency on fold train is fixed")
    if targets["type"] != "minmax" or targets["feature_range"] != [0, 1] or targets["fit_only_on"] != "fold_train":
        raise ConfigError("scaling.targets: MinMaxScaler [0, 1] on fold train is fixed")


def _validate_model(model: dict, signal: dict) -> None:
    fixed = {"name": "unet1d", "output_channels": 5, "normalization": "batchnorm1d", "activation": "gelu",
             "dropout": 0.3, "decoder": "interpolate_linear_plus_conv1d", "skip_connection": "concat",
             "interpolate_to_skip_length": True, "final_conv_kernel": 1, "temporal_downsampling": "maxpool1d",
             "decoder_block_order": "interpolate_conv_norm_activation_dropout_concat", "decoder_dilation": 1,
             "initialization": "pytorch_default"}
    for key, value in fixed.items():
        if model[key] != value:
            raise ConfigError(f"model.{key}: expected {value!r}")
    if model["input_frequency_bins"] != signal["morlet"]["n_frequencies"]:
        raise ConfigError("model.input_frequency_bins must equal the Morlet frequency count")
    kernel = model["spatial_reduce_kernel"]
    if type(kernel) is not int or kernel < 1 or kernel % 2 == 0:
        raise ConfigError("model.spatial_reduce_kernel: expected a positive odd integer")
    ref = model["reference"]
    if (ref["encoder_levels"] != 5 or ref["base_filters"] != 64 or ref["encoder_kernels"] != [7, 7, 5, 5, 5]
            or ref["encoder_dilations"] != [1, 2, 3, 1, 2] or ref["encoder_strides"] != [2, 2, 2, 2, 2]
            or ref["channel_multipliers"] != [1, 1, 2, 4, 8, 8]):
        raise ConfigError("model.reference: the protocol reference U-Net (k=5, L=5, f=64) is fixed")


def _validate_training_and_metrics(training: dict, metrics: dict) -> None:
    if (training["optimizer"] != "adamw" or training["scheduler"] != "none"
            or training["adamw_weight_decay_policy"] != "all_trainable_parameters"):
        raise ConfigError("training: AdamW on all trainable parameters, without a scheduler, is fixed")
    for key in ("batch_size_initial_candidate", "search_epochs_candidate", "profile_warmup_steps",
                "profile_measured_steps"):
        _positive_int(training[key], f"training.{key}")
    for key in ("batch_size", "search_epochs", "final_max_epochs"):
        if training[key] is not None:
            _positive_int(training[key], f"training.{key}")
    loss = training["loss"]
    if (loss["name"] != "mse_plus_temporal_cosine" or loss["mse_weight"] != 0.5 or loss["cosine_weight"] != 0.5
            or loss["cosine_axis"] != "temporal_per_finger"):
        raise ConfigError("training.loss: 0.5 MSE + 0.5 (1 - temporal cosine per finger) is fixed")
    positive(loss["eps"], "training.loss.eps")
    if (metrics["objective"] != "pearson_macro_five_fingers"
            or metrics["pearson_scope"] != "reconstructed_full_valid_trajectories"
            or metrics["pearson_constant_series_policy"] != "mark_invalid"
            or metrics["selection_is_multiobjective"] is not False):
        raise ConfigError("metrics: macro Pearson on reconstructed trajectories, undefined -> invalid, is fixed")
    positive(metrics["degenerate_std_relative"], "metrics.degenerate_std_relative")
    if metrics["degenerate_std_relative"] >= 1:
        raise ConfigError("metrics.degenerate_std_relative must be < 1")


def _validate_verification(verification: dict) -> None:
    if verification["required_test_ids"] != VERIFICATION_IDS:
        raise ConfigError("verification.required_test_ids must list V01-V14")
    if verification["all_pass_required"] is not True or verification["automatically_freeze_on_all_pass"] is not False:
        raise ConfigError("verification: every check must pass and freezing requires explicit approval")
    positive(verification["support_tolerance"], "verification.support_tolerance")
    positive(verification["isolation_tolerance"], "verification.isolation_tolerance", zero=True)
    positive(verification["reproducibility_rtol"], "verification.reproducibility_rtol")
    _positive_int(verification["architecture_batch_size"], "verification.architecture_batch_size")


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    payload = path.read_bytes()
    try:
        data = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ConfigError("Invalid YAML syntax") from exc
    try:
        if not isinstance(data, dict) or data["schema_version"] != 1:
            raise ConfigError("Expected configuration schema_version=1")
        subjects = data["project"]["subjects"]
        if subjects != ["S1", "S2", "S3"]:
            raise ConfigError("Expected independent subjects S1, S2, S3")
        if data["project"]["development_only_until_final_evaluation"] is not True:
            raise ConfigError("Development-only access is mandatory")
        paths = data["paths"]
        for name in ("persistent_storage_dir", *sorted(RESOLVABLE_PATHS)):
            relative_path(paths[name], name)
        selected = paths.get("processed_pipeline_id")
        if selected is not None and (not isinstance(selected, str) or not re.fullmatch(r"[0-9a-f]{64}", selected)):
            raise ConfigError("paths.processed_pipeline_id: expected null or a full SHA-256")
        raw = data["raw_data"]
        for name, value in {"ecog_fs_hz": 1000, "glove_fs_effective_hz": 25,
                            "n_fingers": 5, "train_ecog_key": "train_data",
                            "train_glove_key": "train_dg", "io_policy": "original_readonly"}.items():
            if raw[name] != value:
                raise ConfigError(f"raw_data.{name}: expected {value!r}")
        if raw["expected_channels_nominal"] != {"S1": 62, "S2": 48, "S3": 64}:
            raise ConfigError("Nominal channel counts require a documented adapter to change")
        if raw["train_file_template"] != "sub{subject_number}_comp.mat":
            raise ConfigError("Unsupported file layout; document an adapter first")
        if "quality_control" in data:
            raise ConfigError("quality_control was removed; channel exclusion lives in channels.excluded")
        _validate_channels(data["channels"], raw["expected_channels_nominal"], subjects)
        _validate_signal(data["signal"], raw)
        _validate_eda(data["eda"])
        _validate_validation(data["validation"])
        _validate_scaling(data["scaling"])
        _validate_model(data["model"], data["signal"])
        _validate_training_and_metrics(data["training"], data["metrics"])
        _validate_verification(data["verification"])
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Missing or malformed configuration: {exc}") from exc
    return ExperimentConfig(data, path, hashlib.sha256(payload).hexdigest())
