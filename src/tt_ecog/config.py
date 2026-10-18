"""Configuration validation for the EDA stage; pending values stay pending."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import math

import yaml


class ConfigError(ValueError):
    """Invalid or unsafe stage configuration."""


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


@dataclass(frozen=True)
class ExperimentConfig:
    data: dict[str, Any]
    source: Path
    sha256: str

    def storage_root(self, drive_root: Path) -> Path:
        """drive_root is MyDrive (or an explicit local fixture root)."""
        return inside(drive_root, self.data["paths"]["persistent_storage_dir"])

    def path(self, drive_root: Path, name: str) -> Path:
        if name not in {"raw_dir", "processed_dir", "manifests_dir", "results_dir"}:
            raise ConfigError(f"EDA cannot resolve path {name!r}")
        return inside(self.storage_root(drive_root), self.data["paths"][name])


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
        if data["project"]["subjects"] != ["S1", "S2", "S3"]:
            raise ConfigError("Expected independent subjects S1, S2, S3")
        if data["project"]["development_only_until_final_evaluation"] is not True:
            raise ConfigError("Development-only access is mandatory")
        paths = data["paths"]
        for name in ("persistent_storage_dir", "raw_dir", "processed_dir", "manifests_dir", "results_dir"):
            relative_path(paths[name], name)
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
        qc = data["quality_control"]
        if qc["use_labels_for_detection"] is not False or qc["approval_required"] is not True:
            raise ConfigError("QC must exclude labels and require human approval")
        if (qc["detection_stage"] != "eda" or qc["application_stage"] != "preprocessing"
                or qc["index_convention"] != "zero_based"):
            raise ConfigError("QC stages and zero-based indices must follow the protocol")
        if qc["detection_methods"] != ["robust_amplitude", "discontinuity", "flatline", "spectral", "event_grouping"]:
            raise ConfigError("Unexpected QC detector list")
        for name in ("robust_z", "jump_z", "flatline_fraction", "max_bad_fraction", "spectral_z", "spectral_min_duration_s"):
            value = qc["thresholds"][name]
            if value is not None:
                positive(value, name, zero=name in {"flatline_fraction", "max_bad_fraction"})
                if name.endswith("fraction") and value > 1:
                    raise ConfigError(f"{name}: expected a fraction in [0,1]")
        analysis = qc["analysis"]
        for name in ("window_seconds", "psd_window_seconds", "line_half_width_hz", "numerical_epsilon"):
            positive(analysis[name], name)
        positive(analysis["event_merge_gap_seconds"], "event_merge_gap_seconds", zero=True)
        overlap = analysis["psd_overlap_fraction"]
        positive(overlap, "psd_overlap_fraction", zero=True)
        if overlap >= 1:
            raise ConfigError("psd_overlap_fraction must be <1")
        if type(analysis["plot_max_points"]) is not int or analysis["plot_max_points"] < 10:
            raise ConfigError("plot_max_points must be an integer >=10")
        for key in ("window_seconds", "psd_window_seconds"):
            if analysis[key] * raw["ecog_fs_hz"] < 2:
                raise ConfigError(f"{key}: requires at least two samples")
        if data["signal"]["bandpass_hz"] != [40, 200] or data["signal"]["notch_hz"] != [60, 120, 180]:
            raise ConfigError("EDA spectral diagnostics must use the protocol band and line frequencies")
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Missing or malformed EDA configuration: {exc}") from exc
    return ExperimentConfig(data, path, hashlib.sha256(payload).hexdigest())
