"""Read-only, allowlisted access to development variables in MATLAB files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import h5py
import numpy as np
from scipy.io import loadmat

from .config import ExperimentConfig, inside


class DataContractError(ValueError):
    """The supplied data do not match the documented development contract."""


class MissingDevelopmentData(FileNotFoundError):
    """A development file is absent: the run is BLOCKED, not FAILED."""


class StorageUnavailable(FileNotFoundError):
    """The persistent storage root (mounted Drive) is absent: BLOCKED."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DevelopmentRecord:
    subject: str
    values: np.ndarray  # (channels, time), readonly; original numerical units
    metadata: dict[str, Any]


def _read_variable(path: Path, key: str) -> tuple[np.ndarray, str]:
    """No generic MATLAB reader is exported to the pipeline."""
    if key not in {"train_data", "train_dg"}:
        raise PermissionError(f"EDA forbids MATLAB variable {key!r}")
    if "testlabels" in path.name.lower() or "testlabels" in path.resolve().name.lower():
        raise PermissionError("EDA forbids test-label files, including symlink targets")
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise DataContractError(f"Missing development variable: {key}")
            if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
                raise DataContractError("Development datasets must not follow HDF5 soft/external links")
            dataset = handle[key]
            if not isinstance(dataset, h5py.Dataset) or "MATLAB_class" not in dataset.attrs:
                raise DataContractError("HDF5 layout requires a MATLAB numeric dataset with MATLAB_class")
            if dataset.ndim != 2 or dataset.dtype.kind not in "fiu":
                raise DataContractError(f"{key}: expected a two-dimensional real numeric matrix")
            # MATLAB v7.3 stores dimensions in reverse order in HDF5.
            values = dataset[()].T
        return values, "matlab_v7.3_hdf5_reversed_axes"
    try:
        content = loadmat(path, variable_names=[key], squeeze_me=False)
    except (ValueError, TypeError, OSError) as exc:
        raise DataContractError(f"Cannot read MATLAB development variable {key}: {exc}") from exc
    if key not in content:
        raise DataContractError(f"Missing development variable: {key}")
    return content[key], "matlab_classic_samples_by_channels"


def development_path(config: ExperimentConfig, drive_root: Path, subject: str) -> Path:
    if subject not in config.data["project"]["subjects"]:
        raise DataContractError(f"Unknown subject: {subject}")
    filename = config.data["raw_data"]["train_file_template"].format(subject_number=subject[1:])
    path = inside(config.path(drive_root, "raw_dir"), filename)
    if "testlabels" in path.name.lower():
        raise PermissionError("Development path resolves to test labels")
    if not path.is_file():
        raise MissingDevelopmentData(f"Missing development file for {subject}: {filename}")
    return path


def _matrix(values: np.ndarray, columns: int, key: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.dtype.kind not in "fiu":
        raise DataContractError(f"{key}: expected a two-dimensional real numeric matrix")
    if values.shape[1] != columns or values.shape[0] <= columns:
        raise DataContractError(f"{key}: expected (time,{columns}) with time>{columns}; got {values.shape}; no inferred transpose")
    finite = np.isfinite(values)
    if not finite.all():
        rows, channels = np.where(~finite)
        raise DataContractError(f"{key}: {len(rows)} nonfinite values; first at sample={rows[0]}, channel_0based={channels[0]}")
    result = np.ascontiguousarray(values.T)
    result.setflags(write=False)
    return result


def read_ecog(config: ExperimentConfig, drive_root: Path, subject: str) -> DevelopmentRecord:
    path = development_path(config, drive_root, subject)
    before = sha256_file(path)
    matrix, adapter = _read_variable(path, "train_data")
    channels = config.data["raw_data"]["expected_channels_nominal"][subject]
    values = _matrix(matrix, channels, "train_data")
    if before != sha256_file(path):
        raise DataContractError("Original file changed during reading")
    fs = config.data["raw_data"]["ecog_fs_hz"]
    return DevelopmentRecord(subject, values, {
        "subject": subject, "dataset": config.data["project"]["dataset"],
        "path": (Path(config.data["paths"]["raw_dir"]) / path.name).as_posix(),
        "sha256": before, "loaded_variables": ["train_data"], "adapter": adapter,
        "original_shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "n_ecog_channels_original": channels, "n_train_samples_actual": values.shape[1],
        "sampling_hz_ecog": fs, "time_origin_s": 0.0,
        "last_sample_time_s": (values.shape[1] - 1) / fs,
        "duration_s": values.shape[1] / fs,
        "time_basis": "nominal sampling rate; origin assumed 0; no measured timestamps",
        "units": "original; physical calibration unconfirmed", "nonfinite_count": 0,
    })


def read_development_glove(config: ExperimentConfig, drive_root: Path,
                           record: DevelopmentRecord) -> DevelopmentRecord:
    """Development glove (5, T_storage), readonly. Descriptive EDA and training only.

    The source hash must match the ECoG read so both come from the same file.
    """
    path = development_path(config, drive_root, record.subject)
    if sha256_file(path) != record.metadata["sha256"]:
        raise DataContractError("Source changed since ECoG load")
    matrix, adapter = _read_variable(path, "train_dg")
    values = _matrix(matrix, config.data["raw_data"]["n_fingers"], "train_dg")
    n = values.shape[1]
    ecog_n = record.values.shape[1]
    fs = config.data["raw_data"]["ecog_fs_hz"]
    effective_fs = config.data["raw_data"]["glove_fs_effective_hz"]
    if n == ecog_n:
        storage_fs = fs
        representation = "stored_on_ecog_grid; upsampling pattern verified by preprocessing.inspect_glove_storage"
    elif ecog_n * effective_fs == n * fs:
        storage_fs = effective_fs
        representation = "effective_rate_grid"
    else:
        raise DataContractError(f"train_dg length {n} cannot be reconciled with {ecog_n} ECoG samples")
    if sha256_file(path) != record.metadata["sha256"]:
        raise DataContractError("Original file changed during label validation")
    return DevelopmentRecord(record.subject, values, {
        "subject": record.subject, "sha256": record.metadata["sha256"],
        "loaded_variables": ["train_dg"], "adapter": adapter,
        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "sampling_hz_glove_effective": effective_fs,
        "sampling_hz_glove_storage": storage_fs, "representation": representation,
        "duration_s": n / storage_fs, "nonfinite_count": 0,
        "use": "development only; descriptive EDA and fold training; never test labels"})


def validate_development_labels(config: ExperimentConfig, drive_root: Path,
                                record: DevelopmentRecord) -> dict[str, Any]:
    """Validate train_dg and return its metadata only."""
    return read_development_glove(config, drive_root, record).metadata
