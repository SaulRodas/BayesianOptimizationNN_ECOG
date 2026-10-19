"""Window indices and loading of processed folds (numpy only).

Windows live inside one contiguous run of usable columns of ONE block: they
never cross block boundaries, purged or invalid columns. Column n of a block
pairs ECoG features at x_index[n] with flexion at y_index[n] = x_index[n] + 20,
so a window over columns [s, s + L) is already aligned.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import preprocessing as pp
from .config import ExperimentConfig
from .io import (FOLD_MANIFEST, FOLDS_MANIFEST, StoreError, block_dir, fold_dir,
                    read_json, verify_pipeline, DataContractError)


def _check_window(window: int, stride: int) -> None:
    for name, value in (("window", window), ("stride", stride)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    change = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(change[::2].tolist(), change[1::2].tolist()))


def n_windows(length: int, window: int, stride: int) -> int:
    """0 if length < window, else 1 + (length - window) // stride."""
    _check_window(window, stride)
    return 0 if length < window else 1 + (length - window) // stride


def train_window_starts(mask: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Start columns of every whole window inside each run of the mask."""
    _check_window(window, stride)
    starts = [np.arange(a, b - window + 1, stride) for a, b in contiguous_runs(mask) if b - a >= window]
    return np.concatenate(starts).astype(np.int64) if starts else np.empty(0, dtype=np.int64)


def inference_window_starts(mask: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Strided windows plus an end-anchored one per run, so every True column is covered.

    A run shorter than the window cannot be covered: that is an error, never a
    silent loss of coverage.
    """
    _check_window(window, stride)
    starts = []
    for a, b in contiguous_runs(mask):
        if b - a < window:
            raise ValueError(f"Usable run [{a}, {b}) is shorter than the {window}-column window: incomplete coverage")
        run = np.arange(a, b - window + 1, stride)
        if run[-1] + window < b:
            run = np.r_[run, b - window]
        covered = np.zeros(b - a, dtype=bool)
        for start in run - a:
            covered[start:start + window] = True
        if not covered.all():
            raise ValueError(f"Stride {stride} leaves columns of run [{a}, {b}) uncovered")
        starts.append(run)
    return np.concatenate(starts).astype(np.int64) if starts else np.empty(0, dtype=np.int64)


# --------------------------------------------------------------------------- folds

@dataclass(frozen=True)
class BlockData:
    name: str
    x: np.ndarray  # (C, F, N) float32 scaled with the fold-train feature scaler
    y: np.ndarray  # (5, N) float32 scaled with the fold-train target scaler; NaN where invalid
    y_original: np.ndarray  # (5, N) float64 in original glove units
    x_index: np.ndarray
    y_index: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class FoldData:
    subject: str
    name: str
    val_block: str
    block_names: tuple[str, ...]
    purge_samples: int
    purge_status: str
    window: int
    blocks: dict[str, BlockData]
    train_masks: dict[str, np.ndarray]
    train_windows: np.ndarray  # (n, 2): block number in block_names, start column
    val_windows: np.ndarray  # (m,) start columns in the validation block
    feature_scaler: Any
    target_scaler: Any
    manifest: dict[str, Any]

    @property
    def pipeline_id(self) -> str:
        return self.manifest["pipeline_id"]

    def train_window(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        number, start = (int(v) for v in self.train_windows[i])
        block = self.blocks[self.block_names[number]]
        columns = slice(start, start + self.window)
        return block.x[:, :, columns], block.y[:, columns]

    def val_window(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        block = self.blocks[self.val_block]
        columns = slice(int(self.val_windows[j]), int(self.val_windows[j]) + self.window)
        return block.x[:, :, columns], block.y[:, columns]


def load_fold(config: ExperimentConfig, fold_set_dir: Path, subject: str, val_block: str, *,
              allow_candidate: bool = False, verify: bool = True) -> FoldData:
    """Scaled arrays, masks and windows of one fold.

    A fold set built with a candidate purge is refused unless ``allow_candidate``
    (verification only). With ``verify`` every file of this subject is checked
    against the manifests before use.
    """
    fold_set_dir = Path(fold_set_dir)
    pipeline_dir = fold_set_dir.parent
    if config.data["paths"].get("processed_pipeline_id") != pipeline_dir.name:
        raise StoreError("Select this pipeline explicitly in paths.processed_pipeline_id")
    if verify:
        verify_pipeline(pipeline_dir, config)
    set_manifest = read_json(fold_set_dir / FOLDS_MANIFEST)
    if set_manifest["pipeline_id"] != pipeline_dir.name:
        raise StoreError("Folds belong to a different pipeline")
    purge, status = config.purge_for_build()
    if set_manifest["purge_samples"] != purge or set_manifest["scaling"] != config.data["scaling"]:
        raise StoreError("Fold purge/scaling differs from the effective configuration")
    if status != "approved" and not allow_candidate:
        raise PermissionError(f"Purga candidate: {status}; only verification may load it")
    folder = fold_dir(fold_set_dir, subject, val_block)
    manifest = read_json(folder / FOLD_MANIFEST)
    manifest["pipeline_id"] = set_manifest["pipeline_id"]
    feature_scaler = pp.restore_feature_scaler(config, np.load(folder / "scaler_x_center.npy"),
                                               np.load(folder / "scaler_x_scale.npy"))
    target_scaler = pp.restore_target_scaler(config, np.load(folder / "scaler_y_min.npy"),
                                             np.load(folder / "scaler_y_max.npy"))
    names = tuple(manifest["block_names"])
    blocks = {}
    for name in names:
        source = block_dir(pipeline_dir, subject, name)
        x = np.load(source / "x.npy")
        y = np.load(source / "y.npy")
        channels, frequencies, _ = x.shape
        scaled = feature_scaler.transform(pp.features_to_matrix(x))
        blocks[name] = BlockData(
            name, pp.matrix_to_features(scaled, channels, frequencies).astype(np.float32),
            target_scaler.transform(y.T).T.astype(np.float32), y,
            np.load(source / "x_index.npy"), np.load(source / "y_index.npy"), np.load(source / "valid.npy"))
    train_masks = {name: np.load(folder / f"eligible_{name}.npy") for name in names if name != val_block}
    return FoldData(subject, manifest["fold"], val_block, names, manifest["purge_samples"],
                    status, manifest["window"], blocks, train_masks,
                    np.load(folder / "train_windows.npy"), np.load(folder / "val_windows.npy"),
                    feature_scaler, target_scaler, manifest)


def fit_fold_scalers(config, arrays, eligible, label="fold"):
    """Fit train-only scalers; reject degenerate feature/finger ranges."""
    features = np.concatenate([pp.features_to_matrix(np.asarray(arrays[name]["x"][:, :, eligible[name]]))
                               .astype(np.float64) for name in eligible])
    targets = np.concatenate([arrays[name]["y"][:, eligible[name]].T for name in eligible]).astype(np.float64)
    low, high = config.data["scaling"]["ecog"]["quantile_range_percent"]
    spread = np.diff(np.percentile(features, [low, high], axis=0), axis=0)[0]
    # sklearn silently replaces near-zero scales by 1; here they are a failure.
    tiny = 10 * np.finfo(np.float64).eps
    if (spread < tiny).any():
        raise DataContractError(f"{label}: {int((spread < tiny).sum())} channel-frequency features "
                                f"have a zero {low}-{high} percentile range on fold train")
    finger_range = targets.max(axis=0) - targets.min(axis=0)
    if (finger_range <= 0).any():
        raise DataContractError(f"{label}: constant flexion on fold train for fingers "
                                f"{np.flatnonzero(finger_range <= 0).tolist()}")
    feature_scaler = pp.make_feature_scaler(config).fit(features)
    target_scaler = pp.make_target_scaler(config).fit(targets)
    del features
    return feature_scaler, target_scaler, {"min_spread": float(spread.min()), "fit_rows": int(targets.shape[0])}


class FoldWindows:
    """Index existing windows without copying a whole fold or refitting its scalers.

    PyTorch's DataLoader accepts this map-style dataset; numpy-only users do not
    need to import torch. Validation preserves the persisted start order.
    """

    def __init__(self, fold: FoldData, partition: str):
        if partition not in ("train", "val"):
            raise ValueError("partition must be train or val")
        self.fold, self.partition = fold, partition

    def __len__(self):
        return len(self.fold.train_windows if self.partition == "train" else self.fold.val_windows)

    def __getitem__(self, index):
        x, y = self.fold.train_window(index) if self.partition == "train" else self.fold.val_window(index)
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def full_development(config, subject, arrays, feature_scaler, target_scaler):
    """Refit dataset: all development, retaining each original independent block.

    No validation partition exists here. Scalers were fitted on development only;
    windows still cannot cross invalid samples or the original block boundaries.
    """
    names = tuple(config.data["validation"]["block_names"])
    window = config.data["validation"]["train_window_samples"]
    stride = config.data["validation"]["train_stride_samples"]
    blocks, windows = {}, []
    for number, name in enumerate(names):
        a = arrays[name]
        c, f, _ = a["x"].shape
        x = pp.matrix_to_features(feature_scaler.transform(pp.features_to_matrix(a["x"])), c, f)
        blocks[name] = BlockData(name, x.astype(np.float32), target_scaler.transform(a["y"].T).T.astype(np.float32),
                                 a["y"], a["x_index"], a["y_index"], a["valid"])
        starts = train_window_starts(a["valid"], window, stride)
        windows.extend((number, int(start)) for start in starts)
    if not windows:
        raise DataContractError("No full-development training windows")
    return FoldData(subject, "full_development", "", names, 0, "not_applicable_full_development", window,
                    blocks, {name: arrays[name]["valid"] for name in names}, np.asarray(windows, dtype=np.int64),
                    np.empty(0, dtype=np.int64), feature_scaler, target_scaler,
                    {"pipeline_id": config.data["paths"]["processed_pipeline_id"],
                     "scalers_fit_on": "all_valid_development", "independent_blocks": list(names)})
