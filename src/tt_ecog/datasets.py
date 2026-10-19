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
from .store import (FOLD_MANIFEST, FOLDS_MANIFEST, PROCESSED_MANIFEST, StoreError, block_dir, fold_dir,
                    read_json, verify_manifest)


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
    pipeline_dir = fold_set_dir.parent.parent
    if verify:
        set_manifest = verify_manifest(fold_set_dir, FOLDS_MANIFEST, prefix=f"{subject}/val_{val_block}/")
        verify_manifest(pipeline_dir, PROCESSED_MANIFEST, prefix=f"{subject}/")
    else:
        set_manifest = read_json(fold_set_dir / FOLDS_MANIFEST)
    if set_manifest["pipeline_hash"] != pipeline_dir.name:
        raise StoreError(f"{fold_set_dir.name} belongs to pipeline {set_manifest['pipeline_hash']}, not {pipeline_dir.name}")
    if set_manifest["purge_status"] != "approved" and not allow_candidate:
        raise PermissionError(f"{fold_set_dir.name}: purge status {set_manifest['purge_status']!r}; only the "
                              "step-3 verification may load it (allow_candidate=True)")
    if set_manifest["scaling"] != config.data["scaling"]:
        raise StoreError("The fold scalers were fitted with a different scaling configuration")
    folder = fold_dir(fold_set_dir, subject, val_block)
    manifest = read_json(folder / FOLD_MANIFEST)
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
                    manifest["purge_status"], manifest["window"], blocks, train_masks,
                    np.load(folder / "train_windows.npy"), np.load(folder / "val_windows.npy"),
                    feature_scaler, target_scaler, manifest)
