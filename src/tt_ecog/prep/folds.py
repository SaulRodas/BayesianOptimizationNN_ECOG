"""Fold artifacts on top of the per-block data: purge, scalers and windows.

A fold set depends on the purge; with a null purge it is built with the
verification candidate and labelled ``candidate_pending_V04_V09`` everywhere.
Scalers are fitted ONLY on the eligible training columns of each fold.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .. import preprocessing as pp
from ..config import ExperimentConfig
from ..datasets import contiguous_runs, inference_window_starts, train_window_starts
from ..io import DataContractError
from ..reporting import json_safe, utc_now, write_json
from ..splits import Block, Fold, make_folds, train_eligibility
from ..store import (FOLD_MANIFEST, FOLDS_MANIFEST, HASH_LENGTH, block_dir, canonical_hash, fold_dir,
                     library_versions, read_json, source_hashes, write_manifest)

FOLD_SOURCES = ("splits.py", "datasets.py", "store.py", "prep/folds.py")
PURGE_RULE = ("training column eligible iff valid and, for a block before validation, "
              "y_index <= val.start_target - 1 - purge; for a block after it, x_index >= val.stop_target + purge")


def fold_identity(config: ExperimentConfig, pipeline_hash: str, purge: int, status: str) -> dict[str, Any]:
    validation = config.data["validation"]
    return {"schema_version": 1, "pipeline_hash": pipeline_hash, "purge_samples": purge, "purge_status": status,
            "scaling": config.data["scaling"],
            "windows": {k: validation[k] for k in ("train_window_samples", "train_stride_samples",
                                                   "inference_window_samples", "inference_stride_samples")},
            "validation_blocks": validation["stage2_validation_blocks"],
            "sources": source_hashes(FOLD_SOURCES), "libraries": library_versions(("numpy", "scikit-learn"))}


def fold_hash(identity: dict[str, Any]) -> str:
    return canonical_hash(identity)[:HASH_LENGTH]


def _subject_blocks(pipeline_dir: Path, subject: str, names: list[str]) -> tuple[list[Block], dict[str, dict]]:
    blocks, arrays = [], {}
    for name in names:
        source = block_dir(pipeline_dir, subject, name)
        r = read_json(source / "block_manifest.json")["range"]
        blocks.append(Block(name, r["start_target"], r["stop_target"], r["start_source"], r["stop_source"]))
        arrays[name] = {"x": np.load(source / "x.npy", mmap_mode="r"), "y": np.load(source / "y.npy"),
                        "x_index": np.load(source / "x_index.npy"), "y_index": np.load(source / "y_index.npy"),
                        "valid": np.load(source / "valid.npy")}
    return blocks, arrays


def _gap(before: int | None, after: int | None) -> int | None:
    """Samples strictly between two global indices."""
    return None if before is None or after is None else int(after - before - 1)


def build_fold(config: ExperimentConfig, subject: str, fold: Fold, arrays: dict[str, dict], names: list[str],
               channels: list[int], purge: int, status: str, out: Path) -> dict[str, Any]:
    validation = config.data["validation"]
    window = validation["train_window_samples"]
    eligible = {b.name: train_eligibility(arrays[b.name]["x_index"], arrays[b.name]["y_index"],
                                          arrays[b.name]["valid"], b, fold.val, purge) for b in fold.train}
    train_windows = []
    for block in fold.train:
        starts = train_window_starts(eligible[block.name], window, validation["train_stride_samples"])
        train_windows.append(np.column_stack([np.full(starts.size, names.index(block.name)), starts]))
    train_windows = np.concatenate(train_windows).astype(np.int64)
    if not train_windows.size:
        raise DataContractError(f"{subject} {fold.name}: no whole training window survives masks and purge")
    val = arrays[fold.val.name]
    val_windows = inference_window_starts(val["valid"], validation["inference_window_samples"],
                                          validation["inference_stride_samples"])

    features = np.concatenate([pp.features_to_matrix(np.asarray(arrays[b.name]["x"][:, :, eligible[b.name]]))
                               .astype(np.float64) for b in fold.train])
    targets = np.concatenate([arrays[b.name]["y"][:, eligible[b.name]].T for b in fold.train]).astype(np.float64)
    low, high = config.data["scaling"]["ecog"]["quantile_range_percent"]
    spread = np.diff(np.percentile(features, [low, high], axis=0), axis=0)[0]
    # sklearn silently replaces near-zero scales by 1; here they are a failure.
    tiny = 10 * np.finfo(np.float64).eps
    if (spread < tiny).any():
        raise DataContractError(f"{subject} {fold.name}: {int((spread < tiny).sum())} channel-frequency features "
                                f"have a zero {low}-{high} percentile range on fold train")
    finger_range = targets.max(axis=0) - targets.min(axis=0)
    if (finger_range <= 0).any():
        raise DataContractError(f"{subject} {fold.name}: constant flexion on fold train for fingers "
                                f"{np.flatnonzero(finger_range <= 0).tolist()}")
    feature_scaler = pp.make_feature_scaler(config).fit(features)
    target_scaler = pp.make_target_scaler(config).fit(targets)
    del features

    out.mkdir(parents=True)
    np.save(out / "scaler_x_center.npy", np.asarray(feature_scaler.center_, dtype=np.float64))
    np.save(out / "scaler_x_scale.npy", np.asarray(feature_scaler.scale_, dtype=np.float64))
    np.save(out / "scaler_y_min.npy", np.asarray(target_scaler.data_min_, dtype=np.float64))
    np.save(out / "scaler_y_max.npy", np.asarray(target_scaler.data_max_, dtype=np.float64))
    for name, mask in eligible.items():
        np.save(out / f"eligible_{name}.npy", mask)
    np.save(out / "train_windows.npy", train_windows)
    np.save(out / "val_windows.npy", val_windows)

    before = [b for b in fold.train if b.stop_target <= fold.val.start_target]
    after = [b for b in fold.train if b.start_target >= fold.val.stop_target]
    last_train_y = max((int(arrays[b.name]["y_index"][eligible[b.name]].max()) for b in before
                        if eligible[b.name].any()), default=None)
    first_train_x = min((int(arrays[b.name]["x_index"][eligible[b.name]].min()) for b in after
                         if eligible[b.name].any()), default=None)
    val_x, val_y = val["x_index"][val["valid"]], val["y_index"][val["valid"]]
    manifest = {
        "schema_version": 1, "subject": subject, "fold": fold.name, "val_block": fold.val.name,
        "train_blocks": [b.name for b in fold.train], "block_names": names,
        "purge_samples": purge, "purge_status": status, "purge_rule": PURGE_RULE,
        "window": window, "train_stride": validation["train_stride_samples"],
        "inference_stride": validation["inference_stride_samples"],
        "val": {"start_target": fold.val.start_target, "stop_target": fold.val.stop_target,
                "start_time_s": fold.val.start_source / config.data["raw_data"]["ecog_fs_hz"],
                "stop_time_s": fold.val.stop_source / config.data["raw_data"]["ecog_fs_hz"],
                "valid_columns": int(val["valid"].sum()), "valid_runs": contiguous_runs(val["valid"]),
                "windows": int(val_windows.size), "end_anchored": True, "purged": False},
        "train": {b.name: {"eligible_columns": int(eligible[b.name].sum()),
                           "eligible_runs": contiguous_runs(eligible[b.name]),
                           "windows": int((train_windows[:, 0] == names.index(b.name)).sum())}
                  for b in fold.train},
        "n_train_columns": int(sum(m.sum() for m in eligible.values())),
        "n_train_windows": int(train_windows.shape[0]), "n_val_windows": int(val_windows.size),
        "effective_gap_samples": {"before_val": _gap(last_train_y, int(val_x.min())),
                                  "after_val": _gap(int(val_y.max()), first_train_x),
                                  "definition": "samples strictly between the last training flexion and the first "
                                                "validation ECoG (before), or the last validation flexion and the "
                                                "first training ECoG (after)"},
        "raw_context_shared_with_validation": False,
        "scalers": {"feature": {"type": "RobustScaler", "quantile_range": [low, high],
                                "unit_variance": config.data["scaling"]["ecog"]["unit_variance"],
                                "n_features": int(feature_scaler.center_.size), "column_order": "channel * F + frequency",
                                "fit_rows": int(targets.shape[0]), "min_spread": float(spread.min()),
                                "fitted_on": "eligible training columns of this fold only"},
                    "target": {"type": "MinMaxScaler", "feature_range": config.data["scaling"]["targets"]["feature_range"],
                               "data_min": target_scaler.data_min_, "data_max": target_scaler.data_max_,
                               "fitted_on": "eligible training columns of this fold only"}},
        "channels_kept_original": channels,
        "x_shape_per_block": {name: list(arrays[name]["x"].shape) for name in names},
    }
    write_json(out / FOLD_MANIFEST, json_safe(manifest))
    return {"subject": subject, "fold": fold.name, "purge": purge, "purge_status": status,
            "train_columns": manifest["n_train_columns"], "train_windows": manifest["n_train_windows"],
            "val_columns": manifest["val"]["valid_columns"], "val_windows": manifest["n_val_windows"],
            "gap_before_val": manifest["effective_gap_samples"]["before_val"],
            "gap_after_val": manifest["effective_gap_samples"]["after_val"]}


def build_fold_set(config: ExperimentConfig, pipeline_dir: Path, identity: dict[str, Any], set_hash: str,
                   root: Path, run_id: str, git_commit: str | None) -> list[dict[str, Any]]:
    """Every stage-2 fold of every subject into ``root``; folds_manifest is written last."""
    purge, status = identity["purge_samples"], identity["purge_status"]
    names = config.data["validation"]["block_names"]
    channel_maps = read_json(pipeline_dir / "manifests/channel_map.json")
    rows = []
    for subject in config.data["project"]["subjects"]:
        blocks, arrays = _subject_blocks(pipeline_dir, subject, names)
        for fold in make_folds(blocks, config.data["validation"]["stage2_validation_blocks"]):
            print(f"{subject} {fold.name}: purge {purge} ({status})", flush=True)
            rows.append(build_fold(config, subject, fold, arrays, names,
                                   channel_maps[subject]["kept_original_in_effective_order"], purge, status,
                                   fold_dir(root, subject, fold.val.name)))
    write_manifest(root, FOLDS_MANIFEST, json_safe({
        "schema_version": 1, "fold_hash": set_hash,
        "pipeline_hash": pipeline_dir.name, "identity": identity, "purge_samples": purge, "purge_status": status,
        "scaling": config.data["scaling"], "created_by_run": run_id, "created_at_utc": utc_now(),
        "git_commit": git_commit, "test_accessed": False}))
    return rows
