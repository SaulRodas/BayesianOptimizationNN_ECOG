"""Fold artifacts on top of the per-block data: purge, scalers and windows.

A fold set depends on the effective purge. Approval status lives in execution
reports, so approving the same value never changes the numerical artifacts.
Scalers are fitted ONLY on the eligible training columns of each fold.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..datasets import contiguous_runs, inference_window_starts, train_window_starts, fit_fold_scalers
from ..io import DataContractError
from ..reporting import json_safe, write_json
from ..splits import Block, Fold, train_eligibility
from ..io import FOLD_MANIFEST, block_dir, read_json, save_arrays


PURGE_RULE = ("training column eligible iff valid and, for a block before validation, "
              "y_index <= val.start_target - 1 - purge; for a block after it, x_index >= val.stop_target + purge")


def subject_blocks(pipeline_dir: Path, subject: str, names: list[str]) -> tuple[list[Block], dict[str, dict]]:
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

    feature_scaler, target_scaler, stats = fit_fold_scalers(config, arrays, eligible, f"{subject} {fold.name}")
    return save_fold(config, subject, fold, arrays, names, channels, purge, eligible,
                     feature_scaler, target_scaler, stats, train_windows, val_windows, out)


def save_fold(config, subject, fold, arrays, names, channels, purge, eligible,
              feature_scaler, target_scaler, stats, train_windows, val_windows, out):
    """Persist previously calculated masks, scalers and window starts."""
    validation = config.data["validation"]
    window = validation["train_window_samples"]
    low, high = config.data["scaling"]["ecog"]["quantile_range_percent"]
    val = arrays[fold.val.name]
    if not train_windows.size:
        raise DataContractError(f"{subject} {fold.name}: no whole training window")
    save_arrays(out, scaler_x_center=np.asarray(feature_scaler.center_, dtype=np.float64),
                scaler_x_scale=np.asarray(feature_scaler.scale_, dtype=np.float64),
                scaler_y_min=np.asarray(target_scaler.data_min_, dtype=np.float64),
                scaler_y_max=np.asarray(target_scaler.data_max_, dtype=np.float64),
                train_windows=train_windows, val_windows=val_windows,
                **{f"eligible_{name}": mask for name, mask in eligible.items()})
    before = [b for b in fold.train if b.stop_target <= fold.val.start_target]
    after = [b for b in fold.train if b.start_target >= fold.val.stop_target]
    last_train_y = max((int(arrays[b.name]["y_index"][eligible[b.name]].max()) for b in before
                        if eligible[b.name].any()), default=None)
    first_train_x = min((int(arrays[b.name]["x_index"][eligible[b.name]].min()) for b in after
                         if eligible[b.name].any()), default=None)
    val_x, val_y = val["x_index"][val["valid"]], val["y_index"][val["valid"]]
    manifest = {
        "schema_version": 2, "subject": subject, "fold": fold.name, "val_block": fold.val.name,
        "train_blocks": [b.name for b in fold.train], "block_names": names,
        "purge_samples": purge, "purge_rule": PURGE_RULE,
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
                                "fit_rows": stats["fit_rows"], "min_spread": stats["min_spread"],
                                "fitted_on": "eligible training columns of this fold only"},
                    "target": {"type": "MinMaxScaler", "feature_range": config.data["scaling"]["targets"]["feature_range"],
                               "data_min": target_scaler.data_min_, "data_max": target_scaler.data_max_,
                               "fitted_on": "eligible training columns of this fold only"}},
        "channels_kept_original": channels,
        "x_shape_per_block": {name: list(arrays[name]["x"].shape) for name in names},
    }
    write_json(out / FOLD_MANIFEST, json_safe(manifest))
    return {"subject": subject, "fold": fold.name, "purge": purge,
            "train_columns": manifest["n_train_columns"], "train_windows": manifest["n_train_windows"],
            "val_columns": manifest["val"]["valid_columns"], "val_windows": manifest["n_val_windows"],
            "gap_before_val": manifest["effective_gap_samples"]["before_val"],
            "gap_after_val": manifest["effective_gap_samples"]["after_val"]}
