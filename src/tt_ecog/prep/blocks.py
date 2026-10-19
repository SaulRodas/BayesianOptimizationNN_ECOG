"""Per-block processing: every block B1-B5 is ONE independent segment.

For each block only its own raw ECoG enters the chain (exclusion -> CMR -> FIR ->
Morlet -> antialias decimation) and only its own effective glove samples enter
the spline. Column n then pairs ECoG(x_index[n]) with flexion(y_index[n] =
x_index[n] + delay). Nothing here depends on folds or on the purge.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .. import preprocessing as pp
from ..config import ExperimentConfig
from ..io import DataContractError, DevelopmentRecord
from ..reporting import json_safe, utc_now, write_json
from ..splits import Block, chronological_blocks
from ..store import PROCESSED_MANIFEST, canonical_hash, HASH_LENGTH, library_versions, source_hashes, write_manifest

PROCESSING_SOURCES = ("preprocessing.py", "splits.py", "store.py", "prep/blocks.py")
TIME_CONVENTION = {
    "x_index": "global 100 Hz index of the ECoG feature column",
    "y_index": "global 100 Hz index of the paired flexion target, x_index + delay",
    "source_index": "x_index * factor on the original 1 kHz grid",
    "feature_time_s": "x_index / 100, seconds of the original record",
    "target_time_s": "y_index / 100, seconds of the original record",
    "valid": "full FIR->Morlet->antialias context inside the block AND interpolated flexion at y_index",
}


def pipeline_identity(config: ExperimentConfig, raw_sha256: dict[str, str], reference_run: str) -> dict[str, Any]:
    data = config.data
    raw = data["raw_data"]
    return {"schema_version": 1, "stage": "step2_per_block_features",
            "raw_data": {k: raw[k] for k in ("ecog_fs_hz", "glove_fs_effective_hz", "expected_channels_nominal",
                                             "n_fingers", "train_ecog_key", "train_glove_key")},
            "excluded_channels": {s: list(config.excluded_channels(s)) for s in data["project"]["subjects"]},
            "signal": data["signal"],
            "blocks": {"n_base_blocks": data["validation"]["n_base_blocks"],
                       "block_names": data["validation"]["block_names"]},
            "eda_reference_run": reference_run, "raw_sha256": raw_sha256,
            "sources": source_hashes(PROCESSING_SOURCES), "libraries": library_versions(("numpy", "scipy", "mne"))}


def pipeline_hash(identity: dict[str, Any]) -> str:
    return canonical_hash(identity)[:HASH_LENGTH]


def _summary(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values.astype(np.float64, copy=False), [0, 1, 50, 99, 100])
    return dict(zip(("min", "p1", "median", "p99", "max"), (float(v) for v in q)))


def process_block(spec: pp.PreprocessingSpec, raw: np.ndarray, y25: np.ndarray, knots: np.ndarray,
                  block: Block, out: Path) -> dict[str, Any]:
    x, xvalid, report = pp.segment_features(raw[:, block.start_source:block.stop_source], spec)
    y, yvalid = pp.segment_glove(y25, knots, block.start_source, block.stop_source, spec)
    n_target = block.stop_target - block.start_target
    if x.shape[-1] != n_target or y.shape[-1] != n_target:
        raise DataContractError(f"{spec.subject} {block.name}: features {x.shape[-1]} / targets {y.shape[-1]} "
                                f"columns, expected {n_target}")
    xa, ya, xi, yi = pp.align_delay(x, y, spec.delay_samples)
    xv, yv = xvalid[xi], yvalid[yi]
    valid = xv & yv
    if not valid.any():
        raise DataContractError(f"{spec.subject} {block.name}: no valid ECoG-flexion pair")
    if not np.isfinite(ya[:, valid]).all():
        raise DataContractError(f"{spec.subject} {block.name}: nonfinite flexion on valid columns")
    x_index = (block.start_target + xi).astype(np.int64)
    y_index = (block.start_target + yi).astype(np.int64)
    out.mkdir(parents=True)
    np.save(out / "x.npy", np.ascontiguousarray(xa))
    np.save(out / "y.npy", np.ascontiguousarray(ya, dtype=np.float64))
    np.save(out / "x_index.npy", x_index)
    np.save(out / "y_index.npy", y_index)
    np.save(out / "valid.npy", valid)
    runs = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(np.int8))).reshape(-1, 2)
    manifest = {
        "schema_version": 1, "subject": spec.subject, "block": block.name,
        "range": {"start_source": block.start_source, "stop_source": block.stop_source,
                  "start_target": block.start_target, "stop_target": block.stop_target,
                  "start_time_s": block.start_source / spec.sfreq, "stop_time_s": block.stop_source / spec.sfreq},
        "delay_samples": spec.delay_samples, "factor": spec.factor, "time_convention": TIME_CONVENTION,
        "x": {"shape": list(xa.shape), "dtype": str(xa.dtype), "magnitude": "Morlet power |W|^2, unscaled, no log",
              "axes": ["channel (effective order)", "frequency", "column"]},
        "y": {"shape": list(ya.shape), "dtype": "float64", "units": "original glove units, NaN where invalid",
              "axes": ["finger", "column"]},
        "columns": {"total": int(valid.size), "valid": int(valid.sum()),
                    "invalid_ecog_context": int((~xv).sum()), "invalid_flexion": int((~yv).sum()),
                    "invalid_both": int((~xv & ~yv).sum()), "valid_runs": runs.tolist(),
                    "dropped_by_delay_at_block_end": spec.delay_samples},
        "first_valid": {"x_index": int(x_index[valid][0]),
                        "feature_time_s": float(x_index[valid][0] / spec.target_sfreq)},
        "last_valid": {"x_index": int(x_index[valid][-1]), "y_index": int(y_index[valid][-1])},
        "glove_knots_in_block": int(((knots >= block.start_source) & (knots < block.stop_source)).sum()),
        "morlet": {"negative_values_valid": report["negative_values_valid"], "values_valid": report["values_valid"],
                   "negative_fraction_valid": report["negative_values_valid"] / report["values_valid"],
                   "policy": "reported, never clipped"},
        "x_valid_summary": _summary(xa[:, :, valid]),
        "y_valid_summary": {pl: _summary(ya[f, valid]) for f, pl in enumerate(("thumb", "index", "middle",
                                                                                "ring", "little"))},
    }
    write_json(out / "block_manifest.json", json_safe(manifest))
    return manifest


def process_subject(config: ExperimentConfig, ecog: DevelopmentRecord, glove: DevelopmentRecord,
                    root: Path) -> dict[str, Any]:
    subject = ecog.subject
    spec = pp.PreprocessingSpec.from_config(config, subject)
    if ecog.values.shape[0] != spec.channels.n_original:
        raise DataContractError(f"{subject}: channel count differs from the nominal contract")
    blocks = chronological_blocks(ecog.values.shape[1], config.data["validation"]["block_names"], spec.factor)
    inspection = pp.glove_storage_inspection(glove.values, glove.metadata["sampling_hz_glove_storage"], spec)
    y25, knots = pp.effective_glove(glove.values, inspection, spec)
    for block in blocks:
        print(f"{subject} {block.name}: {block.start_source / spec.sfreq:.0f}-{block.stop_source / spec.sfreq:.0f} s",
              flush=True)
        process_block(spec, ecog.values, y25, knots, block, root / subject / block.name)
    return {"spec": spec, "blocks": blocks, "n_effective_glove": int(knots.size),
            "inspection": {k: inspection[k] for k in ("identified", "method", "phase", "relative_rms_error",
                                                       "tolerance", "factor", "change_fraction_at_phase")}}


def build_pipeline(config: ExperimentConfig, records: dict[str, tuple[DevelopmentRecord, DevelopmentRecord]],
                   identity: dict[str, Any], digest: str, root: Path, run_id: str, git_commit: str | None) -> None:
    """Process every subject into ``root`` and write the manifests (processed_manifest last)."""
    results = {subject: process_subject(config, ecog, glove, root) for subject, (ecog, glove) in records.items()}
    specs = {subject: result["spec"] for subject, result in results.items()}
    any_spec = next(iter(specs.values()))
    analytic = pp.support_report(any_spec)
    measured = pp.measured_support(any_spec, config.data["eda"]["support_tolerance"])
    manifests = root / "manifests"
    manifests.mkdir()
    write_json(manifests / "channel_map.json", {s: spec.channels.to_json() for s, spec in specs.items()})
    write_json(manifests / "filter_support.json", json_safe({
        "filters": [d.to_json() for d in (any_spec.bandpass, any_spec.notch, any_spec.fir, any_spec.antialias)],
        "morlet": {"freqs_hz": any_spec.freqs, "n_cycles": any_spec.n_cycles, "zero_mean": any_spec.zero_mean,
                   "implementation": "mne.time_frequency.tfr_array_morlet, use_fft=True, decim=1"},
        "analytic_support": analytic,
        "measured_support": {"measured_half_support_target": measured["measured_half_support_target"],
                             "tolerance": measured["tolerance"],
                             "method": "unit impulse through FIR -> Morlet -> antialias, max over frequencies"},
        "edge_mask": {"half_support_source": pp.chain_half_support(any_spec),
                      "rule": "column k valid iff k*factor - H >= 0 and k*factor + H <= n_source_block - 1"},
        "purge": {"value": config.data["validation"]["purge_samples_at_100hz"],
                  "note": "Pending V04/V09. Blocks are processed independently, so no feature of a block "
                          "depends on another block; the purge only removes training columns near validation."}}))
    write_json(manifests / "glove_storage.json", json_safe({
        s: {**r["inspection"], "n_effective_samples": r["n_effective_glove"],
            "storage_fs_hz": records[s][1].metadata["sampling_hz_glove_storage"]} for s, r in results.items()}))
    write_json(manifests / "split_manifest.json", json_safe({
        "processing_unit": "chronological block (B1-B5), each processed independently",
        "windows_cross_block_boundaries": False,
        "stage1_validation_blocks": config.data["validation"]["stage1_validation_blocks"],
        "stage2_validation_blocks": config.data["validation"]["stage2_validation_blocks"],
        "time_convention": TIME_CONVENTION,
        "blocks": {s: [{"name": b.name, "start_source": b.start_source, "stop_source": b.stop_source,
                        "start_target": b.start_target, "stop_target": b.stop_target,
                        "start_time_s": b.start_source / specs[s].sfreq, "stop_time_s": b.stop_source / specs[s].sfreq}
                       for b in r["blocks"]] for s, r in results.items()}}))
    write_json(manifests / "preprocessing_config.json", json_safe({
        "identity": identity,
        "order": ["exclude channels.excluded", "common median reference (instantaneous)",
                  "FIR bandpass+notch (zero phase, reflect padding, masked edges)", "Morlet power at 1 kHz",
                  "Kaiser antialias + decimation to 100 Hz", "glove: effective 25 Hz samples -> cubic spline per block",
                  "pair ECoG(t) with flexion(t + delay)"],
        "x_dtype": "float32", "y_dtype": "float64", "scaling": "per fold (folds/), never here"}))
    write_manifest(root, PROCESSED_MANIFEST, json_safe({
        "schema_version": 1, "pipeline_hash": digest, "identity": identity,
        "created_by_run": run_id, "created_at_utc": utc_now(), "git_commit": git_commit,
        "test_accessed": False}))
