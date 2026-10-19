"""Development preparation. Notebook cells and CLI share numerical and persistence functions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from uuid import uuid4

from ..config import ConfigError, load_config
from ..io import (MissingDevelopmentData, StorageUnavailable, read_development, sha256_file,
                  processing_contract, numerical_hashes, pipeline_identity, write_manifest,
                  verify_pipeline, publish, FOLDS_MANIFEST, PROCESSED_MANIFEST, BLOCK_FILES)
from ..reporting import RunContext, environment, write_json, utc_now
from ..splits import make_folds
from .blocks import process_subject, write_processing_metadata
from .folds import subject_blocks, build_fold


class PrepBlocked(RuntimeError):
    """Development data or storage are unavailable."""


class PrepFailed(RuntimeError):
    """Preparation failed; inspect the recorded diagnostic."""


class PrepContext(RunContext):
    def __init__(self, *args):
        super().__init__(*args)
        self.raw_sha256 = {}
        self.pipeline_dir = None
        self.fold_set_dir = None
        self.purge, self.purge_status = self.config.purge_for_build()
        self.status = "PENDING"
        self.reused = False
        self.staging = self.config.path(self.drive_root, "processed_dir") / f".partial_{self.run_id}"


def start_run(config_path, drive_root):
    config = load_config(config_path)
    drive_root = Path(drive_root)
    if not config.storage_root(drive_root).is_dir():
        raise StorageUnavailable("Mount Drive and verify TT_ECoG")
    raw = config.path(drive_root, "raw_dir")
    for name in ("processed_dir", "preprocessing_dir"):
        target = config.path(drive_root, name)
        if target == raw or target.is_relative_to(raw) or raw.is_relative_to(target):
            raise ValueError(f"{name} and raw data must be separate directory trees")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = config.path(drive_root, "preprocessing_dir") / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = environment(config.source.parent.parent)
    env.update(seed_base=config.data["project"]["seed_base"])
    write_json(output / "environment.json", env)
    write_json(output / "effective_config.json", config.data)
    (output / "experiment_config.yaml").write_bytes(config.source.read_bytes())
    ctx = PrepContext(config, drive_root, output, run_id, env)
    ctx.staging.mkdir(parents=True, exist_ok=False)
    return ctx


def load_subjects(ctx: PrepContext) -> dict:
    """Read development ECoG and glove of S1/S2/S3. Test variables are never requested."""
    records = {}
    for subject in ctx.config.data["project"]["subjects"]:
        try:
            ecog, glove = read_development(ctx.config, ctx.drive_root, subject)
        except MissingDevelopmentData as exc:
            ctx.subject_status[subject] = {"status": "BLOCKED", "details": str(exc)}
            continue
        except Exception as exc:
            ctx.subject_status[subject] = {"status": "FAIL", "details": f"{type(exc).__name__}: {exc}"}
            (ctx.output / "errors" / f"{subject}_load.txt").write_text(traceback.format_exc(), encoding="utf-8")
            continue
        ctx.subject_status[subject] = {"status": "LOADED", "n_channels": ecog.values.shape[0],
                                       "n_samples": ecog.values.shape[1]}
        ctx.raw_sha256[subject] = ecog.metadata["sha256"]
        ctx.data_manifest[subject] = {"ecog": ecog.metadata, "glove": glove.metadata}
        records[subject] = (ecog, glove)
    write_json(ctx.output / "data_manifest.json", ctx.data_manifest)
    statuses = {s["status"] for s in ctx.subject_status.values()}
    if statuses != {"LOADED"}:
        ctx.status = "FAIL" if "FAIL" in statuses else "BLOCKED"
        summary = finish_run(ctx, raise_on_failure=False)
        error = PrepFailed if "FAIL" in statuses else PrepBlocked
        raise error(f"{summary['status']}: {json.dumps(ctx.subject_status, ensure_ascii=False)}")
    return records


def publish_pipeline(ctx):
    """Finalize numeric identity, then publish without overwriting an earlier run."""
    config = ctx.config
    # Require all declared subjects, blocks and folds before issuing an identity.
    for subject in config.data["project"]["subjects"]:
        for block in config.data["validation"]["block_names"]:
            for name in BLOCK_FILES:
                if not (ctx.staging / subject / block / name).is_file():
                    raise PrepFailed(f"Missing {subject}/{block}/{name}")
        for val in config.data["validation"]["stage2_validation_blocks"]:
            required = ["fold_manifest.json", "scaler_x_center.npy", "scaler_x_scale.npy",
                        "scaler_y_min.npy", "scaler_y_max.npy", "train_windows.npy", "val_windows.npy"]
            required += [f"eligible_{b}.npy" for b in config.data["validation"]["block_names"] if b != val]
            for name in required:
                if not (ctx.staging / "folds" / subject / f"val_{val}" / name).is_file():
                    raise PrepFailed(f"Missing fold {subject}/{val}/{name}")
    for name in ("channel_map", "preprocessing_config", "split_manifest", "filter_support", "glove_storage"):
        if not (ctx.staging / "manifests" / f"{name}.json").is_file():
            raise PrepFailed(f"Missing {name} metadata")
    contract = processing_contract(config, ctx.raw_sha256)
    digest = pipeline_identity(contract, numerical_hashes(ctx.staging))
    write_manifest(ctx.staging / "folds", FOLDS_MANIFEST, {
        "schema_version": 2, "pipeline_id": digest, "purge_samples": ctx.purge,
        "scaling": config.data["scaling"], "test_accessed": False})
    write_manifest(ctx.staging, PROCESSED_MANIFEST, {
        "schema_version": 2, "pipeline_id": digest, "identity": contract,
        "created_by_run": ctx.run_id, "created_at_utc": utc_now(),
        "git_commit": ctx.env["git_commit"], "source_sha256": ctx.env["source_sha256"],
        "config_sha256": config.sha256, "test_accessed": False})
    final = config.path(ctx.drive_root, "processed_dir") / digest
    if final.exists():
        verify_pipeline(final, config, ctx.raw_sha256)
    ctx.reused = not publish(ctx.staging, final, PROCESSED_MANIFEST)
    verify_pipeline(final, config, ctx.raw_sha256)
    ctx.pipeline_dir, ctx.fold_set_dir = final, final / "folds"
    ctx.status = "COMPLETE"
    return digest


def finish_run(ctx, raise_on_failure=True):
    storage = ctx.config.storage_root(ctx.drive_root)
    summary = {"schema_version": 2, "run_id": ctx.run_id, "status": ctx.status,
               "subjects": ctx.subject_status, "pipeline_id": ctx.pipeline_dir.name if ctx.pipeline_dir else None,
               "pipeline_dir": ctx.pipeline_dir.relative_to(storage).as_posix() if ctx.pipeline_dir else None,
               "fold_set_dir": ctx.fold_set_dir.relative_to(storage).as_posix() if ctx.fold_set_dir else None,
               "purge_samples": ctx.purge, "purge_status": ctx.purge_status,
               "approved_purge": ctx.config.data["validation"]["purge_samples_at_100hz"],
               "reused": ctx.reused, "test_accessed": False,
               "config_sha256": ctx.config.sha256, "git_commit": ctx.env["git_commit"],
               "source_sha256": ctx.env["source_sha256"], "completed_at_utc": utc_now()}
    write_json(ctx.output / "run_summary.json", summary)
    (ctx.output / "index.md").write_text(
        f"# Preprocesamiento {ctx.run_id}\n\nEstado: {ctx.status}\n\n"
        f"Pipeline: `{summary['pipeline_id']}`\n\nPurga: {ctx.purge} ({ctx.purge_status}).\n"
        "Copiar el pipeline_id a paths.processed_pipeline_id para ejecutar 03.\n", encoding="utf8")
    artifacts = {p.relative_to(ctx.output).as_posix(): sha256_file(p)
                 for p in sorted(ctx.output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(ctx.output / "artifact_manifest.json", {"sha256": artifacts, "self_excluded": True})
    if raise_on_failure and ctx.status != "COMPLETE":
        raise PrepFailed(f"Preprocessing {ctx.status}; inspect {ctx.output}")
    return summary


def run_prep(config_path, drive_root):
    ctx = start_run(config_path, drive_root)
    try:
        records = load_subjects(ctx)
        results = {s: process_subject(ctx.config, *pair, ctx.staging) for s, pair in records.items()}
        write_processing_metadata(ctx.config, records, results, ctx.staging)
        names = ctx.config.data["validation"]["block_names"]
        for subject in records:
            blocks, arrays = subject_blocks(ctx.staging, subject, names)
            for fold in make_folds(blocks, ctx.config.data["validation"]["stage2_validation_blocks"]):
                build_fold(ctx.config, subject, fold, arrays, names, list(results[subject]["spec"].channels.kept),
                           ctx.purge, ctx.purge_status, ctx.staging / "folds" / subject / fold.name)
        del arrays  # Release memory maps before renaming the directory on Windows.
        publish_pipeline(ctx)
    except PrepBlocked:
        ctx.status = "BLOCKED"
    except Exception:
        ctx.status = "FAIL"
        (ctx.output / "errors/preparation.txt").write_text(traceback.format_exc(), encoding="utf8")
    finish_run(ctx, raise_on_failure=False)
    return ctx.output


def main():
    parser = argparse.ArgumentParser(description="Prepare development data (EDA optional)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = run_prep(args.config, args.drive_root)
    except StorageUnavailable as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except (ConfigError, ValueError, PermissionError) as exc:
        print(f"FAIL: {exc}")
        return 1
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf8"))
    print(f"{summary['status']}: {output}")
    print("pipeline_id:", summary["pipeline_id"])
    return {"COMPLETE": 0, "BLOCKED": 2}.get(summary["status"], 1)
