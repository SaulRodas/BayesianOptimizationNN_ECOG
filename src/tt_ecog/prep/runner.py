"""Step-2 orchestration, identical from the CLI and the notebook."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from uuid import uuid4

from ..config import ConfigError, load_config
from ..eda.context import RunContext
from ..io import MissingDevelopmentData, StorageUnavailable, read_development_glove, read_ecog, sha256_file
from ..reporting import environment, utc_now, write_json
from .gate import PrepBlocked
from .steps import STEPS


class PrepFailed(RuntimeError):
    """A data contract or a step failed; see the run folder errors/."""


class PrepContext(RunContext):
    """Evidence folder of one step-2 run plus the state shared by its steps."""

    def __init__(self, *args):
        super().__init__(*args)
        self.raw_sha256: dict[str, str] = {}
        self.reference: dict | None = None
        self.pipeline_dir: Path | None = None
        self.fold_set_dir: Path | None = None
        self.purge: int | None = None
        self.purge_status: str | None = None

    def execute(self, name, title, justifies, func, subjects):
        try:
            return super().execute(name, title, justifies, func, subjects)
        except PrepBlocked:
            self.sections[name].status = "BLOCKED"
            raise


def start_run(config_path: str | Path, drive_root: str | Path) -> PrepContext:
    """Validate configuration and storage, then create a new, never-reused run folder."""
    config = load_config(config_path)
    drive_root = Path(drive_root)
    if not config.storage_root(drive_root).is_dir():
        raise StorageUnavailable("Persistent storage root does not exist; mount Drive and verify TT_ECoG")
    raw_dir = config.path(drive_root, "raw_dir")
    for name in ("processed_dir", "preprocessing_dir"):
        target = config.path(drive_root, name)
        if target == raw_dir or target.is_relative_to(raw_dir) or raw_dir.is_relative_to(target):
            raise ValueError(f"{name} and raw data must be separate directory trees")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = config.path(drive_root, "preprocessing_dir") / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = environment(config.source.parent.parent)
    env.update(seed_base=config.data["project"]["seed_base"], random_operations=False)
    write_json(output / "environment.json", env)
    write_json(output / "effective_config.json", config.data)
    (output / "experiment_config.yaml").write_bytes(config.source.read_bytes())
    return PrepContext(config, drive_root, output, run_id, env)


def load_subjects(ctx: PrepContext) -> dict:
    """Read development ECoG and glove of S1/S2/S3. Test variables are never requested."""
    records = {}
    for subject in ctx.config.data["project"]["subjects"]:
        try:
            ecog = read_ecog(ctx.config, ctx.drive_root, subject)
            glove = read_development_glove(ctx.config, ctx.drive_root, ecog)
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
        summary = finish_run(ctx, raise_on_failure=False)
        error = PrepFailed if "FAIL" in statuses else PrepBlocked
        raise error(f"{summary['status']}: {json.dumps(ctx.subject_status, ensure_ascii=False)}")
    return records


def _status(ctx: PrepContext) -> str:
    states = [s["status"] for s in ctx.subject_status.values()]
    steps = [ctx.sections[s.section_name].status if s.section_name in ctx.sections else "NOT_RUN" for s in STEPS]
    if "FAIL" in states + steps:
        return "FAIL"
    if "BLOCKED" in states + steps or not states:
        return "BLOCKED"
    return "COMPLETE" if all(state == "COMPLETE" for state in steps) else "FAIL"


def finish_run(ctx: PrepContext, raise_on_failure: bool = True) -> dict:
    """Write run summary, index and artifact manifest. Raise if the run is not COMPLETE."""
    status = _status(ctx)
    storage = ctx.config.storage_root(ctx.drive_root)
    steps = {s.section_name: {"title": s.title, "justifies": s.justifies,
                              "status": ctx.sections[s.section_name].status if s.section_name in ctx.sections
                              else "NOT_RUN",
                              "artifacts": [a.relative for a in ctx.sections[s.section_name].artifacts]
                              if s.section_name in ctx.sections else []} for s in STEPS}
    summary = {"schema_version": 1, "run_id": ctx.run_id, "status": status, "subjects": ctx.subject_status,
               "steps": steps, "eda_reference_run": ctx.reference["run_id"] if ctx.reference else None,
               "pipeline_hash": ctx.pipeline_dir.name if ctx.pipeline_dir else None,
               "pipeline_dir": ctx.pipeline_dir.relative_to(storage).as_posix() if ctx.pipeline_dir else None,
               "fold_set_dir": ctx.fold_set_dir.relative_to(storage).as_posix() if ctx.fold_set_dir else None,
               "purge_samples": ctx.purge, "purge_status": ctx.purge_status,
               "approved_purge": ctx.config.data["validation"]["purge_samples_at_100hz"],
               "test_accessed": False, "config_sha256": ctx.config.sha256, "git_commit": ctx.env["git_commit"],
               "git_dirty": ctx.env["git_dirty"], "source_sha256": ctx.env["source_sha256"],
               "completed_at_utc": utc_now()}
    write_json(ctx.output / "run_summary.json", summary)
    lines = [f"# Preprocesamiento {ctx.run_id}", "", f"Estado: **{status}** · commit `{ctx.env['git_commit']}`", "",
             f"Datos procesados: `{summary['pipeline_dir']}` · folds: `{summary['fold_set_dir']}` "
             f"(purga {ctx.purge}, {ctx.purge_status}). El test no se lee.", ""]
    for name, record in steps.items():
        lines += [f"## {name} — {record['title']} ({record['status']})", "", f"Justifica: {record['justifies']}", ""]
        source = ctx.sections.get(name)
        for artifact in (source.artifacts if source else []):
            lines.append(f"- `{artifact.relative}` — {artifact.caption}")
        lines.append("")
    (ctx.output / "index.md").write_text("\n".join(lines), encoding="utf-8")
    artifacts = {p.relative_to(ctx.output).as_posix(): sha256_file(p)
                 for p in sorted(ctx.output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(ctx.output / "artifact_manifest.json", {"sha256": artifacts, "self_excluded": True})
    if raise_on_failure and status != "COMPLETE":
        raise PrepFailed(f"Preprocessing {status}; revisar {ctx.output / 'run_summary.json'} y errors/")
    return summary


def run_prep(config_path: str | Path, drive_root: str | Path) -> Path:
    ctx = start_run(config_path, drive_root)
    try:
        records = load_subjects(ctx)
    except (PrepBlocked, PrepFailed):
        return ctx.output
    for step in STEPS:
        try:
            step(ctx, records)
        except Exception:
            break  # recorded with its traceback in errors/; later steps depend on it
    finish_run(ctx, raise_on_failure=False)
    return ctx.output


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: per-block preprocessing and fold artifacts (development only)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True, help="Mounted MyDrive root, or local equivalent")
    args = parser.parse_args()
    try:
        output = run_prep(args.config, args.drive_root)
    except StorageUnavailable as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except (ConfigError, ValueError, PermissionError) as exc:
        print(f"FAIL: {exc}")
        return 1
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    print(f"{summary['status']}: {output}")
    return {"COMPLETE": 0, "BLOCKED": 2}.get(summary["status"], 1)
