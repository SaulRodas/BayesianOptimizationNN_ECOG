"""EDA orchestration, identical from the CLI and the notebook."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from uuid import uuid4

from ..config import ConfigError, load_config
from ..io import (MissingDevelopmentData, StorageUnavailable, read_development,
                  sha256_file)
from ..reporting import environment, utc_now, write_json
from .bundle import SubjectBundle, calculate_overview, calculate_reference, calculate_features_and_glove
from ..reporting import RunContext
from .sections import SECTIONS


class EdaBlocked(RuntimeError):
    """Development data are missing: nothing can be described."""


class EdaFailed(RuntimeError):
    """A data contract or a section failed; see the run folder errors/."""


def start_run(config_path: str | Path, drive_root: str | Path) -> RunContext:
    """Validate configuration and storage, then create a new, never-reused run folder."""
    config = load_config(config_path)
    project_root = config.source.parent.parent
    drive_root = Path(drive_root)
    storage = config.storage_root(drive_root)
    if not storage.is_dir():
        raise StorageUnavailable("Persistent storage root does not exist; mount Drive and verify TT_ECoG")
    raw_dir = config.path(drive_root, "raw_dir")
    eda_dir = config.path(drive_root, "eda_dir")
    if eda_dir == raw_dir or eda_dir.is_relative_to(raw_dir) or raw_dir.is_relative_to(eda_dir):
        raise ValueError("Results and raw data must be separate directory trees")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = eda_dir / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = environment(project_root)
    env.update(seed_base=config.data["project"]["seed_base"], random_operations=False)
    write_json(output / "environment.json", env)
    write_json(output / "effective_config.json", config.data)
    (output / "experiment_config.yaml").write_bytes(config.source.read_bytes())
    return RunContext(config, drive_root, output, run_id, env)


def load_subjects(ctx: RunContext) -> dict[str, SubjectBundle]:
    """Read development ECoG and glove for S1/S2/S3. Test variables are never requested."""
    bundles = {}
    for subject in ctx.config.data["project"]["subjects"]:
        try:
            ecog, glove = read_development(ctx.config, ctx.drive_root, subject)
            bundle = SubjectBundle(ctx.config, ecog, glove)
        except MissingDevelopmentData as exc:
            ctx.subject_status[subject] = {"status": "BLOCKED", "details": str(exc)}
            continue
        except Exception as exc:
            ctx.subject_status[subject] = {"status": "FAIL", "details": f"{type(exc).__name__}: {exc}"}
            (ctx.output / "errors" / f"{subject}_load.txt").write_text(traceback.format_exc(), encoding="utf-8")
            continue
        ctx.subject_status[subject] = {"status": "LOADED", "n_channels": bundle.raw.shape[0],
                                       "n_kept": bundle.cmap.n_kept, "n_samples": bundle.n_samples,
                                       "duration_s": bundle.duration_s}
        ctx.data_manifest[subject] = {"ecog": ecog.metadata, "glove": glove.metadata,
                                      "channel_map": bundle.cmap.to_json()}
        bundles[subject] = bundle
    write_json(ctx.output / "data_manifest.json", ctx.data_manifest)
    statuses = {s["status"] for s in ctx.subject_status.values()}
    if statuses != {"LOADED"}:
        summary = finish_run(ctx, raise_on_failure=False)
        error = EdaFailed if "FAIL" in statuses else EdaBlocked
        raise error(f"{summary['status']}: {json.dumps(ctx.subject_status, ensure_ascii=False)}")
    return bundles


def _status(ctx: RunContext) -> str:
    subjects = {s["status"] for s in ctx.subject_status.values()}
    if "FAIL" in subjects:
        return "FAIL"
    if "BLOCKED" in subjects or not subjects:
        return "BLOCKED"
    expected = [section.SECTION_NAME for section in SECTIONS]
    if any(ctx.sections.get(name) is None or ctx.sections[name].status != "COMPLETE" for name in expected):
        return "FAIL"
    return "COMPLETE"


def finish_run(ctx: RunContext, raise_on_failure: bool = True) -> dict:
    """Write run summary, index and artifact manifest. Raise if the run is not COMPLETE."""
    missing_evidence = []
    if {s["status"] for s in ctx.subject_status.values()} == {"LOADED"}:
        for subject in ctx.config.data["project"]["subjects"]:
            for entry in ctx.config.channel_exclusions(subject):
                if not (ctx.output / entry["evidence_figure"]).is_file():
                    missing_evidence.append(f"{subject}: {entry['evidence_figure']}")
    status = _status(ctx)
    if missing_evidence and status == "COMPLETE":
        status = "FAIL"
    sections = {name: {"title": record.title, "justifies": record.justifies, "status": record.status,
                       "artifacts": [a.relative for a in record.artifacts]}
                for name, record in ctx.sections.items()}
    for section in SECTIONS:
        sections.setdefault(section.SECTION_NAME, {"title": section.TITLE, "justifies": section.JUSTIFIES,
                                                   "status": "NOT_RUN", "artifacts": []})
    summary = {"schema_version": 2, "run_id": ctx.run_id, "status": status,
               "subjects": ctx.subject_status, "sections": sections,
               "missing_exclusion_evidence": missing_evidence,
               "development_labels_use": ctx.config.data["eda"]["development_labels_use"],
               "test_accessed": False, "processed_data_persisted": False,
               "config_sha256": ctx.config.sha256, "git_commit": ctx.env["git_commit"],
               "git_dirty": ctx.env["git_dirty"], "source_sha256": ctx.env["source_sha256"],
               "completed_at_utc": utc_now()}
    write_json(ctx.output / "run_summary.json", summary)
    lines = [f"# EDA {ctx.run_id}", "", f"Estado: **{status}** · commit `{ctx.env['git_commit']}`", "",
             "EDA descriptivo de desarrollo (S1/S2/S3). Las transformaciones se aplican en memoria con "
             "`tt_ecog.preprocessing` solo para graficar; no se persisten datos procesados. Las etiquetas "
             "de desarrollo se usan solo de forma descriptiva; el test no se lee.", ""]
    for name, record in sections.items():
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
        raise EdaFailed(f"EDA {status}; revisar {ctx.output / 'run_summary.json'} y errors/")
    return summary


def run_eda(config_path: str | Path, drive_root: str | Path) -> Path:
    ctx = start_run(config_path, drive_root)
    try:
        subjects = load_subjects(ctx)
    except (EdaBlocked, EdaFailed):
        return ctx.output
    for section in SECTIONS:
        ctx.begin_section(section.SECTION_NAME, section.TITLE, section.JUSTIFIES)
        try:
            if section.SECTION_NAME == "s01_overview":
                calculate_overview(subjects)
            if section.SECTION_NAME == "s03_reference":
                calculate_reference(subjects)
            if section.SECTION_NAME == "s04_band":
                calculate_features_and_glove(subjects)
            if hasattr(section, "prepare_subject"):
                for b in subjects.values():
                    section.prepare_subject(b, ctx.config)
            ctx.execute(section.SECTION_NAME, section.TITLE, section.JUSTIFIES, section.draw, subjects)
        except Exception:
            record = ctx.sections[section.SECTION_NAME]
            record.status, record.error = "FAIL", traceback.format_exc()
            (ctx.output / "errors" / f"{section.SECTION_NAME}.txt").write_text(record.error, encoding="utf8")
            continue  # computation and plotting failures remain visible in this EDA report
    finish_run(ctx, raise_on_failure=False)
    return ctx.output


def main() -> int:
    parser = argparse.ArgumentParser(description="Development-only descriptive EDA (justifies preprocessing)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True, help="Mounted MyDrive root, or local equivalent")
    args = parser.parse_args()
    try:
        output = run_eda(args.config, args.drive_root)
    except StorageUnavailable as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except (ConfigError, ValueError, PermissionError) as exc:
        print(f"FAIL: {exc}")
        return 1
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    print(f"{summary['status']}: {output}")
    return {"COMPLETE": 0, "BLOCKED": 2}.get(summary["status"], 1)
