"""Step-3 orchestration: real-data checks V01-V14 plus the synthetic pytest suite.

A check passes only when its real-data part passes AND every synthetic test
marked with its id passed (none failed, skipped or missing). The run never
freezes anything: with every check in PASS it emits a candidate
``approved_pipeline_manifest.json`` that still needs the user's approval.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from .. import preprocessing as pp
from ..config import VERIFICATION_IDS, ConfigError, load_config
from ..datasets import FoldData, load_fold
from ..reporting import Artifact, RunContext
from ..io import MissingDevelopmentData, StorageUnavailable, read_development, sha256_file, verify_pipeline
from ..reporting import environment, json_safe, utc_now, write_json
from ..splits import Block
from ..io import (FOLDS_MANIFEST, PROCESSED_MANIFEST, StoreError, block_dir, fold_dir, read_json,
                     verify_manifest)

REPOSITORY = Path(__file__).resolve().parents[3]


class VerificationBlocked(RuntimeError):
    """A precondition is missing (data and selected processed pipeline for this code)."""


class BlockedCheck(RuntimeError):
    """A check cannot run because something it needs is missing."""


@dataclass
class CheckResult:
    vid: str
    title: str
    status: str  # PASS | FAIL | BLOCKED
    details: str
    artifacts: list[Artifact] = field(default_factory=list)


class VerificationContext(RunContext):
    """Evidence folder of one verification run plus the state shared by its checks."""

    def __init__(self, *args):
        super().__init__(*args)
        self.records: dict[str, tuple] = {}
        self.raw_sha256: dict[str, str] = {}
        self.pipeline_dir: Path | None = None
        self.fold_set_dir: Path | None = None
        self.purge: int | None = None
        self.purge_status: str | None = None
        self.manifest_sha256: dict[str, str] = {}
        self.pytest: dict | None = None
        self.results: dict[str, CheckResult] = {}
        self.baseline: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self.scratch = Path(tempfile.mkdtemp(prefix="tt_ecog_verify_"))

    # ------------------------------------------------------------- shared accessors
    def require_data(self) -> None:
        if self.pipeline_dir is None or self.fold_set_dir is None or not self.records:
            raise BlockedCheck("Preparation incomplete: run prepare(ctx) first (data and selected processed pipeline)")

    @property
    def subjects(self) -> list[str]:
        return list(self.config.data["project"]["subjects"])

    @property
    def block_names(self) -> list[str]:
        return list(self.config.data["validation"]["block_names"])

    @property
    def val_names(self) -> list[str]:
        return list(self.config.data["validation"]["stage2_validation_blocks"])

    def spec(self, subject: str) -> pp.PreprocessingSpec:
        return pp.PreprocessingSpec.from_config(self.config, subject)

    def blocks(self, subject: str) -> list[Block]:
        result = []
        for name in self.block_names:
            r = read_json(block_dir(self.pipeline_dir, subject, name) / "block_manifest.json")["range"]
            result.append(Block(name, r["start_target"], r["stop_target"], r["start_source"], r["stop_source"]))
        return result

    def block_arrays(self, subject: str, name: str, mmap: bool = True) -> dict[str, np.ndarray]:
        folder = block_dir(self.pipeline_dir, subject, name)
        return {"x": np.load(folder / "x.npy", mmap_mode="r" if mmap else None), "y": np.load(folder / "y.npy"),
                "x_index": np.load(folder / "x_index.npy"), "y_index": np.load(folder / "y_index.npy"),
                "valid": np.load(folder / "valid.npy")}

    def fold_folder(self, subject: str, val_name: str) -> Path:
        return fold_dir(self.fold_set_dir, subject, val_name)

    def load_fold(self, subject: str, val_name: str) -> FoldData:
        return load_fold(self.config, self.fold_set_dir, subject, val_name,
                         allow_candidate=self.purge_status != "approved", verify=False)


def check(vid: str, title: str):
    """Mark ``f(ctx) -> (status, details)`` as check ``vid``; exceptions become FAIL, BlockedCheck BLOCKED."""
    def decorate(func: Callable[[VerificationContext], tuple[str, str]]):
        @functools.wraps(func)
        def run(ctx: VerificationContext) -> CheckResult:
            outcome: dict[str, str] = {}

            def body(c, _):
                outcome["status"], outcome["details"] = func(c)
            try:
                ctx.execute(vid, title, f"{vid}: {title}", body, None)
            except BlockedCheck as exc:
                outcome.update(status="BLOCKED", details=str(exc))
            except Exception as exc:
                outcome.update(status="FAIL", details=f"{type(exc).__name__}: {exc} (traceback en errors/{vid}.txt)")
            if outcome["status"] not in ("PASS", "FAIL", "BLOCKED"):
                outcome.update(status="FAIL", details=f"invalid status {outcome['status']!r}")
            result = CheckResult(vid, title, outcome["status"], outcome["details"], list(ctx.sections[vid].artifacts))
            ctx.results[vid] = result
            return result
        run.vid, run.title = vid, title
        return run
    return decorate


def show_check(result: CheckResult) -> None:
    """Notebook display: status, details and the same files saved to Drive."""
    from IPython.display import Markdown, display
    from ..reporting import show
    display(Markdown(f"**{result.vid} · {result.title}: {result.status}** — {result.details}"))
    show(result.artifacts)


# --------------------------------------------------------------------------- run lifecycle

def start_run(config_path: str | Path, drive_root: str | Path) -> VerificationContext:
    config = load_config(config_path)
    drive_root = Path(drive_root)
    if not config.storage_root(drive_root).is_dir():
        raise StorageUnavailable("Persistent storage root does not exist; mount Drive and verify TT_ECoG")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = config.path(drive_root, "verification_dir") / run_id
    output.mkdir(parents=True, exist_ok=False)
    env = environment(config.source.parent.parent)
    env.update(seed_base=config.data["project"]["seed_base"])
    write_json(output / "environment.json", env)
    write_json(output / "effective_config.json", config.data)
    (output / "experiment_config.yaml").write_bytes(config.source.read_bytes())
    return VerificationContext(config, drive_root, output, run_id, env)


def prepare(ctx: VerificationContext) -> dict[str, Any]:
    """Load the explicitly selected pipeline; never derive its identity from this notebook or environment."""
    config, drive = ctx.config, ctx.drive_root
    selected = config.data["paths"].get("processed_pipeline_id")
    if selected is None:
        raise VerificationBlocked("Select paths.processed_pipeline_id from notebook 02")
    for subject in ctx.subjects:
        try:
            ecog, glove = read_development(config, drive, subject)
        except MissingDevelopmentData as exc:
            raise VerificationBlocked(str(exc)) from exc
        ctx.records[subject] = (ecog, glove)
        ctx.raw_sha256[subject] = ecog.metadata["sha256"]
    pipeline_dir = config.path(drive, "processed_dir") / selected
    if not pipeline_dir.is_dir():
        raise VerificationBlocked("Selected pipeline is absent; run notebook 02")
    try:
        manifest = verify_pipeline(pipeline_dir, config, ctx.raw_sha256)
    except (StoreError, OSError, ValueError, KeyError) as exc:
        raise VerificationBlocked(f"Selected pipeline failed verification: {exc}") from exc
    fold_set = pipeline_dir / "folds"
    ctx.pipeline_dir, ctx.fold_set_dir = pipeline_dir, fold_set
    ctx.purge, ctx.purge_status = config.purge_for_build()
    ctx.manifest_sha256 = {"processed_manifest": sha256_file(pipeline_dir / PROCESSED_MANIFEST),
                           "folds_manifest": sha256_file(fold_set / FOLDS_MANIFEST)}
    storage = config.storage_root(drive)
    summary = {"subjects": {s: {"sha256": d} for s, d in ctx.raw_sha256.items()},
               "pipeline_id": manifest["pipeline_id"],
               "pipeline_dir": pipeline_dir.relative_to(storage).as_posix(),
               "fold_set_dir": fold_set.relative_to(storage).as_posix(), "purge_samples": ctx.purge,
               "purge_status": ctx.purge_status, "manifests_sha256": ctx.manifest_sha256, "manifests_verified": True}
    write_json(ctx.output / "preparation.json", summary)
    return summary


def run_pytest(ctx: VerificationContext, targets: tuple[str, ...] = ()) -> dict[str, Any]:
    """Run the synthetic suite in a subprocess; outcomes per V-id via the conftest recorder."""
    if not targets:
        # Optional later-stage tests have no V01-V14 markers and require Optuna.
        # Verification must remain runnable before installing the search extra.
        targets = ("tests", "--ignore=tests/test_optimization.py", "--ignore=tests/test_final_evaluation.py")
    tests = REPOSITORY / "tests"
    if not tests.is_dir():
        ctx.pytest = {"status": "NOT_AVAILABLE", "details": f"{tests} not found", "by_check": {}}
        return ctx.pytest
    junit = ctx.output / "test_results.xml"
    by_test = ctx.output / "tables" / "pytest_by_check.json"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
               f"--verification-json={by_test}", "--basetemp", str(ctx.scratch / "pytest"), *targets]
    process = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, encoding="utf-8",
                             errors="replace")
    (ctx.output / "tables" / "pytest_output.txt").write_text(process.stdout[-20000:] + process.stderr[-5000:],
                                                              encoding="utf-8")
    if not by_test.is_file():
        ctx.pytest = {"status": "FAIL", "details": f"pytest exited {process.returncode} without results",
                      "by_check": {}, "returncode": process.returncode}
        return ctx.pytest
    records = json.loads(by_test.read_text(encoding="utf-8"))["tests"]
    counts = {k: sum(r["outcome"] == k for r in records.values()) for k in ("passed", "failed", "skipped", "not_run")}
    by_check = {vid: {k: 0 for k in counts} | {"tests": []} for vid in VERIFICATION_IDS}
    for node, record in records.items():
        for vid in record["ids"]:
            by_check[vid][record["outcome"]] += 1
            by_check[vid]["tests"].append(node)
    clean = process.returncode == 0 and counts["failed"] == counts["skipped"] == counts["not_run"] == 0
    ctx.pytest = {"status": "PASS" if clean else "FAIL", "returncode": process.returncode, **counts,
                  "targets": list(targets) or ["tests/"], "junit": "test_results.xml", "by_check": by_check}
    return ctx.pytest


def combine(vid: str, real: CheckResult | None, pytest: dict | None) -> dict[str, Any]:
    tests = (pytest or {}).get("by_check", {}).get(vid)
    synthetic = "NOT_RUN" if tests is None else (
        "FAIL" if tests["failed"] else "INCOMPLETE" if tests["skipped"] or tests["not_run"] or not tests["passed"] else "PASS")
    if real is None:
        status = "BLOCKED"
    elif real.status == "FAIL" or synthetic == "FAIL":
        status = "FAIL"
    elif real.status == "BLOCKED" or synthetic != "PASS":
        status = "BLOCKED"
    else:
        status = "PASS"
    return {"status": status, "real_data": real.status if real else "NOT_RUN", "synthetic_tests": synthetic,
            "synthetic_counts": {k: v for k, v in (tests or {}).items() if k != "tests"},
            "evidence": [a.relative for a in real.artifacts] if real else [],
            "details": real.details if real else "check not run"}


def finish_run(ctx: VerificationContext) -> dict[str, Any]:
    """Write the report, summary, candidate approval manifest (only if all pass) and hashes."""
    tests = {vid: combine(vid, ctx.results.get(vid), ctx.pytest) for vid in VERIFICATION_IDS}
    pytest_status = (ctx.pytest or {}).get("status", "NOT_RUN")
    all_pass = all(t["status"] == "PASS" for t in tests.values()) and pytest_status == "PASS"
    storage = ctx.config.storage_root(ctx.drive_root)
    report = {"schema_version": 2, "run_id": ctx.run_id, "pipeline_git_commit": ctx.env["git_commit"],
              "git_dirty": ctx.env["git_dirty"], "config_sha256": ctx.config.sha256,
              "source_sha256": ctx.env["source_sha256"], "data_sha256_by_subject": ctx.raw_sha256,
              "execution_timestamp_utc": utc_now(),
              "pipeline_id": ctx.pipeline_dir.name if ctx.pipeline_dir else None,
              "pipeline_dir": ctx.pipeline_dir.relative_to(storage).as_posix() if ctx.pipeline_dir else None,
              "fold_set_dir": ctx.fold_set_dir.relative_to(storage).as_posix() if ctx.fold_set_dir else None,
              "manifests_sha256": ctx.manifest_sha256, "purge_samples": ctx.purge, "purge_status": ctx.purge_status,
              "pytest": {k: v for k, v in (ctx.pytest or {"status": "NOT_RUN"}).items() if k != "by_check"},
              "tests": tests, "all_pass": all_pass, "approved": False,
              "approval_note": "Freezing needs the user's explicit approval (automatically_freeze_on_all_pass: false)"}
    write_json(ctx.output / "verification_report.json", json_safe(report))
    lines = [f"# Verificación {ctx.run_id}", "",
             f"Commit `{ctx.env['git_commit']}` (sin commitear: {ctx.env['git_dirty']}) · datos `{report['pipeline_dir']}` "
             f"· folds `{report['fold_set_dir']}` (purga {ctx.purge}, {ctx.purge_status}) · pytest **{pytest_status}**", "",
             "| Prueba | Estado | Datos reales | Pruebas sintéticas | Detalle |", "|---|---|---|---|---|"]
    for vid, t in tests.items():
        title = ctx.results[vid].title if vid in ctx.results else ""
        lines.append(f"| {vid} {title} | **{t['status']}** | {t['real_data']} | {t['synthetic_tests']} "
                     f"| {t['details'].replace('|', '/')[:300]} |")
    lines += ["", "**Todas PASS:** " + ("sí — se emitió `approved_pipeline_manifest.json` como candidato; la "
                                        "congelación requiere aprobación explícita." if all_pass else
                                        "no — el pipeline no se aprueba ni se congela."), ""]
    (ctx.output / "verification_summary.md").write_text("\n".join(lines), encoding="utf-8")
    if all_pass:
        write_json(ctx.output / "approved_pipeline_manifest.json", json_safe({
            "status": "all_pass_pending_user_approval", "verification_run": ctx.run_id,
            "git_commit": ctx.env["git_commit"], "source_sha256": ctx.env["source_sha256"],
            "config_sha256": ctx.config.sha256, "data_sha256_by_subject": ctx.raw_sha256,
            "pipeline_id": report["pipeline_id"], "pipeline_dir": report["pipeline_dir"],
            "fold_set_dir": report["fold_set_dir"], "manifests_sha256": ctx.manifest_sha256,
            "purge_samples": ctx.purge, "versions": ctx.env["versions"],
            "methodological_components": {
                "model": ctx.config.data["model"], "loss": ctx.config.data["training"]["loss"],
                "metrics": ctx.config.data["metrics"], "validation": ctx.config.data["validation"],
                "verification": ctx.config.data["verification"]},
            "component_source_sha256": {name: digest for name, digest in ctx.env["source_files"].items()
                                         if name.startswith("src/tt_ecog/")}}))
    summary = {"schema_version": 2, "run_id": ctx.run_id, "status": "ALL_PASS" if all_pass else "NOT_APPROVED",
               "pipeline_id": report["pipeline_id"],
               "tests": {vid: t["status"] for vid, t in tests.items()}, "pytest": pytest_status,
               "test_accessed": False, "completed_at_utc": utc_now()}
    write_json(ctx.output / "run_summary.json", summary)
    artifacts = {p.relative_to(ctx.output).as_posix(): sha256_file(p)
                 for p in sorted(ctx.output.rglob("*")) if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(ctx.output / "artifact_manifest.json", {"sha256": artifacts, "self_excluded": True})
    shutil.rmtree(ctx.scratch, ignore_errors=True)
    return report


def run_verification(config_path: str | Path, drive_root: str | Path, *, run_tests: bool = True) -> Path:
    from .checks import CHECKS
    ctx = start_run(config_path, drive_root)
    try:
        prepare(ctx)
    except VerificationBlocked as exc:
        (ctx.output / "errors" / "prepare.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"BLOCKED: {exc}")
    if run_tests:
        run_pytest(ctx)
    for run in CHECKS:
        run(ctx)
    finish_run(ctx)
    return ctx.output


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: verification V01-V14 (development data only)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true", help="Do not run pytest (the report cannot pass)")
    args = parser.parse_args()
    try:
        output = run_verification(args.config, args.drive_root, run_tests=not args.skip_tests)
    except StorageUnavailable as exc:
        print(f"BLOCKED: {exc}")
        return 2
    except (ConfigError, ValueError, PermissionError) as exc:
        print(f"FAIL: {exc}")
        return 1
    report = json.loads((output / "verification_report.json").read_text(encoding="utf-8"))
    print(("ALL_PASS" if report["all_pass"] else "NOT_APPROVED") + f": {output}")
    return 0 if report["all_pass"] else 1
