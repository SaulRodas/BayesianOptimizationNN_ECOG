"""Precondition of step 2: a COMPLETE EDA run over the same raw files holds the evidence.

The reference run lives in ``channels.reference_run``. It must be COMPLETE, must
not have accessed the test set, must describe the same .mat files (SHA-256) with
the same exclusion list and signal configuration, and every artifact (including
the evidence figure of each excluded channel) must match its manifest.
"""

from pathlib import Path
from typing import Any

from ..config import RUN_ID, ExperimentConfig
from ..io import sha256_file
from ..store import read_json


class PrepBlocked(RuntimeError):
    """A precondition is missing: development data or a valid EDA reference run."""


def eda_run_problems(config: ExperimentConfig, drive_root: Path, run_id: str,
                     raw_sha256: dict[str, str]) -> list[str]:
    """Reasons why an EDA run cannot back step 2 (empty list: valid evidence)."""
    run = config.path(drive_root, "eda_dir") / run_id
    if not run.is_dir():
        return [f"folder results/eda/{run_id} not found"]
    try:
        summary = read_json(run / "run_summary.json")
        data = read_json(run / "data_manifest.json")
        effective = read_json(run / "effective_config.json")
        artifacts = read_json(run / "artifact_manifest.json")["sha256"]
    except (OSError, ValueError, KeyError) as exc:
        return [f"unreadable evidence ({type(exc).__name__}: {exc})"]
    problems = []
    if summary.get("status") != "COMPLETE":
        problems.append(f"status {summary.get('status')!r}, expected 'COMPLETE'")
    if summary.get("test_accessed") is not False:
        problems.append("test_accessed is not false")
    for subject, digest in raw_sha256.items():
        if data.get(subject, {}).get("ecog", {}).get("sha256") != digest:
            problems.append(f"{subject}: the .mat SHA-256 differs from the one described by the EDA")
    if effective.get("channels", {}).get("excluded") != config.data["channels"]["excluded"]:
        problems.append("channels.excluded differs from the current configuration")
    if effective.get("signal") != config.data["signal"]:
        problems.append("the signal configuration differs from the current one")
    for subject in config.data["project"]["subjects"]:
        for entry in config.channel_exclusions(subject):
            if entry["evidence_figure"] not in artifacts:
                problems.append(f"{subject}: evidence figure {entry['evidence_figure']} not in artifact_manifest")
    for relative, digest in artifacts.items():
        path = run / relative
        if not path.is_file():
            problems.append(f"artifact missing: {relative}")
        elif sha256_file(path) != digest:
            problems.append(f"artifact SHA-256 mismatch: {relative}")
    return problems


def discover_eda_runs(config: ExperimentConfig, drive_root: Path, raw_sha256: dict[str, str]) -> list[dict[str, Any]]:
    """Every EDA run folder, newest first, with the reasons it is (not) usable."""
    eda = config.path(drive_root, "eda_dir")
    runs = sorted((p for p in eda.iterdir() if p.is_dir() and RUN_ID.match(p.name)), reverse=True) \
        if eda.is_dir() else []
    rows = []
    for run in runs:
        problems = eda_run_problems(config, drive_root, run.name, raw_sha256)
        rows.append({"run_id": run.name, "valid": not problems, "problems": problems})
    return rows


def _candidates(config: ExperimentConfig, drive_root: Path, raw_sha256: dict[str, str]) -> str:
    valid = [row["run_id"] for row in discover_eda_runs(config, drive_root, raw_sha256) if row["valid"]]
    if not valid:
        return "No EDA run is valid evidence for these files: run notebook 01 until it is COMPLETE."
    return (f"Valid EDA runs for these files (newest first): {valid}. Set channels.reference_run to one of "
            "them in configs/experiment_config.yaml, commit, push and run again.")


def validate_reference_run(config: ExperimentConfig, drive_root: Path, raw_sha256: dict[str, str]) -> dict[str, Any]:
    run_id = config.data["channels"]["reference_run"]
    if run_id is None:
        raise PrepBlocked("channels.reference_run is null (pending). " + _candidates(config, drive_root, raw_sha256))
    problems = eda_run_problems(config, drive_root, run_id, raw_sha256)
    if problems:
        raise PrepBlocked(f"EDA reference run {run_id} is not valid evidence: {problems[:5]}. "
                          + _candidates(config, drive_root, raw_sha256))
    run = config.path(drive_root, "eda_dir") / run_id
    summary = read_json(run / "run_summary.json")
    return {"run_id": run_id, "path": f"{config.data['paths']['eda_dir']}/{run_id}", "status": summary["status"],
            "git_commit": summary.get("git_commit"), "completed_at_utc": summary.get("completed_at_utc"),
            "artifact_manifest_sha256": sha256_file(run / "artifact_manifest.json"),
            "verified_artifacts": len(read_json(run / "artifact_manifest.json")["sha256"]),
            "checks": ["status COMPLETE", "test_accessed false", "raw SHA-256 per subject",
                       "channels.excluded", "signal configuration", "artifact SHA-256 incl. exclusion evidence"]}
