"""Step 3 runner and checks on the synthetic step-2 output (candidate purge, then an approved one)."""

import json
import subprocess
import sys

import nbformat
import pytest

from tt_ecog.config import VERIFICATION_IDS, load_config
from tt_ecog.prep import run_prep
from tt_ecog import verification as ver
from tt_ecog.verification.runner import CheckResult, combine
from conftest import ROOT, project

NOTEBOOK = ROOT / "notebooks/03_verificacion_pipeline.ipynb"
PURGE_DEPENDENT = {"V04", "V09"}


def clean_git(ctx):
    """The temporary project shell is not a git checkout; Colab's clone is."""
    ctx.env.update(git_commit="0" * 40, git_dirty=False)


@pytest.fixture(scope="module")
def candidate_run(prepared):
    ctx = ver.start_run(prepared["config_path"], prepared["drive"])
    clean_git(ctx)
    ver.prepare(ctx)
    results = {run.vid: run(ctx) for run in ver.checks.CHECKS}
    return ctx, results


def test_every_check_runs_on_real_artifacts_and_the_purge_blocks(candidate_run):
    ctx, results = candidate_run
    assert list(results) == VERIFICATION_IDS
    statuses = {vid: r.status for vid, r in results.items()}
    assert {vid for vid, s in statuses.items() if s != "PASS"} == PURGE_DEPENDENT, \
        {vid: (r.status, r.details) for vid, r in results.items() if r.status != "PASS"}
    assert all(statuses[vid] == "BLOCKED" for vid in PURGE_DEPENDENT)
    assert "purga mínima que aísla contextos y etiquetas: 0" in results["V04"].details.lower()
    for result in results.values():
        assert result.artifacts and all(a.path.is_file() for a in result.artifacts)
    assert (ctx.output / "filter_support.json").is_file()
    assert not list((ctx.output / "errors").iterdir())


def test_report_never_approves_with_a_pending_purge(candidate_run):
    ctx, _ = candidate_run
    ctx.pytest = {"status": "PASS", "by_check": {vid: {"passed": 1, "failed": 0, "skipped": 0, "not_run": 0}
                                                 for vid in VERIFICATION_IDS}}
    report = ver.finish_run(ctx)
    assert report["all_pass"] is False and report["approved"] is False
    assert {vid for vid, t in report["tests"].items() if t["status"] != "PASS"} == PURGE_DEPENDENT
    for name in ("verification_report.json", "verification_summary.md", "filter_support.json", "artifact_manifest.json"):
        assert (ctx.output / name).is_file()
    assert not (ctx.output / "approved_pipeline_manifest.json").exists()
    assert json.loads((ctx.output / "run_summary.json").read_text(encoding="utf-8"))["status"] == "NOT_APPROVED"


def test_approved_purge_unblocks_v04_and_v09(prepared, tmp_path):
    config_path = project(tmp_path, purge_samples_at_100hz=256)
    drive = prepared["drive"]
    assert load_config(config_path).purge_for_build() == (256, "approved")
    output = run_prep(config_path, drive)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE" and summary["purge_status"] == "approved"
    assert summary["pipeline_dir"] == prepared["summary"]["pipeline_dir"]  # blocks reused, folds rebuilt
    ctx = ver.start_run(config_path, drive)
    ver.prepare(ctx)
    assert ctx.purge_status == "approved"
    assert ver.checks.v04_splits(ctx).status == "PASS"
    assert ver.checks.v09_filters(ctx).status == "PASS"


def result(status):
    return CheckResult("V01", "x", status, "d")


@pytest.mark.parametrize("real,tests,expected", [
    ("PASS", {"passed": 2, "failed": 0, "skipped": 0, "not_run": 0}, "PASS"),
    ("PASS", {"passed": 2, "failed": 1, "skipped": 0, "not_run": 0}, "FAIL"),
    ("PASS", {"passed": 2, "failed": 0, "skipped": 1, "not_run": 0}, "BLOCKED"),
    ("PASS", None, "BLOCKED"),
    ("BLOCKED", {"passed": 2, "failed": 0, "skipped": 0, "not_run": 0}, "BLOCKED"),
    ("FAIL", {"passed": 2, "failed": 0, "skipped": 0, "not_run": 0}, "FAIL"),
    (None, {"passed": 2, "failed": 0, "skipped": 0, "not_run": 0}, "BLOCKED")])
def test_a_check_passes_only_with_real_and_synthetic_evidence(real, tests, expected):
    pytest_summary = None if tests is None else {"by_check": {"V01": tests}}
    assert combine("V01", result(real) if real else None, pytest_summary)["status"] == expected


def test_pytest_outcomes_are_mapped_to_checks(prepared):
    ctx = ver.start_run(prepared["config_path"], prepared["drive"])
    summary = ver.run_pytest(ctx, ("tests/test_datasets.py",))
    assert summary["status"] == "PASS" and summary["failed"] == summary["skipped"] == 0
    assert summary["by_check"]["V10"]["passed"] >= 10 and summary["by_check"]["V12"]["passed"] == 2
    assert summary["by_check"]["V13"]["passed"] == 0
    assert (ctx.output / "test_results.xml").is_file()


def test_missing_processed_data_blocks_preparation(prepared, tmp_path):
    config_path = project(tmp_path, purge_samples_at_100hz=300)  # no fold set was built for this purge
    ctx = ver.start_run(config_path, prepared["drive"])
    with pytest.raises(ver.VerificationBlocked, match="notebook 02"):
        ver.prepare(ctx)
    assert ver.checks.v01_integrity(ctx).status == "BLOCKED"


def test_cli_without_storage_is_blocked(tmp_path):
    command = subprocess.run([sys.executable, "-m", "tt_ecog.verification", "--config", str(project(tmp_path)),
                              "--drive-root", str(tmp_path / "nowhere"), "--skip-tests"],
                             capture_output=True, text=True)
    assert command.returncode == 2 and "BLOCKED" in command.stdout


def test_notebook_03_is_light_and_valid():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in code:
        compile(source, "notebook", "exec")
        assert "def " not in source and "class " not in source
    assert "--ff-only" in code[0] and "pull" in code[0]
    joined = "\n".join(code)
    for run in ver.checks.CHECKS:
        assert f"checks.{run.__name__}" in joined
    for name in ("prepare", "run_pytest", "finish_run"):
        assert name in joined


def test_committed_notebook_03_has_no_outputs():
    shown = subprocess.run(["git", "show", "HEAD:notebooks/03_verificacion_pipeline.ipynb"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
    if shown.returncode != 0:
        pytest.skip("Notebook 03 not committed yet")
    for cell in nbformat.reads(shown.stdout, as_version=4).cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
