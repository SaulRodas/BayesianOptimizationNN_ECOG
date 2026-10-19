import json
import shutil
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest

from tt_ecog.config import load_config
from tt_ecog.eda import runner
from tt_ecog.eda.context import section
from tt_ecog.eda.sections import SECTIONS
from tt_ecog.io import sha256_file
from conftest import ROOT, write_subject

NOTEBOOK = ROOT / "notebooks/01_exploracion_ecog.ipynb"


def project_copy(base):
    project = base / "p"
    (project / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/experiment_config.yaml", project / "configs/experiment_config.yaml")
    return project / "configs/experiment_config.yaml"


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One complete EDA over three synthetic 30 s subjects with injected artifacts."""
    base = tmp_path_factory.mktemp("eda")
    config_path = project_copy(base)
    config = load_config(config_path)
    drive = base / "d"
    config.path(drive, "raw_dir").mkdir(parents=True)
    paths = [write_subject(drive, config, s, n=30000, artifacts=True) for s in ("S1", "S2", "S3")]
    originals = [sha256_file(p) for p in paths]
    output = runner.run_eda(config_path, drive)
    return config, output, paths, originals


def test_full_run_is_complete_and_traceable(full_run):
    config, output, paths, originals = full_run
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE", summary
    assert summary["test_accessed"] is False and summary["processed_data_persisted"] is False
    assert [name for name in summary["sections"]] == [s.section_name for s in SECTIONS]
    for name, record in summary["sections"].items():
        assert record["status"] == "COMPLETE", name
        assert record["artifacts"], name
        for relative in record["artifacts"]:
            assert (output / relative).is_file(), relative
    assert not summary["missing_exclusion_evidence"]
    for subject in ("S1", "S2"):
        for entry in config.channel_exclusions(subject):
            assert (output / entry["evidence_figure"]).is_file()
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["sha256"].items():
        assert sha256_file(output / relative) == digest
        assert len(relative) <= 100
    assert [sha256_file(p) for p in paths] == originals
    data = json.loads((output / "data_manifest.json").read_text(encoding="utf-8"))
    assert {s: data[s]["channel_map"]["n_kept"] for s in data} == {"S1": 61, "S2": 46, "S3": 64}
    assert data["S2"]["channel_map"]["excluded_original"] == [20, 37]
    assert not list((output / "errors").iterdir())
    assert (output / "index.md").is_file()


def test_excluded_channels_lead_the_ranking(full_run):
    _, output, _, _ = full_run
    summary = pd.read_csv(output / "tables/channel_summary.csv")
    for subject, group in summary.groupby("subject"):
        excluded = int(group["excluded"].sum())
        if excluded:
            top = group.nlargest(excluded, "peak_over_typical")
            assert top["excluded"].all(), subject


def test_glove_and_delay_evidence_on_synthetic_data(full_run):
    _, output, _, _ = full_run
    glove = pd.read_csv(output / "tables/glove_storage.csv")
    assert glove["identified"].all() and (glove["method"] == "zero_order_hold").all()
    assert (glove["phase"] == 13).all()
    lags = pd.read_csv(output / "tables/lag_xcorr.csv")
    best = lags.loc[lags.groupby("subject")["median_abs_r"].idxmax()]
    assert (best["lag_s"].between(0.1, 0.35)).all()
    support = json.loads((output / "tables/support.json").read_text(encoding="utf-8"))
    assert support["measured_half_support_target"] <= support["analytic"]["total_half_support_samples_target"]


def test_new_run_never_reuses_a_folder(tmp_path):
    config_path = project_copy(tmp_path)
    load_config(config_path).storage_root(tmp_path).mkdir()
    first = runner.start_run(config_path, tmp_path)
    second = runner.start_run(config_path, tmp_path)
    assert first.output != second.output and first.output.is_dir() and second.output.is_dir()


def test_missing_data_is_blocked_not_pass(tmp_path):
    config_path = project_copy(tmp_path)
    load_config(config_path).storage_root(tmp_path).mkdir()
    output = runner.run_eda(config_path, tmp_path)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "BLOCKED"
    assert all(s["status"] == "BLOCKED" for s in summary["subjects"].values())
    assert all(s["status"] == "NOT_RUN" for s in summary["sections"].values())
    command = subprocess.run([sys.executable, "-m", "tt_ecog.eda", "--config", str(config_path),
                              "--drive-root", str(tmp_path)], capture_output=True, text=True)
    assert command.returncode == 2 and "BLOCKED" in command.stdout


def test_missing_storage_is_blocked(tmp_path):
    config_path = project_copy(tmp_path)
    command = subprocess.run([sys.executable, "-m", "tt_ecog.eda", "--config", str(config_path),
                              "--drive-root", str(tmp_path / "nowhere")], capture_output=True, text=True)
    assert command.returncode == 2 and "BLOCKED" in command.stdout


def test_bad_subject_produces_fail_with_diagnostic(tmp_path):
    config_path = project_copy(tmp_path)
    config = load_config(config_path)
    config.path(tmp_path, "raw_dir").mkdir(parents=True)
    write_subject(tmp_path, config, values=np.full((2000, 62), np.nan))
    output = runner.run_eda(config_path, tmp_path)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["subjects"]["S1"]["status"] == "FAIL"
    assert "nonfinite" in summary["subjects"]["S1"]["details"]
    assert (output / "errors/S1_load.txt").is_file()


def test_file_not_found_inside_a_section_is_fail_not_blocked(tmp_path, monkeypatch):
    """Regression: a write failure must never be reported as missing data (BLOCKED)."""
    config_path = project_copy(tmp_path)
    config = load_config(config_path)
    config.path(tmp_path, "raw_dir").mkdir(parents=True)
    for subject in ("S1", "S2", "S3"):
        write_subject(tmp_path, config, subject, n=4000)

    @section("s_fake", "Falla simulada", "regresión")
    def failing(ctx, subjects):
        raise FileNotFoundError("cannot write figure")

    monkeypatch.setattr(runner, "SECTIONS", [failing])
    output = runner.run_eda(config_path, tmp_path)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["sections"]["s_fake"]["status"] == "FAIL"
    assert "cannot write figure" in (output / "errors/s_fake.txt").read_text(encoding="utf-8")


def test_overlong_artifact_path_fails_explicitly(tmp_path, monkeypatch):
    from tt_ecog.eda import context
    config_path = project_copy(tmp_path)
    load_config(config_path).storage_root(tmp_path).mkdir()
    ctx = runner.start_run(config_path, tmp_path)
    monkeypatch.setattr(context, "MAX_ABSOLUTE_PATH", 10)
    with pytest.raises(OSError, match="too long"):
        ctx.save_table(pd.DataFrame({"a": [1]}), "x", "caption")


def test_notebook_is_light_and_valid():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in code:
        compile(source, "notebook", "exec")
        assert "def " not in source and "class " not in source
    assert "--ff-only" in code[0] and "pull" in code[0]
    joined = "\n".join(code)
    for function in SECTIONS:
        assert function.section_name in joined
    assert "finish_run" in joined


def test_committed_notebook_has_no_outputs():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.ipynb filter=nbstripout" in attributes
    shown = subprocess.run(["git", "show", "HEAD:notebooks/01_exploracion_ecog.ipynb"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
    if shown.returncode != 0:
        pytest.skip("No committed notebook available (git or HEAD missing)")
    committed = nbformat.reads(shown.stdout, as_version=4)
    for cell in committed.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
