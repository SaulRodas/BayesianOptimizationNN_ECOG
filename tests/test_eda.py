import json
import subprocess
import sys

import nbformat
import pandas as pd

from tt_ecog.config import load_config
from tt_ecog.eda import run_eda
from tt_ecog.io import sha256_file
from conftest import ROOT, write_subject


def test_three_subject_reports_and_no_overwrite(config_copy, tmp_path):
    config = load_config(config_copy)
    drive = tmp_path / "drive"
    config.path(drive, "raw_dir").mkdir(parents=True)
    paths = [write_subject(drive, config, subject) for subject in ("S1", "S2", "S3")]
    originals = [sha256_file(p) for p in paths]
    output = run_eda(config_copy, drive)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PENDING_THRESHOLDS"
    assert not summary["qc_approved"] and not summary["pipeline_verified"]
    assert set(pd.read_csv(output / "qc_channels.csv").subject) == {"S1", "S2", "S3"}
    decisions = json.loads((output / "qc_decisions.json").read_text(encoding="utf-8"))
    assert decisions["status"] == "pending"
    assert all(not d["excluded_channel_indices_0based"] and not d["intervals"] for d in decisions["subjects"].values())
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert len(list((output / "qc_figures").glob("*.png"))) == 15
    for relative, digest in manifest["sha256"].items():
        assert sha256_file(output / relative) == digest
    assert [sha256_file(p) for p in paths] == originals
    # A repeated execution must get a distinct directory and preserve old artifacts.
    second = run_eda(config_copy, drive)
    assert second != output
    for relative, digest in manifest["sha256"].items():
        assert sha256_file(output / relative) == digest


def test_missing_data_is_blocked_not_pass(config_copy, tmp_path):
    config = load_config(config_copy)
    config.storage_root(tmp_path).mkdir()
    output = run_eda(config_copy, tmp_path)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "BLOCKED"
    assert all(s["status"] == "BLOCKED" for s in summary["subjects"])
    assert not summary["qc_approved"]
    command = subprocess.run([sys.executable, "-m", "tt_ecog.eda", "--config", str(config_copy),
                              "--drive-root", str(tmp_path)], capture_output=True, text=True)
    assert command.returncode == 2
    assert "BLOCKED" in command.stdout


def test_bad_subject_produces_fail_with_diagnostic(config_copy, tmp_path):
    import numpy as np
    config = load_config(config_copy)
    config.path(tmp_path, "raw_dir").mkdir(parents=True)
    write_subject(tmp_path, config, values=np.full((2000, 62), np.nan))
    output = run_eda(config_copy, tmp_path)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["subjects"][0]["status"] == "FAIL"
    assert "nonfinite" in summary["subjects"][0]["details"]
    assert (output / "S1_error.txt").is_file()


def test_notebook_valid_and_clean():
    notebook = nbformat.read(ROOT / "notebooks/01_exploracion_ecog.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
            compile(cell.source, "notebook", "exec")
