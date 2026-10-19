"""Step 2 on three synthetic 60 s subjects (blocks of 12 s) with a synthetic EDA reference."""

from copy import deepcopy
import json
import re
import shutil
import subprocess
import sys

import nbformat
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
import yaml

from tt_ecog import preprocessing as pp
from tt_ecog.config import ExperimentConfig, load_config
from tt_ecog.datasets import load_fold
from tt_ecog.io import DataContractError, DevelopmentRecord, read_development_glove, read_ecog, sha256_file
from tt_ecog.prep import PrepBlocked, run_prep
from tt_ecog.prep import blocks as prep_blocks
from tt_ecog.prep import folds as prep_folds
from tt_ecog.prep.gate import eda_run_problems, validate_reference_run
from tt_ecog.prep.steps import STEPS
from tt_ecog.reporting import write_json
from tt_ecog.splits import make_folds
from tt_ecog.store import (FOLDS_MANIFEST, PROCESSED_MANIFEST, StoreError, block_dir, fold_dir, publish,
                           verify_manifest)
from conftest import ROOT, write_subject

RUN_ID = "20260101T000000Z_abcdef"
SUBJECTS = ("S1", "S2", "S3")
NOTEBOOK = ROOT / "notebooks/02_preprocesamiento_ecog.ipynb"


def project(base, reference_run=RUN_ID):
    (base / "p/configs").mkdir(parents=True)
    data = yaml.safe_load((ROOT / "configs/experiment_config.yaml").read_text(encoding="utf-8"))
    data["channels"]["reference_run"] = reference_run
    path = base / "p/configs/experiment_config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def write_eda_evidence(config, drive, raw_paths, run_id=RUN_ID):
    """Minimal COMPLETE EDA run folder describing ``raw_paths``."""
    run = config.path(drive, "eda_dir") / run_id
    (run / "fig").mkdir(parents=True)
    for subject in SUBJECTS:
        for entry in config.channel_exclusions(subject):
            (run / entry["evidence_figure"]).write_bytes(b"evidence " + entry["evidence_figure"].encode())
    write_json(run / "run_summary.json", {"status": "COMPLETE", "test_accessed": False, "git_commit": "eda"})
    write_json(run / "data_manifest.json", {s: {"ecog": {"sha256": sha256_file(p)}} for s, p in raw_paths.items()})
    write_json(run / "effective_config.json", config.data)
    write_json(run / "artifact_manifest.json", {"sha256": {
        p.relative_to(run).as_posix(): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()}})
    return run


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    base = tmp_path_factory.mktemp("prep")
    config_path = project(base)
    config = load_config(config_path)
    drive = base / "d"
    config.path(drive, "raw_dir").mkdir(parents=True)
    raw = {s: write_subject(drive, config, s, n=60000) for s in SUBJECTS}
    originals = {s: sha256_file(p) for s, p in raw.items()}
    write_eda_evidence(config, drive, raw)
    output = run_prep(config_path, drive)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    storage = config.storage_root(drive)
    return {"config": config, "config_path": config_path, "drive": drive, "raw": raw, "originals": originals,
            "output": output, "summary": summary, "pipeline": storage / summary["pipeline_dir"],
            "folds": storage / summary["fold_set_dir"]}


def records(prepared, subject):
    ecog = read_ecog(prepared["config"], prepared["drive"], subject)
    return ecog, read_development_glove(prepared["config"], prepared["drive"], ecog)


def test_run_is_complete_and_traceable(prepared):
    summary, output = prepared["summary"], prepared["output"]
    assert summary["status"] == "COMPLETE", summary
    assert [name for name in summary["steps"]] == [s.section_name for s in STEPS]
    assert all(step["status"] == "COMPLETE" for step in summary["steps"].values())
    assert summary["eda_reference_run"] == RUN_ID and summary["test_accessed"] is False
    assert summary["purge_samples"] == 256 and summary["purge_status"] == "candidate_pending_V04_V09"
    assert summary["approved_purge"] is None
    assert re.fullmatch(r"data/processed/[0-9a-f]{16}", summary["pipeline_dir"])
    assert re.fullmatch(r"purge_256_[0-9a-f]{16}", prepared["folds"].name)
    verify_manifest(prepared["pipeline"], PROCESSED_MANIFEST)
    folds_manifest = verify_manifest(prepared["folds"], FOLDS_MANIFEST)
    assert folds_manifest["purge_status"] == "candidate_pending_V04_V09"
    for relative, digest in json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))["sha256"].items():
        assert sha256_file(output / relative) == digest
    assert not list((output / "errors").iterdir())
    assert {s: sha256_file(p) for s, p in prepared["raw"].items()} == prepared["originals"]
    for name in ("channel_map", "filter_support", "glove_storage", "split_manifest", "preprocessing_config"):
        assert (prepared["pipeline"] / f"manifests/{name}.json").is_file()
    channel_map = json.loads((prepared["pipeline"] / "manifests/channel_map.json").read_text(encoding="utf-8"))
    assert {s: channel_map[s]["n_kept"] for s in SUBJECTS} == {"S1": 61, "S2": 46, "S3": 64}
    assert channel_map["S2"]["excluded_original"] == [20, 37]


def test_block_arrays_contract(prepared):
    config = prepared["config"]
    for subject, channels in zip(SUBJECTS, (61, 46, 64)):
        spec = pp.PreprocessingSpec.from_config(config, subject)
        half = pp.chain_half_support(spec)
        for k, name in enumerate(config.data["validation"]["block_names"]):
            folder = block_dir(prepared["pipeline"], subject, name)
            x, y = np.load(folder / "x.npy"), np.load(folder / "y.npy")
            x_index, y_index, valid = (np.load(folder / f) for f in ("x_index.npy", "y_index.npy", "valid.npy"))
            assert x.shape == (channels, 40, 1180) and x.dtype == np.float32
            assert y.shape == (5, 1180) and y.dtype == np.float64
            np.testing.assert_array_equal(x_index, 1200 * k + np.arange(1180))
            np.testing.assert_array_equal(y_index - x_index, 20)
            local = np.flatnonzero(valid)
            assert local[0] == -(-half // 10) and local[-1] == (12000 - 1 - half) // 10
            assert np.isfinite(y[:, valid]).all() and np.isfinite(x).all()
            manifest = json.loads((folder / "block_manifest.json").read_text(encoding="utf-8"))
            assert manifest["columns"]["valid"] == valid.sum()
            assert manifest["morlet"]["policy"] == "reported, never clipped"


def test_alignment_ecog_t_to_flexion_t_plus_delay(prepared):
    """Stored y[n] is the flexion at time y_index[n] = x_index[n] + 20, from B2's own samples (V06)."""
    config = prepared["config"]
    _, glove = records(prepared, "S1")
    spec = pp.PreprocessingSpec.from_config(config, "S1")
    y25, knots = pp.effective_glove(glove.values, pp.glove_storage_inspection(
        glove.values, glove.metadata["sampling_hz_glove_storage"], spec), spec)
    folder = block_dir(prepared["pipeline"], "S1", "B2")
    y, x_index, y_index, valid = (np.load(folder / f) for f in ("y.npy", "x_index.npy", "y_index.npy", "valid.npy"))
    inside = (knots >= 12000) & (knots < 24000)
    spline = CubicSpline(knots[inside] / 1000, y25[:, inside], axis=1, bc_type="not-a-knot")
    scale = np.abs(y25).max()
    np.testing.assert_allclose(y[:, valid], spline(y_index[valid] / 100), rtol=0, atol=1e-9 * scale)
    assert np.abs(y[:, valid] - spline(x_index[valid] / 100)).max() > 1e-3 * scale


def test_blocks_are_processed_independently(prepared, tmp_path):
    """Perturbing raw ECoG and glove inside B3 leaves every other block bitwise identical (V07/V08)."""
    ecog, glove = records(prepared, "S3")
    values = np.array(ecog.values, dtype=np.float64)
    for sample in (24000, 24005, 30000, 35999):  # first, interior and last samples of B3
        values[:, sample] += 1e7
    dg = np.array(glove.values)
    dg[:, 26013:26053] += 1e3  # one whole 40-row hold of the phase-13 staircase: one B3 sample
    perturbed = (DevelopmentRecord("S3", values, ecog.metadata), DevelopmentRecord("S3", dg, glove.metadata))
    prep_blocks.process_subject(prepared["config"], *perturbed, tmp_path)
    for name in ("B1", "B2", "B3", "B4", "B5"):
        for array in ("x.npy", "y.npy", "valid.npy"):
            a = np.load(block_dir(prepared["pipeline"], "S3", name) / array)
            b = np.load(tmp_path / "S3" / name / array)
            if name == "B3" and array != "valid.npy":
                assert not np.array_equal(a, b, equal_nan=True)
            else:
                np.testing.assert_array_equal(a, b)


def _arrays(prepared, subject):
    blocks, arrays = prep_folds._subject_blocks(prepared["pipeline"], subject,
                                                prepared["config"].data["validation"]["block_names"])
    return blocks, {n: {k: np.array(v) for k, v in a.items()} for n, a in arrays.items()}


def test_validation_perturbation_leaves_fold_scalers_unchanged(prepared, tmp_path):
    """Scalers and training masks of fold val_B3 do not see B3 (V05)."""
    config = prepared["config"]
    blocks, arrays = _arrays(prepared, "S2")
    fold = make_folds(blocks, ["B3"])[0]
    names = config.data["validation"]["block_names"]
    prep_folds.build_fold(config, "S2", fold, arrays, names, [], 256, "test", tmp_path / "a")
    arrays["B3"]["x"] = arrays["B3"]["x"] * 1000
    arrays["B3"]["y"] = -5 * arrays["B3"]["y"]
    prep_folds.build_fold(config, "S2", fold, arrays, names, [], 256, "test", tmp_path / "b")
    for file in ("scaler_x_center.npy", "scaler_x_scale.npy", "scaler_y_min.npy", "scaler_y_max.npy",
                 "train_windows.npy", "val_windows.npy", "eligible_B2.npy", "eligible_B4.npy"):
        np.testing.assert_array_equal(np.load(tmp_path / "a" / file), np.load(tmp_path / "b" / file))


def test_zero_range_features_or_fingers_fail(prepared, tmp_path):
    config = prepared["config"]
    names = config.data["validation"]["block_names"]
    blocks, arrays = _arrays(prepared, "S1")
    fold = make_folds(blocks, ["B2"])[0]
    constant = deepcopy(arrays)
    for name in ("B1", "B3", "B4", "B5"):
        constant[name]["x"][3, 7] = 1.0
    with pytest.raises(DataContractError, match="zero 10-90"):
        prep_folds.build_fold(config, "S1", fold, constant, names, [], 256, "test", tmp_path / "a")
    still = deepcopy(arrays)
    for name in ("B1", "B3", "B4", "B5"):
        still[name]["y"][2] = 0.5
    with pytest.raises(DataContractError, match="constant flexion"):
        prep_folds.build_fold(config, "S1", fold, still, names, [], 256, "test", tmp_path / "b")


def test_load_fold_refuses_candidates_and_returns_scaled_windows(prepared):
    config = prepared["config"]
    with pytest.raises(PermissionError, match="candidate"):
        load_fold(config, prepared["folds"], "S2", "B4")
    fold = load_fold(config, prepared["folds"], "S2", "B4", allow_candidate=True)
    assert fold.purge_samples == 256 and fold.val_block == "B4" and set(fold.train_masks) == {"B1", "B2", "B3", "B5"}
    x, y = fold.train_window(0)
    assert x.shape == (46, 40, 256) and y.shape == (5, 256) and x.dtype == np.float32
    for number, start in fold.train_windows:
        assert fold.train_masks[fold.block_names[number]][start:start + 256].all()
    xs = np.concatenate([pp.features_to_matrix(fold.blocks[n].x[:, :, m]) for n, m in fold.train_masks.items()])
    ys = np.concatenate([fold.blocks[n].y[:, m].T for n, m in fold.train_masks.items()])
    assert np.abs(np.median(xs, axis=0)).max() < 1e-4
    np.testing.assert_allclose([ys.min(), ys.max()], [0, 1], atol=1e-6)
    coverage = np.zeros(fold.blocks["B4"].valid.size, dtype=bool)
    for start in fold.val_windows:
        coverage[start:start + 256] = True
    np.testing.assert_array_equal(coverage, fold.blocks["B4"].valid)
    restored = fold.target_scaler.inverse_transform(fold.blocks["B4"].y[:, fold.blocks["B4"].valid].T.astype(np.float64))
    np.testing.assert_allclose(restored.T, fold.blocks["B4"].y_original[:, fold.blocks["B4"].valid], rtol=1e-5,
                               atol=1e-6)


def test_second_run_reuses_verified_directories(prepared):
    leftover = prepared["pipeline"].parent / f"{prepared['pipeline'].name}.partial_crashed"
    leftover.mkdir()
    output = run_prep(prepared["config_path"], prepared["drive"])
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "COMPLETE"
    assert summary["pipeline_dir"] == prepared["summary"]["pipeline_dir"]
    assert summary["fold_set_dir"] == prepared["summary"]["fold_set_dir"]
    for name in ("p02_pipeline", "p03_fold_set"):
        assert json.loads((output / f"tables/{name}.json").read_text(encoding="utf-8"))["reused"] is True


def test_corrupt_directory_is_never_reused_or_overwritten(prepared, tmp_path):
    copy = tmp_path / prepared["pipeline"].name
    shutil.copytree(prepared["pipeline"], copy, ignore=shutil.ignore_patterns("folds"))
    verify_manifest(copy, PROCESSED_MANIFEST)
    with open(block_dir(copy, "S1", "B1") / "y.npy", "ab") as stream:
        stream.write(b"0")
    with pytest.raises(StoreError, match="mismatch"):
        verify_manifest(copy, PROCESSED_MANIFEST)
    partial = tmp_path / "partial"
    shutil.copytree(prepared["pipeline"], partial, ignore=shutil.ignore_patterns("folds"))
    with pytest.raises(StoreError):
        publish(partial, copy, PROCESSED_MANIFEST)
    (copy / PROCESSED_MANIFEST).unlink()
    with pytest.raises(StoreError, match="missing"):
        verify_manifest(copy, PROCESSED_MANIFEST)


def _with(config, **channels):
    data = deepcopy(config.data)
    data["channels"].update(channels)
    return ExperimentConfig(data, config.source, config.sha256)


def test_gate_refusals(prepared, tmp_path):
    config, drive = prepared["config"], prepared["drive"]
    digests = dict(prepared["originals"])
    assert eda_run_problems(config, drive, RUN_ID, digests) == []
    with pytest.raises(PrepBlocked, match=RUN_ID):
        validate_reference_run(_with(config, reference_run=None), drive, digests)
    assert eda_run_problems(config, drive, "20260101T000000Z_000000", digests)[0].startswith("folder")
    assert any("S2" in p for p in eda_run_problems(config, drive, RUN_ID, {**digests, "S2": "0" * 64}))
    excluded = deepcopy(config.data["channels"]["excluded"])
    excluded["S3"] = [{"index": 1, "reason": "x", "evidence_figure": "fig/x.png"}]
    problems = eda_run_problems(_with(config, excluded=excluded), drive, RUN_ID, digests)
    assert any("channels.excluded" in p for p in problems) and any("fig/x.png" in p for p in problems)
    run = config.path(drive, "eda_dir") / RUN_ID
    figure = run / "fig/s02_S2_c037.png"
    original = figure.read_bytes()
    figure.write_bytes(b"tampered")
    try:
        assert any("mismatch: fig/s02_S2_c037.png" in p for p in eda_run_problems(config, drive, RUN_ID, digests))
    finally:
        figure.write_bytes(original)


def test_null_reference_run_blocks_before_processing(tmp_path, prepared):
    config_path = project(tmp_path, reference_run=None)
    drive = tmp_path / "d"
    shutil.copytree(prepared["config"].path(prepared["drive"], "raw_dir"),
                    load_config(config_path).path(drive, "raw_dir"))
    output = run_prep(config_path, drive)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "BLOCKED" and summary["steps"]["p01_gate"]["status"] == "BLOCKED"
    assert {summary["steps"][s]["status"] for s in ("p02_blocks", "p03_folds")} == {"NOT_RUN"}
    assert summary["pipeline_dir"] is None
    assert "reference_run is null" in (output / "errors/p01_gate.txt").read_text(encoding="utf-8")
    command = subprocess.run([sys.executable, "-m", "tt_ecog.prep", "--config", str(config_path),
                              "--drive-root", str(drive)], capture_output=True, text=True)
    assert command.returncode == 2 and "BLOCKED" in command.stdout


def test_step2_sources_never_name_test_variables():
    sources = [ROOT / "src/tt_ecog" / f for f in ("datasets.py", "splits.py", "store.py")]
    sources += sorted((ROOT / "src/tt_ecog/prep").glob("*.py"))
    for path in sources:
        text = path.read_text(encoding="utf-8").lower()
        for name in ("test_data", "test_dg", "testlabels", "test_labels"):
            assert name not in text, (path.name, name)


def test_notebook_02_is_light_and_valid():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    for source in code:
        compile(source, "notebook", "exec")
        assert "def " not in source and "class " not in source
    assert "--ff-only" in code[0] and "pull" in code[0]
    joined = "\n".join(code)
    for step in STEPS:
        assert step.section_name in joined
    assert "finish_run" in joined and "256" not in joined


def test_committed_notebook_02_has_no_outputs():
    shown = subprocess.run(["git", "show", "HEAD:notebooks/02_preprocesamiento_ecog.ipynb"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8")
    if shown.returncode != 0:
        pytest.skip("Notebook 02 not committed yet")
    for cell in nbformat.reads(shown.stdout, as_version=4).cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
