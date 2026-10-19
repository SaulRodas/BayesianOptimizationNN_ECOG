"""Content identity is shared across notebooks and independent of execution metadata."""
from copy import deepcopy
import json

import numpy as np
import pytest

from tt_ecog.config import ExperimentConfig
from tt_ecog.io import (PROCESSED_MANIFEST, StoreError, numerical_hashes, pipeline_identity,
                        processing_contract, read_development, sha256_file, verify_pipeline)
from tt_ecog.prep import start_run, publish_pipeline, PrepFailed
from tt_ecog.verification import start_run as start_verification, prepare, VerificationBlocked
from conftest import project, write_subject


def changed(config, edit):
    data = deepcopy(config.data)
    edit(data)
    return ExperimentConfig(data, config.source, "different-snapshot-hash")


@pytest.mark.verification("V01", "V14")
def test_joint_reader_hashes_twice_and_never_requests_test(config, data_root, monkeypatch):
    from tt_ecog import io
    path = write_subject(data_root, config, n=4000)
    hashes, requests = [], []
    original_hash, original_load = io.sha256_file, io.loadmat

    def digest(p):
        hashes.append(p)
        return original_hash(p)

    def load(p, **kwargs):
        requests.append(kwargs["variable_names"])
        return original_load(p, **kwargs)

    monkeypatch.setattr(io, "sha256_file", digest)
    monkeypatch.setattr(io, "loadmat", load)
    ecog, glove = read_development(config, data_root, "S1")
    assert hashes == [path, path]
    assert requests == [["train_data", "train_dg"]]
    assert ecog.metadata["sha256"] == glove.metadata["sha256"]
    assert not ecog.values.flags.writeable and not glove.values.flags.writeable


@pytest.mark.verification("V14")
def test_identity_ignores_presentation_training_paths_and_approval(prepared):
    config = prepared["config"]
    original = processing_contract(config, prepared["originals"])

    def edit(d):
        d["eda"]["figure_dpi"] = 180
        d["channels"]["excluded"]["S1"][0]["reason"] = "Rephrased acquisition evidence"
        d["channels"]["excluded"]["S1"][0]["evidence_figure"] = "fig/another-name.png"
        d["paths"]["processed_dir"] = "another/location"
        d["paths"]["processed_pipeline_id"] = None
        d["training"].update(search_epochs=5, batch_size=8)
        d["validation"]["purge_samples_at_100hz"] = 256

    assert processing_contract(changed(config, edit), prepared["originals"]) == original
    numeric = numerical_hashes(prepared["pipeline"])
    assert pipeline_identity(original, numeric) == prepared["summary"]["pipeline_id"]


@pytest.mark.verification("V14")
@pytest.mark.parametrize("edit", [
    lambda d: d["validation"].update(purge_samples_at_100hz=300),
    lambda d: d["signal"]["filtering"].update(notch_width_hz=3),
    lambda d: d["channels"]["excluded"].update(S1=[]),
    lambda d: d["scaling"]["ecog"].update(unit_variance=False),
    lambda d: d["validation"].update(train_stride_samples=2),
])
def test_effective_contract_changes_identity(config, edit):
    original = processing_contract(config, {"S1": "raw"})
    other = processing_contract(changed(config, edit), {"S1": "raw"})
    assert pipeline_identity(original, {"S1/B1/x.npy": "data"}) != pipeline_identity(
        other, {"S1/B1/x.npy": "data"})


@pytest.mark.verification("V11", "V14")
def test_numerical_changes_change_identity_and_metadata_do_not(config, tmp_path):
    np.save(tmp_path / "x.npy", np.arange(10))
    contract = processing_contract(config, {"S1": "raw"})
    first = pipeline_identity(contract, numerical_hashes(tmp_path))
    (tmp_path / "environment.json").write_text('{"run_id": "later", "commit": "new"}')
    (tmp_path / "figure.png").write_bytes(b"presentation")
    assert pipeline_identity(contract, numerical_hashes(tmp_path)) == first
    np.save(tmp_path / "x.npy", np.arange(10) + 1)
    assert pipeline_identity(contract, numerical_hashes(tmp_path)) != first
    assert pipeline_identity({**contract, "raw_sha256": {"S1": "different"}},
                             numerical_hashes(tmp_path)) != first


@pytest.mark.verification("V14")
def test_03_inherits_identity_without_eda_or_rewriting_pipeline(prepared):
    manifest_path = prepared["pipeline"] / PROCESSED_MANIFEST
    before = sha256_file(manifest_path)
    ctx = start_verification(prepared["config_path"], prepared["drive"])
    ctx.env.update(git_commit="another-code-version", source_sha256="another-code-hash")
    summary = prepare(ctx)
    assert summary["pipeline_id"] == prepared["summary"]["pipeline_id"]
    assert sha256_file(manifest_path) == before
    assert not ctx.config.path(ctx.drive_root, "eda_dir").exists()


def test_selection_is_required_and_never_falls_back(prepared, tmp_path):
    config_path = project(tmp_path)
    ctx = start_verification(config_path, prepared["drive"])
    with pytest.raises(VerificationBlocked, match="processed_pipeline_id"):
        prepare(ctx)
    assert ctx.pipeline_dir is None


def test_incomplete_preparation_cannot_publish(prepared, tmp_path):
    config_path = project(tmp_path)
    ctx = start_run(config_path, prepared["drive"])
    with pytest.raises(PrepFailed, match="Missing"):
        publish_pipeline(ctx)
    assert ctx.staging.is_dir() and ctx.pipeline_dir is None


@pytest.mark.verification("V11", "V14")
def test_pipeline_contract_mismatch_is_rejected(prepared):
    other = changed(prepared["config"], lambda d: d["validation"].update(purge_samples_at_100hz=300))
    with pytest.raises(StoreError, match="different effective contract"):
        verify_pipeline(prepared["pipeline"], other)


def test_historical_format_is_not_migrated(tmp_path):
    folder = tmp_path / "historical"
    (folder / "manifests").mkdir(parents=True)
    (folder / PROCESSED_MANIFEST).write_text(json.dumps({"schema_version": 1, "files": {}}))
    before = sha256_file(folder / PROCESSED_MANIFEST)
    with pytest.raises(StoreError, match="Historical format"):
        verify_pipeline(folder)
    assert sha256_file(folder / PROCESSED_MANIFEST) == before


def test_fold_loader_requires_the_selected_identity(prepared):
    from tt_ecog.datasets import load_fold
    config = changed(prepared["config"], lambda d: d["paths"].update(processed_pipeline_id=None))
    with pytest.raises(StoreError, match="processed_pipeline_id"):
        load_fold(config, prepared["folds"], "S1", "B2", allow_candidate=True)
    fold = load_fold(prepared["config"], prepared["folds"], "S1", "B2", allow_candidate=True)
    assert fold.pipeline_id == prepared["summary"]["pipeline_id"]


def test_incomplete_fold_cannot_publish(config_copy, tmp_path):
    from tt_ecog.config import load_config
    from tt_ecog.io import BLOCK_FILES
    config = load_config(config_copy)
    config.storage_root(tmp_path).mkdir()
    ctx = start_run(config_copy, tmp_path)
    # An interrupted save may have block files but no scaler/window files.
    subject = config.data["project"]["subjects"][0]
    for block in config.data["validation"]["block_names"]:
        folder = ctx.staging / subject / block
        folder.mkdir(parents=True)
        for name in BLOCK_FILES:
            (folder / name).touch()
    with pytest.raises(PrepFailed, match="Missing fold"):
        publish_pipeline(ctx)
    assert not (ctx.staging / PROCESSED_MANIFEST).exists()
