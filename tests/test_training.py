"""Training behavior on tiny synthetic folds; never experimental approval."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

from tt_ecog.config import ExperimentConfig, ConfigError, load_config
from tt_ecog.datasets import BlockData, FoldData, FoldWindows
from tt_ecog.io import (StoreError, approved_pipeline, read_json, sha256_file,
                        save_training_checkpoint, load_training_checkpoint)
from tt_ecog.losses import make_loss
from tt_ecog.reporting import write_json
from tt_ecog.training import (seed_run, make_loaders, make_optimizer, train_epoch, evaluate,
                              profile_updates, profile_architecture, representative_architectures)
from tt_ecog.validation import score_validation
from conftest import ROOT


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_fold():
    rng = np.random.default_rng(21)
    scaler = MinMaxScaler().fit(np.array([[-2.] * 5, [2.] * 5]))
    blocks = {}
    for number, name in enumerate(("B1", "B2")):
        x = rng.normal(size=(2, 40, 320)).astype(np.float32)
        y = (0.2 * x[0, :5] + 0.5).astype(np.float32)
        blocks[name] = BlockData(name, x, y, scaler.inverse_transform(y.T).T,
                                 np.arange(320) + number * 320, np.arange(320) + number * 320 + 20,
                                 np.ones(320, dtype=bool))
    return FoldData("S1", "val_B2", "B2", ("B1", "B2"), 256, "approved", 256, blocks,
                    {"B1": np.ones(320, dtype=bool)}, np.array([[0, 0], [0, 32], [0, 64]]),
                    np.array([0, 64]), None, scaler, {"pipeline_id": "a" * 64})


def small_model():
    return torch.nn.Sequential(torch.nn.Flatten(1, 2), torch.nn.Conv1d(80, 5, 1), torch.nn.Dropout(0.1))


def test_epoch_updates_weights_and_keeps_validation_order(config, tiny_fold):
    seed_run(7)
    train, val = make_loaders(tiny_fold, 2, 7)
    model = small_model()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = make_optimizer(model, 1e-3, 1e-4)
    assert isinstance(optimizer, torch.optim.AdamW)
    loss = train_epoch(model, optimizer, make_loss(config), train, "cpu")
    assert np.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert optimizer.param_groups[0]["lr"] == 1e-3
    metrics = evaluate(model, make_loss(config), tiny_fold, val, "cpu", 1e-9)
    with torch.no_grad():
        predictions = model(torch.tensor(np.stack([tiny_fold.val_window(i)[0] for i in range(2)]))).numpy()
    expected = score_validation(tiny_fold, predictions, 1e-9)
    assert metrics["val_pearson_full"] == pytest.approx(expected["val_pearson_full"])
    assert metrics["coverage_min_valid"] > 0 and len(metrics["pearson"]) == 5
    np.testing.assert_array_equal(FoldWindows(tiny_fold, "train")[2][0], tiny_fold.train_window(2)[0])


def test_nan_and_undefined_pearson_are_not_silenced(config, tiny_fold):
    train, val = make_loaders(tiny_fold, 2, 1)
    model = small_model()
    tiny_fold.blocks["B1"].x[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="loss"):
        train_epoch(model, make_optimizer(model, 1e-3, 0), make_loss(config), train, "cpu")
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    with pytest.raises(ValueError, match="Pearson"):
        evaluate(model, make_loss(config), tiny_fold, val, "cpu", 1e-9)


def test_resume_restores_adamw_dropout_and_window_order(config, tiny_fold, tmp_path):
    seed_run(11)
    train, _ = make_loaders(tiny_fold, 2, 11)
    model = small_model()
    optimizer = make_optimizer(model, 1e-3, 1e-4)
    loss_fn = make_loss(config)
    train_epoch(model, optimizer, loss_fn, train, "cpu")
    path = tmp_path / "last.pt"
    save_training_checkpoint(path, model, optimizer, 1, {"pipeline_id": tiny_fold.pipeline_id}, train)
    expected_loss = train_epoch(model, optimizer, loss_fn, train, "cpu")
    expected = deepcopy(model.state_dict())
    fresh = small_model()
    fresh_optimizer = make_optimizer(fresh, 1e-3, 1e-4)
    fresh_train, _ = make_loaders(tiny_fold, 2, 11)
    state = load_training_checkpoint(path, fresh, fresh_optimizer, fresh_train, tiny_fold.pipeline_id)
    assert state["epoch"] == 1
    assert train_epoch(fresh, fresh_optimizer, loss_fn, fresh_train, "cpu") == expected_loss
    assert all(torch.equal(expected[k], fresh.state_dict()[k]) for k in expected)
    with pytest.raises(StoreError, match="another pipeline"):
        load_training_checkpoint(path, fresh, fresh_optimizer, fresh_train, "b" * 64)
    other_train, _ = make_loaders(replace(tiny_fold, subject="S2"), 2, 11)
    with pytest.raises(StoreError, match="subject, fold or batch"):
        load_training_checkpoint(path, fresh, fresh_optimizer, other_train, tiny_fold.pipeline_id)


def test_interrupted_checkpoint_preserves_last_complete(config, tiny_fold, tmp_path, monkeypatch):
    train, _ = make_loaders(tiny_fold, 2, 1)
    model = small_model()
    optimizer = make_optimizer(model, 1e-3, 0)
    path = tmp_path / "last.pt"
    save_training_checkpoint(path, model, optimizer, 1, {"pipeline_id": tiny_fold.pipeline_id}, train)
    before = sha256_file(path)

    def interrupted(payload, partial):
        Path(partial).write_bytes(b"interrupted")
        raise OSError("simulated interruption")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError):
        save_training_checkpoint(path, model, optimizer, 2, {"pipeline_id": tiny_fold.pipeline_id}, train)
    assert sha256_file(path) == before


def test_profile_measures_optimizer_updates_and_does_not_claim_gpu(config, tiny_fold):
    model = small_model()
    optimizer = make_optimizer(model, 1e-3, 1e-4)
    x, y = next(iter(make_loaders(tiny_fold, 2, 1)[0]))
    result = profile_updates(model, optimizer, make_loss(config), x, y, "cpu", 2, 3)
    assert result["seconds_per_batch"] > 0
    assert result["peak_allocated_bytes"] is None and result["peak_reserved_bytes"] is None
    assert all(state["step"].item() == 5 for state in optimizer.state.values())
    arches = representative_architectures(config)
    assert len(arches) == 3 and arches[-1].key == "k7_L5_f96"


def test_profile_reports_oom_without_changing_batch(config, tiny_fold, monkeypatch):
    from tt_ecog import training

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(training, "build_model", oom)
    result = profile_architecture(config, tiny_fold, representative_architectures(config)[0], 2, "cpu", 1)
    assert result["status"] == "OOM" and result["batch_size"] == 2


def test_training_blocked_before_any_approval(config, tmp_path):
    with pytest.raises(StoreError, match="V01-V14"):
        approved_pipeline(config, tmp_path)


def approval_fixture(prepared, tmp_path):
    """Fabricated evidence ONLY to exercise the loader, never scientific approval."""
    data = deepcopy(prepared["config"].data)
    # Keep the existing prepared data while writing isolated synthetic approval evidence.
    storage = prepared["config"].storage_root(prepared["drive"])
    folder = storage / "results" / tmp_path.name
    folder.mkdir(parents=True)
    data["paths"]["persistent_storage_dir"] = "TT_ECoG"
    data["validation"]["purge_samples_at_100hz"] = 256
    data["verification"].update(status="approved", approved_manifest_file=
        (folder / "approved_pipeline_manifest.json").relative_to(storage).as_posix())
    config = ExperimentConfig(data, prepared["config"].source, "fixture")
    report = {"pipeline_id": prepared["pipeline"].name, "all_pass": True,
              "tests": {vid: {"status": "PASS"} for vid in data["verification"]["required_test_ids"]}}
    from tt_ecog.io import PROCESSED_MANIFEST
    approval = {"pipeline_id": prepared["pipeline"].name,
                "methodological_components": {"model": data["model"], "loss": data["training"]["loss"],
                                               "metrics": data["metrics"]},
                "component_source_sha256": {f"src/tt_ecog/{p.relative_to(ROOT / 'src/tt_ecog').as_posix()}": sha256_file(p)
                                            for p in (ROOT / 'src/tt_ecog').rglob('*.py')},
                "manifests_sha256": {"processed_manifest": sha256_file(prepared["pipeline"] / PROCESSED_MANIFEST)}}
    write_json(folder / "verification_report.json", report)
    write_json(folder / "approved_pipeline_manifest.json", approval)
    write_json(folder / "artifact_manifest.json", {"sha256": {p.name: sha256_file(p) for p in folder.glob('*.json')}})
    return config, folder


def test_approval_requires_matching_complete_unchanged_evidence(prepared, tmp_path):
    config, folder = approval_fixture(prepared, tmp_path)
    assert approved_pipeline(config, prepared["drive"]) == prepared["pipeline"]
    report = read_json(folder / "verification_report.json")
    report["tests"]["V04"]["status"] = "BLOCKED"
    write_json(folder / "verification_report.json", report)
    with pytest.raises(StoreError, match="PASS"):
        approved_pipeline(config, prepared["drive"])


def test_approval_rejects_changed_methodology(prepared, tmp_path):
    config, _ = approval_fixture(prepared, tmp_path)
    config.data["model"]["dropout"] = 0.2
    with pytest.raises(StoreError, match="Frozen model"):
        approved_pipeline(config, prepared["drive"])


def test_notebook_training_cells_execute_on_synthetic_cpu_data(config_copy, tiny_fold, tmp_path):
    """Executes real cells; only Colab/GPU and real-data admission are excluded."""
    import yaml
    data = yaml.safe_load(config_copy.read_text(encoding="utf8"))
    data["training"].update(batch_size_initial_candidate=2, search_epochs_candidate=2,
                           profile_warmup_steps=1, profile_measured_steps=1)
    config_copy.write_text(yaml.safe_dump(data), encoding="utf8")
    namespace = {"CONFIG": config_copy, "DRIVE_ROOT": tmp_path, "fold": tiny_fold,
                 "device": torch.device("cpu"), "display": lambda *args: None}
    notebook = nbformat.read(ROOT / "notebooks/04_entrenamiento.ipynb", 4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and set(cell.metadata.get("tags", [])) & {"setup", "analysis"}:
            exec(compile(cell.source, f"04:cell_{i}", "exec"), namespace)
    summary = read_json(namespace["output"] / "run_summary.json")
    assert summary["status"] == "COMPLETE" and summary["completed_epochs"] == 2
    assert summary["pipeline_id"] == tiny_fold.pipeline_id
    assert summary["last_val_pearson_full"] == namespace["history"][-1]["val_pearson_full"]
    assert summary["checkpoint_selection"] == "last_completed_epoch"
    assert (namespace["output"] / "learning_curves.png").is_file()
    assert (namespace["output"] / "checkpoint/last.pt").is_file()
    assert load_config(config_copy).data["training"]["search_epochs"] is None


@pytest.mark.parametrize("field,value", [("search_epochs_candidate", 0), ("batch_size", -1),
                                         ("optimizer", "sgd"), ("scheduler", "cosine")])
def test_training_rejects_invalid_execution_config(config_copy, field, value):
    import yaml
    data = yaml.safe_load(config_copy.read_text(encoding="utf8"))
    data["training"][field] = value
    config_copy.write_text(yaml.safe_dump(data), encoding="utf8")
    with pytest.raises(ConfigError):
        load_config(config_copy)
