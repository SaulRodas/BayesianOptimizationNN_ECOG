"""Convergence diagnostic on synthetic curves and tiny folds; never a search_epochs decision."""
from copy import deepcopy

import nbformat
import pandas as pd
import pytest
import torch
import yaml

from tt_ecog.config import ConfigError, load_config
from tt_ecog.convergence import GROUPS, convergence_decision, convergence_plan, load_convergence_results, registry_dir
from tt_ecog.io import StoreError, atomic_json, read_json
from tt_ecog.reporting import write_json
from tt_ecog.training import fold_seed
from conftest import ROOT
from test_config import write
from test_training import tiny_fold, cpu_threads


def test_plan_has_seven_runs_per_subject(config):
    plan = convergence_plan(config)
    assert len(plan) == 21 and {run["subject"] for run in plan} == {"S1", "S2", "S3"}
    runs = {run["name"]: run for run in plan if run["subject"] == "S3"}
    assert list(runs) == ["reference", "reference_seed_b", "lr_3e-05", "lr_0.00025", "lr_0.00075",
                          "arch_k3_L3_f32", "arch_k7_L5_f96"]
    seed = fold_seed(config, "S3", "B2")
    assert runs["reference"]["seed"] == seed and runs["reference_seed_b"]["seed"] == seed + 1000
    assert all(run["seed"] == seed for name, run in runs.items() if name != "reference_seed_b")
    reference = {"kernel_size": 5, "encoder_levels": 5, "base_filters": 64}
    assert {n for n, r in runs.items() if "learning_rate" in r["groups"]} == {
        "reference", "lr_3e-05", "lr_0.00025", "lr_0.00075"}
    assert all(r["architecture"] == reference for n, r in runs.items() if n.startswith(("lr_", "reference")))
    assert {n for n, r in runs.items() if "architecture" in r["groups"]} == {
        "reference", "arch_k3_L3_f32", "arch_k7_L5_f96"}
    assert all(r["lr"] == 8.42e-5 for n, r in runs.items() if not n.startswith("lr_"))
    assert all(run["weight_decay"] == 1e-4 for run in plan) and runs["reference_seed_b"]["groups"] == []


@pytest.mark.parametrize("change", [
    lambda s: s.update(candidate_epochs=[30, 40, 56]),               # candidate inside the horizon
    lambda s: s.update(horizon_epochs=[56, 59]),                     # horizon must end at epochs
    lambda s: s.update(candidate_epochs=[40, 30, 50]),
    lambda s: s.update(learning_rates=[3.0e-5, 2.0e-3]),             # outside the stage-1 range
    lambda s: s.update(learning_rates=[8.42e-5]),                    # the reference is already a run
    lambda s: s.update(architectures=[{"kernel_size": 9, "encoder_levels": 3, "base_filters": 32}]),
    lambda s: s.update(architectures=[{"kernel_size": 5, "encoder_levels": 5, "base_filters": 64}]),
    lambda s: s.update(subjects=["S4"]),
    lambda s: s.update(validation_block="B6")])
def test_invalid_diagnostic_configuration_is_rejected(config, tmp_path, change):
    data = deepcopy(config.data)
    change(data["convergence_study"])
    with pytest.raises(ConfigError, match="convergence_study"):
        load_config(write(tmp_path, data))


STUDY = {"subjects": ["S1"], "epochs": 6, "candidate_epochs": [2, 3, 4], "horizon_epochs": [5, 6]}
PLAN = [{"subject": "S1", "name": name, "groups": list(groups)} for name, groups in (
    ("reference", GROUPS), ("reference_seed_b", ()), ("fast", ("learning_rate",)),
    ("slow", ("learning_rate",)), ("small", ("architecture",)))]
REFERENCE = [.1, .2, .3, .4, .5, .5]


def curves(**overrides):
    values = {"reference": REFERENCE, "reference_seed_b": REFERENCE, "fast": [0, .1, .2, .3, .4, .4],
              "slow": [0, .1, .2, .3, .4, .4], "small": [0, .1, .2, .3, .4, .4], **overrides}
    return {("S1", name): [{"epoch": e + 1, "val_pearson_full": v} for e, v in enumerate(curve)]
            for name, curve in values.items()}


def rows(decision, group, epoch):
    return next(r for r in decision["rows"] if r["group"] == group and r["epoch"] == epoch)


def test_same_winner_at_every_candidate_selects_the_smallest():
    decision = convergence_decision(STUDY, PLAN, curves())
    assert decision["selected_epochs"] == 2 and decision["accepted_epochs"] == [2, 3, 4]
    assert all(r["winner"] == "reference" and r["loss"] == 0 for r in decision["rows"])


def test_a_fast_early_winner_is_rejected_until_the_right_one_leads():
    # "fast" leads up to epoch 3, but "slow" is better at the horizon.
    decision = convergence_decision(STUDY, PLAN, curves(fast=[.3, .4, .5, .5, .5, .5], slow=[0, .1, .2, .55, .7, .7]))
    assert decision["selected_epochs"] == 4 and decision["accepted_epochs"] == [4]
    early = rows(decision, "learning_rate", 2)
    assert early["winner"] == "fast" and early["loss"] == pytest.approx(.2) and not early["acceptable"]
    assert early["inversions"] == 2  # fast and reference both above slow at epoch 2
    assert rows(decision, "learning_rate", 4)["winner"] == "slow"
    assert all(r["acceptable"] for r in decision["rows"] if r["group"] == "architecture")


def test_no_candidate_passes_when_the_winner_is_still_wrong():
    decision = convergence_decision(STUDY, PLAN, curves(fast=[.3, .4, .5, .6, .5, .5], slow=[0, .1, .2, .3, .7, .7]))
    assert decision["selected_epochs"] is None and decision["accepted_epochs"] == []


def test_losses_within_the_seed_noise_are_accepted():
    decision = convergence_decision(STUDY, PLAN, curves(reference_seed_b=[.1, .2, .3, .4, .45, .45],
                                                        fast=[.3, .4, .5, .5, .48, .48]))
    early = rows(decision, "learning_rate", 2)
    assert early["delta"] == pytest.approx(.05) and early["loss"] == pytest.approx(.02)
    assert early["winner"] == "fast" and early["acceptable"] and decision["selected_epochs"] == 2


def test_incomplete_or_nonfinite_curves_fail():
    short = curves()
    short[("S1", "slow")] = short[("S1", "slow")][:5]
    with pytest.raises(ValueError, match="epochs 1..6"):
        convergence_decision(STUDY, PLAN, short)
    with pytest.raises(ValueError, match="finite"):
        convergence_decision(STUDY, PLAN, curves(slow=[0, .1, float("nan"), .3, .4, .4]))


def test_results_require_complete_runs_on_one_gpu(config, tmp_path):
    plan = [run for run in convergence_plan(config) if run["subject"] == "S1"][:2]
    folder = registry_dir(config, tmp_path, "a" * 64, 2)
    storage = config.storage_root(tmp_path)
    registry = {}
    for index, run in enumerate(plan):
        output = storage / "results/training/S1" / f"run{index}"
        output.mkdir(parents=True)
        pd.DataFrame([{"epoch": 1, "val_pearson_full": 0.5}]).to_csv(output / "epoch_metrics.csv", index=False)
        write_json(output / "run_summary.json", {"status": "COMPLETE"})
        write_json(output / "environment.json", {"gpu": "NVIDIA L4", "resumptions": []})
        registry[f"S1/{run['name']}"] = output.relative_to(storage).as_posix()
    atomic_json(folder / "registry.json", registry)
    histories, gpu = load_convergence_results(config, tmp_path, "a" * 64, 2, plan)
    assert gpu == "NVIDIA L4" and len(histories) == 2
    write_json(output / "environment.json", {"gpu": "NVIDIA L4", "resumptions": [{"gpu": "NVIDIA A100"}]})
    with pytest.raises(StoreError, match="mixed GPU"):
        load_convergence_results(config, tmp_path, "a" * 64, 2, plan)
    write_json(output / "run_summary.json", {"status": "INTERRUPTED"})
    with pytest.raises(StoreError, match="not complete"):
        load_convergence_results(config, tmp_path, "a" * 64, 2, plan)
    write_json(output / "run_summary.json", {"status": "COMPLETE"})
    with pytest.raises(StoreError, match="has not started"):
        load_convergence_results(config, tmp_path, "a" * 64, 2, convergence_plan(config))


def test_notebook_diagnostic_trains_resumes_and_decides(config_copy, tiny_fold, tmp_path):
    """Executes the real 04 cells and U-Nets; only data admission and the GPU are synthetic."""
    data = yaml.safe_load(config_copy.read_text(encoding="utf8"))
    data["convergence_study"].update(subjects=["S1"], epochs=3, candidate_epochs=[1, 2], horizon_epochs=[3, 3])
    data["training"].update(batch_size_initial_candidate=2, loader_workers=0)
    config_copy.write_text(yaml.safe_dump(data), encoding="utf8")
    notebook = nbformat.read(ROOT / "notebooks/04_entrenamiento.ipynb", 4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if "convergence" in c.metadata.get("tags", []))

    def namespace():
        ns = {"CONFIG": config_copy, "DRIVE_ROOT": tmp_path, "display": lambda *args: None}
        for cell in notebook.cells:
            if "setup" in cell.metadata.get("tags", []):
                exec(compile(cell.source, "04:setup", "exec"), ns)
        ns.update(device=torch.device("cpu"), pipeline_dir=tmp_path / tiny_fold.pipeline_id,
                  load_fold=lambda config, root, subject, val_block, **kwargs: tiny_fold)
        return ns

    ns = namespace()
    real_epoch, calls = ns["train_epoch"], 0

    def disconnect_in_second_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:  # run 2, epoch 2
            raise KeyboardInterrupt("synthetic disconnect")
        return real_epoch(*args, **kwargs)

    ns["train_epoch"] = disconnect_in_second_run
    with pytest.raises(KeyboardInterrupt):
        exec(code, ns)
    interrupted = ns["output"]
    assert read_json(interrupted / "run_summary.json")["status"] == "INTERRUPTED"

    ns = namespace()
    exec(code, ns)
    assert read_json(interrupted / "environment.json")["resumptions"][0]["after_epoch"] == 1
    assert pd.read_csv(interrupted / "epoch_metrics.csv")["epoch"].tolist() == [1, 2, 3]
    folder = registry_dir(ns["config"], tmp_path, tiny_fold.pipeline_id, 2)
    decision = read_json(folder / "convergence_decision.json")
    assert len(read_json(folder / "registry.json")) == 7 and (folder / "convergence_curves.png").is_file()
    assert decision["candidate_epochs"] == [1, 2] and len(decision["rows"]) == 2 * 2
    assert decision["batch_size"] == 2 and decision["pipeline_id"] == tiny_fold.pipeline_id
    assert load_config(config_copy).data["training"]["search_epochs"] is None

    ns = namespace()
    def no_more_training(*args, **kwargs):
        pytest.fail("Completed diagnostic runs must not train again")
    ns["train_epoch"] = no_more_training
    exec(code, ns)
