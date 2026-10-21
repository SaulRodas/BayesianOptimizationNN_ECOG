"""Real Optuna on synthetic objectives/folds; never scientific approval or real searches."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle
import zipfile

import nbformat
import numpy as np
import optuna
import pandas as pd
import pytest
import torch

from tt_ecog.config import ConfigError
from tt_ecog.io import StoreError, read_json
from tt_ecog.optimization import (search_budget, search_session, next_trial, suggest_parameters,
                                  record_fold, complete_trial, fail_trial, completed_selection,
                                  reevaluation_candidates, reevaluation_state, reevaluation_training_run,
                                  record_reevaluation, fail_reevaluation, stage1_selection)
from tt_ecog.reporting import report_search
from conftest import ROOT
from test_training import tiny_fold, cpu_threads


@pytest.fixture
def search_config(config):
    data = deepcopy(config.data)
    data["paths"]["processed_pipeline_id"] = "a" * 64
    data["validation"]["purge_samples_at_100hz"] = 256
    data["training"].update(search_epochs=2, batch_size=2)
    return replace(config, data=data)


def history(score, epochs=2):
    return [{"epoch": e, "val_pearson_full": 0.99 if e < epochs else score,
             "elapsed_seconds": 1.0, **{f"pearson_{k}": score for k in range(5)}}
            for e in range(1, epochs + 1)]


def finish_synthetic_trial(session, trial, score=0.2):
    for block in session.contract["policy"]["validation_blocks"]:
        record_fold(session, trial, block, history(score), {"seed": 42, "run_id": "synthetic"}, {"device": "cpu"})
    return complete_trial(session, trial)


def fill_study(session):
    while (trial := next_trial(session)) is not None:
        arch, lr, wd = suggest_parameters(session, trial)
        score = float(np.sin(np.log(lr)) * 0.3 + wd)
        finish_synthetic_trial(session, trial, score)


def fill_reevaluation(session, score=lambda trial, block: 0.2):
    """Synthetic scores for the stage-1 finalists on the folds stage 1 left out."""
    for trial in reevaluation_candidates(session):
        for block in session.contract["policy"]["reevaluation"]["validation_blocks"]:
            record_reevaluation(session, trial, block, history(score(trial, block)),
                                {"seed": 42, "run_id": "synthetic"}, {"device": "cpu"})
    session.save()
    return stage1_selection(session)


def test_pending_values_do_not_use_pilot_candidates(config):
    with pytest.raises(ConfigError, match="batch_size"):
        search_budget(config)
    config.data["training"]["batch_size"] = 2
    with pytest.raises(ConfigError, match="search_epochs"):
        search_budget(config)


def test_tpe_resume_preserves_suggestions_and_total_budget(search_config, tmp_path):
    baseline, resumed = tmp_path / "base", tmp_path / "resume"
    with search_session(search_config, baseline, "S1", "stage1") as session:
        fill_study(session)
        expected = [t.params for t in session.study.trials]
        assert len(expected) == 20
        assert all(t.distributions["learning_rate"].log and t.distributions["weight_decay"].log
                   for t in session.study.trials)
    with search_session(search_config, resumed, "S1", "stage1") as session:
        for _ in range(8):
            trial = next_trial(session)
            _, lr, wd = suggest_parameters(session, trial)
            finish_synthetic_trial(session, trial, float(np.sin(np.log(lr)) * 0.3 + wd))
    with search_session(search_config, resumed, "S1", "stage1") as session:
        fill_study(session)
        assert [t.params for t in session.study.trials] == expected
        assert next_trial(session) is None
        assert report_search(session)["status"] == "REEVALUATION_PENDING"
    with search_session(search_config, resumed, "S1", "stage1") as session:
        assert next_trial(session) is None and len(session.study.trials) == 20


def test_two_stages_cover_297_fits_and_include_reference(search_config, tmp_path):
    fits = 0
    study_seeds = []
    for subject in search_config.data["project"]["subjects"]:
        with search_session(search_config, tmp_path, subject, "stage1") as session:
            study_seeds.append(session.contract["seed"])
            fill_study(session)
            selected = fill_reevaluation(session)
            fits += sum(len(t.user_attrs["folds"]) for t in session.study.trials)
            fits += sum(len(f) for f in session.study.user_attrs["reevaluation"]["folds"].values())
            assert len(session.study.trials) == 20
        with search_session(search_config, tmp_path, subject, "stage2") as session:
            assert session.contract["parent"] == selected
            while (trial := next_trial(session)) is not None:
                arch, lr, wd = suggest_parameters(session, trial)
                assert lr == selected["parameters"]["learning_rate"]
                assert wd == selected["parameters"]["weight_decay"]
                assert set(trial.params) == {"kernel_size", "encoder_levels", "base_filters"}
                if trial.number == 0:
                    assert arch.key == "k5_L5_f64"
                finish_synthetic_trial(session, trial)
            assert len(session.study.trials) == 10
            fits += sum(len(t.user_attrs["folds"]) for t in session.study.trials)
    assert fits == search_config.data["optimization"]["budgets"]["combined_search_fits"] == 297
    assert len(set(study_seeds)) == 3


def test_stage2_rejects_missing_or_incomplete_parent(search_config, tmp_path):
    with pytest.raises(StoreError, match="stage1"):
        with search_session(search_config, tmp_path, "S1", "stage2"):
            pass
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        pass
    with pytest.raises(StoreError, match="All budgeted"):
        with search_session(search_config, tmp_path, "S1", "stage2"):
            pass
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        fill_study(session)
        finalist = reevaluation_candidates(session)[0]
        record_reevaluation(session, finalist, "B1", history(0.2), {"seed": 42, "run_id": "r"}, {})
        session.save()
    with pytest.raises(StoreError, match="Re-evaluate every stage-1 finalist"):
        with search_session(search_config, tmp_path, "S1", "stage2"):
            pass


def test_finalists_are_rescored_on_the_remaining_folds(search_config, tmp_path):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        fill_study(session)
        ranked = sorted(session.study.trials, key=lambda t: (-t.value, t.number))
        finalists = reevaluation_candidates(session)
        assert [t.number for t in finalists] == [t.number for t in ranked[:3]]
        assert completed_selection(session)["trial"] == finalists[0].number
        # The B2/B4 winner falls behind on B1/B3/B5: the five-fold mean decides.
        third = finalists[2].number
        selected = fill_reevaluation(session, lambda trial, block: 0.9 if trial.number == third else -0.5)
        assert selected["trial"] == third and selected["parameters"] == finalists[2].params
        expected = (sum(finalists[2].user_attrs["folds"][b]["val_pearson_full"] for b in ("B2", "B4")) + 3 * 0.9) / 5
        assert selected["value"] == pytest.approx(expected)
        assert [f["trial"] for f in selected["finalists"]] == [t.number for t in finalists]
        assert set(selected["finalists"][0]["fold_scores"]) == {"B1", "B2", "B3", "B4", "B5"}
        summary = report_search(session)
        assert summary["status"] == "COMPLETE" and summary["selection"] == selected
        assert summary["best_search_trial"]["trial"] == finalists[0].number
        assert len(session.study.trials) == 20 and next_trial(session) is None
    with search_session(search_config, tmp_path, "S1", "stage2") as session:
        assert session.contract["parent"] == selected
        trial = next_trial(session)
        _, lr, wd = suggest_parameters(session, trial)
        assert (lr, wd) == (finalists[2].params["learning_rate"], finalists[2].params["weight_decay"])


def test_finalist_tie_keeps_the_stage1_ranking(search_config, tmp_path):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        for _ in range(20):
            trial = next_trial(session)
            suggest_parameters(session, trial)
            finish_synthetic_trial(session, trial, 0.3 if trial.number in (4, 7) else 0.1)
        assert [t.number for t in reevaluation_candidates(session)][:2] == [4, 7]
        assert fill_reevaluation(session)["trial"] == 4


def test_reevaluation_runs_resume_and_failures_stop(search_config, tiny_fold, tmp_path):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        fill_study(session)
        finalists = [t.number for t in reevaluation_candidates(session)]
        finalist = reevaluation_candidates(session)[0]
        outside = next(t for t in session.study.trials if t.number not in finalists)
        fold = replace(tiny_fold, name="val_B1", val_block="B1")
        output, parameters, arch = reevaluation_training_run(session, finalist, fold, "cpu")
        assert arch.key == "k5_L5_f64" and parameters["purpose"] == "optuna_reevaluation"
        assert parameters["lr"] == finalist.params["learning_rate"] and parameters["epochs"] == 2
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        assert reevaluation_training_run(session, finalist, fold, "cpu")[0] == output  # same run after reconnecting
        with pytest.raises(StoreError, match="finalists"):
            reevaluation_training_run(session, outside, fold, "cpu")
        with pytest.raises(StoreError, match="finalists"):
            reevaluation_training_run(session, finalist, replace(fold, name="val_B2", val_block="B2"), "cpu")
        fail_reevaluation(session, FloatingPointError("undefined Pearson"))
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        with pytest.raises(StoreError, match="diagnose"):
            reevaluation_state(session)
        assert report_search(session)["status"] == "REEVALUATION_FAILED"
        assert len(session.study.trials) == 20


def test_running_trial_resumes_same_params_and_finished_fold(search_config, tmp_path):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        trial = next_trial(session)
        suggest_parameters(session, trial)
        params = trial.params
        record_fold(session, trial, "B2", history(-0.2), {"seed": 42, "run_id": "r"}, {})
        session.save()
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        trial = next_trial(session)
        assert trial.number == 0 and trial.params == params
        assert list(trial.user_attrs["folds"]) == ["B2"]
        with pytest.raises(ValueError, match="every configured fold"):
            complete_trial(session, trial)
        record_fold(session, trial, "B4", history(0.4), {"seed": 44, "run_id": "r2"}, {})
        assert complete_trial(session, trial) == pytest.approx(0.1)  # not the earlier .99 peak


def test_failures_stop_without_extra_trials_or_partial_scores(search_config, tmp_path):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        trial = next_trial(session)
        suggest_parameters(session, trial)
        with pytest.raises(ValueError, match="Nonfinite"):
            record_fold(session, trial, "B2", history(float("nan")), {"seed": 42, "run_id": "r"}, {})
        fail_trial(session, trial, FloatingPointError("undefined Pearson"))
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        with pytest.raises(StoreError, match="failed trials"):
            next_trial(session)
        assert len(session.study.trials) == 1
        assert session.study.trials[0].value is None
        assert report_search(session)["selection"] is None


@pytest.mark.parametrize("field", ["search_epochs", "batch_size"])
def test_resume_rejects_changed_training_budget(search_config, tmp_path, field):
    with search_session(search_config, tmp_path, "S1", "stage1"):
        pass
    search_config.data["training"][field] += 1
    with pytest.raises(StoreError, match="contract changed"):
        with search_session(search_config, tmp_path, "S1", "stage1"):
            pass


def test_corruption_is_not_replaced_by_an_empty_study(search_config, tmp_path):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        path = session.archive.path
    path.write_bytes(b"broken archive")
    with pytest.raises(zipfile.BadZipFile):
        with search_session(search_config, tmp_path, "S1", "stage1"):
            pass
    assert path.read_bytes() == b"broken archive"


def test_interrupted_snapshot_preserves_previous_database_and_sampler(search_config, tmp_path, monkeypatch):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        previous = session.archive.path.read_bytes()
        trial = next_trial(session)
        suggest_parameters(session, trial)
        with monkeypatch.context() as scoped:
            def interrupted(*args, **kwargs):
                raise OSError("interrupted snapshot")
            scoped.setattr(pickle, "dumps", interrupted)
            with pytest.raises(OSError):
                session.save()
        assert session.archive.path.read_bytes() == previous
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        assert not session.study.trials


def test_interruption_after_tell_recovers_without_retraining_folds(search_config, tmp_path, monkeypatch):
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        trial = next_trial(session)
        suggest_parameters(session, trial)
        for block in ("B2", "B4"):
            record_fold(session, trial, block, history(0.2), {"seed": 42, "run_id": "r"}, {})
        session.save()
        error = OSError("snapshot interrupted after tell")
        with monkeypatch.context() as scoped:
            def interrupted(*args):
                raise error
            scoped.setattr(session.archive, "save", interrupted)
            with pytest.raises(OSError):
                complete_trial(session, trial)
            fail_trial(session, trial, error)  # retain the original I/O diagnosis
    with search_session(search_config, tmp_path, "S1", "stage1") as session:
        trial = next_trial(session)
        assert trial.number == 0 and set(trial.user_attrs["folds"]) == {"B2", "B4"}
        assert complete_trial(session, trial) == pytest.approx(0.2)


def test_notebooks_train_resume_and_transfer_stage1_on_synthetic_cpu(search_config, tiny_fold, tmp_path):
    """Execute the actual orchestration and U-Net; only data admission/GPU use synthetic fixtures."""
    search_config.data["project"]["subjects"] = ["S1"]
    for stage in ("stage1", "stage2"):
        search_config.data["optimization"][stage]["trials_per_subject"] = 1
    search_config.data["optimization"]["stage1"]["reevaluation"]["top_trials"] = 1
    blocks = {f"B{k}": replace(tiny_fold.blocks["B1" if k % 2 else "B2"], name=f"B{k}")
              for k in range(1, 6)}

    def synthetic_fold(config, root, subject, val_block, **kwargs):
        train_block = "B1" if val_block != "B1" else "B2"
        indices = tiny_fold.train_windows.copy()
        indices[:, 0] = list(blocks).index(train_block)
        return replace(tiny_fold, subject=subject, name=f"val_{val_block}", val_block=val_block,
                       block_names=tuple(blocks), blocks=blocks, train_windows=indices)

    def notebook_namespace(name):
        notebook = nbformat.read(ROOT / "notebooks" / name, 4)
        nbformat.validate(notebook)
        ns = {"CONFIG": search_config.source, "DRIVE_ROOT": tmp_path, "display": lambda *args: None}
        for cell in notebook.cells:
            if "setup" in cell.metadata.get("tags", []):
                exec(cell.source, ns)
        ns.update(config=search_config, load_fold=synthetic_fold, device=torch.device("cpu"),
                  pipeline_dir=tmp_path / "unused", batch_size=2, epochs=2)
        code = "\n".join(c.source for c in notebook.cells if "analysis" in c.metadata.get("tags", []))
        return ns, code

    ns, code = notebook_namespace("05_optuna_entrenamiento.ipynb")
    real_epoch = ns["train_epoch"]
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("synthetic disconnect")
        return real_epoch(*args, **kwargs)

    ns["train_epoch"] = interrupt_second_epoch
    with pytest.raises(KeyboardInterrupt):
        exec(code, ns)
    output = ns["output"]
    assert read_json(output / "run_summary.json")["status"] == "INTERRUPTED"
    metrics_path = output / "epoch_metrics.csv"
    frame = pd.read_csv(metrics_path)
    stale = frame.copy()
    stale["epoch"] = 999
    pd.concat([frame, stale]).to_csv(metrics_path, index=False)

    ns, code = notebook_namespace("05_optuna_entrenamiento.ipynb")
    exec(code, ns)
    assert ns["summaries"][0]["status"] == "COMPLETE"
    assert pd.read_csv(metrics_path)["epoch"].tolist() == [1, 2]
    assert read_json(output / "environment.json")["resumptions"][0]["after_epoch"] == 1
    selected = ns["summaries"][0]["selection"]
    ns, code = notebook_namespace("06_optuna_arquitectura.ipynb")
    exec(code, ns)
    assert ns["summaries"][0]["status"] == "COMPLETE"
    assert ns["summaries"][0]["parent"] == selected
    assert ns["summaries"][0]["pipeline_id"] == tiny_fold.pipeline_id
    with search_session(search_config, tmp_path, "S1", "stage2") as session:
        folds = session.study.trials[0].user_attrs["folds"]
        assert len(folds) == 5 and all(row["epoch"] == 2 for row in folds.values())
    # A completed notebook can be reopened without another optimizer update.
    ns, code = notebook_namespace("05_optuna_entrenamiento.ipynb")
    def no_more_training(*args, **kwargs):
        pytest.fail("Completed studies must not train again")
    ns["train_epoch"] = no_more_training
    exec(code, ns)
