"""Sequential TPE decisions. The notebooks expose subjects, trials, folds and epochs."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math

import optuna
from optuna.trial import TrialState

from .config import ConfigError, inside
from .io import StudyArchive, StoreError, read_json
from .model import architecture, reference_architecture
from .reporting import start_training_run
from .training import fold_seed


def search_budget(config):
    """Pending values never inherit pilot candidates or final-training settings."""
    settings = config.data["training"]
    for key in ("batch_size", "search_epochs"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ConfigError(f"training.{key} is pending; decide it from notebook 04 before Optuna")
    if (settings["search_early_stopping"] is not False
            or settings["search_checkpoint_selection"] != "last_epoch"
            or settings["search_checkpoint_metric"] != "val_pearson_full"
            or settings["common_seeds_between_configurations"] is not True):
        raise ConfigError("Search requires fixed epochs, last-epoch Pearson and common fold seeds")
    policy = config.data["optimization"]
    if policy["algorithm"] != "optuna_tpe" or policy["study_direction"] != "maximize":
        raise ConfigError("Search requires single-objective TPE maximization")
    return settings["batch_size"], settings["search_epochs"]


def study_contract(config, subject, stage, parent=None):
    batch, epochs = search_budget(config)
    if subject not in config.data["project"]["subjects"] or stage not in ("stage1", "stage2"):
        raise ConfigError("Unknown subject or optimization stage")
    pipeline_id = config.data["paths"]["processed_pipeline_id"]
    if pipeline_id is None:
        raise ConfigError("Select paths.processed_pipeline_id before Optuna")
    config.approved_purge()
    policy = config.data["optimization"][stage]
    if type(policy["trials_per_subject"]) is not int or policy["trials_per_subject"] < 1:
        raise ConfigError("trials_per_subject must be positive")
    return deepcopy({"pipeline_id": pipeline_id, "subject": subject, "stage": stage,
                     "batch_size": batch, "epochs": epochs, "policy": policy,
                     "model": config.data["model"], "loss": config.data["training"]["loss"],
                     "metrics": config.data["metrics"], "parent": parent,
                     "seed": config.data["project"]["seed_base"] + config.data["project"]["subjects"].index(subject),
                     "optimizer": "adamw", "scheduler": "none", "precision": "float32",
                     "checkpoint_selection": "last_epoch"})


@dataclass
class SearchSession:
    config: object
    drive_root: object
    archive: StudyArchive
    storage: object
    study: object
    contract: dict

    def save(self):
        self.archive.save(self.study.sampler)


@contextmanager
def search_session(config, drive_root, subject, stage, *, existing_only=False):
    """One local writer and a persistent snapshot per pipeline/subject/stage.

    Call approved_pipeline before entering an experimental session. Keeping this
    data-free layer separate lets synthetic tests exercise Optuna without faking
    scientific approval or reading any raw/test labels.
    """
    parent = None
    if stage == "stage2":
        with search_session(config, drive_root, subject, "stage1", existing_only=True) as previous:
            parent = stage1_selection(previous)
    contract = study_contract(config, subject, stage, parent)
    folder = config.path(drive_root, "optimization_dir") / contract["pipeline_id"] / subject / stage
    if existing_only and not (folder / "study.zip").is_file():
        raise StoreError("Complete stage1 for this pipeline and subject before stage2")
    with StudyArchive(folder) as archive:
        sampler = archive.sampler
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=contract["seed"],
                       n_startup_trials=contract["policy"]["n_startup_trials"])
        storage = optuna.storages.RDBStorage("sqlite:///" + archive.database.as_posix())
        try:
            study = optuna.create_study(storage=storage, study_name=f"{subject}_{stage}",
                    sampler=sampler, pruner=optuna.pruners.NopPruner(),
                    direction="maximize", load_if_exists=True)
            recorded = study.user_attrs.get("contract")
            if recorded is not None and recorded != contract:
                raise StoreError("Study contract changed: do not mix epochs, batch, pipeline or search settings")
            if recorded is None and archive.path.exists():
                raise StoreError("Existing study has no execution contract")
            session = SearchSession(config, drive_root, archive, storage, study, contract)
            if recorded is None:
                study.set_user_attr("contract", contract)
                if stage == "stage2":
                    study.enqueue_trial(contract["policy"]["enqueue_reference_architecture_within_budget"])
                session.save()
            yield session
        finally:
            storage.remove_session()
            storage.engine.dispose()


def next_trial(session):
    """Resume RUNNING, otherwise ask once; reopening never adds another budget."""
    study, trials = session.study, session.study.get_trials()
    if any(t.state in (TrialState.FAIL, TrialState.PRUNED) for t in trials):
        raise StoreError("Study contains failed trials; diagnose and document recovery before continuing")
    running = [t for t in trials if t.state == TrialState.RUNNING]
    if len(running) > 1:
        raise StoreError("Sequential search cannot resume multiple RUNNING trials")
    if running:
        study_id = session.storage.get_study_id_from_name(study.study_name)
        trial_id = session.storage.get_trial_id_from_study_id_trial_number(study_id, running[0].number)
        return optuna.Trial(study, trial_id)
    budget = session.contract["policy"]["trials_per_subject"]
    if sum(t.state == TrialState.COMPLETE for t in trials) == budget:
        return None
    if len(trials) >= budget and not any(t.state == TrialState.WAITING for t in trials):
        raise StoreError("Trial budget exhausted without a complete study")
    return study.ask()


def suggest_parameters(session, trial):
    config, contract = session.config, session.contract
    space = contract["policy"]["search"]
    if contract["stage"] == "stage1":
        values = {name: trial.suggest_float(name, limits["min"], limits["max"], log=True)
                  for name, limits in space.items()}
        arch = reference_architecture(config)
    else:
        selected = {name: trial.suggest_categorical(name, choices) for name, choices in space.items()}
        arch = architecture(config, **selected)
        values = contract["parent"]["parameters"]
    repeats = [t.number for t in session.study.get_trials()
               if t.number < trial.number and t.params == trial.params]
    trial.set_user_attr("duplicate_of", repeats)
    trial.set_user_attr("pipeline_id", contract["pipeline_id"])
    trial.set_user_attr("epochs", contract["epochs"])
    return arch, values["learning_rate"], values["weight_decay"]


def trial_training_run(session, trial, fold, arch, lr, weight_decay, device):
    """Locate the same fold run after a reconnect; model/checkpoint loading stays visible."""
    contract = session.contract
    parameters = {"purpose": "optuna", "stage": contract["stage"], "study": session.study.study_name,
                  "trial": trial.number, "architecture": arch.key,
                  "seed": fold_seed(session.config, fold.subject, fold.val_block),
                  "batch_size": contract["batch_size"], "epochs": contract["epochs"],
                  "lr": lr, "weight_decay": weight_decay}
    runs = trial.user_attrs.get("runs", {})
    if fold.val_block in runs:
        output = inside(session.config.storage_root(session.drive_root), runs[fold.val_block])
        recorded = read_json(output / "config.json")["run"]
        if any(recorded.get(key) != value for key, value in parameters.items()):
            raise StoreError("Stored fold run differs from the current trial")
        return output, recorded
    output, parameters = start_training_run(session.config, session.drive_root, fold, parameters, device)
    runs[fold.val_block] = output.relative_to(session.config.storage_root(session.drive_root)).as_posix()
    trial.set_user_attr("runs", runs)
    session.save()  # Publish the run location before the first optimizer update.
    return output, parameters


def _fold_record(session, history, parameters, device_metadata):
    epochs = session.contract["epochs"]
    if [row["epoch"] for row in history] != list(range(1, epochs + 1)):
        raise ValueError("A fold must finish the fixed epoch budget before scoring")
    last = history[-1]
    score = last["val_pearson_full"]
    fingers = {k: v for k, v in last.items() if k.startswith("pearson_")}
    if len(fingers) != 5 or not all(math.isfinite(v) for v in [score, *fingers.values()]):
        raise ValueError("Nonfinite or incomplete fold Pearson")
    return {"val_pearson_full": score, "pearson_per_finger": fingers,
            "epoch": epochs, "seed": parameters["seed"], "run_id": parameters["run_id"],
            "elapsed_seconds": sum(row["elapsed_seconds"] for row in history),
            "device": device_metadata}


def record_fold(session, trial, val_block, history, parameters, device_metadata):
    folds = trial.user_attrs.get("folds", {})
    folds[val_block] = _fold_record(session, history, parameters, device_metadata)
    trial.set_user_attr("folds", folds)


def complete_trial(session, trial):
    folds = trial.user_attrs.get("folds", {})
    expected = session.contract["policy"]["validation_blocks"]
    if set(folds) != set(expected):
        raise ValueError("Score requires every configured fold, with no omitted folds")
    scores = [folds[name]["val_pearson_full"] for name in expected]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Nonfinite trial objective")
    score = sum(scores) / len(scores)
    session.study.tell(trial, score)
    session.save()
    return score


def fail_trial(session, trial, exc):
    if session.study.trials[trial.number].state.is_finished():
        # A publication error after tell() must not overwrite a valid result or
        # mask the original I/O exception with "cannot update a finished trial".
        return
    trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
    session.study.tell(trial, state=TrialState.FAIL)
    session.save()


def _complete_trials(session):
    trials = session.study.get_trials()
    budget = session.contract["policy"]["trials_per_subject"]
    if len(trials) != budget or any(t.state != TrialState.COMPLETE for t in trials):
        raise StoreError("All budgeted trials must complete before selecting the next stage")
    return trials


def completed_selection(session):
    """Best trial by the objective of its own folds; stage 2 hands this to C."""
    _complete_trials(session)
    best = session.study.best_trial
    return {"trial": best.number, "parameters": best.params, "value": best.value}


# --------------------------------------------------------------------------- stage-1 finalists

def reevaluation_candidates(session):
    """Best stage-1 trials by their B2/B4 objective; the earlier trial wins a tie."""
    if session.contract["stage"] != "stage1":
        raise ValueError("Only stage 1 re-evaluates its finalists")
    top = session.contract["policy"]["reevaluation"]["top_trials"]
    return sorted(_complete_trials(session), key=lambda t: (-t.value, t.number))[:top]


def reevaluation_state(session):
    """Finalists, run locations and fold results, kept in the stage-1 study archive.

    The finalists are fixed once the study is complete; a recorded failure stops the
    re-evaluation for diagnosis, like a failed trial, without automatic replacement.
    """
    candidates = [t.number for t in reevaluation_candidates(session)]
    state = session.study.user_attrs.get("reevaluation")
    if state is None:
        state = {"candidates": candidates, "runs": {}, "folds": {}, "failure": None}
        session.study.set_user_attr("reevaluation", state)
        session.save()
    if state["candidates"] != candidates:
        raise StoreError("Stage-1 finalists changed after the study was complete")
    if state["failure"]:
        raise StoreError("Re-evaluation failed; diagnose and document recovery before continuing: "
                         + state["failure"])
    return state


def reevaluation_training_run(session, trial, fold, device):
    """Same contract as a stage-1 fold: reference U-Net, the finalist's LR/WD, fold seed and epochs."""
    contract = session.contract
    state = reevaluation_state(session)
    if (trial.number not in state["candidates"] or fold.subject != contract["subject"]
            or fold.val_block not in contract["policy"]["reevaluation"]["validation_blocks"]):
        raise StoreError("Only the stage-1 finalists are re-evaluated, on the folds stage 1 left out")
    arch = reference_architecture(session.config)
    parameters = {"purpose": "optuna_reevaluation", "stage": contract["stage"],
                  "study": session.study.study_name, "trial": trial.number, "architecture": arch.key,
                  "seed": fold_seed(session.config, fold.subject, fold.val_block),
                  "batch_size": contract["batch_size"], "epochs": contract["epochs"],
                  "lr": trial.params["learning_rate"], "weight_decay": trial.params["weight_decay"]}
    storage = session.config.storage_root(session.drive_root)
    runs = state["runs"].setdefault(str(trial.number), {})
    if fold.val_block in runs:
        output = inside(storage, runs[fold.val_block])
        recorded = read_json(output / "config.json")["run"]
        if any(recorded.get(key) != value for key, value in parameters.items()):
            raise StoreError("Stored re-evaluation run differs from the current finalist")
        return output, recorded, arch
    output, parameters = start_training_run(session.config, session.drive_root, fold, parameters, device)
    runs[fold.val_block] = output.relative_to(storage).as_posix()
    session.study.set_user_attr("reevaluation", state)
    session.save()  # Publish the run location before the first optimizer update.
    return output, parameters, arch


def record_reevaluation(session, trial, val_block, history, parameters, device_metadata):
    state = reevaluation_state(session)
    state["folds"].setdefault(str(trial.number), {})[val_block] = _fold_record(
        session, history, parameters, device_metadata)
    session.study.set_user_attr("reevaluation", state)


def fail_reevaluation(session, exc):
    state = session.study.user_attrs.get("reevaluation")
    if state is None or state["failure"]:
        return  # Nothing was trained yet, or keep the original diagnosis.
    state["failure"] = f"{type(exc).__name__}: {exc}"
    session.study.set_user_attr("reevaluation", state)
    session.save()


def stage1_selection(session):
    """LR/WD inherited by stage 2, B and C: best five-fold mean among the finalists.

    The search folds come from each trial and the others from its re-evaluation; nothing is
    retrained here. A tie keeps the finalist that stage 1 ranked higher.
    """
    candidates = reevaluation_candidates(session)
    state = session.study.user_attrs.get("reevaluation") or {}
    if state.get("failure"):
        raise StoreError("Re-evaluation failed: " + state["failure"])
    policy = session.contract["policy"]
    added = policy["reevaluation"]["validation_blocks"]
    finalists = []
    for trial in candidates:
        extra = state.get("folds", {}).get(str(trial.number), {})
        if set(extra) != set(added):
            raise StoreError("Re-evaluate every stage-1 finalist on " + ", ".join(added) + " before selecting")
        scores = {block: trial.user_attrs["folds"][block]["val_pearson_full"] for block in policy["validation_blocks"]}
        scores.update({block: extra[block]["val_pearson_full"] for block in added})
        finalists.append({"trial": trial.number, "parameters": trial.params, "search_value": trial.value,
                          "value": sum(scores.values()) / len(scores), "fold_scores": scores})
    best = max(finalists, key=lambda finalist: finalist["value"])
    return {"trial": best["trial"], "parameters": best["parameters"], "value": best["value"],
            "search_value": best["search_value"], "finalists": finalists}
