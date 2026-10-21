"""Convergence diagnostic for ``training.search_epochs``; notebook 04 exposes the epoch loop.

Representative configurations train to a fixed horizon. A candidate number of epochs is
acceptable when the configuration that wins at that epoch, as Optuna would score it, loses
no more than the seed noise against the best horizon value. Nothing here writes the YAML.
"""

import math
from statistics import mean, pstdev

import pandas as pd

from .config import inside
from .io import StoreError, atomic_json, read_json
from .model import reference_architecture
from .reporting import start_training_run
from .training import fold_seed

GROUPS = ("learning_rate", "architecture")


def convergence_plan(config):
    """Seven runs per subject: the reference (two seeds), the other LRs and architectures."""
    study, training = config.data["convergence_study"], config.data["training"]
    ref = reference_architecture(config)
    reference = {"kernel_size": ref.kernel_size, "encoder_levels": ref.encoder_levels,
                 "base_filters": ref.base_filters}
    rate, decay = training["reference_learning_rate"], training["reference_weight_decay"]
    plan = []
    for subject in study["subjects"]:
        seed = fold_seed(config, subject, study["validation_block"])

        def add(name, arch, lr, run_seed, groups):
            plan.append({"subject": subject, "name": name, "architecture": dict(arch), "lr": lr,
                         "weight_decay": decay, "seed": run_seed, "groups": list(groups)})

        add("reference", reference, rate, seed, GROUPS)
        add("reference_seed_b", reference, rate, seed + study["second_seed_offset"], ())
        for lr in study["learning_rates"]:
            add(f"lr_{lr:g}", reference, lr, seed, ("learning_rate",))
        for arch in study["architectures"]:
            key = f"k{arch['kernel_size']}_L{arch['encoder_levels']}_f{arch['base_filters']}"
            add(f"arch_{key}", arch, rate, seed, ("architecture",))
    return plan


def registry_dir(config, drive_root, pipeline_id, batch_size):
    """One diagnostic per pipeline and batch; another batch is another diagnostic."""
    return config.path(drive_root, "training_dir") / "convergence" / pipeline_id / f"batch_{batch_size}"


def _registry(folder):
    path = folder / "registry.json"
    return read_json(path) if path.exists() else {}


def convergence_run(config, drive_root, fold, run, architecture, batch_size, device):
    """Locate the same run after a reconnect; publish its location before the first update."""
    study = config.data["convergence_study"]
    if fold.subject != run["subject"] or fold.val_block != study["validation_block"]:
        raise StoreError("The fold does not belong to this diagnostic run")
    parameters = {"purpose": "convergence", "name": run["name"], "architecture": architecture.key,
                  "seed": run["seed"], "batch_size": batch_size, "epochs": study["epochs"],
                  "lr": run["lr"], "weight_decay": run["weight_decay"]}
    folder = registry_dir(config, drive_root, fold.pipeline_id, batch_size)
    registry, key = _registry(folder), f"{run['subject']}/{run['name']}"
    storage = config.storage_root(drive_root)
    if key in registry:
        output = inside(storage, registry[key])
        recorded = read_json(output / "config.json")["run"]
        if any(recorded.get(name) != value for name, value in parameters.items()):
            raise StoreError(f"Stored diagnostic run {key} differs from the configuration")
        return output, recorded
    output, parameters = start_training_run(config, drive_root, fold, parameters, device)
    registry[key] = output.relative_to(storage).as_posix()
    atomic_json(folder / "registry.json", registry)
    return output, parameters


def load_convergence_results(config, drive_root, pipeline_id, batch_size, plan):
    """Complete epoch curves of every planned run; one GPU model across all of them."""
    folder = registry_dir(config, drive_root, pipeline_id, batch_size)
    registry, storage = _registry(folder), config.storage_root(drive_root)
    histories, gpus = {}, {}
    for run in plan:
        key = f"{run['subject']}/{run['name']}"
        if key not in registry:
            raise StoreError(f"Diagnostic run {key} has not started")
        output = inside(storage, registry[key])
        summary = read_json(output / "run_summary.json") if (output / "run_summary.json").exists() else {}
        if summary.get("status") != "COMPLETE":
            raise StoreError(f"Diagnostic run {key} is not complete: {summary.get('status', 'RUNNING')}")
        histories[(run["subject"], run["name"])] = pd.read_csv(output / "epoch_metrics.csv").to_dict("records")
        env = read_json(output / "environment.json")
        gpus[key] = {env.get("gpu"), *(r.get("gpu") for r in env.get("resumptions", []))}
    used = set().union(*gpus.values())
    if len(used) != 1:
        raise StoreError(f"Diagnostic runs mixed GPU models: {sorted(map(str, used))}")
    return histories, used.pop()


def _curve(history, epochs):
    curve = {int(row["epoch"]): float(row["val_pearson_full"]) for row in history}
    if sorted(curve) != list(range(1, epochs + 1)) or not all(math.isfinite(v) for v in curve.values()):
        raise ValueError(f"Expected a finite Pearson for epochs 1..{epochs}")
    return curve


def convergence_decision(study, plan, histories):
    """Smallest candidate whose last-epoch winner loses at most the noise at the horizon.

    Per subject, delta is the larger of the gap between the two reference seeds at the
    horizon and the epoch-to-epoch spread of either seed inside the horizon window.
    For each group (learning rates; architectures) and candidate epoch E, the winner is the
    best value at epoch E (earlier run in the plan on ties) and its loss is the best
    horizon value minus the winner's. Inversions among pairs farther apart than delta are
    reported but do not decide.
    """
    first, last = study["horizon_epochs"]
    window = range(first, last + 1)
    rows = []
    for subject in study["subjects"]:
        runs = [run for run in plan if run["subject"] == subject]
        curves = {run["name"]: _curve(histories[(subject, run["name"])], study["epochs"]) for run in runs}
        horizon = {name: mean(curve[e] for e in window) for name, curve in curves.items()}
        delta = max(abs(horizon["reference"] - horizon["reference_seed_b"]),
                    *(pstdev([curves[name][e] for e in window]) for name in ("reference", "reference_seed_b")))
        for group in GROUPS:
            names = [run["name"] for run in runs if group in run["groups"]]
            best = max(horizon[name] for name in names)
            for epoch in study["candidate_epochs"]:
                winner = max(names, key=lambda name: curves[name][epoch])
                inversions = sum(1 for i, a in enumerate(names) for b in names[i + 1:]
                                 if abs(horizon[a] - horizon[b]) > delta
                                 and (curves[a][epoch] - curves[b][epoch]) * (horizon[a] - horizon[b]) < 0)
                loss = best - horizon[winner]
                rows.append({"subject": subject, "group": group, "epoch": epoch, "winner": winner,
                             "winner_at_epoch": curves[winner][epoch], "winner_horizon": horizon[winner],
                             "best_horizon": best, "loss": loss, "delta": delta,
                             "inversions": inversions, "acceptable": loss <= delta})
    accepted = [e for e in study["candidate_epochs"]
                if all(row["acceptable"] for row in rows if row["epoch"] == e)]
    return {"selected_epochs": min(accepted) if accepted else None, "accepted_epochs": accepted,
            "candidate_epochs": list(study["candidate_epochs"]), "horizon_epochs": [first, last],
            "rows": rows,
            "rule": "smallest candidate whose last-epoch winner loses <= delta at the horizon, "
                    "in every subject and group"}
