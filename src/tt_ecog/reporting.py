"""Portable, timestamped evidence: environment, hashes and JSON output."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import traceback
from typing import Any, Callable

import pandas as pd

from .config import ExperimentConfig
from importlib import metadata
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess

from .io import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def json_safe(value: object) -> object:
    """Plain JSON types: numpy values become Python values, non-finite floats None."""
    import math
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in (value.tolist() if isinstance(value, np.ndarray) else value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def start_training_run(config, drive_root, fold, parameters, device):
    """One evidence folder for the visible notebook epoch loop."""
    from uuid import uuid4
    from .training import device_info
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = config.path(drive_root, "training_dir") / fold.subject / run_id
    output.mkdir(parents=True, exist_ok=False)
    parameters = {**parameters, "pipeline_id": fold.pipeline_id, "run_id": run_id,
                  "subject": fold.subject, "fold": fold.name}
    write_json(output / "config.json", {"run": parameters, "configuration": config.data})
    env = environment(config.source.parent.parent)
    env.update(device_info(device), gpu_used=str(device).startswith("cuda"))
    write_json(output / "environment.json", env)
    write_json(output / "data_manifest.json", {"pipeline_id": fold.pipeline_id,
               "processed_dir": config.data["paths"]["processed_dir"],
               "approval": config.data["verification"]["approved_manifest_file"], "test_accessed": False})
    write_json(output / "split_manifest.json", json_safe(fold.manifest))
    write_json(output / "run_summary.json", {**parameters, "status": "RUNNING", "started_at_utc": utc_now()})
    return output, parameters


def record_training_epoch(output, parameters, epoch, train_loss, metrics, seconds, memory):
    """Append actual measurements; an interrupted epoch never gets a metric row."""
    import csv
    row = {**parameters, "epoch": epoch, "train_loss": train_loss,
           "val_loss": metrics["val_loss"], "val_pearson_full": metrics["val_pearson_full"],
           "elapsed_seconds": seconds, **memory}
    row.update({f"pearson_{finger}": value for finger, value in zip(metrics["fingers"], metrics["pearson"])})
    row.update({"val_mse": metrics["mse_mean"], "val_rmse": metrics["rmse_mean"], "val_ccc": metrics["ccc_macro"]})
    path = output / "epoch_metrics.csv"
    exists = path.exists()
    with path.open("a", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def finish_training_run(output, parameters, status, completed_epochs, last_score=None, details=None):
    from .io import write_manifest
    summary = {**parameters, "status": status, "completed_epochs": completed_epochs,
               "last_val_pearson_full": last_score, "checkpoint_selection": "last_completed_epoch",
               "details": details, "test_accessed": False, "completed_at_utc": utc_now()}
    write_json(output / "run_summary.json", summary)
    write_manifest(output, "artifact_manifest.json", {"pipeline_id": parameters["pipeline_id"],
                                                     "run_id": parameters["run_id"]})
    return summary


def environment(project_root: Path) -> dict:
    versions = {}
    for package in ("tt-ecog", "numpy", "scipy", "pandas", "h5py", "PyYAML", "matplotlib",
                    "mne", "scikit-learn", "IPython", "pytest", "torch", "optuna"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    installed = sorted({f"{d.metadata.get('Name')}=={d.version}" for d in metadata.distributions() if d.metadata.get('Name')})
    commit = None
    dirty = None
    remote = None
    branch = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                                text=True, capture_output=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
        state = subprocess.run(["git", "status", "--porcelain"], cwd=project_root,
                               text=True, capture_output=True, check=False)
        if state.returncode == 0:
            dirty = bool(state.stdout.strip())
        for name, args in (("remote", ["remote", "get-url", "origin"]),
                           ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
            result = subprocess.run(["git", *args], cwd=project_root, text=True,
                                    capture_output=True, check=False)
            if result.returncode == 0:
                if name == "remote":
                    # Never persist credentials embedded in a remote URL.
                    remote = re.sub(r"//[^/@]*@", "//", result.stdout.strip())
                else:
                    branch = result.stdout.strip()
    except FileNotFoundError:
        pass  # Git is optional for local synthetic runs; source hashes remain mandatory.
    files = []
    for directory, patterns in {"src": ["*.py"], "configs": ["*.yaml"], "scripts": ["*.py"],
                                 "notebooks": ["*.ipynb"], "tests": ["*.py"]}.items():
        for pattern in patterns:
            files.extend((project_root / directory).rglob(pattern))
    files.extend(p for p in (project_root / "pyproject.toml", project_root / "requirements.txt") if p.exists())
    hashes = {p.relative_to(project_root).as_posix(): sha256_file(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"timestamp_utc": utc_now(), "python": platform.python_version(), "platform": platform.system(),
            "versions": versions, "installed_distributions": installed, "git_commit": commit, "git_dirty": dirty,
            "git_remote": remote, "git_branch": branch,
            "source_sha256": digest, "source_files": hashes,
            "device": "CPU", "gpu_used": False}



# Windows MAX_PATH is 260; keep every artifact path comfortably below it.
MAX_ABSOLUTE_PATH = 240


@dataclass(frozen=True)
class Artifact:
    relative: str  # posix path relative to the run folder
    path: Path
    kind: str  # figure | table | json
    caption: str
    section: str


@dataclass
class SectionRecord:
    name: str
    title: str
    justifies: str
    status: str = "PENDING"
    artifacts: list[Artifact] = field(default_factory=list)
    error: str | None = None


class RunContext:
    """Everything a section needs to save evidence inside one run folder."""

    def __init__(self, config: ExperimentConfig, drive_root: Path, output: Path,
                 run_id: str, env: dict[str, Any]):
        self.config = config
        self.drive_root = drive_root
        self.output = output
        self.run_id = run_id
        self.env = env
        self.eda = config.data["eda"]
        self.sections: dict[str, SectionRecord] = {}
        self.subject_status: dict[str, dict[str, Any]] = {}
        self.data_manifest: dict[str, Any] = {}
        self._current: SectionRecord | None = None
        for name in ("fig", "tables", "errors"):
            (output / name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- artifacts
    def _target(self, folder: str, filename: str) -> tuple[Path, str]:
        path = self.output / folder / filename
        if len(str(path.resolve())) >= MAX_ABSOLUTE_PATH:
            raise OSError(f"Artifact path too long for Windows ({len(str(path))} chars): {path}")
        return path, f"{folder}/{filename}"

    def _register(self, relative: str, path: Path, kind: str, caption: str) -> Artifact:
        section = self._current.name if self._current else "run"
        artifact = Artifact(relative, path, kind, caption, section)
        if self._current is not None:
            self._current.artifacts.append(artifact)
        return artifact

    def save_fig(self, fig, name: str, caption: str) -> Artifact:
        path, relative = self._target("fig", f"{name}.png")
        fig.savefig(path, dpi=self.eda["figure_dpi"], facecolor=fig.get_facecolor())
        return self._register(relative, path, "figure", caption)

    def save_table(self, frame: pd.DataFrame, name: str, caption: str) -> Artifact:
        path, relative = self._target("tables", f"{name}.csv")
        frame.to_csv(path, index=False)
        return self._register(relative, path, "table", caption)

    def save_json(self, value: Any, name: str, caption: str) -> Artifact:
        path, relative = self._target("tables", f"{name}.json")
        write_json(path, value)
        return self._register(relative, path, "json", caption)

    # ------------------------------------------------------------- sections
    def begin_section(self, name: str, title: str, justifies: str = "") -> None:
        self._current = SectionRecord(name, title, justifies)
        self.sections[name] = self._current

    def end_section(self) -> list[Artifact]:
        if self._current is None:
            raise RuntimeError("No active section")
        self._current.status = "COMPLETE"
        artifacts = self._current.artifacts
        self._current = None
        return artifacts

    def execute(self, name: str, title: str, justifies: str,
                func: Callable[["RunContext", dict], None], subjects: dict) -> list[Artifact]:
        import matplotlib
        from .plotting import RC
        record = SectionRecord(name, title, justifies)
        self.sections[name] = record
        self._current = record
        try:
            with matplotlib.rc_context(RC):
                func(self, subjects)
        except Exception:
            record.status = "FAIL"
            record.error = traceback.format_exc()
            (self.output / "errors" / f"{name}.txt").write_text(record.error, encoding="utf-8")
            raise
        else:
            record.status = "COMPLETE"
        finally:
            self._current = None
        return record.artifacts


def show(artifacts: list[Artifact], max_table_rows: int = 20) -> None:
    from IPython.display import Image, Markdown, display
    import pandas as pd
    for artifact in artifacts:
        if artifact.kind == "figure":
            display(Markdown(f"**{artifact.relative}** — {artifact.caption}"))
            display(Image(filename=str(artifact.path)))
            continue
        display(Markdown(f"Tabla guardada: `{artifact.relative}` — {artifact.caption}"))
        if artifact.kind == "table":
            frame = pd.read_csv(artifact.path)
            if len(frame) <= max_table_rows:
                display(frame)


def report_search(session):
    """Export readable results; the atomic study archive remains the recovery source."""
    from .io import write_manifest
    from .optimization import completed_selection, stage1_selection
    from .io import StoreError
    folder = session.archive.folder
    trials = session.study.get_trials()
    rows = [{"number": t.number, "state": t.state.name, "value": t.value,
             "parameters": t.params, "evidence": t.user_attrs} for t in trials]
    write_json(folder / "trials.json", rows)
    session.study.trials_dataframe().to_csv(folder / "trials.csv", index=False)
    try:
        selection = completed_selection(session)
    except StoreError:
        selection = None
    status = "COMPLETE" if selection is not None else "INCOMPLETE"
    stage1 = {}
    if session.contract["stage"] == "stage1":
        # Stage 1 hands on the finalist with the best five-fold mean, not its best B2/B4 trial.
        reevaluation = session.study.user_attrs.get("reevaluation")
        stage1 = {"best_search_trial": selection, "reevaluation": reevaluation}
        try:
            selection = stage1_selection(session)
        except StoreError:
            selection = None
            if status == "COMPLETE":
                status = "REEVALUATION_FAILED" if (reevaluation or {}).get("failure") else "REEVALUATION_PENDING"
    summary = {**session.contract, "status": status, "selection": selection, **stage1,
               "trials_recorded": len(trials), "test_accessed": False,
               "updated_at_utc": utc_now()}
    write_json(folder / "study_summary.json", summary)
    write_manifest(folder, "artifact_manifest.json", {"pipeline_id": session.contract["pipeline_id"]})
    return summary


def search_learning_curves(config, drive_root, session):
    """Validation Pearson and training loss per epoch of every fit of a completed study.

    Descriptive only: reads the saved epoch metrics and never changes the selection. Stage 1
    shows each search fold (colour = learning rate; finalists opaque) and the finalists'
    five-fold mean; stage 2 shows each fold and the five-fold mean of every trial. The
    selected trial is drawn thick. Saved outside the study folder, so its manifest is intact.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from .config import inside
    from .optimization import completed_selection, stage1_selection
    storage = config.storage_root(drive_root)
    trials, contract = session.study.get_trials(), session.contract
    blocks = contract["policy"]["validation_blocks"]
    panels = [(f"validación {b}", {t.number: [t.user_attrs["runs"][b]] for t in trials}) for b in blocks]
    if contract["stage"] == "stage1":
        selection = stage1_selection(session)
        highlight = [f["trial"] for f in selection["finalists"]]
        extra = session.study.user_attrs["reevaluation"]["runs"]
        panels.append(("finalistas · media de 5 folds",
                       {n: [*trials[n].user_attrs["runs"].values(), *extra[str(n)].values()] for n in highlight}))
        rates = [t.params["learning_rate"] for t in trials]
        norm = colors.LogNorm(min(rates), max(rates))
        color = {t.number: cm.viridis(norm(t.params["learning_rate"])) for t in trials}
        label = {t.number: f"trial {t.number} (lr {t.params['learning_rate']:.1e}, "
                           f"wd {t.params['weight_decay']:.1e})" for t in trials}
    else:
        selection, highlight = completed_selection(session), [t.number for t in trials]
        panels.append(("media de 5 folds", {t.number: [t.user_attrs["runs"][b] for b in blocks] for t in trials}))
        color = {t.number: plt.get_cmap("tab10")(t.number % 10) for t in trials}
        label = {t.number: "trial {} (k{kernel_size} L{encoder_levels} f{base_filters})".format(t.number, **t.params)
                 for t in trials}
    fig, axes = plt.subplots(2, len(panels), figsize=(4.4 * len(panels), 6.6), sharex=True, squeeze=False,
                             layout="constrained")
    for col, (title, groups) in enumerate(panels):
        for number, paths in groups.items():
            frames = [pd.read_csv(inside(storage, p) / "epoch_metrics.csv").set_index("epoch") for p in paths]
            pearson = pd.concat([f["val_pearson_full"] for f in frames], axis=1).mean(axis=1)
            loss = pd.concat([f["train_loss"] for f in frames], axis=1).mean(axis=1)
            top = number == selection["trial"]
            style = dict(color=color[number], lw=2.4 if top else 0.9, alpha=1.0 if number in highlight else 0.35,
                         zorder=3 if top else 1)
            axes[0, col].plot(pearson.index, pearson, label=label[number] + (" ← elegido" if top else ""), **style)
            axes[1, col].plot(loss.index, loss, **style)
        axes[0, col].set_title(f"{contract['subject']} · {contract['stage']} · {title}", fontsize=10)
    axes[0, -1].legend(fontsize=7)
    for ax in axes[0]:
        ax.set_ylabel("Pearson reconstruido (val)")
    for ax in axes[1]:
        ax.set(yscale="log", ylabel="Pérdida de entrenamiento", xlabel="Época")
    if contract["stage"] == "stage1":
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes[:, :-1], label="learning rate")
    folder = config.path(drive_root, "optimization_dir") / contract["pipeline_id"] / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{contract['subject']}_{contract['stage']}_learning_curves.png"
    fig.savefig(path, dpi=140)
    return fig, path


def record_training_resume(output, device, epoch):
    from .training import device_info
    from .io import read_json
    path = output / "environment.json"
    env = read_json(path)
    env.setdefault("resumptions", []).append({**device_info(device), "after_epoch": epoch,
                                              "timestamp_utc": utc_now()})
    write_json(path, env)


def restore_epoch_metrics(output, history):
    """Discard only CSV rows newer than the last complete checkpoint."""
    path = output / "epoch_metrics.csv"
    if not history:
        path.unlink(missing_ok=True)
        return
    partial = path.with_suffix(".partial")
    pd.DataFrame(history).to_csv(partial, index=False)
    partial.replace(path)


def report_final_evaluation(config, output):
    """Tables and figures from saved predictions only; never reopen test or train.

    A/B/C/D form the comparison. Every seed of D (C is member 0) and their mean before
    smoothing go to a separate table: all repeated seeds are reported (protocol §11.2).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from .io import read_json, verify_manifest, write_manifest
    from .final_evaluation import VARIANTS, ENSEMBLE
    plan = read_json(output / "plan.json")
    rows, detail = [], []
    for subject in config.data["project"]["subjects"]:
        folder = output / subject / "test"
        verify_manifest(folder, "artifact_manifest.json")
        report = read_json(folder / "metrics.json")
        valid = np.load(folder / "valid.npy")
        truth = np.load(folder / "truth.npy")
        time = np.load(folder / "y_index.npy") / config.data["signal"]["target_fs_hz"]
        fig, axes = plt.subplots(5, 1, figsize=(14, 11), sharex=True)
        for k, ax in enumerate(axes):
            ax.plot(time, np.where(valid, truth[k], np.nan), color="black", label="Test", linewidth=1)
        for variant in (*VARIANTS, ENSEMBLE):
            result = report["variants"][variant]
            prediction = np.load(folder / f"prediction_{variant}.npy")
            for k, finger in enumerate(result["fingers"]):
                rows.append({"pipeline_id": plan["pipeline_id"], "run_id": plan["run_id"], "subject": subject,
                             "variant": variant, "finger": finger, "n_samples": result["n_samples"],
                             **{key: result[key][k] for key in ("pearson", "ccc", "mse", "rmse")},
                             **{key: result[key] for key in ("inference_seconds", "peak_allocated_bytes", "peak_reserved_bytes")}})
                axes[k].plot(time, prediction[k], label=variant, alpha=0.7, linewidth=0.8)
                axes[k].set_ylabel(finger)
        ensemble = report["ensemble"]
        for model, result in (*ensemble["members"].items(), ("mean_before_smoothing", ensemble["mean_before_smoothing"])):
            for k, finger in enumerate(result["fingers"]):
                detail.append({"pipeline_id": plan["pipeline_id"], "run_id": plan["run_id"], "subject": subject,
                               "model": model, "seed": result.get("seed"), "finger": finger,
                               "n_samples": result["n_samples"],
                               **{key: result[key][k] for key in ("pearson", "ccc", "mse", "rmse")}})
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Tiempo de la flexión (s); ECoG adelantado 0.2 s")
        fig.suptitle(subject + " — comparación final A/B/C/D, unidades originales")
        fig.tight_layout()
        fig.savefig(output / f"{subject}_predictions.png", dpi=140)
        plt.close(fig)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics_per_finger.csv", index=False)
    summary = frame.groupby(["subject", "variant"], sort=False)[["pearson", "ccc", "mse", "rmse"]].mean()
    summary.to_csv(output / "metrics_per_subject.csv")
    members = pd.DataFrame(detail)
    members.to_csv(output / "metrics_ensemble_detail.csv", index=False)
    members = members.groupby(["subject", "model"], sort=False)[["pearson", "ccc", "mse", "rmse"]].mean()
    write_json(output / "evaluation_summary.json", {"status": "COMPLETE", "pipeline_id": plan["pipeline_id"],
               "run_id": plan["run_id"], "test_accessed": True, "completed_at_utc": utc_now(),
               "rmse_summary": "mean of the five per-finger RMSE values", "objective": "Pearson macro",
               "variants": [*VARIANTS, ENSEMBLE], "ensemble_detail": "metrics_ensemble_detail.csv"})
    write_manifest(output, "artifact_manifest.json", {"pipeline_id": plan["pipeline_id"], "run_id": plan["run_id"]})
    return frame, summary, members
