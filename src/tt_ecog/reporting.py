"""Portable, timestamped evidence and raw-data visualizations."""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import DevelopmentRecord, sha256_file
from .quality_control import QCResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def environment(project_root: Path) -> dict:
    versions = {}
    for package in ("tt-ecog", "numpy", "scipy", "pandas", "h5py", "PyYAML", "matplotlib",
                    "jsonschema", "pytest", "torch", "mne", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    installed = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata['Name']})
    commit = None
    dirty = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                                text=True, capture_output=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
        state = subprocess.run(["git", "status", "--porcelain"], cwd=project_root,
                               text=True, capture_output=True, check=False)
        if state.returncode == 0:
            dirty = bool(state.stdout.strip())
    except FileNotFoundError:
        pass  # Git is optional for local synthetic runs; source hashes remain mandatory.
    files = []
    for directory, patterns in {"src": ["*.py"], "configs": ["*.yaml", "*.json"],
                                 "schemas": ["*.json"], "scripts": ["*.py"],
                                 "notebooks": ["*.ipynb"], "tests": ["*.py"]}.items():
        for pattern in patterns:
            files.extend((project_root / directory).rglob(pattern))
    files.extend(p for p in (project_root / "pyproject.toml", project_root / "requirements-eda.txt") if p.exists())
    hashes = {p.relative_to(project_root).as_posix(): sha256_file(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"timestamp_utc": utc_now(), "python": platform.python_version(), "platform": platform.system(),
            "versions": versions, "installed_distributions": installed, "git_commit": commit, "git_dirty": dirty,
            "source_sha256": digest, "source_files": hashes,
            "device": "CPU", "gpu_used": False}


def plot_subject(record: DevelopmentRecord, qc: QCResult, output: Path, max_points: int) -> list[str]:
    output.mkdir(exist_ok=True)
    saved = []

    def save(figure, name):
        path = output / f"{record.subject}_{name}.png"
        figure.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        saved.append(path.relative_to(output.parent).as_posix())

    values = record.values
    fs = record.metadata["sampling_hz_ecog"]
    step = max(1, int(np.ceil(values.shape[1] / max_points)))
    fig, axis = plt.subplots(figsize=(12, 5))
    view = values[:, ::step]
    extent = [0, values.shape[1] / fs, values.shape[0] - .5, -.5]
    image = axis.imshow(view, aspect="auto", extent=extent, interpolation="nearest")
    axis.set(title=f"{record.subject}: ECoG cruda (visualización cada {step} muestras)",
             xlabel="Tiempo original (s)", ylabel="Canal (0-based)")
    fig.colorbar(image, ax=axis, label="Unidades originales")
    save(fig, "raw")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, metric in zip(axes.flat, ("median", "mad", "max_abs_difference", "std")):
        axis.plot(qc.channels["channel_0based"], qc.channels[metric], ".-")
        axis.set(xlabel="Canal (0-based)", ylabel=metric, title=metric)
    fig.suptitle(f"{record.subject}: diagnósticos de adquisición (sin limpieza)")
    fig.tight_layout()
    save(fig, "amplitude_jumps")

    fig, axis = plt.subplots(figsize=(12, 5))
    for spectrum in qc.psd:
        axis.semilogy(qc.frequencies, np.maximum(spectrum, np.finfo(float).tiny), alpha=.4)
    axis.set(xlabel="Frecuencia (Hz)", ylabel="PSD (unidades originales²/Hz)",
             title=f"{record.subject}: PSD Welch de todos los canales")
    save(fig, "psd")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for axis, metric in zip(axes, ("std", "line_power_fraction")):
        table = qc.temporal.pivot(index="channel_0based", columns="start_s", values=metric)
        image = axis.imshow(table.to_numpy(), aspect="auto", extent=extent, interpolation="nearest")
        axis.set(title=metric, xlabel="Tiempo original (s)", ylabel="Canal (0-based)")
        fig.colorbar(image, ax=axis)
    fig.suptitle(f"{record.subject}: variabilidad local y ruido de línea")
    fig.tight_layout()
    save(fig, "temporal")

    fig, axis = plt.subplots(figsize=(12, 5))
    for detector, group in qc.events.groupby("detector"):
        axis.scatter(group.start_s, group.channel_0based, s=12, label=detector)
    if qc.events.empty:
        axis.text(.5, .5, "Sin eventos emitidos; consultar estado de detectores",
                  ha="center", transform=axis.transAxes)
    else:
        axis.legend()
    axis.set(xlabel="Inicio original (s)", ylabel="Canal (0-based)",
             title=f"{record.subject}: candidatos para revisión, no exclusiones")
    save(fig, "events")
    return saved
