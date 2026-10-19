"""Step-2 stages with status tracking, called in this order by the notebook and the CLI."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..datasets import contiguous_runs
from ..eda import plotting as pl
from ..eda.context import section
from ..store import (FOLD_MANIFEST, FOLDS_MANIFEST, PROCESSED_MANIFEST, block_dir, fold_dir, fold_set_name, publish,
                     read_json, verify_manifest)
from . import blocks as prep_blocks
from . import folds as prep_folds
from .gate import PrepBlocked, validate_reference_run


def _publish_or_reuse(final: Path, run_id: str, manifest: str, build: Callable[[Path], object]) -> bool:
    """Reuse a verified published directory or build a partial one and publish it.
    Returns True when an existing directory was reused."""
    if final.exists():
        verify_manifest(final, manifest)
        print(f"Reutilizado (manifiesto verificado): {final}", flush=True)
        return True
    final.parent.mkdir(parents=True, exist_ok=True)
    partial = final.parent / f"{final.name}.partial_{run_id}"
    build(partial)
    return not publish(partial, final, manifest)


def _relative(ctx, path: Path) -> str:
    return path.relative_to(ctx.config.storage_root(ctx.drive_root)).as_posix()


@section("p01_gate", "Compuerta del EDA", "Exclusión fija respaldada por una corrida EDA COMPLETE de los mismos .mat")
def p01_gate(ctx, records):
    ctx.reference = validate_reference_run(ctx.config, ctx.drive_root, ctx.raw_sha256)
    ctx.save_json(ctx.reference, "p01_eda_reference",
                  f"Corrida EDA {ctx.reference['run_id']} validada: estado, hashes de los .mat, exclusiones, "
                  "configuración de señal y SHA-256 de sus artefactos")


@section("p02_blocks", "Bloques procesados", "Características y flexión por bloque independiente, con máscaras de validez")
def p02_blocks(ctx, records):
    if ctx.reference is None:
        raise PrepBlocked("The EDA gate has not passed")
    identity = prep_blocks.pipeline_identity(ctx.config, ctx.raw_sha256, ctx.reference["run_id"])
    digest = prep_blocks.pipeline_hash(identity)
    final = ctx.config.path(ctx.drive_root, "processed_dir") / digest
    reused = _publish_or_reuse(final, ctx.run_id, PROCESSED_MANIFEST, lambda root: prep_blocks.build_pipeline(
        ctx.config, records, identity, digest, root, ctx.run_id, ctx.env["git_commit"]))
    ctx.pipeline_dir = final
    rows = []
    for subject in ctx.config.data["project"]["subjects"]:
        for name in ctx.config.data["validation"]["block_names"]:
            m = read_json(block_dir(final, subject, name) / "block_manifest.json")
            rows.append({"subject": subject, "block": name, "x_shape": "x".join(map(str, m["x"]["shape"])),
                         "valid": m["columns"]["valid"], "invalid_ecog_context": m["columns"]["invalid_ecog_context"],
                         "invalid_flexion": m["columns"]["invalid_flexion"],
                         "valid_runs": len(m["columns"]["valid_runs"]),
                         "negative_power_fraction": m["morlet"]["negative_fraction_valid"],
                         "x_median": m["x_valid_summary"]["median"], "x_max": m["x_valid_summary"]["max"]})
    ctx.save_table(pd.DataFrame(rows), "p02_blocks", "Forma, columnas válidas por motivo, negativos de Morlet "
                   "(sin recortar) y escala de la potencia por sujeto y bloque")
    ctx.save_json({"pipeline_hash": digest, "path": _relative(ctx, final), "reused": reused, "identity": identity},
                  "p02_pipeline", "Identidad y ubicación de los datos procesados por bloque")


def _timeline(ctx, fold_set: Path):
    subjects = ctx.config.data["project"]["subjects"]
    names = ctx.config.data["validation"]["block_names"]
    fs = ctx.config.data["signal"]["target_fs_hz"]
    fig, axes = pl.figure(len(subjects), 1, width=13, height=2.6)
    for row, subject in enumerate(subjects):
        ax = axes[row, 0]
        index = {n: np.load(block_dir(ctx.pipeline_dir, subject, n) / "x_index.npy") for n in names}
        valid = {n: np.load(block_dir(ctx.pipeline_dir, subject, n) / "valid.npy") for n in names}
        for k, val_name in enumerate(names):
            folder = fold_dir(fold_set, subject, val_name)
            for name in names:
                mask = valid[name] if name == val_name else np.load(folder / f"eligible_{name}.npy")
                spans = [(index[name][a] / fs, (index[name][b - 1] + 1 - index[name][a]) / fs)
                         for a, b in contiguous_runs(mask)]
                ax.broken_barh(spans, (len(names) - 1 - k - 0.35, 0.7),
                               facecolors=pl.SLOTS[3] if name == val_name else pl.KEPT)
        for name in names:
            ax.axvline(index[name][0] / fs, color=pl.AXIS, lw=0.8)
        ax.set(yticks=range(len(names)), yticklabels=[f"val {n}" for n in reversed(names)], ylabel=subject,
               xlabel="Tiempo original (s)" if row == len(subjects) - 1 else None)
        ax.grid(False)
    axes[0, 0].legend(handles=[Patch(color=pl.KEPT, label="Train elegible"),
                               Patch(color=pl.SLOTS[3], label="Validación válida")],
                      loc="upper right", ncols=2, bbox_to_anchor=(1, 1.3))
    return fig


def _purge_zoom(ctx, fold_set: Path, val_name: str, seconds: float = 8.0):
    """Columns around the start of ``val_name``: block validity, purge and validation."""
    subjects = ctx.config.data["project"]["subjects"]
    names = ctx.config.data["validation"]["block_names"]
    previous = names[names.index(val_name) - 1]
    fs = ctx.config.data["signal"]["target_fs_hz"]
    fig, axes = pl.figure(1, len(subjects), width=4.6, height=2.6, sharey=True)
    for col, subject in enumerate(subjects):
        ax = axes[0, col]
        folder = fold_dir(fold_set, subject, val_name)
        manifest = read_json(folder / FOLD_MANIFEST)
        boundary = manifest["val"]["start_target"] / fs
        rows = [("Train válido", np.load(block_dir(ctx.pipeline_dir, subject, previous) / "valid.npy"), previous, pl.MUTED),
                ("Train elegible", np.load(folder / f"eligible_{previous}.npy"), previous, pl.KEPT),
                ("Val válida", np.load(block_dir(ctx.pipeline_dir, subject, val_name) / "valid.npy"), val_name,
                 pl.SLOTS[3])]
        for y, (label, mask, block, color) in enumerate(rows):
            index = np.load(block_dir(ctx.pipeline_dir, subject, block) / "x_index.npy")
            spans = [(index[a] / fs, (index[b - 1] + 1 - index[a]) / fs) for a, b in contiguous_runs(mask)]
            ax.broken_barh(spans, (len(rows) - 1 - y - 0.3, 0.6), facecolors=color)
        ax.axvline(boundary, color=pl.INK, lw=0.9)
        ax.set(xlim=(boundary - seconds, boundary + seconds), yticks=range(len(rows)),
               yticklabels=[r[0] for r in reversed(rows)], xlabel="Tiempo original (s)",
               title=f"{subject}: hueco efectivo {manifest['effective_gap_samples']['before_val']} muestras")
        ax.grid(False)
    return fig


@section("p03_folds", "Folds y escaladores", "Purga, escaladores ajustados solo con train del fold y ventanas por tramo")
def p03_folds(ctx, records):
    if ctx.pipeline_dir is None:
        raise PrepBlocked("No processed blocks are available")
    purge, status = ctx.config.purge_for_build()
    identity = prep_folds.fold_identity(ctx.config, ctx.pipeline_dir.name, purge, status)
    digest = prep_folds.fold_hash(identity)
    final = ctx.pipeline_dir / "folds" / fold_set_name(purge, digest)
    reused = _publish_or_reuse(final, ctx.run_id, FOLDS_MANIFEST, lambda root: prep_folds.build_fold_set(
        ctx.config, ctx.pipeline_dir, identity, digest, root, ctx.run_id, ctx.env["git_commit"]))
    ctx.fold_set_dir, ctx.purge, ctx.purge_status = final, purge, status
    rows = []
    for subject in ctx.config.data["project"]["subjects"]:
        for name in ctx.config.data["validation"]["stage2_validation_blocks"]:
            m = read_json(fold_dir(final, subject, name) / FOLD_MANIFEST)
            rows.append({"subject": subject, "fold": m["fold"], "purge": m["purge_samples"],
                         "purge_status": m["purge_status"], "train_columns": m["n_train_columns"],
                         "train_windows": m["n_train_windows"], "val_columns": m["val"]["valid_columns"],
                         "val_windows": m["n_val_windows"],
                         "gap_before_val": m["effective_gap_samples"]["before_val"],
                         "gap_after_val": m["effective_gap_samples"]["after_val"],
                         "min_feature_spread": m["scalers"]["feature"]["min_spread"]})
    ctx.save_table(pd.DataFrame(rows), "p03_folds", f"Folds con purga {purge} ({status}): columnas y ventanas de "
                   "train/validación, huecos efectivos y menor rango 10–90 de las características")
    ctx.save_fig(_timeline(ctx, final), "p03_folds_timeline",
                 f"Columnas usadas por fold (purga {purge}, {status}): train elegible y validación válida; "
                 "los huecos son bordes de bloque, retardo y purga")
    first = ctx.config.data["validation"]["stage1_validation_blocks"][0]
    ctx.save_fig(_purge_zoom(ctx, final, first), "p03_purge_zoom",
                 f"Frontera {first}: columnas válidas del bloque previo, las que sobreviven a la purga y la validación")
    ctx.save_json({"fold_set": final.name, "path": _relative(ctx, final), "reused": reused, "purge_samples": purge,
                   "purge_status": status, "identity": identity}, "p03_fold_set",
                  "Identidad y ubicación del conjunto de folds")


STEPS = [p01_gate, p02_blocks, p03_folds]
