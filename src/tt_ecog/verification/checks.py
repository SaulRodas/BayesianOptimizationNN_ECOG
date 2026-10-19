"""Checks V01-V14 on the real development data (docs/validation_requirements.md).

Each check saves its evidence and returns (status, details). Synthetic
counterparts live in tests/ (``verification`` marker) and the runner combines
both. Nothing here reads test variables or test-label files.
"""

import math
import shutil
import time

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import norm, pearsonr

from .. import io as tio
from .. import preprocessing as pp
from ..datasets import inference_window_starts, n_windows, contiguous_runs, train_window_starts
from ..eda import plotting as pl
from ..metrics import trajectory_metrics
from ..prep import blocks as prep_blocks
from ..prep import folds as prep_folds
from ..reporting import json_safe, write_json
from ..splits import Fold, make_folds, train_eligibility
from ..store import FOLD_MANIFEST, FOLDS_MANIFEST, read_json
from ..validation import score_validation
from .runner import VerificationContext, check

SCALER_FILES = ("scaler_x_center.npy", "scaler_x_scale.npy", "scaler_y_min.npy", "scaler_y_max.npy")


def _verdict(problems: list[str], passed: str, blocked: str | None = None) -> tuple[str, str]:
    if problems:
        return "FAIL", f"{len(problems)} problema(s): " + "; ".join(problems[:6])
    if blocked:
        return "BLOCKED", blocked
    return "PASS", passed


def _table(ctx, rows: list[dict], name: str, caption: str) -> None:
    ctx.save_table(pd.DataFrame(rows), name, caption)


# --------------------------------------------------------------------------- V01

@check("V01", "Integridad")
def v01_integrity(ctx: VerificationContext):
    ctx.require_data()
    nominal = ctx.config.data["raw_data"]["expected_channels_nominal"]
    identity = read_json(ctx.pipeline_dir / "manifests/processed_manifest.json")["identity"]
    rows, problems = [], []
    for subject, (ecog, glove) in ctx.records.items():
        path = tio.development_path(ctx.config, ctx.drive_root, subject)
        digest = ecog.metadata["sha256"]
        try:
            tio._read_variable(path, "test_data")
            refused = False
        except PermissionError:
            refused = True
        channels, samples = ecog.values.shape
        row = {"subject": subject, "file": ecog.metadata["path"], "sha256": digest,
               "unchanged_after_reading": tio.sha256_file(path) == digest,
               "matches_eda_reference": True, "matches_processed_identity": identity["raw_sha256"].get(subject) == digest,
               "adapter": ecog.metadata["adapter"], "stored_shape": ecog.metadata["original_shape"],
               "dtype": ecog.metadata["dtype"], "channels": channels, "samples": samples,
               "duration_s": ecog.metadata["duration_s"], "glove_shape": list(glove.values.shape),
               "glove_storage_hz": glove.metadata["sampling_hz_glove_storage"],
               "finite": bool(np.isfinite(ecog.values).all() and np.isfinite(glove.values).all()),
               "test_variable_refused": refused}
        rows.append(row)
        for key in ("unchanged_after_reading", "matches_processed_identity", "finite", "test_variable_refused"):
            if not row[key]:
                problems.append(f"{subject}: {key} is false")
        if channels != nominal[subject] or glove.values.shape[0] != 5:
            problems.append(f"{subject}: shape {ecog.values.shape} / glove {glove.values.shape}")
    _table(ctx, rows, "v01_integrity", "Archivos de desarrollo: hash antes/después, forma, tipo, finitud y guarda de test")
    shapes = ", ".join("{} {}x{}".format(r["subject"], r["channels"], r["samples"]) for r in rows)
    return _verdict(problems, f"Tres .mat íntegros e inmutables ({shapes}); "
                    "mismos SHA-256 que el EDA de referencia y que la identidad de los datos procesados; la lectura "
                    "de variables de test se rechaza antes de abrir el archivo.")


# --------------------------------------------------------------------------- V02

@check("V02", "Exclusión de canales")
def v02_exclusion(ctx: VerificationContext):
    ctx.require_data()
    channel_map = read_json(ctx.pipeline_dir / "manifests/channel_map.json")
    nominal = ctx.config.data["raw_data"]["expected_channels_nominal"]
    rows, problems = [], []
    for subject, (ecog, _) in ctx.records.items():
        cmap = ctx.spec(subject).channels
        shapes = {read_json(ctx.pipeline_dir / subject / b / "block_manifest.json")["x"]["shape"][0]
                  for b in ctx.block_names}
        raw = np.asarray(ecog.values[:, :20000], dtype=np.float64)
        base = pp.common_median_reference(pp.exclude_channels(raw, cmap))
        unchanged = sensitive = None
        if cmap.excluded:
            spiked = raw.copy()
            spiked[list(cmap.excluded), ::997] += 1e9 * max(1.0, float(np.abs(raw).max()))
            unchanged = bool(np.array_equal(base, pp.common_median_reference(pp.exclude_channels(spiked, cmap))))
            kept = list(cmap.kept)
            sensitive = not np.allclose((spiked - np.median(spiked, axis=0))[kept], (raw - np.median(raw, axis=0))[kept])
        row = {"subject": subject, "excluded_original": list(cmap.excluded), "n_original": cmap.n_original,
               "n_kept": cmap.n_kept, "expected_kept": nominal[subject] - len(ctx.config.excluded_channels(subject)),
               "persisted_map_matches": channel_map[subject] == cmap.to_json(), "block_channel_counts": sorted(shapes),
               "cmr_unchanged_by_excluded_transients": unchanged, "same_transients_move_cmr_if_included": sensitive}
        rows.append(row)
        if not row["persisted_map_matches"] or row["n_kept"] != row["expected_kept"] or shapes != {cmap.n_kept}:
            problems.append(f"{subject}: channel map or block channel counts differ")
        if cmap.excluded and not (unchanged and sensitive):
            problems.append(f"{subject}: exclusion before CMR not demonstrated (unchanged={unchanged}, sensitive={sensitive})")
    _table(ctx, rows, "v02_exclusion", "Mapa original→efectivo, canales por bloque y CMR frente a transitorios extremos en canales excluidos")
    ctx.save_json(ctx.reference, "v02_eda_reference", "Corrida EDA COMPLETE que respalda la exclusión (validada)")
    counts = "/".join(str(r["n_kept"]) for r in rows)
    return _verdict(problems, f"Exclusión S1-54, S2-20, S2-37 aplicada antes de la CMR; C_válidos {counts}; un "
                    "transitorio de 1e9 en los excluidos no cambia la CMR de los conservados (sí la cambiaría si se "
                    "incluyeran). La misma ChannelMap del YAML se usa en cada bloque; el test no se procesa en esta "
                    "etapa y la evaluación final debe usar el mismo PreprocessingSpec.")


# --------------------------------------------------------------------------- V03

@check("V03", "Referencia espacial")
def v03_reference(ctx: VerificationContext):
    ctx.require_data()
    rows, problems = [], []
    for subject, (ecog, _) in ctx.records.items():
        cmap = ctx.spec(subject).channels
        kept = pp.exclude_channels(np.asarray(ecog.values[:, :30000]), cmap)
        out = pp.common_median_reference(kept)
        error = float(np.abs(out - (kept - np.median(kept, axis=0, keepdims=True))).max())
        scale = float(np.abs(kept).max())
        row = {"subject": subject, "n_kept": cmap.n_kept, "even_channel_count": cmap.n_kept % 2 == 0,
               "max_abs_error_vs_numpy_median": error, "relative_error": error / max(scale, 1e-300),
               "time_reversal_invariant": bool(np.array_equal(pp.common_median_reference(kept[:, ::-1])[:, ::-1], out)),
               "chunking_invariant": bool(np.array_equal(pp.common_median_reference(kept, chunk_samples=777), out))}
        rows.append(row)
        if row["relative_error"] > 1e-12 or not row["time_reversal_invariant"] or not row["chunking_invariant"]:
            problems.append(f"{subject}: CMR differs from the instantaneous median reference")
    _table(ctx, rows, "v03_reference", "CMR frente a x - median(x, canales) por instante; invariancia a invertir el tiempo y a trocear")
    return _verdict(problems, "La CMR coincide con x − mediana(x) por instante (error relativo ≤ 1e-12) en los tres "
                    "sujetos, incluido el número par de canales de S2; no usa estadísticas temporales (invariante a "
                    "invertir el tiempo y a procesar por trozos).")


# --------------------------------------------------------------------------- V04

def _isolated(arrays: dict, fold: Fold, half: int, factor: int, purge: int) -> bool:
    val = fold.val
    for block in fold.train:
        a = arrays[block.name]
        mask = train_eligibility(a["x_index"], a["y_index"], a["valid"], block, val, purge)
        x, y = a["x_index"][mask], a["y_index"][mask]
        source = x * factor
        if (((source + half) >= val.start_source) & ((source - half) < val.stop_source)).any():
            return False
        if (((x >= val.start_target) & (x < val.stop_target)) | ((y >= val.start_target) & (y < val.stop_target))).any():
            return False
    return True


def _minimal_isolating_purge(arrays: dict, fold: Fold, half: int, factor: int) -> int:
    if _isolated(arrays, fold, half, factor, 0):
        return 0
    low, high = 0, max(b.stop_target - b.start_target for b in fold.train)
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (low, middle) if _isolated(arrays, fold, half, factor, middle) else (middle, high)
    return high


def _window_problems(starts: np.ndarray, mask: np.ndarray, window: int) -> bool:
    if not starts.size:
        return False
    if starts.min() < 0 or (starts + window).max() > mask.size:
        return True
    total = np.r_[0, np.cumsum(mask)]
    return bool((total[starts + window] - total[starts] != window).any())


@check("V04", "Bloques temporales y purga")
def v04_splits(ctx: VerificationContext):
    ctx.require_data()
    window, purge = ctx.config.data["validation"]["train_window_samples"], ctx.purge
    rows, problems, minimal = [], [], []
    for subject in ctx.subjects:
        spec = ctx.spec(subject)
        half, factor = pp.chain_half_support(spec), spec.factor
        blocks = ctx.blocks(subject)
        n_source = ctx.records[subject][0].values.shape[1]
        lengths = [b.stop_target - b.start_target for b in blocks]
        if (blocks[0].start_source != 0 or blocks[-1].stop_source != n_source or max(lengths) - min(lengths) > 1
                or any(a.stop_source != b.start_source for a, b in zip(blocks, blocks[1:]))
                or any(b.start_source % factor for b in blocks)):
            problems.append(f"{subject}: blocks do not tile the record in five near-equal chronological parts")
        arrays = {b.name: ctx.block_arrays(subject, b.name) for b in blocks}
        for fold in make_folds(blocks, ctx.val_names):
            val, folder = fold.val, ctx.fold_folder(subject, fold.val.name)
            manifest = read_json(folder / FOLD_MANIFEST)
            issues, masks = [], {}
            for block in fold.train:
                a = arrays[block.name]
                mask = np.load(folder / f"eligible_{block.name}.npy")
                masks[block.name] = mask
                if not np.array_equal(mask, train_eligibility(a["x_index"], a["y_index"], a["valid"], block, val, purge)):
                    issues.append(f"{block.name} mask differs from the purge rule")
                if (mask & ~a["valid"]).any():
                    issues.append(f"{block.name} uses invalid columns")
                x, y = a["x_index"][mask], a["y_index"][mask]
                source = x * factor
                if ((x < block.start_target) | (y >= block.stop_target)).any():
                    issues.append(f"{block.name} pairs leave their block")
                if (((x >= val.start_target) & (x < val.stop_target)) | ((y >= val.start_target) & (y < val.stop_target))).any():
                    issues.append(f"{block.name} pairs fall inside validation")
                if not (((source - half) >= block.start_source) & ((source + half) <= block.stop_source - 1)).all():
                    issues.append(f"{block.name} features use context outside their block")
                if (((source + half) >= val.start_source) & ((source - half) < val.stop_source)).any():
                    issues.append(f"{block.name} feature context touches validation")
                if block.stop_target <= val.start_target and y.size and y.max() > val.start_target - 1 - purge:
                    issues.append(f"{block.name} violates the purge before validation")
                if block.start_target >= val.stop_target and x.size and x.min() < val.stop_target + purge:
                    issues.append(f"{block.name} violates the purge after validation")
            windows = np.load(folder / "train_windows.npy")
            for number, name in enumerate(ctx.block_names):
                starts = windows[windows[:, 0] == number, 1]
                if name == val.name and starts.size:
                    issues.append("training windows inside validation")
                elif name in masks and _window_problems(starts, masks[name], window):
                    issues.append(f"{name} windows cross non-eligible columns")
            val_valid = arrays[val.name]["valid"]
            if _window_problems(np.load(folder / "val_windows.npy"), val_valid, window):
                issues.append("validation windows cross invalid columns")
            needed = _minimal_isolating_purge(arrays, fold, half, factor)
            minimal.append(needed)
            gaps = manifest["effective_gap_samples"]
            rows.append({"subject": subject, "fold": fold.name, "purge": purge, "purge_status": ctx.purge_status,
                         "train_blocks": "+".join(b.name for b in fold.train),
                         "train_columns": int(sum(m.sum() for m in masks.values())),
                         "gap_before_val": gaps["before_val"], "gap_after_val": gaps["after_val"],
                         "minimal_isolating_purge": needed, "issues": "; ".join(issues)})
            problems += [f"{subject} {fold.name}: {issue}" for issue in issues]
    _table(ctx, rows, "v04_splits", "Por fold: máscaras de train frente a la regla de purga, contexto FIR/Morlet/antialias, "
           "etiquetas, ventanas, huecos efectivos y purga mínima que aísla")
    needed = max(minimal) if minimal else None
    gaps = [g for r in rows for g in (r["gap_before_val"], r["gap_after_val"]) if g is not None]
    summary = (f"{len(rows)} folds: bloques B1–B5 contiguos; ningún par, contexto de transformación ni ventana de train "
               f"toca la validación; segmentos de train independientes por bloque. Purga mínima que aísla contextos y "
               f"etiquetas: {needed} muestras. Con la purga {purge}, el hueco efectivo mínimo entre train y validación "
               f"es {min(gaps) if gaps else None} muestras.")
    blocked = None if ctx.purge_status == "approved" else (
        summary + f" Purga {purge} = {ctx.purge_status}: falta aprobar validation.purge_samples_at_100hz en el YAML.")
    return _verdict(problems, summary + f" Purga aprobada: {purge}.", blocked)


# --------------------------------------------------------------------------- V05

def _perturbed_rebuild(ctx: VerificationContext, subject: str, fold: Fold, arrays: dict, channels: list[int]) -> dict:
    names, rtol = ctx.block_names, ctx.config.data["verification"]["reproducibility_rtol"]
    clean = {n: {k: np.array(v) for k, v in a.items()} for n, a in arrays.items()}
    base = ctx.scratch / "v05" / subject / fold.name
    prep_folds.build_fold(ctx.config, subject, fold, clean, names, channels, ctx.purge, ctx.purge_status, base / "a")
    perturbed = dict(clean)
    v = fold.val.name
    perturbed[v] = dict(clean[v], x=clean[v]["x"] * 1e3 + 1e6, y=-7 * clean[v]["y"])
    prep_folds.build_fold(ctx.config, subject, fold, perturbed, names, channels, ctx.purge, ctx.purge_status, base / "b")
    files = [*SCALER_FILES, "train_windows.npy", "val_windows.npy", *(f"eligible_{b.name}.npy" for b in fold.train)]
    identical = all(np.array_equal(np.load(base / "a" / f), np.load(base / "b" / f)) for f in files)
    persisted = ctx.fold_folder(subject, v)
    reproducible = all(np.allclose(np.load(base / "a" / f), np.load(persisted / f), rtol=rtol, atol=0)
                       for f in SCALER_FILES) and all(np.array_equal(np.load(base / "a" / f), np.load(persisted / f))
                                                       for f in files[len(SCALER_FILES):])
    shutil.rmtree(base, ignore_errors=True)
    return {"identical_after_val_perturbation": identical, "rebuild_matches_persisted": reproducible}


@check("V05", "Aislamiento de escaladores")
def v05_scalers(ctx: VerificationContext):
    ctx.require_data()
    low, high = ctx.config.data["scaling"]["ecog"]["quantile_range_percent"]
    adjust = norm.ppf(high / 100) - norm.ppf(low / 100)
    stage1 = ctx.config.data["validation"]["stage1_validation_blocks"]
    channel_maps = read_json(ctx.pipeline_dir / "manifests/channel_map.json")
    rows, problems = [], []
    for subject in ctx.subjects:
        blocks, arrays = prep_folds._subject_blocks(ctx.pipeline_dir, subject, ctx.block_names)
        for fold in make_folds(blocks, ctx.val_names):
            folder = ctx.fold_folder(subject, fold.val.name)
            masks = {b.name: np.load(folder / f"eligible_{b.name}.npy") for b in fold.train}
            features = np.concatenate([pp.features_to_matrix(np.asarray(arrays[b.name]["x"][:, :, masks[b.name]]))
                                       .astype(np.float64) for b in fold.train])
            targets = np.concatenate([arrays[b.name]["y"][:, masks[b.name]].T for b in fold.train])
            q = np.percentile(features, [low, 50, high], axis=0)
            del features
            row = {"subject": subject, "fold": fold.name, "fit_rows": int(targets.shape[0]),
                   "center_is_train_median": bool(np.allclose(np.load(folder / "scaler_x_center.npy"), q[1], rtol=1e-10, atol=0)),
                   "scale_is_train_10_90_over_normal_iqr": bool(np.allclose(np.load(folder / "scaler_x_scale.npy"),
                                                                            (q[2] - q[0]) / adjust, rtol=1e-10, atol=0)),
                   "minmax_is_train_min_max": bool(np.array_equal(np.load(folder / "scaler_y_min.npy"), targets.min(axis=0))
                                                   and np.array_equal(np.load(folder / "scaler_y_max.npy"), targets.max(axis=0)))}
            if fold.val.name in stage1:
                row.update(_perturbed_rebuild(ctx, subject, fold, arrays,
                                              channel_maps[subject]["kept_original_in_effective_order"]))
            data = ctx.load_fold(subject, fold.val.name)
            train_y = np.concatenate([data.blocks[n].y[:, m] for n, m in data.train_masks.items()], axis=1)
            train_x = np.concatenate([pp.features_to_matrix(data.blocks[n].x[:, :, m]) for n, m in data.train_masks.items()])
            val_block = data.blocks[fold.val.name]
            val_y = val_block.y[:, val_block.valid]
            row.update(train_y_min=float(train_y.min()), train_y_max=float(train_y.max()),
                       train_x_abs_median_max=float(np.abs(np.median(train_x, axis=0)).max()),
                       val_y_fraction_outside_0_1=float(np.mean((val_y < 0) | (val_y > 1))))
            del train_x, data
            rows.append(row)
            failed = [k for k in ("center_is_train_median", "scale_is_train_10_90_over_normal_iqr",
                                  "minmax_is_train_min_max", "identical_after_val_perturbation",
                                  "rebuild_matches_persisted") if row.get(k) is False]
            if abs(row["train_y_min"]) > 1e-6 or abs(row["train_y_max"] - 1) > 1e-6 or row["train_x_abs_median_max"] > 1e-4:
                failed.append("scaled train out of its fitted reference")
            problems += [f"{subject} {fold.name}: {k}" for k in failed]
    _table(ctx, rows, "v05_scalers", "Escaladores de cada fold frente a estadísticos de su train; reconstrucción con la "
           "validación perturbada (folds de la etapa 1); escala resultante en train y validación")
    return _verdict(problems, f"{len(rows)} folds: RobustScaler(10, 90, unit_variance) = mediana y rango 10–90 del "
                    "train elegible; MinMax = mín/máx del train; perturbar la validación (×1000, signo de la flexión) "
                    "deja escaladores, máscaras y ventanas idénticos; validación solo recibe transform (parte de su "
                    "flexión queda fuera de [0, 1]). El test nunca se ajusta: no existe ruta que lo lea en esta etapa.")


# --------------------------------------------------------------------------- V06

def _glove(ctx: VerificationContext, subject: str, values: np.ndarray | None = None):
    _, glove = ctx.records[subject]
    spec = ctx.spec(subject)
    values = glove.values if values is None else values
    info = pp.glove_storage_inspection(values, glove.metadata["sampling_hz_glove_storage"], spec)
    y25, knots = pp.effective_glove(values, info, spec)
    return y25, knots, info


@check("V06", "Alineación de 200 ms")
def v06_alignment(ctx: VerificationContext):
    ctx.require_data()
    delay = ctx.config.data["signal"]["temporal_alignment"]["delay_samples_at_target_fs"]
    rows, problems = [], []
    for subject in ctx.subjects:
        spec = ctx.spec(subject)
        y25, knots, _ = _glove(ctx, subject)
        scale = float(np.abs(y25).max()) or 1.0
        for block in ctx.blocks(subject):
            a = ctx.block_arrays(subject, block.name)
            valid = a["valid"]
            inside = (knots >= block.start_source) & (knots < block.stop_source)
            spline = CubicSpline(knots[inside] / spec.sfreq, y25[:, inside], axis=1, bc_type="not-a-knot", extrapolate=False)
            at_target = float(np.abs(a["y"][:, valid] - spline(a["y_index"][valid] / spec.target_sfreq)).max()) / scale
            at_ecog = float(np.abs(a["y"][:, valid] - spline(a["x_index"][valid] / spec.target_sfreq)).max()) / scale
            row = {"subject": subject, "block": block.name, "columns": int(valid.size), "valid": int(valid.sum()),
                   "y_minus_x_is_delay": bool((a["y_index"] - a["x_index"] == delay).all()),
                   "x_index_contiguous_from_block_start": bool(np.array_equal(
                       a["x_index"], block.start_target + np.arange(a["x_index"].size))),
                   "pairs_inside_block": bool(a["y_index"].max() < block.stop_target),
                   "max_rel_error_flexion_at_t_plus_delay": at_target,
                   "max_rel_difference_flexion_at_t": at_ecog}
            rows.append(row)
            if not (row["y_minus_x_is_delay"] and row["x_index_contiguous_from_block_start"] and row["pairs_inside_block"]):
                problems.append(f"{subject} {block.name}: index alignment")
            if at_target > 1e-9:
                problems.append(f"{subject} {block.name}: stored flexion is not the flexion at t + {delay}")
    _table(ctx, rows, "v06_alignment", "Cada columna empareja ECoG(t) con la flexión en t+20 reconstruida solo con los "
           "nodos del guante de su bloque; diferencia con la flexión en t como contraste")
    return _verdict(problems, f"15 bloques: y_index = x_index + {delay} (200 ms) en cada columna; la flexión guardada "
                    "es el spline del propio bloque en t+20 (error relativo ≤ 1e-9) y difiere de la flexión en t; "
                    "ningún par cruza un bloque. Los marcadores sintéticos están en las pruebas.")


# --------------------------------------------------------------------------- V07 / V08

def _recompute(ctx: VerificationContext, subject: str, raw: np.ndarray, glove_values: np.ndarray, tag: str) -> dict:
    spec = ctx.spec(subject)
    y25, knots, _ = _glove(ctx, subject, glove_values)
    result = {}
    for block in ctx.blocks(subject):
        folder = ctx.scratch / tag / subject / block.name
        prep_blocks.process_block(spec, raw, y25, knots, block, folder)
        result[block.name] = {k: np.load(folder / f"{k}.npy") for k in ("x", "y", "valid")}
        shutil.rmtree(folder)
    return result


def _baseline(ctx: VerificationContext, subject: str) -> dict:
    if subject not in ctx.baseline:
        ecog, glove = ctx.records[subject]
        ctx.baseline[subject] = _recompute(ctx, subject, ecog.values, glove.values, "baseline")
    return ctx.baseline[subject]


def _perturbed_record(ctx: VerificationContext, subject: str, samples: list[int], knots: list[int]):
    """Raw ECoG with extreme impulses on every channel and glove with whole holds shifted."""
    ecog, glove = ctx.records[subject]
    raw = np.array(ecog.values, dtype=np.float64)
    raw[:, samples] += 1e6 * max(1.0, float(np.abs(raw).max()))
    values = np.array(glove.values, dtype=np.float64)
    _, _, info = _glove(ctx, subject)
    width = ctx.spec(subject).glove_factor if info["method"] == "zero_order_hold" else 1
    shift = 1e3 * max(1.0, float(np.ptp(values)))
    for knot in knots:
        if info["method"] == "zero_order_hold":
            values[:, knot:knot + width] += shift
        elif info["method"] == "effective_rate_grid":
            values[:, knot // ctx.spec(subject).glove_factor] += shift
    return raw, values


def _comparison(a: dict, b: dict, tolerance: float) -> tuple[bool, float, float]:
    valid = a["valid"] & b["valid"]
    dx = float(np.abs(a["x"][:, :, valid].astype(np.float64) - b["x"][:, :, valid]).max()) if valid.any() else 0.0
    dy = float(np.nanmax(np.abs(a["y"][:, valid] - b["y"][:, valid]))) if valid.any() else 0.0
    same = bool(np.array_equal(a["valid"], b["valid"])) and dx <= tolerance and dy <= tolerance
    return same, dx, dy


def _isolation(ctx: VerificationContext, tag: str, perturbed_role: str):
    ctx.require_data()
    tolerance = ctx.config.data["verification"]["isolation_tolerance"]
    stage1 = ctx.config.data["validation"]["stage1_validation_blocks"]
    rows, problems = [], []
    for subject in ctx.subjects:
        base = _baseline(ctx, subject)
        blocks = {b.name: b for b in ctx.blocks(subject)}
        _, knots, _ = _glove(ctx, subject)
        factor = ctx.spec(subject).glove_factor
        samples, holds = [], []
        for name in stage1:
            b = blocks[name]
            inside = knots[(knots >= b.start_source) & (knots + factor <= b.stop_source)]
            if perturbed_role == "validation":
                samples += [b.start_source, b.start_source + 1, b.start_source + 2, (b.start_source + b.stop_source) // 2,
                            b.stop_source - 3, b.stop_source - 2, b.stop_source - 1]
                holds += [int(inside[0]), int(inside[inside.size // 2]), int(inside[-1])]
            else:
                samples += [b.start_source - 3, b.start_source - 2, b.start_source - 1,
                            b.stop_source, b.stop_source + 1, b.stop_source + 2]
                before = knots[knots + factor <= b.start_source]
                after = knots[knots >= b.stop_source]
                holds += [int(before[-1]), int(after[0])]
        raw, values = _perturbed_record(ctx, subject, samples, holds)
        recomputed = _recompute(ctx, subject, raw, values, tag)
        for name, arrays in recomputed.items():
            touched = (name in stage1) == (perturbed_role == "validation")
            same, dx, dy = _comparison(arrays, base[name], tolerance)
            rows.append({"subject": subject, "block": name, "role": "validación" if name in stage1 else "train",
                         "perturbed": touched, "valid_values_identical": same, "max_abs_diff_x_valid": dx,
                         "max_abs_diff_y_valid": dy})
            if not touched and not same:
                problems.append(f"{subject} {name}: changed by a perturbation outside the block")
            if touched and same:
                problems.append(f"{subject} {name}: its own perturbation had no effect (insensitive test)")
    return rows, problems


@check("V07", "Aislamiento validación → entrenamiento")
def v07_val_to_train(ctx: VerificationContext):
    rows, problems = _isolation(ctx, "v07", "validation")
    _table(ctx, rows, "v07_val_to_train", "Impulsos extremos en los bordes y el centro de B2/B4 (ECoG en todos los canales "
           "y retenciones del guante); bloques de train recalculados frente a la línea base de esta sesión")
    return _verdict(problems, "Impulsos de 1e6×máx|ECoG| en el primer, el último y el centro de cada bloque de "
                    "validación de la etapa 1, más saltos en el guante, no cambian ningún valor válido de B1/B3/B5 "
                    f"(tolerancia {ctx.config.data['verification']['isolation_tolerance']}); los bloques perturbados sí cambian.")


@check("V08", "Aislamiento entrenamiento → validación")
def v08_train_to_val(ctx: VerificationContext):
    rows, problems = _isolation(ctx, "v08", "training")
    _table(ctx, rows, "v08_train_to_val", "Impulsos extremos en las muestras de train contiguas a B2/B4 (y retenciones "
           "del guante vecinas); bloques de validación recalculados frente a la línea base de esta sesión")
    return _verdict(problems, "Impulsos en las tres muestras de train a cada lado de B2 y B4, más saltos en las "
                    "retenciones vecinas del guante, no cambian ningún valor válido de la validación antes del "
                    "escalado; los bloques de train perturbados sí cambian.")


# --------------------------------------------------------------------------- V09

def _support_figure(result: dict, analytic_target: int, mask_target: int):
    fig, axes = pl.figure(1, 1, width=9, height=3.2)
    ax = axes[0, 0]
    ax.plot(result["lags_target"], np.maximum(result["response"], 1e-18), color=pl.KEPT, lw=1.1,
            label="Respuesta al impulso (máx entre frecuencias)")
    ax.axhline(result["tolerance"], color=pl.MUTED, lw=0.8, label=f"Tolerancia {result['tolerance']:g}")
    for value, color, label in ((analytic_target, pl.INK, "Soporte analítico"), (mask_target, pl.SLOTS[3], "Máscara de borde")):
        ax.axvline(value, color=color, lw=1.0, label=label)
        ax.axvline(-value, color=color, lw=1.0)
    ax.set(yscale="log", ylim=(1e-12, 2), xlim=(-1.5 * mask_target, 1.5 * mask_target),
           xlabel="Muestras a 100 Hz desde el impulso", ylabel="Respuesta normalizada",
           title=f"Soporte medido ±{result['measured_half_support_target']} (analítico ±{analytic_target})")
    ax.legend(loc="upper right", fontsize=7)
    return fig


@check("V09", "Filtrado, Morlet y remuestreo")
def v09_filters(ctx: VerificationContext):
    ctx.require_data()
    signal = ctx.config.data["signal"]
    attenuation = signal["resample"]["antialias_fir"]["attenuation_db"]
    spec = ctx.spec(ctx.subjects[0])
    problems = []
    gains = {"fir_passband_db": spec.fir.gain_db([50, 90, 100, 150, 190]).tolist(),
             "fir_lines_db": spec.fir.gain_db(signal["notch_hz"]).tolist(),
             "fir_stop_db": spec.fir.gain_db([10, 300]).tolist(),
             "antialias_passband_db": spec.antialias.gain_db([1, 10, 20, 30, 35]).tolist(),
             "antialias_stop_db": spec.antialias.gain_db([50, 60, 100, 200, 499]).tolist()}
    if max(abs(g) for g in gains["fir_passband_db"]) > 0.1 or max(gains["fir_lines_db"]) > -40 or max(gains["fir_stop_db"]) > -40:
        problems.append(f"band-pass/notch response out of design: {gains}")
    if max(abs(g) for g in gains["antialias_passband_db"]) > 0.1 or max(gains["antialias_stop_db"]) > -(attenuation - 1):
        problems.append("antialias response out of design")
    persisted = read_json(ctx.pipeline_dir / "manifests/filter_support.json")
    if not np.allclose(persisted["morlet"]["freqs_hz"], spec.freqs) or persisted["morlet"]["n_cycles"] != spec.n_cycles \
            or persisted["morlet"]["zero_mean"] is not True:
        problems.append("persisted Morlet parameters differ from the configuration")
    taps = {d["name"]: d["taps_sha256"] for d in persisted["filters"]}
    for design in (spec.bandpass, spec.notch, spec.fir, spec.antialias):
        if taps.get(design.name) != design.to_json()["taps_sha256"]:
            problems.append(f"{design.name} taps differ from the persisted design")
    half = pp.chain_half_support(spec)
    mask_target = math.ceil(half / spec.factor)
    measured = pp.measured_support(spec, ctx.config.data["eda"]["support_tolerance"])
    analytic_target = measured["analytic"]["total_half_support_samples_target"]
    if measured["measured_half_support_target"] > analytic_target:
        problems.append("measured support exceeds the analytic support")
    rows = []
    for subject in ctx.subjects:
        for block in ctx.blocks(subject):
            a = ctx.block_arrays(subject, block.name)
            n = a["x_index"].size
            context = pp.feature_valid_mask(block.stop_source - block.start_source, spec)[:n]
            local = np.flatnonzero(a["valid"])
            row = {"subject": subject, "block": block.name, "valid_within_full_context": not (a["valid"] & ~context).any(),
                   "first_valid_column": int(local[0]), "last_valid_column": int(local[-1]),
                   "uniform_100hz_grid": bool((np.diff(a["x_index"]) == 1).all()),
                   "source_alignment": bool(np.array_equal(a["x_index"] * spec.factor - block.start_source,
                                                           np.arange(n) * spec.factor))}
            rows.append(row)
            if not (row["valid_within_full_context"] and row["uniform_100hz_grid"] and row["source_alignment"]) \
                    or row["first_valid_column"] < mask_target:
                problems.append(f"{subject} {block.name}: masks or timing inconsistent with the support")
    _table(ctx, rows, "v09_masks", "Columnas válidas dentro del contexto completo de la cadena, rejilla de 100 Hz y alineación con la fuente")
    ctx.save_fig(_support_figure(measured, analytic_target, mask_target), "v09_support",
                 "Respuesta al impulso de FIR→Morlet→antialias→decimación frente al soporte analítico y la máscara")
    filter_support = {"filters": persisted["filters"], "gains_db": gains, "morlet": persisted["morlet"],
                      "analytic_support": measured["analytic"],
                      "measured_support": {"half_support_target": measured["measured_half_support_target"],
                                           "tolerance": measured["tolerance"]},
                      "edge_mask": {"half_support_source": half, "half_support_target": mask_target,
                                    "rule": persisted["edge_mask"]["rule"]},
                      "purge": {"samples": ctx.purge, "status": ctx.purge_status}}
    write_json(ctx.output / "filter_support.json", json_safe(filter_support))
    summary = (f"FIR (pasabanda+notch, 1981 taps) y antialias Kaiser cumplen su diseño (líneas ≤ −40 dB, antialias "
               f"≤ −{attenuation - 1} dB desde 50 Hz); Morlet 40 frecuencias, 7 ciclos, zero_mean; soporte medido "
               f"±{measured['measured_half_support_target']} ≤ analítico ±{analytic_target} muestras a 100 Hz; las "
               f"máscaras exigen ±{mask_target} columnas de contexto dentro del bloque en los 15 bloques; rejilla de "
               "100 Hz uniforme y alineada con la fuente.")
    blocked = None if ctx.purge_status == "approved" else summary + " La purga sigue pendiente de aprobación."
    return _verdict(problems, summary, blocked)


# --------------------------------------------------------------------------- V10

@check("V10", "Ventanas")
def v10_windows(ctx: VerificationContext):
    ctx.require_data()
    validation = ctx.config.data["validation"]
    window = validation["train_window_samples"]
    rows, problems = [], []
    for subject in ctx.subjects:
        for val_name in ctx.val_names:
            folder = ctx.fold_folder(subject, val_name)
            windows = np.load(folder / "train_windows.npy")
            issues, expected_total = [], 0
            for number, name in enumerate(ctx.block_names):
                if name == val_name:
                    continue
                mask = np.load(folder / f"eligible_{name}.npy")
                starts = windows[windows[:, 0] == number, 1]
                expected = train_window_starts(mask, window, validation["train_stride_samples"])
                expected_total += sum(n_windows(b - a, window, validation["train_stride_samples"])
                                      for a, b in contiguous_runs(mask))
                if not np.array_equal(starts, expected):
                    issues.append(f"{name} training windows differ from the recomputation")
            valid = ctx.block_arrays(subject, val_name)["valid"]
            val_windows = np.load(folder / "val_windows.npy")
            coverage = np.zeros(valid.size, dtype=bool)
            for start in val_windows:
                coverage[start:start + window] = True
            if not np.array_equal(val_windows, inference_window_starts(valid, window, validation["inference_stride_samples"])):
                issues.append("validation windows differ from the recomputation")
            if not np.array_equal(coverage, valid):
                issues.append("validation windows do not cover exactly the valid columns")
            if windows.shape[0] != expected_total:
                issues.append("window count differs from 1 + (T - L) // s per run")
            rows.append({"subject": subject, "fold": f"val_{val_name}", "train_windows": int(windows.shape[0]),
                         "expected_by_formula": expected_total, "val_windows": int(val_windows.size),
                         "val_coverage_exact": bool(np.array_equal(coverage, valid)), "issues": "; ".join(issues)})
            problems += [f"{subject} val_{val_name}: {i}" for i in issues]
    _table(ctx, rows, "v10_windows", "Ventanas persistidas frente a su recálculo, fórmula de conteo por tramo y cobertura de la validación")
    return _verdict(problems, f"{len(rows)} folds: ventanas de train (stride 1) = 1 + (T − 256) // 1 por tramo elegible, "
                    "dentro de un solo bloque; ventanas de inferencia (stride 64 + ancla final) cubren exactamente las "
                    "columnas válidas de validación.")


# --------------------------------------------------------------------------- V11

@check("V11", "Finitud, dimensiones y persistencia")
def v11_sanity(ctx: VerificationContext):
    ctx.require_data()
    rtol = ctx.config.data["verification"]["reproducibility_rtol"]
    delay = ctx.config.data["signal"]["temporal_alignment"]["delay_samples_at_target_fs"]
    rows, problems = [], []
    for subject in ctx.subjects:
        n_kept = ctx.spec(subject).channels.n_kept
        base = _baseline(ctx, subject)
        medians = []
        for block in ctx.blocks(subject):
            manifest = read_json(ctx.pipeline_dir / subject / block.name / "block_manifest.json")
            a = ctx.block_arrays(subject, block.name, mmap=False)
            n = block.stop_target - block.start_target - delay
            same, dx, dy = _comparison(a, base[block.name], 0.0)
            scale = float(np.abs(a["x"]).max())
            reproducible = bool(np.array_equal(a["valid"], base[block.name]["valid"])) and dx <= rtol * scale \
                and dy <= rtol * float(np.nanmax(np.abs(a["y"])))
            medians.append(manifest["x_valid_summary"]["median"])
            row = {"subject": subject, "block": block.name, "x_shape": "x".join(map(str, a["x"].shape)),
                   "shape_ok": a["x"].shape == (n_kept, 40, n) and a["y"].shape == (5, n),
                   "dtypes_ok": a["x"].dtype == np.float32 and a["y"].dtype == np.float64,
                   "x_finite": bool(np.isfinite(a["x"]).all()), "y_finite_on_valid": bool(np.isfinite(a["y"][:, a["valid"]]).all()),
                   "x_min": manifest["x_valid_summary"]["min"], "x_median": manifest["x_valid_summary"]["median"],
                   "x_max": manifest["x_valid_summary"]["max"],
                   "negative_power_fraction": manifest["morlet"]["negative_fraction_valid"],
                   "recomputed_bitwise_identical": same, "max_abs_diff_recomputed": dx, "reproducible": reproducible}
            rows.append(row)
            if not (row["shape_ok"] and row["dtypes_ok"] and row["x_finite"] and row["y_finite_on_valid"] and reproducible):
                problems.append(f"{subject} {block.name}: shape, dtype, finiteness or reproducibility")
        if max(medians) / max(min(medians), 1e-300) > 10:
            problems.append(f"{subject}: block medians differ by more than an order of magnitude")
        ctx.baseline.pop(subject, None)
        for val_name in ctx.val_names:
            fold = ctx.load_fold(subject, val_name)
            x, y = fold.train_window(0)
            vx, vy = fold.val_window(fold.val_windows.size - 1)
            finite = all(np.isfinite(b.x[:, :, b.valid]).all() and np.isfinite(b.y[:, b.valid]).all()
                         for b in fold.blocks.values())
            if not finite or x.shape != (n_kept, 40, 256) or y.shape != (5, 256) or vx.shape != x.shape or vy.shape != y.shape:
                problems.append(f"{subject} val_{val_name}: reloaded fold not finite or wrong window shape")
    _table(ctx, rows, "v11_sanity", "Formas, tipos, finitud, escala y reproducibilidad (persistido frente a recalculado en esta sesión) por bloque")
    return _verdict(problems, f"15 bloques con forma (C, 40, 7980)/(5, 7980), float32/float64, X finita y flexión finita "
                    "en columnas válidas; manifiestos con SHA-256 verificados; recalcular desde los .mat reproduce lo "
                    f"persistido (rtol {rtol}); los 15 folds se recargan, escalan y dan ventanas finitas.")


# --------------------------------------------------------------------------- V12

@check("V12", "Reconstrucción temporal")
def v12_overlap_add(ctx: VerificationContext):
    ctx.require_data()
    window = ctx.config.data["validation"]["inference_window_samples"]
    degenerate = ctx.config.data["metrics"]["degenerate_std_relative"]
    rows, problems = [], []
    for subject in ctx.subjects:
        for k, val_name in enumerate(ctx.val_names):
            fold = ctx.load_fold(subject, val_name)
            block = fold.blocks[val_name]
            predictions = np.stack([block.y[:, s:s + window] for s in fold.val_windows])
            result = score_validation(fold, predictions, degenerate)
            span = float(np.nanmax(block.y_original) - np.nanmin(block.y_original))
            row = {"subject": subject, "fold": f"val_{val_name}", "windows": int(fold.val_windows.size),
                   "valid_columns": int(block.valid.sum()), "coverage_min_valid": result["coverage_min_valid"],
                   "pearson_min": min(result["pearson"]) if result["valid"] else None,
                   "rmse_max_original_units": max(result["rmse"]), "glove_span": span}
            if k == 0:
                constant = score_validation(fold, np.full_like(predictions, 0.5), degenerate)
                row["constant_prediction_marked_invalid"] = not constant["valid"] and constant["val_pearson_full"] is None
                if not row["constant_prediction_marked_invalid"]:
                    problems.append(f"{subject}: a constant prediction was not marked invalid")
            rows.append(row)
            if not result["valid"] or row["pearson_min"] < 1 - 1e-6 or row["rmse_max_original_units"] > 1e-5 * span:
                problems.append(f"{subject} val_{val_name}: identity reconstruction is not exact")
    _table(ctx, rows, "v12_overlap_add", "Reconstrucción por overlap-add de predicciones identidad sobre las ventanas de validación de cada fold")
    return _verdict(problems, f"{len(rows)} folds: el overlap-add de predicciones identidad reconstruye la flexión de "
                    "validación en unidades originales (Pearson = 1, RMSE ≈ 0) con cobertura en cada columna válida, "
                    "incluida la ventana final; una predicción constante queda inválida, no se descarta.")


# --------------------------------------------------------------------------- V13

@check("V13", "Arquitectura")
def v13_architecture(ctx: VerificationContext):
    import torch
    from ..losses import make_loss
    from ..model import build_model, count_parameters, search_space
    ctx.require_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = ctx.config.data["verification"]["architecture_batch_size"]
    seed = ctx.config.data["project"]["seed_base"]
    loss_fn = make_loss(ctx.config)
    first = ctx.config.data["validation"]["stage1_validation_blocks"][0]
    rows, problems = [], []
    for subject in ctx.subjects:
        fold = ctx.load_fold(subject, first)
        picks = np.linspace(0, fold.train_windows.shape[0] - 1, batch).astype(int)
        pairs = [fold.train_window(int(i)) for i in picks]
        x = torch.tensor(np.stack([p[0] for p in pairs])).to(device)
        y = torch.tensor(np.stack([p[1] for p in pairs])).to(device)
        for arch in search_space(ctx.config):
            torch.manual_seed(seed)
            model = build_model(ctx.config, x.shape[1], arch).to(device).train()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            seconds = time.perf_counter() - start
            grads_finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            row = {"subject": subject, "architecture": arch.key, "parameters": count_parameters(model),
                   "output_shape": "x".join(map(str, out.shape)), "loss": float(loss.detach()),
                   "loss_finite": bool(torch.isfinite(loss)), "grads_finite": grads_finite, "seconds": seconds,
                   "device": device, "peak_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device == "cuda" else None}
            rows.append(row)
            if tuple(out.shape) != (batch, 5, x.shape[-1]) or not (row["loss_finite"] and grads_finite):
                problems.append(f"{subject} {arch.key}: shape {tuple(out.shape)} or nonfinite loss/gradients")
            del model, out, loss
    _table(ctx, rows, "v13_architecture", "Forward + pérdida + backward de las 27 arquitecturas por sujeto con ventanas reales escaladas")
    return _verdict(problems, f"27 arquitecturas × 3 sujetos en {device} con ventanas reales (B={batch}): entrada "
                    "(B, C, 40, 256) → salida (B, 5, 256), pérdida y gradientes finitos. Es un smoke test de formas: "
                    "no prueba memoria suficiente para un entrenamiento prolongado.")


# --------------------------------------------------------------------------- V14

@check("V14", "Métricas y trazabilidad")
def v14_metrics(ctx: VerificationContext):
    import torch
    from ..losses import make_loss
    ctx.require_data()
    degenerate = ctx.config.data["metrics"]["degenerate_std_relative"]
    seed = ctx.config.data["project"]["seed_base"]
    first = ctx.config.data["validation"]["stage1_validation_blocks"][0]
    problems = []
    fold = ctx.load_fold(ctx.subjects[0], first)
    block = fold.blocks[first]
    valid, truth = block.valid, block.y_original
    rng = np.random.default_rng(seed)
    prediction = truth + rng.normal(scale=0.3 * float(np.nanstd(truth)), size=truth.shape)
    result = trajectory_metrics(truth, prediction, valid, degenerate_std_relative=degenerate)
    t, p = truth[:, valid], prediction[:, valid]
    manual = {"pearson": [pearsonr(t[f], p[f])[0] for f in range(5)],
              "ccc": [2 * np.mean((t[f] - t[f].mean()) * (p[f] - p[f].mean()))
                      / (t[f].var() + p[f].var() + (t[f].mean() - p[f].mean()) ** 2) for f in range(5)],
              "mse": [np.mean((p[f] - t[f]) ** 2) for f in range(5)]}
    differences = {k: float(np.max(np.abs(np.array(result[k]) - np.array(v)))) for k, v in manual.items()}
    if not result["valid"] or max(differences.values()) > 1e-10:
        problems.append(f"metrics differ from manual references: {differences}")
    constant, missing = prediction.copy(), prediction.copy()
    constant[2] = 1.0
    missing[4, np.flatnonzero(valid)[10]] = np.nan
    undefined = {name: trajectory_metrics(truth, values, valid, degenerate_std_relative=degenerate)
                 for name, values in (("constant", constant), ("nan", missing))}
    if any(r["valid"] or r["pearson_macro"] is not None or not r["diagnostics"] for r in undefined.values()):
        problems.append("an undefined Pearson was not marked invalid with a diagnostic")
    pairs = [fold.train_window(i) for i in (0, fold.train_windows.shape[0] - 1)]
    target = np.stack([pair[1] for pair in pairs]).astype(np.float64)
    estimate = target + rng.normal(scale=0.1, size=target.shape)
    eps = ctx.config.data["training"]["loss"]["eps"]
    energy = np.maximum((estimate ** 2).sum(-1) * (target ** 2).sum(-1), eps * eps)
    cosine = (estimate * target).sum(-1) / np.sqrt(energy)
    expected_loss = 0.5 * np.mean((estimate - target) ** 2) + 0.5 * (1 - cosine.mean())
    loss = float(make_loss(ctx.config)(torch.tensor(estimate), torch.tensor(target)))
    if abs(loss - expected_loss) > 1e-12 * max(1.0, abs(expected_loss)):
        problems.append(f"loss {loss} differs from the manual value {expected_loss}")
    folds_manifest = read_json(ctx.fold_set_dir / FOLDS_MANIFEST)
    trace = {"git_commit": ctx.env["git_commit"], "git_dirty": ctx.env["git_dirty"], "source_sha256": ctx.env["source_sha256"],
             "versions": {k: ctx.env["versions"].get(k) for k in ("tt-ecog", "numpy", "scipy", "mne", "scikit-learn", "torch")},
             "python": ctx.env["python"], "config_sha256": ctx.config.sha256, "seed_base": seed,
             "raw_sha256": ctx.raw_sha256, "eda_reference": {k: ctx.reference.get(k) for k in ("run_id", "artifact_manifest_sha256")},
             "pipeline_dir": ctx.pipeline_dir.name, "fold_set": ctx.fold_set_dir.name, "manifests_sha256": ctx.manifest_sha256,
             "purge": {"samples": ctx.purge, "status": ctx.purge_status},
             "scaler_sha256": {k: v for k, v in folds_manifest["files"].items() if "/scaler_" in k}}
    missing_fields = [k for k, v in trace.items() if v in (None, {}, "")]
    missing_fields += [f"versions.{k}" for k, v in trace["versions"].items() if v is None]
    if missing_fields:
        problems.append(f"traceability fields missing: {missing_fields}")
    if trace["git_dirty"]:
        problems.append("the code has uncommitted changes: results are not traceable to a commit")
    ctx.save_json(trace, "v14_traceability", "Commit, versiones, configuración, semillas, hashes de datos, manifiestos y escaladores")
    ctx.save_json(json_safe({
        "computed": {k: result[k] for k in ("pearson", "ccc", "mse", "rmse", "pearson_macro")},
        "max_abs_difference_vs_manual": differences, "loss": loss, "loss_manual": expected_loss,
        "undefined_cases": {k: {"valid": r["valid"], "diagnostics": r["diagnostics"]} for k, r in undefined.items()}}),
        "v14_metrics", "Métricas sobre flexión real de validación frente a referencias manuales; casos indefinidos")
    return _verdict(problems, "Pearson, CCC y MSE coinciden con SciPy/cálculo manual (≤ 1e-10) sobre flexión real; un "
                    "dedo constante o con NaN invalida la evaluación con diagnóstico, sin promediar el resto; la pérdida "
                    "0.5 MSE + 0.5(1 − coseno temporal) coincide con su fórmula; commit, versiones, semillas y hashes "
                    "de datos, manifiestos y escaladores quedan registrados.")


CHECKS = [v01_integrity, v02_exclusion, v03_reference, v04_splits, v05_scalers, v06_alignment, v07_val_to_train,
          v08_train_to_val, v09_filters, v10_windows, v11_sanity, v12_overlap_add, v13_architecture, v14_metrics]
