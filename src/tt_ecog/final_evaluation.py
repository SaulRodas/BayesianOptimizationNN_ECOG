"""Final development-only decisions and fixed-model inference, exposed in notebook 07."""

from copy import deepcopy
import math
from time import perf_counter

import numpy as np
import torch

from .config import ConfigError, positive
from .datasets import contiguous_runs
from .model import reference_architecture
from .metrics import trajectory_metrics
from .validation import overlap_add
from .training import fold_seed, synchronize, reset_peak_memory, peak_memory


VARIANTS = ("baseline_A", "training_optimized_B", "architecture_optimized_C")
ENSEMBLE = "ensemble_smoothed_D"  # protocol v1.4: C over predefined seeds, averaged and smoothed


def final_policy(config):
    """No pilot/search duration becomes an implicit final-training decision."""
    t = config.data["training"]
    for key in ("batch_size", "final_max_epochs", "final_early_stopping_patience"):
        if type(t[key]) is not int or t[key] < 1:
            raise ConfigError(f"training.{key} is pending; decide it from development")
    positive(t["final_early_stopping_min_delta"], "training.final_early_stopping_min_delta", zero=True)
    block = t.get("final_validation_block")
    if block not in config.data["validation"]["block_names"]:
        raise ConfigError("training.final_validation_block is pending; select a common internal fold")
    if (t["final_early_stopping"] is not True
            or t["final_checkpoint_metric"] != "val_pearson_full"
            or t["final_retrain_full_development_after_epoch_selection"] is not True
            or t["common_seeds_between_configurations"] is not True):
        raise ConfigError("07 requires common seeds, internal Pearson selection and full-development refit")
    if config.data["final_evaluation"]["variants"] != list(VARIANTS):
        raise ConfigError("07 requires the predefined A/B/C comparison")
    return {"batch_size": t["batch_size"], "max_epochs": t["final_max_epochs"],
            "patience": t["final_early_stopping_patience"], "min_delta": t["final_early_stopping_min_delta"],
            "validation_block": block}


def final_contract(config):
    """Frozen experimental settings; output paths and presentation are provenance only."""
    final_policy(config)
    data = config.data
    return deepcopy({"pipeline_id": data["paths"]["processed_pipeline_id"],
                     **{k: data[k] for k in ("training", "model", "metrics", "optimization", "final_evaluation")},
                     "subjects": data["project"]["subjects"], "seed_base": data["project"]["seed_base"]})


def selected_variants(config, drive_root, subject):
    """Read both completed studies for this subject/pipeline; never launch a trial."""
    from .optimization import search_session, completed_selection, stage1_selection
    from .io import StoreError
    with search_session(config, drive_root, subject, "stage1", existing_only=True) as session:
        first = stage1_selection(session)
    with search_session(config, drive_root, subject, "stage2", existing_only=True) as session:
        second = completed_selection(session)
        if session.contract["parent"] != first:
            raise StoreError("Stage2 does not inherit the selected stage1")
    ref = reference_architecture(config)
    arch = {k: getattr(ref, k) for k in ("kernel_size", "encoder_levels", "base_filters")}
    return {
        VARIANTS[0]: {"architecture": arch, "learning_rate": config.data["training"]["reference_learning_rate"],
                      "weight_decay": config.data["training"]["reference_weight_decay"], "source_trials": []},
        VARIANTS[1]: {"architecture": arch, **first["parameters"], "source_trials": [first]},
        VARIANTS[2]: {"architecture": second["parameters"], **first["parameters"], "source_trials": [first, second]},
    }


def ensemble_members(config):
    """Run folders averaged by D: member 0 is C itself, then one folder per extra seed."""
    ensemble = config.data["final_evaluation"]["ensemble"]
    extra = range(1, len(ensemble["seed_offsets"]))
    return [ensemble["base_variant"], *(f"{ENSEMBLE}/member_{k}" for k in extra)]


def final_runs(config, plans, subject):
    """Every model 07 trains for one subject, keyed by its folder: A/B/C and D's extra members.

    A/B/C share the seed of the internal fold. Member k of D trains C's configuration with
    that seed plus its predefined offset (offset 0 is C), under the same duration rule.
    """
    seed = fold_seed(config, subject, config.data["training"]["final_validation_block"])
    ensemble = config.data["final_evaluation"]["ensemble"]
    runs = {variant: {"parameters": plans[variant], "seed": seed} for variant in VARIANTS}
    for name, offset in zip(ensemble_members(config)[1:], ensemble["seed_offsets"][1:]):
        runs[name] = {"parameters": plans[ensemble["base_variant"]], "seed": seed + offset}
    return runs


def epoch_selection(history, policy):
    """Highest internal Pearson (earliest tie); patience counts significant improvements.

    min_delta affects the stopping clock, not which observed epoch has best Pearson.
    Reconstructing this small state from checkpoint history makes resume deterministic.
    """
    if not history:
        return {"selected_epoch": None, "stop": False}
    if [r["epoch"] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError("Epoch history must be contiguous")
    scores = [r["val_pearson_full"] for r in history]
    if not all(math.isfinite(v) for v in scores):
        raise ValueError("Nonfinite internal Pearson")
    anchor, stale = -math.inf, 0
    for score in scores:
        if score > anchor + policy["min_delta"]:
            anchor, stale = score, 0
        else:
            stale += 1
    best = int(np.argmax(scores))
    return {"selected_epoch": best + 1, "selected_val_pearson_full": scores[best],
            "completed_epochs": len(history), "stale_epochs": stale,
            "stop": stale >= policy["patience"] or len(history) >= policy["max_epochs"]}


def smooth_trajectory(prediction, valid, sigma, truncate):
    """Gaussian smoothing of every finger inside each contiguous run of valid columns.

    Normalized convolution: near a run edge the kernel uses only that run's samples, so
    padding invents no values and runs separated by invalid columns never mix. Symmetric
    (non-causal), like the rest of the offline evaluation. sigma 0 leaves values unchanged.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction.ndim != 2 or valid.shape != prediction.shape[1:]:
        raise ValueError(f"Expected (fingers, T) predictions and a (T,) mask; got {prediction.shape}, {valid.shape}")
    positive(sigma, "sigma", zero=True)
    positive(truncate, "truncate")
    if not np.isfinite(prediction[:, valid]).all():
        raise ValueError("Nonfinite prediction inside the valid columns")
    smoothed = np.full_like(prediction, np.nan)
    if sigma == 0:
        smoothed[:, valid] = prediction[:, valid]
        return smoothed
    from scipy.ndimage import gaussian_filter1d
    for start, stop in contiguous_runs(valid):
        weight = gaussian_filter1d(np.ones(stop - start), sigma, mode="constant", cval=0.0, truncate=truncate)
        smoothed[:, start:stop] = gaussian_filter1d(prediction[:, start:stop], sigma, axis=-1, mode="constant",
                                                    cval=0.0, truncate=truncate) / weight
    return smoothed


def smoothing_selection(config, truth, valid, members):
    """D's sigma from the internal block only: highest Pearson macro (smallest sigma on ties).

    ``members`` are the internal-block trajectories (glove units) of D's members at their
    selected epochs, trained without that block. Recomputing it from the stored arrays must
    give the same decision, so freezing can check it like the epoch selections.
    """
    ensemble = config.data["final_evaluation"]["ensemble"]
    smoothing = ensemble["smoothing"]
    members = np.asarray(members, dtype=np.float64)
    if members.ndim != 3 or members.shape[0] != len(ensemble["seed_offsets"]) or members.shape[1:] != np.shape(truth):
        raise ValueError(f"Expected one (fingers, T) internal trajectory per member; got {members.shape}")
    if not np.isfinite(members[:, :, valid]).all():
        raise ValueError("A member does not cover every valid internal column")
    relative = config.data["metrics"]["degenerate_std_relative"]
    combined = members.mean(axis=0)
    table = []
    for sigma in smoothing["sigma_samples"]:
        result = trajectory_metrics(truth, smooth_trajectory(combined, valid, sigma, smoothing["truncate_sigmas"]),
                                    valid, degenerate_std_relative=relative)
        if not result["valid"]:
            raise ValueError(f"Invalid internal Pearson with sigma {sigma}: " + "; ".join(result["diagnostics"]))
        table.append({"sigma_samples": sigma, "pearson_macro": result["pearson_macro"],
                      **dict(zip(result["fingers"], result["pearson"]))})
    best = int(np.argmax([row["pearson_macro"] for row in table]))  # first maximum: smallest sigma
    return {"selected_sigma_samples": table[best]["sigma_samples"],
            "selected_internal_pearson": table[best]["pearson_macro"],
            "truncate_sigmas": smoothing["truncate_sigmas"], "members": ensemble_members(config),
            "member_internal_pearson": [trajectory_metrics(truth, m, valid, degenerate_std_relative=relative)
                                        ["pearson_macro"] for m in members],
            "sigma_table": table}


@torch.no_grad()
def predict_trajectory(model, x, starts, valid, target_scaler, batch_size, device):
    """Uniform overlap-add; no label is needed for inference. Return original glove units."""
    window = 256
    model.eval()
    reset_peak_memory(device)
    synchronize(device)
    started = perf_counter()
    predictions = []
    for offset in range(0, len(starts), batch_size):
        batch = np.stack([x[..., int(s):int(s) + window] for s in starts[offset:offset + batch_size]])
        result = model(torch.as_tensor(batch, dtype=torch.float32, device=device))
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite test prediction")
        predictions.append(result.cpu().numpy())
    if not predictions:
        raise ValueError("No test windows")
    scaled, coverage = overlap_add(np.concatenate(predictions), starts, len(valid), valid)
    predicted = np.full_like(scaled, np.nan)
    predicted[:, valid] = target_scaler.inverse_transform(scaled[:, valid].T).T
    synchronize(device)
    return predicted, coverage, {"inference_seconds": perf_counter() - started, **peak_memory(device)}


def internal_prediction(config, model, fold, batch_size, device, val_pearson_full):
    """Internal-block trajectory (glove units) of the epoch just scored, kept for D's sigma.

    Same windows, overlap-add and inverse scaling as the validation score: the stored
    trajectory must reproduce that epoch's val_pearson_full, or the run fails.
    """
    block = fold.blocks[fold.val_block]
    predicted, _, _ = predict_trajectory(model, block.x, fold.val_windows, block.valid, fold.target_scaler,
                                         batch_size, device)
    score = trajectory_metrics(block.y_original, predicted, block.valid,
                               degenerate_std_relative=config.data["metrics"]["degenerate_std_relative"])
    if not score["valid"] or not math.isclose(score["pearson_macro"], val_pearson_full, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(f"Internal trajectory does not reproduce val_pearson_full: "
                         f"{score['pearson_macro']} vs {val_pearson_full}")
    return predicted


def score_test(truth, predicted, valid, config):
    result = trajectory_metrics(truth, predicted, valid,
                               degenerate_std_relative=config.data["metrics"]["degenerate_std_relative"])
    if not result["valid"]:
        raise ValueError("Invalid test Pearson: " + "; ".join(result["diagnostics"]))
    return result


def combined_timing(timings):
    """D's inference cost: every member's forward passes; memory peak of the largest one."""
    peaks = {key: [t[key] for t in timings] for key in ("peak_allocated_bytes", "peak_reserved_bytes")}
    return {"inference_seconds": sum(t["inference_seconds"] for t in timings),
            **{key: None if None in values else max(values) for key, values in peaks.items()}}
