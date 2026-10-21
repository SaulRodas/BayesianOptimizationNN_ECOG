"""Final development-only decisions and fixed-model inference, exposed in notebook 07."""

from copy import deepcopy
import math
from time import perf_counter

import numpy as np
import torch

from .config import ConfigError, positive
from .model import reference_architecture
from .metrics import trajectory_metrics
from .validation import overlap_add
from .training import synchronize, reset_peak_memory, peak_memory


VARIANTS = ("baseline_A", "training_optimized_B", "architecture_optimized_C")


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


def score_test(truth, predicted, valid, config):
    result = trajectory_metrics(truth, predicted, valid,
                               degenerate_std_relative=config.data["metrics"]["degenerate_std_relative"])
    if not result["valid"]:
        raise ValueError("Invalid test Pearson: " + "; ".join(result["diagnostics"]))
    return result
