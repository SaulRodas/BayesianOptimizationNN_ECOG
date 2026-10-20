"""Shared training operations and a short resource measurement, without Optuna.

No preprocessing, scaler fitting, early stopping or best-epoch selection occurs
here. Notebooks expose the epoch loop; future trials can use the same functions.
"""

import gc
import math
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import positive
from .datasets import FoldWindows
from .losses import make_loss
from .model import build_model, search_space, reference_architecture
from .validation import score_validation


def seed_run(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def fold_seed(config, subject, val_block):
    """Same subject/fold seed across architectures and learning rates."""
    subjects = config.data["project"]["subjects"]
    blocks = config.data["validation"]["block_names"]
    return config.data["project"]["seed_base"] + subjects.index(subject) * len(blocks) + blocks.index(val_block)


def make_loaders(fold, batch_size: int, seed: int, workers: int = 0):
    """Batches and their order depend only on ``seed``. ``workers`` (training.loader_workers)
    only assembles training batches in parallel and pins them for an asynchronous GPU copy."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    generator = torch.Generator().manual_seed(seed)
    # Workers restart every epoch: persistent workers would stop drawing the per-epoch
    # base seed from ``generator`` and change the shuffle (and resume state) after epoch 1.
    train = DataLoader(FoldWindows(fold, "train"), batch_size=batch_size, shuffle=True,
                       generator=generator, num_workers=workers, drop_last=False,
                       pin_memory=workers > 0 and torch.cuda.is_available())
    val = DataLoader(FoldWindows(fold, "val"), batch_size=batch_size, shuffle=False,
                     num_workers=0, drop_last=False)
    return train, val


def make_optimizer(model, learning_rate: float, weight_decay: float):
    positive(learning_rate, "learning_rate")
    positive(weight_decay, "weight_decay", zero=True)
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=learning_rate, weight_decay=weight_decay)


def train_batch(model, optimizer, loss_fn, x, y, device):
    """One complete update; used identically for profiling and training."""
    model.train()
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None for g in grads) or not torch.stack([torch.isfinite(g).all() for g in grads]).all():
        raise FloatingPointError("Missing or nonfinite training gradient")
    optimizer.step()
    return float(loss.detach())


def train_epoch(model, optimizer, loss_fn, loader, device):
    total, samples = 0.0, 0
    for x, y in loader:
        total += train_batch(model, optimizer, loss_fn, x, y, device) * len(x)
        samples += len(x)
    if not samples:
        raise ValueError("No training windows")
    return total / samples


@torch.no_grad()
def evaluate(model, loss_fn, fold, loader, device, degenerate_std_relative):
    """Score the reconstructed validation trajectory, never average batch Pearson."""
    model.eval()
    predictions, total, samples = [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predicted = model(x)
        loss = loss_fn(predicted, y)
        if not torch.isfinite(predicted).all() or not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation prediction or loss")
        predictions.append(predicted.cpu().numpy())
        total += float(loss) * len(x)
        samples += len(x)
    if not samples:
        raise ValueError("No validation windows")
    predictions = np.concatenate(predictions)
    metrics = score_validation(fold, predictions, degenerate_std_relative)
    if not metrics["valid"]:
        raise ValueError("Invalid validation Pearson: " + "; ".join(metrics["diagnostics"]))
    return {**metrics, "val_loss": total / samples}


def device_info(device):
    device = torch.device(device)
    info = {"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "precision": "float32", "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(gpu=props.name, total_bytes=props.total_memory)
    else:
        info.update(gpu=None, total_bytes=None)
    return info


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory(device):
    if torch.device(device).type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def representative_architectures(config):
    space = search_space(config)
    key = lambda a: (a.encoder_levels, a.base_filters, a.kernel_size)
    return [min(space, key=key), reference_architecture(config), max(space, key=key)]


def profile_updates(model, optimizer, loss_fn, x, y, device, warmup_steps, measured_steps):
    """Measure complete updates on a fixed real batch; no epoch or convergence claim."""
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError("At least one warmup and one measured step are required")
    reset_peak_memory(device)
    for _ in range(warmup_steps):
        train_batch(model, optimizer, loss_fn, x, y, device)
    synchronize(device)
    started = perf_counter()
    for _ in range(measured_steps):
        train_batch(model, optimizer, loss_fn, x, y, device)
    synchronize(device)
    return {"seconds_per_batch": (perf_counter() - started) / measured_steps,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps, **peak_memory(device)}


def profile_architecture(config, fold, arch, batch_size, device, seed):
    """Disposable model: profiling never supplies trained weights to a trial."""
    model = optimizer = x = y = None
    row = {"pipeline_id": fold.pipeline_id, "subject": fold.subject, "fold": fold.name,
           "architecture": arch.key, "batch_size": batch_size, "seed": seed,
           "device": str(device), "precision": "float32"}
    try:
        if len(fold.train_windows) < batch_size:
            raise ValueError("Not enough windows to measure the requested full batch")
        seed_run(seed)
        loader, _ = make_loaders(fold, batch_size, seed)
        x, y = next(iter(loader))
        model = build_model(config, x.shape[1], arch).to(device)
        optimizer = make_optimizer(model, config.data["training"]["reference_learning_rate"],
                                    config.data["training"]["reference_weight_decay"])
        measured = profile_updates(model, optimizer, make_loss(config), x, y, device,
                                    config.data["training"]["profile_warmup_steps"],
                                    config.data["training"]["profile_measured_steps"])
        estimate = measured["seconds_per_batch"] * math.ceil(len(fold.train_windows) / batch_size)
        return {**row, **measured, "status": "OK", "train_epoch_seconds_estimate": estimate,
                "estimate_excludes": "validation, full data-loading time and startup"}
    except torch.cuda.OutOfMemoryError:
        return {**row, "status": "OOM", "details": "Reduce the common batch candidate and repeat all profiles"}
    finally:
        del model, optimizer, x, y
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
