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


class DeviceWindows:
    """Training batches cut from the fold's blocks, which stay on ``device`` for the whole run.

    Same windows, batches and order as ``DataLoader(FoldWindows(fold, "train"), shuffle=True)``
    with the same generator: a DataLoader over window numbers keeps PyTorch's sampler and its
    use of the generator, so checkpoints save and restore the same state. Only the per-batch
    host assembly and copy (~335 MB at batch 128) disappear.
    """

    def __init__(self, fold, batch_size: int, generator: torch.Generator, device):
        self.dataset, self.batch_size, self.generator = FoldWindows(fold, "train"), batch_size, generator
        self.device = torch.device(device)
        self._numbers = DataLoader(range(len(self.dataset)), batch_size=batch_size, shuffle=True,
                                   generator=generator, num_workers=0, drop_last=False)
        self._windows = np.asarray(fold.train_windows, dtype=np.int64).reshape(-1, 2)
        self._x, self._y = {}, {}
        for number in np.unique(self._windows[:, 0]).tolist():
            block = fold.blocks[fold.block_names[number]]
            self._x[number] = torch.as_tensor(block.x, device=self.device).unfold(-1, fold.window, 1)
            self._y[number] = torch.as_tensor(block.y, device=self.device).unfold(-1, fold.window, 1)

    def __len__(self):
        return len(self._numbers)

    def __iter__(self):
        for numbers in self._numbers:
            rows = self._windows[numbers.numpy()]
            x = y = None
            for number in np.unique(rows[:, 0]).tolist():
                xs, ys = self._x[number], self._y[number]
                if x is None:
                    x = torch.empty((len(rows), xs.shape[0], xs.shape[1], xs.shape[3]), dtype=xs.dtype, device=self.device)
                    y = torch.empty((len(rows), ys.shape[0], ys.shape[2]), dtype=ys.dtype, device=self.device)
                chosen = np.flatnonzero(rows[:, 0] == number)
                where = torch.as_tensor(chosen, device=self.device)
                starts = torch.as_tensor(rows[chosen, 1], device=self.device)
                x[where] = xs[:, :, starts].permute(2, 0, 1, 3)
                y[where] = ys[:, starts].permute(1, 0, 2)
            yield x, y


def make_loaders(fold, batch_size: int, seed: int, workers: int = 0, device=None):
    """Batches and their order depend only on ``seed``.

    With ``device`` the training windows are cut where the fold already lives (DeviceWindows);
    otherwise ``workers`` (training.loader_workers) only assembles training batches in parallel
    and pins them for an asynchronous GPU copy. Validation is a small host-side loader.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a nonnegative integer")
    generator = torch.Generator().manual_seed(seed)
    if device is not None:
        train = DeviceWindows(fold, batch_size, generator, device)
    else:
        # Workers restart every epoch: persistent workers would stop drawing the per-epoch
        # base seed from ``generator`` and change the shuffle (and resume state) after epoch 1.
        train = DataLoader(FoldWindows(fold, "train"), batch_size=batch_size, shuffle=True,
                           generator=generator, num_workers=workers, drop_last=False,
                           pin_memory=workers > 0 and torch.cuda.is_available())
    val = DataLoader(FoldWindows(fold, "val"), batch_size=batch_size, shuffle=False,
                     num_workers=0, drop_last=False)
    return train, val


def training_loaders(config, fold, batch_size: int, seed: int, device):
    """Loaders for experiment runs, following training.windows_on_device and loader_workers."""
    settings = config.data["training"]
    return make_loaders(fold, batch_size, seed, settings["loader_workers"],
                        device if settings["windows_on_device"] else None)


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
        # Batches already live on the device during training (DeviceWindows): time the update only.
        x, y = (tensor.to(device) for tensor in next(iter(loader)))
        model = build_model(config, x.shape[1], arch).to(device)
        optimizer = make_optimizer(model, config.data["training"]["reference_learning_rate"],
                                    config.data["training"]["reference_weight_decay"])
        measured = profile_updates(model, optimizer, make_loss(config), x, y, device,
                                    config.data["training"]["profile_warmup_steps"],
                                    config.data["training"]["profile_measured_steps"])
        estimate = measured["seconds_per_batch"] * math.ceil(len(fold.train_windows) / batch_size)
        return {**row, **measured, "status": "OK", "train_epoch_seconds_estimate": estimate,
                "estimate_excludes": "validation, batch assembly and host-to-device copy, and startup"}
    except torch.cuda.OutOfMemoryError:
        return {**row, "status": "OOM", "details": "Reduce the common batch candidate and repeat all profiles"}
    finally:
        del model, optimizer, x, y
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
