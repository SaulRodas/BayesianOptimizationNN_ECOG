"""Training loss (protocol §8.1): 0.5 MSE + 0.5 (1 - temporal cosine similarity).

The cosine is computed along time for each finger of each window, then averaged
over fingers and windows. It updates weights only; model selection uses the
reconstructed Pearson (``metrics``), never this loss.
"""

from typing import Callable

import torch

from .config import ExperimentConfig


def temporal_cosine(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """Cosine similarity along the last (time) axis, shape (B, fingers).

    The denominator is clamped at ``eps`` before the square root, so an all-zero
    window gives a finite value and a finite gradient.
    """
    _check(pred, target)
    product = (pred * target).sum(dim=-1)
    energy = (pred * pred).sum(dim=-1) * (target * target).sum(dim=-1)
    return product / torch.sqrt(torch.clamp(energy, min=eps * eps))


def mse_cosine_loss(pred: torch.Tensor, target: torch.Tensor, *, mse_weight: float, cosine_weight: float,
                    eps: float) -> torch.Tensor:
    _check(pred, target)
    mse = torch.mean((pred - target) ** 2)
    return mse_weight * mse + cosine_weight * (1 - temporal_cosine(pred, target, eps).mean())


def make_loss(config: ExperimentConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    loss = config.data["training"]["loss"]

    def compute(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return mse_cosine_loss(pred, target, mse_weight=loss["mse_weight"], cosine_weight=loss["cosine_weight"],
                               eps=loss["eps"])
    return compute


def _check(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(f"Expected matching (B, fingers, T) tensors; got {tuple(pred.shape)} and {tuple(target.shape)}")
