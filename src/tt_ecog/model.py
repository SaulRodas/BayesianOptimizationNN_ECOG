"""Parametrizable 1D U-Net (protocol §7): one implementation for every variant.

Input (B, C, 40, T) -> trainable channel-frequency reduction shared in time
(Conv1d over C*40 inputs) -> encoder of dilated Conv1d blocks, each followed by
MaxPool1d(2), with the skip taken before pooling -> mirrored decoder: linear
interpolation to the exact skip length, Conv1d, BatchNorm, GELU, Dropout, then
concatenation of the skip -> Conv1d(kernel 1) to five fingers. Every conv block
is Conv1d (no bias) -> BatchNorm1d -> GELU -> Dropout. Weights use PyTorch's
default initialisation for every variant.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import torch
from torch import nn

from .config import ConfigError, ExperimentConfig


@dataclass(frozen=True)
class Architecture:
    kernel_size: int
    encoder_levels: int
    base_filters: int
    channels: tuple[int, ...]  # encoder_levels + 1 entries, the last one is the deep representation
    kernels: tuple[int, ...]  # one per encoder level; the decoder mirrors them
    dilations: tuple[int, ...]  # one per encoder level

    @property
    def key(self) -> str:
        return f"k{self.kernel_size}_L{self.encoder_levels}_f{self.base_filters}"


def architecture(config: ExperimentConfig, kernel_size: int, encoder_levels: int, base_filters: int) -> Architecture:
    """Kernels [7,7] + [k]*(L-2), channels [f,f,2f,4f,8f,8f][:L+1], dilations [1,2,3,1,2][:L]."""
    ref = config.data["model"]["reference"]
    multipliers, dilations = ref["channel_multipliers"], ref["encoder_dilations"]
    for name, value in (("kernel_size", kernel_size), ("encoder_levels", encoder_levels),
                        ("base_filters", base_filters)):
        if type(value) is not int or value < 1:
            raise ConfigError(f"{name} must be a positive integer; got {value!r}")
    if not 2 <= encoder_levels <= min(len(dilations), len(multipliers) - 1):
        raise ConfigError(f"encoder_levels must lie in [2, {min(len(dilations), len(multipliers) - 1)}]")
    return Architecture(kernel_size, encoder_levels, base_filters,
                        tuple(m * base_filters for m in multipliers[:encoder_levels + 1]),
                        tuple(ref["encoder_kernels"][:2]) + (kernel_size,) * (encoder_levels - 2),
                        tuple(dilations[:encoder_levels]))


def reference_architecture(config: ExperimentConfig) -> Architecture:
    ref = config.data["model"]["reference"]
    arch = architecture(config, ref["encoder_kernels"][-1], ref["encoder_levels"], ref["base_filters"])
    if list(arch.kernels) != ref["encoder_kernels"] or list(arch.dilations) != ref["encoder_dilations"]:
        raise ConfigError("The reference U-Net does not follow the stage-2 construction rule")
    return arch


def search_space(config: ExperimentConfig) -> list[Architecture]:
    """The 27 stage-2 architectures (kernel_size x encoder_levels x base_filters)."""
    search = config.data["optimization"]["stage2"]["search"]
    return [architecture(config, k, levels, f)
            for k, levels, f in product(search["kernel_size"], search["encoder_levels"], search["base_filters"])]


@lru_cache(maxsize=64)
def _linear_weights(length: int, size: int, dtype: torch.dtype, device: torch.device):
    # area_pixel_compute_source_index of PyTorch's upsample_linear1d, align_corners=False.
    scale = torch.tensor(length, dtype=dtype) / size
    position = ((torch.arange(size, dtype=dtype) + 0.5) * scale - 0.5).clamp_min(0)
    left = position.floor().long()
    right = (left + 1).clamp_max(length - 1)
    weight = position - left
    return left.to(device), right.to(device), weight.to(device)


def linear_upsample(x: torch.Tensor, size: int) -> torch.Tensor:
    """``F.interpolate(x, size, mode="linear", align_corners=False)`` on the last axis.

    Same formula, written as gathers: PyTorch's CUDA kernel parallelises only over
    output positions and loops over batch and channels, which took ~96 % of a step.
    """
    compute = torch.promote_types(x.dtype, torch.float32)
    left, right, weight = _linear_weights(x.shape[-1], size, compute, x.device)
    out = x[..., left].to(compute) * (1 - weight) + x[..., right].to(compute) * weight
    return out.to(x.dtype)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same", bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.norm(self.conv(x))))


class UNet1D(nn.Module):
    def __init__(self, n_channels: int, n_frequencies: int, n_outputs: int, arch: Architecture,
                 dropout: float, reduce_kernel: int, decoder_dilation: int = 1):
        super().__init__()
        self.n_channels, self.n_frequencies, self.arch = n_channels, n_frequencies, arch
        ch, levels = arch.channels, arch.encoder_levels
        self.reduce = ConvBlock(n_channels * n_frequencies, ch[0], reduce_kernel, 1, dropout)
        self.encoder = nn.ModuleList(ConvBlock(ch[i], ch[i + 1], arch.kernels[i], arch.dilations[i], dropout)
                                     for i in range(levels))
        self.pool = nn.MaxPool1d(2)
        self.decoder = nn.ModuleList(
            ConvBlock(ch[levels] if i == levels - 1 else 2 * ch[i + 1], ch[i], arch.kernels[i], decoder_dilation,
                      dropout) for i in reversed(range(levels)))
        self.head = nn.Conv1d(2 * ch[0], n_outputs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.n_channels or x.shape[2] != self.n_frequencies:
            raise ValueError(f"Expected (B, {self.n_channels}, {self.n_frequencies}, T); got {tuple(x.shape)}")
        batch, channels, frequencies, time = x.shape
        x = self.reduce(x.reshape(batch, channels * frequencies, time))
        skips = []
        for block in self.encoder:
            skips.append(x)
            x = self.pool(block(x))
        for block, skip in zip(self.decoder, reversed(skips)):
            x = linear_upsample(x, skip.shape[-1])
            x = torch.cat([block(x), skip], dim=1)
        return self.head(x)


def build_model(config: ExperimentConfig, n_channels: int, arch: Architecture | None = None) -> UNet1D:
    """U-Net for one subject; ``n_channels`` is the effective channel count (61/46/64)."""
    model = config.data["model"]
    return UNet1D(n_channels, model["input_frequency_bins"], model["output_channels"],
                  arch or reference_architecture(config), model["dropout"], model["spatial_reduce_kernel"],
                  model["decoder_dilation"])


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
