"""Chronological development blocks. Folds, purge and segments come in step 2."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Block:
    """Half-open block [start, stop) on the target (100 Hz) and source (1 kHz) grids."""

    name: str
    start_target: int
    stop_target: int
    start_source: int
    stop_source: int


def chronological_blocks(n_source: int, names: list[str], factor: int) -> list[Block]:
    """Contiguous blocks of approximately equal length.

    Boundaries are defined on the decimated grid, so every source boundary is a
    multiple of ``factor`` and maps exactly to a target sample.
    """
    if not names:
        raise ValueError("At least one block name is required")
    n_target = math.ceil(n_source / factor)
    if n_target < len(names):
        raise ValueError("Fewer target samples than blocks")
    edges = np.round(np.linspace(0, n_target, len(names) + 1)).astype(int)
    return [Block(name, int(a), int(b), int(a) * factor, min(int(b) * factor, n_source))
            for name, a, b in zip(names, edges[:-1], edges[1:])]
