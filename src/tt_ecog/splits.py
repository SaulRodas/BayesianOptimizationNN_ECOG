"""Chronological development blocks, folds and the purge rule.

Each block is processed independently (``preprocessing.segment_features``), so
features never depend on another block's raw data. A fold validates one block
and trains on the others; the purge only removes training columns near the
validation block, so changing it never requires reprocessing.
"""

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


@dataclass(frozen=True)
class Fold:
    """One validation block; every other block is a separate training segment."""

    name: str
    val: Block
    train: tuple[Block, ...]


def make_folds(blocks: list[Block], val_names: list[str]) -> list[Fold]:
    names = [block.name for block in blocks]
    unknown = [name for name in val_names if name not in names]
    if unknown or len(set(val_names)) != len(val_names):
        raise ValueError(f"Validation blocks must be distinct block names; got {val_names}")
    return [Fold(f"val_{name}", blocks[names.index(name)], tuple(b for b in blocks if b.name != name))
            for name in val_names]


def check_purge(purge: object) -> int:
    """The purge must be an explicit integer >= 0; a pending (None) purge is an error."""
    if purge is None:
        raise ValueError("The purge is pending (null); pass the approved value or the labelled candidate")
    if (type(purge) is not int and not isinstance(purge, np.integer)) or purge < 0:
        raise ValueError(f"The purge must be an integer >= 0; got {purge!r}")
    return int(purge)


def train_eligibility(x_index: np.ndarray, y_index: np.ndarray, valid: np.ndarray,
                      block: Block, val: Block, purge: int) -> np.ndarray:
    """Training columns of ``block`` usable when ``val`` is validated.

    Column n pairs ECoG at global 100 Hz index x_index[n] with flexion at
    y_index[n]; both lie inside ``block``. The purge keeps ``purge`` samples
    between the validation block and the training side: before validation the
    flexion must satisfy y <= val.start - 1 - purge; after it, the ECoG must
    satisfy x >= val.stop + purge. Validation columns are never purged.
    """
    purge = check_purge(purge)
    x_index, y_index, valid = np.asarray(x_index), np.asarray(y_index), np.asarray(valid, dtype=bool)
    if not x_index.shape == y_index.shape == valid.shape or x_index.ndim != 1:
        raise ValueError("x_index, y_index and valid must be one-dimensional and aligned")
    if x_index.size and (x_index.min() < block.start_target or y_index.max() >= block.stop_target):
        raise ValueError(f"Columns of {block.name} reference samples outside the block")
    if block.stop_target <= val.start_target:
        side = y_index <= val.start_target - 1 - purge
    elif block.start_target >= val.stop_target:
        side = x_index >= val.stop_target + purge
    else:
        raise ValueError(f"Training block {block.name} overlaps the validation block {val.name}")
    return valid & side
