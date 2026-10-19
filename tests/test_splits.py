import numpy as np
import pytest

from tt_ecog.splits import chronological_blocks, check_purge, make_folds, train_eligibility

NAMES = ["B1", "B2", "B3", "B4", "B5"]
DELAY = 20


@pytest.fixture
def blocks():
    # 1000 target samples -> five blocks of 200 columns ([0,200), [200,400), ...).
    return chronological_blocks(10000, NAMES, 10)


def columns(block, valid=None):
    """Aligned columns of one block: x in [start, stop - delay), y = x + delay."""
    x = np.arange(block.start_target, block.stop_target - DELAY)
    return x, x + DELAY, np.ones(x.size, dtype=bool) if valid is None else valid


def test_stage_folds(blocks):
    folds = make_folds(blocks, ["B2", "B4"])
    assert [f.name for f in folds] == ["val_B2", "val_B4"]
    assert [b.name for b in folds[0].train] == ["B1", "B3", "B4", "B5"]
    assert len(make_folds(blocks, NAMES)) == 5
    for bad in (["B6"], ["B2", "B2"]):
        with pytest.raises(ValueError):
            make_folds(blocks, bad)


@pytest.mark.parametrize("purge", [0, 10, 37])
def test_purge_before_validation_bounds_the_flexion_time(blocks, purge):
    b2, b3 = blocks[1], blocks[2]
    x, y, valid = columns(b2)
    mask = train_eligibility(x, y, valid, b2, b3, purge)
    # Last usable flexion sits exactly purge samples before validation.
    assert y[mask].max() == b3.start_target - 1 - purge
    assert mask.sum() == np.count_nonzero(y <= b3.start_target - 1 - purge)
    assert mask[:np.flatnonzero(mask)[-1] + 1].all()


@pytest.mark.parametrize("purge", [0, 10, 37])
def test_purge_after_validation_bounds_the_ecog_time(blocks, purge):
    b3, b4 = blocks[2], blocks[3]
    x, y, valid = columns(b4)
    mask = train_eligibility(x, y, valid, b4, b3, purge)
    assert x[mask].min() == b3.stop_target + purge
    assert mask.sum() == x.size - purge


def test_distant_blocks_keep_every_valid_column(blocks):
    x, y, valid = columns(blocks[0])
    valid = valid.copy()
    valid[:5] = False
    mask = train_eligibility(x, y, valid, blocks[0], blocks[2], 37)
    np.testing.assert_array_equal(mask, valid)


def test_purge_larger_than_a_block_removes_it(blocks):
    for train in (blocks[1], blocks[3]):
        x, y, valid = columns(train)
        assert not train_eligibility(x, y, valid, train, blocks[2], 500).any()


@pytest.mark.parametrize("val", [0, 4])
def test_edge_blocks_as_validation(blocks, val):
    for train in blocks:
        if train.name == blocks[val].name:
            continue
        x, y, valid = columns(train)
        mask = train_eligibility(x, y, valid, train, blocks[val], 10)
        adjacent = abs(blocks.index(train) - val) == 1
        assert mask.sum() == (x.size - 10 if adjacent else x.size)


@pytest.mark.parametrize("purge", [None, True, -1, 2.5, "256"])
def test_pending_or_invalid_purge_is_rejected(blocks, purge):
    x, y, valid = columns(blocks[1])
    with pytest.raises(ValueError):
        train_eligibility(x, y, valid, blocks[1], blocks[2], purge)


def test_numpy_integer_purge_is_accepted():
    assert check_purge(np.int64(5)) == 5


def test_overlapping_block_and_foreign_columns_are_rejected(blocks):
    x, y, valid = columns(blocks[2])
    with pytest.raises(ValueError, match="overlaps"):
        train_eligibility(x, y, valid, blocks[2], blocks[2], 0)
    with pytest.raises(ValueError, match="outside"):
        train_eligibility(x + DELAY, y + DELAY, valid, blocks[2], blocks[0], 0)
