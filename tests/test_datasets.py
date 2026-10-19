import numpy as np
import pytest

from tt_ecog.datasets import contiguous_runs, inference_window_starts, n_windows, train_window_starts

L = 256


@pytest.mark.parametrize("length,stride,expected", [
    (L - 1, 1, 0), (L, 1, 1), (L + 1, 1, 2), (L + 63, 64, 1), (L + 64, 64, 2), (0, 1, 0)])
def test_window_count_formula(length, stride, expected):
    assert n_windows(length, L, stride) == expected
    assert train_window_starts(np.ones(length, dtype=bool), L, stride).size == expected


@pytest.mark.parametrize("window,stride", [(0, 1), (L, 0), (L, 1.0)])
def test_invalid_window_parameters(window, stride):
    with pytest.raises(ValueError):
        n_windows(300, window, stride)


def test_contiguous_runs():
    assert contiguous_runs(np.zeros(5, dtype=bool)) == []
    assert contiguous_runs(np.ones(5, dtype=bool)) == [(0, 5)]
    assert contiguous_runs(np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)) == [(0, 2), (4, 5), (6, 8)]


def two_runs():
    mask = np.zeros(1000, dtype=bool)
    mask[10:400] = True  # 390 columns
    mask[405:661] = True  # exactly one window
    mask[700:900] = True  # shorter than a window
    return mask


def test_training_windows_stay_inside_one_run():
    mask = two_runs()
    starts = train_window_starts(mask, L, 1)
    assert starts.size == n_windows(390, L, 1) + 1
    for start in starts:
        assert mask[start:start + L].all()
    assert 405 in starts and not np.isin(np.arange(700, 900), starts).any()


def test_inference_windows_cover_every_valid_column_with_an_end_anchor():
    mask = np.zeros(2000, dtype=bool)
    mask[100:1100] = True
    mask[1200:1456] = True
    starts = inference_window_starts(mask, L, 64)
    coverage = np.zeros(mask.size, dtype=int)
    for start in starts:
        assert mask[start:start + L].all()
        coverage[start:start + L] += 1
    np.testing.assert_array_equal(coverage > 0, mask)
    assert 1100 - L in starts and 1200 in starts
    assert np.diff(starts[starts < 1100])[:-1].tolist() == [64] * (np.sum(starts < 1100) - 2)


def test_inference_refuses_incomplete_coverage():
    with pytest.raises(ValueError, match="shorter"):
        inference_window_starts(two_runs(), L, 64)
    with pytest.raises(ValueError, match="uncovered"):
        inference_window_starts(np.ones(1000, dtype=bool), L, L + 10)
