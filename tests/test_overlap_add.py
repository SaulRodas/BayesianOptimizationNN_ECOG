import numpy as np
import pytest

from tt_ecog.datasets import inference_window_starts, load_fold
from tt_ecog.validation import IncompleteCoverage, overlap_add, score_validation

pytestmark = pytest.mark.verification("V12")
L = 256


def windows(trajectory, starts):
    return np.stack([trajectory[:, s:s + L] for s in starts])


def test_identity_reconstruction_with_gaps_and_end_anchor():
    rng = np.random.default_rng(0)
    trajectory = rng.normal(size=(5, 1500))
    mask = np.zeros(1500, dtype=bool)
    mask[10:900] = True
    mask[950:1500] = True  # reaches the end of the timeline
    starts = inference_window_starts(mask, L, 64)
    rebuilt, coverage = overlap_add(windows(trajectory, starts), starts, 1500, mask)
    assert rebuilt.shape == (5, 1500)
    np.testing.assert_allclose(rebuilt[:, mask], trajectory[:, mask], rtol=0, atol=1e-12)
    assert (coverage[mask] >= 1).all() and np.isnan(rebuilt[:, coverage == 0]).all()
    assert 1500 - L in starts and coverage[-1] >= 1


def test_overlapping_predictions_are_averaged():
    starts = np.array([0, 64])
    predictions = np.stack([np.full((1, L), 1.0), np.full((1, L), 3.0)])
    rebuilt, coverage = overlap_add(predictions, starts, 400)
    assert rebuilt[0, 10] == 1.0 and rebuilt[0, 100] == 2.0 and rebuilt[0, 300] == 3.0
    assert coverage.tolist()[:65] == [1] * 64 + [2]


def test_incomplete_coverage_and_bad_windows_fail():
    mask = np.ones(600, dtype=bool)
    with pytest.raises(IncompleteCoverage):
        overlap_add(np.zeros((1, 5, L)), np.array([0]), 600, mask)
    with pytest.raises(ValueError):
        overlap_add(np.zeros((1, 5, L)), np.array([500]), 600)
    with pytest.raises(ValueError):
        overlap_add(np.zeros((2, 5, L)), np.array([0]), 600)


def test_identity_prediction_scores_one_on_a_processed_fold(prepared):
    fold = load_fold(prepared["config"], prepared["folds"], "S2", "B3", allow_candidate=True)
    block = fold.blocks["B3"]
    predictions = windows(block.y, fold.val_windows)
    result = score_validation(fold, predictions, 1e-9)
    assert result["valid"] and result["coverage_min_valid"] >= 1
    assert result["val_pearson_full"] == pytest.approx(1.0, abs=1e-6)
    assert max(result["rmse"]) < 1e-5 * np.nanmax(np.abs(block.y_original))
    constant = score_validation(fold, np.full_like(predictions, 0.5), 1e-9)
    assert not constant["valid"] and constant["val_pearson_full"] is None
