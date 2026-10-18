import numpy as np
import pandas as pd
import pytest

from tt_ecog.quality_control import analyze_ecog, grouped_intervals, robust_scores
from tt_ecog.io import read_ecog, validate_development_labels
from conftest import write_subject


def test_synthetic_spike_jump_flat_and_spectral(config, candidate_qc):
    fs, n = 1000, 4000
    rng = np.random.default_rng(42)
    values = rng.normal(size=(8, n))
    values[0, 500] = 100
    values[1, 2000:] += 100
    values[2] = 0
    values[3] += 30 * np.sin(2 * np.pi * 60 * np.arange(n) / fs)
    before = values.copy()
    result = analyze_ecog(values, "S1", fs, candidate_qc, config.data["signal"])
    events = result.events
    assert ((events.channel_0based == 0) & (events.detector == "robust_amplitude")
            & (events.start_sample <= 500) & (events.end_sample_exclusive > 500)).any()
    assert ((events.channel_0based == 1) & (events.detector == "discontinuity")
            & (events.start_sample <= 2000) & (events.end_sample_exclusive > 2000)).any()
    assert ((events.channel_0based == 2) & (events.detector == "flatline")).any()
    assert ((events.channel_0based == 3) & (events.detector == "spectral")).any()
    np.testing.assert_array_equal(values, before)
    assert not result.channels.automatic_exclusion.any()
    np.testing.assert_allclose(events.start_s, events.start_sample / fs)
    np.testing.assert_allclose(events.end_s, events.end_sample_exclusive / fs)


def test_pending_thresholds_are_not_defaults(config):
    values = np.zeros((4, 2000))
    result = analyze_ecog(values, "S1", 1000, config.data["quality_control"], config.data["signal"])
    assert result.events.empty
    assert all(status.startswith("PENDING") for status in result.detector_status.values())
    assert (result.channels.mad == 0).all()


def test_mad_zero_is_finite_and_still_detects_spike():
    values = np.zeros(100)
    values[50] = 20
    scores, median, mad = robust_scores(values, 1e-12)
    assert median == mad == 0
    assert np.isfinite(scores).all()
    assert scores[50] > 8
    assert np.count_nonzero(scores) == 1


@pytest.mark.parametrize("mask,gap,expected", [([], 0, []), ([0, 1, 1, 0], 0, [(1, 3)]),
    ([1, 0, 1], 0, [(0, 1), (2, 3)]), ([1, 0, 1], 1, [(0, 3)]), ([1, 1], 0, [(0, 2)])])
def test_grouping_half_open(mask, gap, expected):
    assert grouped_intervals(np.array(mask, dtype=bool), gap) == expected


def test_labels_cannot_change_qc(config, candidate_qc, data_root):
    write_subject(data_root, config)
    first = read_ecog(config, data_root, "S1")
    validate_development_labels(config, data_root, first)
    a = analyze_ecog(first.values, "S1", 1000, candidate_qc, config.data["signal"])
    write_subject(data_root, config, values=first.values.T, labels=np.full((2000, 5), 1e8))
    second = read_ecog(config, data_root, "S1")
    validate_development_labels(config, data_root, second)
    b = analyze_ecog(second.values, "S1", 1000, candidate_qc, config.data["signal"])
    pd.testing.assert_frame_equal(a.channels, b.channels)
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.temporal, b.temporal)
