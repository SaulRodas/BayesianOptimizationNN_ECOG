import numpy as np
import pytest

from tt_ecog.descriptive import band_line_fraction, grouped_intervals, robust_scores, welch_psd, windowed_std
from tt_ecog.io import read_development_glove, read_ecog
from conftest import write_subject


def test_mad_zero_is_finite_and_still_flags_spike():
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


def test_windowed_std_and_line_fraction():
    fs = 1000
    t = np.arange(4000) / fs
    x = np.vstack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 60 * t)])
    x[0, 1000:2000] *= 10
    stds, starts = windowed_std(x, 1000)
    np.testing.assert_array_equal(starts, [0, 1000, 2000, 3000])
    assert stds[0, 1] / stds[0, 0] == pytest.approx(10, rel=1e-6)
    freq, psd = welch_psd(x, fs, 1.0)
    _, fraction = band_line_fraction(freq, psd, [40, 200], [60, 120, 180], 1.0, 1e-12)
    assert fraction[0] < 0.01 and fraction[1] > 0.95


def test_channel_summary_does_not_depend_on_glove(config, data_root):
    from tt_ecog.eda.bundle import SubjectBundle, calculate_overview
    write_subject(data_root, config, n=4000)
    first = read_ecog(config, data_root, "S1")
    a = SubjectBundle(config, first, read_development_glove(config, data_root, first))
    calculate_overview({"S1": a})
    write_subject(data_root, config, n=4000, values=first.values.T, labels=np.full((4000, 5), 1e8))
    second = read_ecog(config, data_root, "S1")
    b = SubjectBundle(config, second, read_development_glove(config, data_root, second))
    calculate_overview({"S1": b})
    import pandas as pd
    pd.testing.assert_frame_equal(a.channel_summary, b.channel_summary)
