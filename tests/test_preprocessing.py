import ast
import math

import numpy as np
import pytest

from tt_ecog import preprocessing as pp
from tt_ecog.io import DataContractError
from tt_ecog.splits import chronological_blocks
from conftest import ROOT

FS = 1000.0


@pytest.fixture
def spec(config):
    return pp.PreprocessingSpec.from_config(config, "S2")


def tone(freq, n, fs=FS, amplitude=1.0):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / fs)


# ------------------------------------------------------------------ channels

def test_exclusion_reindexes_and_copies():
    x = np.repeat(np.arange(6, dtype=float)[:, None], 10, axis=1)
    x.setflags(write=False)
    cmap = pp.channel_map("S1", 6, [1, 4])
    kept = pp.exclude_channels(x, cmap)
    np.testing.assert_array_equal(kept[:, 0], [0, 2, 3, 5])
    assert kept.flags.writeable and not np.shares_memory(kept, x)
    assert cmap.new_index(5) == 3
    assert cmap.to_json()["kept_original_in_effective_order"] == [0, 2, 3, 5]
    with pytest.raises(KeyError):
        cmap.new_index(4)


@pytest.mark.parametrize("excluded", [[1, 1], [-1], [6], [0, 1, 2, 3, 4, 5], [True]])
def test_invalid_exclusions_rejected(excluded):
    with pytest.raises(ValueError):
        pp.channel_map("S1", 6, excluded)


def test_spec_channel_counts(config):
    counts = {s: pp.PreprocessingSpec.from_config(config, s).channels.n_kept for s in ("S1", "S2", "S3")}
    assert counts == {"S1": 61, "S2": 46, "S3": 64}


# ------------------------------------------------------------------ CMR

@pytest.mark.parametrize("n_channels", [5, 6])
def test_cmr_matches_numpy_median_and_chunks(n_channels):
    x = np.random.default_rng(0).normal(size=(n_channels, 1003))
    expected = x - np.median(x, axis=0, keepdims=True)
    np.testing.assert_allclose(pp.common_median_reference(x), expected, atol=1e-12)
    np.testing.assert_allclose(pp.common_median_reference(x, chunk_samples=17), expected, atol=1e-12)


def test_cmr_is_instantaneous_and_robust_to_one_extreme_channel():
    x = np.random.default_rng(1).normal(size=(7, 500))
    base = pp.common_median_reference(x)
    perturbed = x.copy()
    perturbed[3, 200] += 1e6
    out = pp.common_median_reference(perturbed)
    changed = np.flatnonzero(np.any(np.abs(out - base) > 1e-9, axis=0))
    np.testing.assert_array_equal(changed, [200])
    others = np.delete(np.arange(7), 3)
    assert np.abs(out[others, 200] - base[others, 200]).max() < 10


# ------------------------------------------------------------------ FIR

def test_bandpass_and_notch_designs(spec):
    assert spec.bandpass.n_taps == 331 and spec.notch.n_taps == 1651
    assert spec.fir.n_taps == 1981 and spec.fir.half_support == 990
    assert abs(spec.bandpass.gain_db([100.0])[0]) < 0.1
    assert (spec.bandpass.gain_db([10.0, 300.0]) <= -40).all()
    assert (spec.notch.gain_db([60.0, 120.0, 180.0]) <= -40).all()
    assert (np.abs(spec.notch.gain_db([70.0, 100.0, 150.0])) < 0.1).all()
    np.testing.assert_allclose(spec.fir.taps, spec.fir.taps[::-1])


def test_apply_fir_keeps_only_in_band_tone(spec):
    n = 20000
    x = tone(100, n) + tone(10, n) + tone(60, n) + tone(300, n)
    y = pp.apply_fir(x, spec.fir)
    interior = pp.valid_slice(n, spec.fir.half_support)
    np.testing.assert_allclose(y[interior], tone(100, n)[interior], atol=0.02)


def test_apply_fir_matches_convolution_and_support(spec):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 6000))
    y = pp.apply_fir(x, spec.bandpass)
    half = spec.bandpass.half_support
    reference = np.vstack([np.convolve(row, spec.bandpass.taps, mode="same") for row in x])
    np.testing.assert_allclose(y[:, half:-half], reference[:, half:-half], atol=1e-10)
    impulse = np.zeros(8001)
    impulse[4000] = 1
    response = pp.apply_fir(impulse, spec.fir)
    # FFT-based convolution leaves round-off (~1e-17) outside the kernel.
    support = np.flatnonzero(np.abs(response) > 1e-12 * np.abs(response).max())
    assert support.min() >= 4000 - spec.fir.half_support
    assert support.max() <= 4000 + spec.fir.half_support
    assert np.argmax(np.abs(response)) == 4000


def test_segments_are_filtered_independently(spec):
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    ya = pp.apply_fir(a, spec.fir)
    b_huge = b.copy()
    b_huge[0] = 1e9
    pp.apply_fir(b_huge, spec.fir)
    np.testing.assert_array_equal(pp.apply_fir(a, spec.fir), ya)


# ------------------------------------------------------------------ Morlet

def test_morlet_frequency_grid(spec):
    np.testing.assert_allclose(spec.freqs, np.geomspace(40, 200, 40))
    support = pp.morlet_half_support(spec.freqs, spec.n_cycles, FS)
    assert support[0] == max(support) and 135 <= support[0] <= 140


def test_morlet_tone_peak_and_impulse_localization(spec):
    n = 4000
    x = np.vstack([tone(spec.freqs[20], n), np.zeros(n)])
    x[1, 2000] = 1.0
    power = pp.morlet_power(x, FS, spec.freqs, spec.n_cycles, True)
    assert power.shape == (2, 40, n)
    assert int(np.argmax(power[0, :, 1000:3000].mean(axis=-1))) == 20
    assert abs(int(np.argmax(power[1, 20])) - 2000) <= 1
    constant = pp.morlet_power(np.full((1, n), 5.0), FS, spec.freqs, spec.n_cycles, True)
    assert constant[0, :, 500:3500].max() < 1e-10


def test_morlet_per_channel_equals_single_call(spec):
    x = np.random.default_rng(4).normal(size=(3, 3000))
    joint = pp.morlet_power(x, FS, spec.freqs, spec.n_cycles, True)
    separate = np.vstack([pp.morlet_power(x[c:c + 1], FS, spec.freqs, spec.n_cycles, True) for c in range(3)])
    np.testing.assert_allclose(joint, separate, rtol=1e-10, atol=1e-14)


# ------------------------------------------------------------------ decimation

def test_antialias_design_and_alias_rejection(spec):
    aa = spec.antialias
    assert aa.n_taps % 2 == 1 and 200 <= aa.n_taps <= 300
    n = 20000
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 70 * t)
    y = pp.decimate(x, aa, 10)
    assert y.size == math.ceil(n / 10)
    naive = x[::10]
    edge = math.ceil(aa.half_support / 10)
    t100 = t[::10]
    alias = np.sin(2 * np.pi * 30 * t100)  # 70 Hz folds onto 30 Hz at 100 Hz
    interior = slice(edge, -edge)
    alias_naive = abs(np.dot(naive[interior], alias[interior]))
    alias_aa = abs(np.dot(y[interior], alias[interior]))
    assert 20 * np.log10(alias_aa / alias_naive) <= -50
    np.testing.assert_allclose(y[interior], np.sin(2 * np.pi * 5 * t100)[interior], atol=0.01)


def test_decimation_timing_on_ramp(spec):
    ramp = np.arange(5000, dtype=float)
    y = pp.decimate(ramp, spec.antialias, 10)
    edge = math.ceil(spec.antialias.half_support / 10)
    np.testing.assert_allclose(y[edge:-edge], ramp[::10][edge:-edge], atol=1e-6)
    np.testing.assert_allclose(pp.output_times(5000, 10, FS)[:3], [0, 0.01, 0.02])


def test_morlet_power_decimated_shape_and_report(spec):
    x = np.random.default_rng(5).normal(size=(2, 3000))
    power, report = pp.morlet_power_decimated(x, spec)
    assert power.shape == (2, 40, 300) and power.dtype == np.float32
    assert report["total_values"] == power.size


def test_support_report_is_informational(spec):
    report = pp.support_report(spec)
    assert report["total_half_support_samples_source"] == 990 + report["morlet_half_support_samples_source_max"] \
        + spec.antialias.half_support
    assert report["total_half_support_samples_target"] < 256


def test_measured_support_within_analytic(spec):
    result = pp.measured_support(spec, 1e-6)
    assert 0 < result["measured_half_support_target"] <= result["analytic"]["total_half_support_samples_target"]


@pytest.mark.parametrize("n_source", [6000, 6007])
def test_valid_columns_do_not_depend_on_data_outside_the_segment(spec, n_source):
    """A segment processed alone equals the same samples inside a longer record on
    every column of feature_valid_mask (V07/V08/V09)."""
    rng = np.random.default_rng(9)
    before, after = 3000, 2000  # before is a multiple of the factor: same decimation grid
    record = rng.normal(scale=20.0, size=(spec.channels.n_original, before + n_source + after))
    alone, valid, report = pp.segment_features(record[:, before:before + n_source], spec)
    embedded, _, _ = pp.segment_features(record, spec)
    embedded = embedded[:, :, before // 10:before // 10 + alone.shape[-1]]
    half = pp.chain_half_support(spec)
    assert half == 1250 and alone.shape == (46, 40, math.ceil(n_source / 10))
    assert np.flatnonzero(valid)[0] == math.ceil(half / 10)
    assert np.flatnonzero(valid)[-1] == (n_source - 1 - half) // 10
    np.testing.assert_allclose(alone[:, :, valid], embedded[:, :, valid], rtol=1e-5,
                               atol=1e-6 * float(np.abs(embedded).max()))
    assert report["valid_columns"] == valid.sum() and 0 <= report["negative_values_valid"] < report["values_valid"]
    assert not np.allclose(alone[:, :, 0], embedded[:, :, 0], rtol=1e-3)


def test_segment_features_rejects_wrong_channel_count(spec):
    with pytest.raises(DataContractError):
        pp.segment_features(np.zeros((47, 3000)), spec)


# ------------------------------------------------------------------ glove

def staircase(n, factor=40, phase=7, seed=6):
    rng = np.random.default_rng(seed)
    samples = np.cumsum(rng.normal(size=(5, n // factor + 2)), axis=1)
    samples[:, 10:14] = samples[:, 9:10]  # repeated values at rest
    stored = np.repeat(samples, factor, axis=1)[:, factor - phase:factor - phase + n]
    return stored, samples


def test_glove_staircase_detected_and_recovered():
    stored, _ = staircase(4000)
    info = pp.inspect_glove_storage(stored, 40)
    assert info["identified"] and info["method"] == "zero_order_hold" and info["phase"] == 7
    values, times = pp.recover_glove_samples(stored, 40, info["phase"], FS, info)
    np.testing.assert_array_equal(values, stored[:, 7::40])
    np.testing.assert_allclose(times[:2], [0.007, 0.047])
    assert set(np.unique(info["run_lengths"][1:-1] % 40)) == {0}


def test_glove_linear_upsampling_detected():
    knots = np.arange(3, 4000, 40)
    rng = np.random.default_rng(7)
    values = np.cumsum(rng.normal(size=(5, knots.size)), axis=1)
    stored = np.vstack([np.interp(np.arange(4000), knots, row) for row in values])
    info = pp.inspect_glove_storage(stored, 40)
    assert info["method"] == "linear" and info["phase"] == 3


def test_unknown_glove_storage_raises():
    noise = np.random.default_rng(8).normal(size=(5, 4000))
    info = pp.inspect_glove_storage(noise, 40)
    assert not info["identified"] and info["method"] == "unknown"
    with pytest.raises(DataContractError):
        pp.recover_glove_samples(noise, 40, info["phase"], FS, info)


def test_effective_glove_on_both_storage_grids(spec):
    stored, samples = staircase(8000, phase=0)
    info = pp.glove_storage_inspection(stored, 1000.0, spec)
    y25, knots = pp.effective_glove(stored, info, spec)
    assert info["method"] == "zero_order_hold" and knots[:3].tolist() == [0, 40, 80]
    np.testing.assert_array_equal(y25, stored[:, ::40])
    effective = samples[:, :50]
    info = pp.glove_storage_inspection(effective, 25.0, spec)
    y25, knots = pp.effective_glove(effective, info, spec)
    assert info["method"] == "effective_rate_grid" and knots[:3].tolist() == [0, 40, 80]
    with pytest.raises(DataContractError):
        pp.glove_storage_inspection(effective, 100.0, spec)


def test_segment_glove_uses_only_absolute_knots_inside_the_segment(spec):
    stored, _ = staircase(8000, phase=0)
    y25, knots = pp.effective_glove(stored, pp.glove_storage_inspection(stored, 1000.0, spec), spec)
    start, stop = 3010, 6000  # not a multiple of 40: the first knot inside is 3040
    y, valid = pp.segment_glove(y25, knots, start, stop, spec)
    targets = start + np.arange(y.shape[1]) * 10
    assert y.shape == (5, 299) and y.dtype == np.float64
    assert targets[valid][0] == 3040 and targets[valid][-1] == 5960 and np.isnan(y[:, ~valid]).all()
    on_knot = valid & (targets % 40 == 0)
    np.testing.assert_allclose(y[:, on_knot], stored[:, targets[on_knot]], rtol=0, atol=1e-12)
    outside = y25.copy()
    outside[:, (knots < start) | (knots >= stop)] += 1e6
    np.testing.assert_array_equal(pp.segment_glove(outside, knots, start, stop, spec)[0], y)
    with pytest.raises(DataContractError):
        pp.segment_glove(y25, knots, 0, 100, spec)


def test_cubic_interpolation_interior_and_no_extrapolation():
    t25 = np.arange(0, 4, 0.04)
    y25 = np.vstack([np.sin(2 * np.pi * 1 * t25)] * 5)
    t100 = np.arange(-0.05, 4.0, 0.01)
    y100, valid = pp.interpolate_glove(t25, y25, t100)
    assert not valid[:5].any() and np.isnan(y100[:, 0]).all()
    interior = (t100 > 0.2) & (t100 < 3.7)
    np.testing.assert_allclose(y100[:, interior], np.sin(2 * np.pi * t100[interior])[None].repeat(5, 0), atol=1e-3)
    on_knots = np.isin(np.round(t100, 6), np.round(t25, 6))
    np.testing.assert_allclose(y100[:, on_knots & valid], np.sin(2 * np.pi * t100[on_knots & valid])[None].repeat(5, 0), atol=1e-12)


# ------------------------------------------------------------------ alignment

def test_delay_pairs_ecog_t_with_flexion_t_plus_delay():
    x = np.zeros((2, 100))
    y = np.zeros((5, 100))
    x[:, 30] = 1
    y[:, 50] = 1
    xa, ya, xi, yi = pp.align_delay(x, y, 20)
    assert xa.shape[-1] == ya.shape[-1] == 80
    assert np.flatnonzero(xa[0])[0] == np.flatnonzero(ya[0])[0] == 30
    np.testing.assert_array_equal(yi - xi, 20)
    full = pp.align_delay(x, y, 0)
    assert full[0].shape[-1] == 100
    with pytest.raises(ValueError):
        pp.align_delay(x, y, -1)
    with pytest.raises(TypeError):
        pp.align_delay(x, y, 2.0)


# ------------------------------------------------------------------ blocks and scalers

def test_five_chronological_blocks():
    blocks = chronological_blocks(400000, ["B1", "B2", "B3", "B4", "B5"], 10)
    assert [(b.start_source, b.stop_source) for b in blocks] == [(k * 80000, (k + 1) * 80000) for k in range(5)]
    assert all(b.start_source == b.start_target * 10 for b in blocks)


def test_scaler_factories_and_reshape(config):
    scaler = pp.make_feature_scaler(config)
    assert scaler.quantile_range == (10.0, 90.0) and scaler.unit_variance is True
    assert pp.make_target_scaler(config).feature_range == (0.0, 1.0)
    features = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    matrix = pp.features_to_matrix(features)
    assert matrix.shape == (4, 6) and matrix[0, 1 * 3 + 2] == features[1, 2, 0]
    np.testing.assert_array_equal(pp.matrix_to_features(matrix, 2, 3), features)


def test_restored_scalers_match_the_fitted_ones(config):
    rng = np.random.default_rng(10)
    x = rng.lognormal(size=(500, 12))
    y = rng.normal(size=(500, 5))
    fitted_x = pp.make_feature_scaler(config).fit(x)
    fitted_y = pp.make_target_scaler(config).fit(y)
    restored_x = pp.restore_feature_scaler(config, fitted_x.center_, fitted_x.scale_)
    restored_y = pp.restore_target_scaler(config, fitted_y.data_min_, fitted_y.data_max_)
    probe = rng.lognormal(size=(50, 12)).astype(np.float32)
    np.testing.assert_array_equal(restored_x.transform(probe), fitted_x.transform(probe))
    np.testing.assert_array_equal(restored_y.transform(y[:7]), fitted_y.transform(y[:7]))
    np.testing.assert_array_equal(restored_y.inverse_transform(y[:7]), fitted_y.inverse_transform(y[:7]))
    with pytest.raises(ValueError):
        pp.restore_feature_scaler(config, fitted_x.center_, fitted_x.scale_[:-1])


def test_preprocessing_does_not_depend_on_eda():
    tree = ast.parse((ROOT / "src/tt_ecog/preprocessing.py").read_text(encoding="utf-8"))
    imported = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any("eda" in name for name in imported)
