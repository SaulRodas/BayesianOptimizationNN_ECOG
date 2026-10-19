"""Synthetic final evaluation: these tests never approve a real pipeline or read real test labels."""
from copy import deepcopy
from dataclasses import replace

import h5py
import nbformat
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import savemat

from tt_ecog import io, preprocessing as pp
from tt_ecog.config import ConfigError
from tt_ecog.datasets import fit_fold_scalers, full_development, inference_window_starts
from tt_ecog.final_evaluation import (VARIANTS, final_policy, epoch_selection, selected_variants,
                                      predict_trajectory, score_test)
from tt_ecog.training import make_loaders
from conftest import ROOT
from test_training import tiny_fold, cpu_threads, small_model


@pytest.fixture
def final_config(config):
    config = replace(config, data=deepcopy(config.data))
    config.data['project']['subjects'] = ['S1']
    config.data['paths'].update(processed_pipeline_id='a' * 64, test_labels_dir='testlabels')
    config.data['validation'].update(purge_samples_at_100hz=256, train_stride_samples=64)
    config.data['training'].update(batch_size=2, search_epochs=2, final_max_epochs=2,
                                  final_early_stopping_patience=1, final_early_stopping_min_delta=0.0)
    config.data['raw_data']['expected_channels_nominal']['S1'] = 3
    config.data['channels']['excluded']['S1'] = []
    return config


def plans():
    return {'S1': {v: {'architecture': {'kernel_size': 3, 'encoder_levels': 2, 'base_filters': 2},
                       'learning_rate': 1e-3, 'weight_decay': 1e-4, 'source_trials': []} for v in VARIANTS}}


@pytest.mark.parametrize('key', ['batch_size', 'final_max_epochs', 'final_early_stopping_patience',
                                'final_early_stopping_min_delta', 'final_validation_block'])
def test_pending_final_choices_do_not_inherit_pilot_or_search(final_config, key):
    final_config.data['training'][key] = None
    with pytest.raises(ConfigError, match=key):
        final_policy(final_config)


def test_epoch_selection_stopping_ties_and_small_improvements():
    policy = dict(max_epochs=20, patience=2, min_delta=0.01)
    history = [{'epoch': k + 1, 'val_pearson_full': s} for k, s in enumerate([0.1, 0.2, 0.205, 0.204])]
    decision = epoch_selection(history, policy)
    assert decision['stop'] and decision['selected_epoch'] == 3 and decision['stale_epochs'] == 2
    assert not epoch_selection(history[:3], policy)['stop']
    assert epoch_selection([{'epoch': 1, 'val_pearson_full': 0.2},
                            {'epoch': 2, 'val_pearson_full': 0.2}], policy)['selected_epoch'] == 1
    with pytest.raises(ValueError, match='Nonfinite'):
        epoch_selection([{'epoch': 1, 'val_pearson_full': float('nan')}], policy)


def development_arrays(tiny_fold):
    return {f'B{k}': {'x': tiny_fold.blocks['B1' if k % 2 else 'B2'].x.copy(),
                      'y': tiny_fold.blocks['B1' if k % 2 else 'B2'].y_original.copy(),
                      'valid': np.ones(320, dtype=bool), 'x_index': np.arange(320) + (k - 1) * 320,
                      'y_index': np.arange(320) + (k - 1) * 320 + 20} for k in range(1, 6)}


def test_full_development_scalers_and_windows_stay_inside_blocks(final_config, tiny_fold):
    arrays = development_arrays(tiny_fold)
    arrays['B1']['valid'][:20] = False
    masks = {k: a['valid'] for k, a in arrays.items()}
    sx, sy, stats = fit_fold_scalers(final_config, arrays, masks)
    fold = full_development(final_config, 'S1', arrays, sx, sy)
    assert stats['fit_rows'] == 5 * 320 - 20
    assert fold.pipeline_id == 'a' * 64
    assert fold.val_windows.size == 0
    for number, start in fold.train_windows:
        a = arrays[fold.block_names[number]]
        assert a['valid'][start:start + 256].all() and start + 256 <= 320
    expected_min = np.concatenate([a['y'][:, a['valid']] for a in arrays.values()], axis=1).min(axis=1)
    np.testing.assert_allclose(sy.data_min_, expected_min)


@pytest.fixture
def frozen_run(final_config, tiny_fold, tmp_path):
    output = io.open_final_run(final_config, tmp_path, plans(), 'cpu')
    arrays = development_arrays(tiny_fold)
    sx, sy, _ = fit_fold_scalers(final_config, arrays, {k: a['valid'] for k, a in arrays.items()})
    fold = full_development(final_config, 'S1', arrays, sx, sy)
    io.save_final_scalers(output / 'S1', sx, sy, (0, 1))
    train, _ = make_loaders(fold, 2, 1)
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters())
    history = [{'epoch': 1, 'val_pearson_full': 0.3}, {'epoch': 2, 'val_pearson_full': 0.2}]
    for variant, params in plans()['S1'].items():
        folder = output / 'S1' / variant
        run = {'pipeline_id': fold.pipeline_id, 'variant_parameters': params}
        io.save_training_checkpoint(folder / 'selection.pt', model, optimizer, 2, run, train, history)
        io.atomic_json(folder / 'selection.json', epoch_selection(history, final_policy(final_config)))
        io.save_training_checkpoint(folder / 'last.pt', model, optimizer, 1, run, train, [{'epoch': 1}])
        pd.DataFrame(history).to_csv(folder / 'selection.csv', index=False)
        pd.DataFrame([{'epoch': 1}]).to_csv(folder / 'refit.csv', index=False)
    io.freeze_final_run(final_config, output)
    return output


def test_freezing_integrity_reuse_and_explicit_resume(final_config, frozen_run, tmp_path):
    assert io.verify_final_run(final_config, frozen_run)['pipeline_id'] == 'a' * 64
    final_config.data['paths']['final_run_id'] = frozen_run.name
    assert io.open_final_run(final_config, tmp_path, plans(), 'cpu') == frozen_run
    assert io.freeze_final_run(final_config, frozen_run)['run_id'] == frozen_run.name
    (frozen_run / 'S1' / VARIANTS[0] / 'last.pt').write_bytes(b'corrupt')
    with pytest.raises(io.StoreError, match='sha256'):
        io.verify_final_run(final_config, frozen_run)


def test_changed_duration_and_missing_variant_block_test(final_config, frozen_run):
    final_config.data['training']['final_max_epochs'] += 1
    with pytest.raises(io.StoreError, match='settings'):
        io.verify_final_run(final_config, frozen_run)
    final_config.data['training']['final_max_epochs'] -= 1
    (frozen_run / 'S1' / VARIANTS[2] / 'last.pt').unlink()
    with pytest.raises(io.StoreError, match='missing'):
        io.verify_final_run(final_config, frozen_run)


def test_missing_freeze_never_resolves_or_reads_test(final_config, tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'approved_pipeline', lambda *a: tmp_path)
    monkeypatch.setattr(io, '_read_numeric_variables', lambda *a: pytest.fail('test was opened'))
    final_config.data['paths']['test_labels_dir'] = None
    with pytest.raises(io.StoreError, match='Freeze'):
        io.read_final_test(final_config, tmp_path, tmp_path / 'incomplete', 'S1')


@pytest.mark.parametrize('hdf5', [False, True])
def test_final_reader_allowlist_orientation_and_unchanged_development_guard(final_config, frozen_run, tmp_path, monkeypatch, hdf5):
    raw = final_config.path(tmp_path, 'raw_dir')
    raw.mkdir(parents=True)
    labels = final_config.storage_root(tmp_path) / 'testlabels'
    labels.mkdir()
    x = np.arange(12000, dtype=float).reshape(4000, 3)
    y = np.arange(500, dtype=float).reshape(100, 5)
    ecog = raw / 'sub1_comp.mat'
    target = labels / 'sub1_testlabels.mat'
    if hdf5:
        for path, key, values in ((ecog, 'test_data', x), (target, 'test_dg', y)):
            with h5py.File(path, 'w') as handle:
                dataset = handle.create_dataset(key, data=values.T)
                dataset.attrs['MATLAB_class'] = np.bytes_('double')
    else:
        savemat(ecog, {'test_data': x, 'train_dg': np.array([[float('nan')]])})
        savemat(target, {'test_dg': y})
    monkeypatch.setattr(io, 'approved_pipeline', lambda *a: tmp_path)
    io.atomic_json(tmp_path / io.PROCESSED_MANIFEST, {'identity': {'raw_sha256': {'S1': io.sha256_file(ecog)}}})
    original = io._read_numeric_variables
    keys_read = []
    def capture(path, keys):
        keys_read.extend(keys)
        return original(path, keys)
    monkeypatch.setattr(io, '_read_numeric_variables', capture)
    actual_x, actual_y, meta = io.read_final_test(final_config, tmp_path, frozen_run, 'S1')
    np.testing.assert_array_equal(actual_x, x.T)
    np.testing.assert_array_equal(actual_y, y.T)
    assert keys_read == ['test_data', 'test_dg'] and meta['sampling_hz_glove_storage'] == 25
    assert (frozen_run / 'test_access.json').is_file()
    with pytest.raises(PermissionError):
        io._read_variables(target, ('train_dg',))
    with pytest.raises(PermissionError):
        io._read_variables(ecog, ('test_data',))
    with ecog.open('ab') as stream:
        stream.write(b'changed source')
    with pytest.raises(io.StoreError, match='Competition file differs'):
        io.read_final_test(final_config, tmp_path, frozen_run, 'S1')


def test_prediction_inverse_scaling_and_full_coverage(final_config, tiny_fold):
    model = small_model()
    block = tiny_fold.blocks['B2']
    starts = inference_window_starts(block.valid, 256, 64)
    before = tiny_fold.target_scaler.data_min_.copy()
    y, coverage, timing = predict_trajectory(model, block.x, starts, block.valid, tiny_fold.target_scaler, 2, 'cpu')
    assert y.shape == (5, 320) and coverage.min() == 1
    assert timing['inference_seconds'] > 0 and timing['peak_allocated_bytes'] is None
    np.testing.assert_array_equal(before, tiny_fold.target_scaler.data_min_)
    result = score_test(block.y_original, y, block.valid, final_config)
    np.testing.assert_allclose(result['mse'], ((block.y_original - y) ** 2).mean(axis=1))
    with pytest.raises(ValueError, match='Invalid test Pearson'):
        score_test(block.y_original, np.zeros_like(y), block.valid, final_config)


def test_variants_use_completed_subject_studies(final_config, tmp_path):
    from tt_ecog.optimization import search_session
    from test_optimization import fill_study
    for stage in ('stage1', 'stage2'):
        final_config.data['optimization'][stage]['trials_per_subject'] = 1
        with search_session(final_config, tmp_path, 'S1', stage) as session:
            fill_study(session)
    result = selected_variants(final_config, tmp_path, 'S1')
    a, b, c = (result[v] for v in VARIANTS)
    assert a['learning_rate'] == 8.42e-5 and a['weight_decay'] == 1e-4
    assert b['learning_rate'] == c['learning_rate'] and b['weight_decay'] == c['weight_decay']
    assert a['architecture'] == b['architecture']
    assert c['source_trials'][1]['parameters'] == c['architecture']


def test_notebook07_synthetic_training_resume_freeze_and_test(final_config, tiny_fold, tmp_path, monkeypatch):
    """Actual notebook operations/U-Net/preprocessing; synthetic data admission, never real PASS."""
    pipeline = tmp_path / ('a' * 64)
    arrays = development_arrays(tiny_fold)
    for k, (name, a) in enumerate(arrays.items()):
        folder = pipeline / 'S1' / name
        io.save_arrays(folder, **a)
        io.atomic_json(folder / 'block_manifest.json', {'range': {'start_target': k * 320, 'stop_target': (k + 1) * 320,
            'start_source': k * 3200, 'stop_source': (k + 1) * 3200}})
    # Prepared fixture has two effective input channels, matching three raw minus a fixed excluded one.
    final_config.data['channels']['excluded']['S1'] = [{'index': 2, 'reason': 'synthetic', 'evidence_figure': 'unused.png'}]
    source = final_config.path(tmp_path, 'raw_dir')
    source.mkdir(parents=True)
    labels = final_config.storage_root(tmp_path) / 'testlabels'
    labels.mkdir()
    rng = np.random.default_rng(8)
    raw_test, glove_test = rng.normal(size=(16000, 3)), rng.normal(size=(400, 5))
    savemat(source / 'sub1_comp.mat', {'test_data': raw_test})
    savemat(labels / 'sub1_testlabels.mat', {'test_dg': glove_test})
    io.atomic_json(pipeline / io.PROCESSED_MANIFEST, {'identity': {'raw_sha256': {'S1': io.sha256_file(source / 'sub1_comp.mat')}}})
    monkeypatch.setattr(io, 'approved_pipeline', lambda *a: pipeline)
    notebook = nbformat.read(ROOT / 'notebooks/07_evaluacion_final.ipynb', 4)
    nbformat.validate(notebook)
    def run(tag, ns):
        for c in notebook.cells:
            if tag in c.metadata.get('tags', []):
                exec(c.source, ns)
    ns = {'CONFIG': final_config.source, 'DRIVE_ROOT': tmp_path, 'display': lambda *a: None}
    run('setup', ns)
    ns.update(config=final_config, pipeline_dir=pipeline, device=torch.device('cpu'),
              policy=final_policy(final_config), selected_variants=lambda *a: plans()['S1'],
              load_fold=lambda *a, **kw: replace(tiny_fold, val_block='B2'))
    run('plan', ns)
    real_train = ns['train_epoch']
    calls = 0
    def interrupted(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('synthetic disconnect')
        return real_train(*a, **kw)
    ns['train_epoch'] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run('selection', ns)
    checkpoint = ns['output'] / 'S1' / VARIANTS[0] / 'selection.pt'
    assert torch.load(checkpoint, weights_only=True)['epoch'] == 1
    ns['train_epoch'] = real_train
    run('selection', ns)
    calls = 0
    ns['train_epoch'] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run('refit', ns)
    assert not (ns['output'] / 'frozen.json').exists()
    ns['train_epoch'] = real_train
    run('refit', ns)
    run('freeze', ns)
    assert ns['frozen']['pipeline_id'] == pipeline.name
    real_predict = ns['predict_trajectory']
    inputs = []
    def capture_inputs(model, x, *args, **kwargs):
        inputs.append(x.copy())
        return real_predict(model, x, *args, **kwargs)
    ns['predict_trajectory'] = capture_inputs
    run('test', ns)
    # Visible per-channel operations equal the common preprocessing primitive.
    spec = pp.PreprocessingSpec.from_config(final_config, 'S1')
    features, fv, _ = pp.segment_features(raw_test.T, spec)
    expected_y, yv = pp.segment_glove(glove_test.T, np.arange(400) * 40, 0, 16000, spec)
    features, expected_y, xi, yi = pp.align_delay(features, expected_y, 20)
    sx, _ = io.load_final_scalers(final_config, ns['output'] / 'S1')
    expected_x = pp.matrix_to_features(sx.transform(pp.features_to_matrix(features)), 2, 40).astype(np.float32)
    for actual_x in inputs:
        np.testing.assert_array_equal(actual_x, expected_x)
    # x is released by the notebook; the shared mask/times/targets remain on disk.
    saved = ns['output'] / 'S1' / 'test'
    np.testing.assert_array_equal(np.load(saved / 'valid.npy'), fv[xi] & yv[yi])
    np.testing.assert_allclose(np.load(saved / 'truth.npy'), expected_y, equal_nan=True)
    table, summary = ns['report_final_evaluation'](final_config, ns['output'])
    assert len(table) == 15 and len(summary) == 3
    assert table.groupby('variant')['n_samples'].first().nunique() == 1
    manifest = io.verify_manifest(ns['output'], 'artifact_manifest.json')
    assert manifest['pipeline_id'] == pipeline.name
    for variant in VARIANTS:
        folder = ns['output'] / 'S1' / variant
        selected = io.read_json(folder / 'selection.json')['selected_epoch']
        state = torch.load(folder / 'last.pt', weights_only=True)
        assert state['epoch'] == selected
    def forbidden(*a, **kw):
        pytest.fail('Frozen/completed run must neither retrain nor reopen test')
    ns.update(train_epoch=forbidden, read_final_test=forbidden)
    run('selection', ns)
    run('refit', ns)
    run('test', ns)


def test_interrupted_publication_never_exposes_partial_results(tmp_path, monkeypatch):
    destination = tmp_path / 'test'
    arrays = {'truth': np.ones((5, 300)), 'valid': np.ones(300, dtype=bool)}
    report = {'pipeline_id': 'a' * 64, 'run_id': 'synthetic'}
    original = io.write_manifest
    def interrupted(*args, **kwargs):
        raise OSError('interrupted before commit marker')
    monkeypatch.setattr(io, 'write_manifest', interrupted)
    with pytest.raises(OSError):
        io.save_test_result(destination, arrays, report)
    assert not destination.exists()
    monkeypatch.setattr(io, 'write_manifest', original)
    io.save_test_result(destination, arrays, report)
    assert io.verify_manifest(destination, 'artifact_manifest.json')['pipeline_id'] == 'a' * 64
    np.testing.assert_array_equal(np.load(destination / 'truth.npy'), arrays['truth'])
