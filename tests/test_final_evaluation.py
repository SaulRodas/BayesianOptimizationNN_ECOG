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
from tt_ecog.final_evaluation import (VARIANTS, ENSEMBLE, final_policy, epoch_selection, selected_variants,
                                      final_runs, ensemble_members, smooth_trajectory, smoothing_selection,
                                      internal_prediction, predict_trajectory, score_test)
from tt_ecog.losses import make_loss
from tt_ecog.metrics import trajectory_metrics
from tt_ecog.training import make_loaders, evaluate, fold_seed
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


def test_d_members_train_c_with_predefined_seeds(final_config):
    runs = final_runs(final_config, plans()['S1'], 'S1')
    members = ensemble_members(final_config)
    seed = fold_seed(final_config, 'S1', 'B5')
    offsets = final_config.data['final_evaluation']['ensemble']['seed_offsets']
    assert members[0] == VARIANTS[2] and list(runs) == [*VARIANTS, *members[1:]]
    assert all(runs[variant]['seed'] == seed for variant in VARIANTS)
    assert [runs[name]['seed'] for name in members] == [seed + offset for offset in offsets]
    assert all(runs[name]['parameters'] == plans()['S1'][VARIANTS[2]] for name in members)


def test_smoothing_is_normalized_and_never_crosses_invalid_columns():
    from scipy.ndimage import gaussian_filter1d
    rng = np.random.default_rng(3)
    valid = np.ones(400, dtype=bool)
    valid[:2] = valid[200:210] = False
    prediction = np.where(valid, rng.normal(size=(5, 400)), np.nan)
    np.testing.assert_array_equal(smooth_trajectory(prediction, valid, 0, 4.0), prediction)
    smoothed = smooth_trajectory(prediction, valid, 5, 4.0)
    assert np.isnan(smoothed[:, ~valid]).all() and np.isfinite(smoothed[:, valid]).all()
    # Far from the run edges it is the ordinary Gaussian filter.
    np.testing.assert_allclose(smoothed[:, 50:150], gaussian_filter1d(prediction[:, 2:200], 5, axis=-1)[:, 48:148])
    # The kernel is renormalized at the edges, so a constant survives up to them.
    np.testing.assert_allclose(smooth_trajectory(np.where(valid, 3.0, np.nan) + np.zeros((5, 1)), valid, 40, 4.0)
                               [:, valid], 3.0)
    changed = prediction.copy()
    changed[:, 210:] += 100.0
    np.testing.assert_array_equal(smooth_trajectory(changed, valid, 40, 4.0)[:, :200],
                                  smooth_trajectory(prediction, valid, 40, 4.0)[:, :200])
    prediction[0, 100] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        smooth_trajectory(prediction, valid, 5, 4.0)


def test_smoothing_selection_maximizes_internal_pearson_smallest_sigma_on_ties(final_config):
    truth, valid = internal_truth()
    rng = np.random.default_rng(11)
    members = np.stack([np.where(valid, truth + rng.normal(size=truth.shape), np.nan)
                        for _ in final_config.data['final_evaluation']['ensemble']['seed_offsets']])
    decision = smoothing_selection(final_config, truth, valid, members)
    table = decision['sigma_table']
    assert [row['sigma_samples'] for row in table] == [0, 5, 10, 20, 40]
    assert decision['selected_sigma_samples'] == max(table, key=lambda r: r['pearson_macro'])['sigma_samples'] > 0
    mean = trajectory_metrics(truth, members.mean(axis=0), valid, degenerate_std_relative=1e-9)['pearson_macro']
    assert table[0]['pearson_macro'] == mean > max(decision['member_internal_pearson'])
    assert decision['members'] == ensemble_members(final_config)
    # A sigma far below one sample leaves the trajectory unchanged: the tie keeps the smaller sigma.
    final_config.data['final_evaluation']['ensemble']['smoothing']['sigma_samples'] = [0, 1e-12]
    tie = smoothing_selection(final_config, truth, valid, members)
    assert tie['sigma_table'][0]['pearson_macro'] == tie['sigma_table'][1]['pearson_macro']
    assert tie['selected_sigma_samples'] == 0
    with pytest.raises(ValueError, match='one'):
        smoothing_selection(final_config, truth, valid, members[:2])
    members[1, 0, 20] = np.nan
    with pytest.raises(ValueError, match='cover'):
        smoothing_selection(final_config, truth, valid, members)


def test_internal_prediction_reproduces_the_scored_epoch(final_config, tiny_fold):
    model = small_model()
    _, val = make_loaders(tiny_fold, 2, 1)
    metrics = evaluate(model, make_loss(final_config), tiny_fold, val, 'cpu', 1e-9)
    prediction = internal_prediction(final_config, model, tiny_fold, 2, 'cpu', metrics['val_pearson_full'])
    block = tiny_fold.blocks['B2']
    assert prediction.shape == block.y_original.shape and np.isfinite(prediction[:, block.valid]).all()
    with pytest.raises(ValueError, match='reproduce'):
        internal_prediction(final_config, model, tiny_fold, 2, 'cpu', metrics['val_pearson_full'] + 0.01)


def test_only_the_selected_epoch_trajectory_is_kept(tmp_path):
    for epoch in (1, 2, 3):
        io.save_selection_prediction(tmp_path, epoch, np.full((5, 4), float(epoch)))
    io.save_selection_prediction(tmp_path, 2, np.full((5, 4), 20.0))  # an epoch retrained after a disconnect
    io.keep_selected_prediction(tmp_path, 2)
    epoch, prediction = io.load_selected_prediction(tmp_path)
    assert epoch == 2 and (prediction == 20.0).all()
    assert not (tmp_path / 'selection_predictions').exists()
    io.keep_selected_prediction(tmp_path, 2)  # repeating the cell after a reconnection
    with pytest.raises(io.StoreError, match='another epoch'):
        io.keep_selected_prediction(tmp_path, 3)


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


def internal_truth():
    """Smooth flexion of a synthetic internal block, with an invalid gap between two runs."""
    t = np.arange(320) / 100
    truth = np.stack([np.sin(2 * np.pi * (0.3 + 0.1 * k) * t) + 2 for k in range(5)])
    valid = np.ones(320, dtype=bool)
    valid[150:160] = False
    return truth, valid


def write_selection(folder, config, truth, valid, rng):
    """Two selection epochs whose stored internal trajectories give the recorded scores."""
    history = []
    for epoch in (1, 2):
        prediction = np.where(valid, truth + rng.normal(scale=0.3 * epoch, size=truth.shape), np.nan)
        io.save_selection_prediction(folder, epoch, prediction)
        score = trajectory_metrics(truth, prediction, valid,
                                   degenerate_std_relative=config.data['metrics']['degenerate_std_relative'])
        history.append({'epoch': epoch, 'val_pearson_full': score['pearson_macro']})
    decision = epoch_selection(history, final_policy(config))
    io.atomic_json(folder / 'selection.json', decision)
    io.keep_selected_prediction(folder, decision['selected_epoch'])
    return history, decision


@pytest.fixture
def final_run(final_config, tiny_fold, tmp_path):
    """Every model of A/B/C and D completed, with D's sigma chosen, but not yet frozen."""
    output = io.open_final_run(final_config, tmp_path, plans(), 'cpu')
    arrays = development_arrays(tiny_fold)
    sx, sy, _ = fit_fold_scalers(final_config, arrays, {k: a['valid'] for k, a in arrays.items()})
    fold = full_development(final_config, 'S1', arrays, sx, sy)
    io.save_final_scalers(output / 'S1', sx, sy, (0, 1))
    train, _ = make_loaders(fold, 2, 1)
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters())
    truth, valid = internal_truth()
    io.save_internal_validation(output / 'S1', 'B5', truth, valid)
    rng = np.random.default_rng(5)
    for name, run_plan in final_runs(final_config, plans()['S1'], 'S1').items():
        folder = output / 'S1' / name
        history, decision = write_selection(folder, final_config, truth, valid, rng)
        run = {'pipeline_id': fold.pipeline_id, 'variant_parameters': run_plan['parameters'], 'seed': run_plan['seed']}
        io.save_training_checkpoint(folder / 'selection.pt', model, optimizer, 2, run, train, history)
        io.save_training_checkpoint(folder / 'last.pt', model, optimizer, decision['selected_epoch'], run, train,
                                    [{'epoch': 1}])
        pd.DataFrame(history).to_csv(folder / 'selection.csv', index=False)
        pd.DataFrame([{'epoch': 1}]).to_csv(folder / 'refit.csv', index=False)
    members = [io.load_selected_prediction(output / 'S1' / name)[1] for name in ensemble_members(final_config)]
    io.atomic_json(output / 'S1' / ENSEMBLE / 'selection.json', smoothing_selection(final_config, truth, valid, members))
    return output


@pytest.fixture
def frozen_run(final_config, final_run):
    io.freeze_final_run(final_config, final_run)
    return final_run


def test_freezing_integrity_reuse_and_explicit_resume(final_config, frozen_run, tmp_path):
    frozen = io.verify_final_run(final_config, frozen_run)
    assert frozen['pipeline_id'] == 'a' * 64
    assert 'src/tt_ecog/final_evaluation.py' in frozen['source_sha256']
    assert 'src/tt_ecog/reporting.py' not in frozen['source_sha256']
    assert not any('/eda/' in p for p in frozen['source_sha256'])
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


def test_freezing_rederives_d_from_stored_development_evidence(final_config, final_run):
    path = final_run / 'S1' / ENSEMBLE / 'selection.json'
    decision = io.read_json(path)
    other = next(row['sigma_samples'] for row in decision['sigma_table']
                 if row['sigma_samples'] != decision['selected_sigma_samples'])
    io.atomic_json(path, {**decision, 'selected_sigma_samples': other})
    with pytest.raises(io.StoreError, match='smoothing'):
        io.freeze_final_run(final_config, final_run)
    io.atomic_json(path, decision)
    member = final_run / 'S1' / ensemble_members(final_config)[1]
    kept = (member / 'selection_prediction.npz').read_bytes()
    epoch, prediction = io.load_selected_prediction(member)
    with (member / 'selection_prediction.npz').open('wb') as stream:
        np.savez(stream, epoch=epoch, prediction=prediction[::-1])
    with pytest.raises(io.StoreError, match='reproduce'):
        io.freeze_final_run(final_config, final_run)
    (member / 'selection_prediction.npz').write_bytes(kept)
    state = torch.load(member / 'last.pt', weights_only=True)
    state['parameters']['seed'] += 1
    torch.save(state, member / 'last.pt')
    with pytest.raises(io.StoreError, match='Refit checkpoint'):
        io.freeze_final_run(final_config, final_run)
    state['parameters']['seed'] -= 1
    torch.save(state, member / 'last.pt')
    frozen = io.freeze_final_run(final_config, final_run)
    assert frozen['ensemble']['S1'] == decision
    assert set(frozen['decisions']['S1']) == set(final_runs(final_config, plans()['S1'], 'S1'))
    assert f'S1/{ENSEMBLE}/selection.json' in frozen['files'] and 'S1/internal_validation.npz' in frozen['files']


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
    from test_optimization import fill_study, fill_reevaluation
    final_config.data['optimization']['stage1']['reevaluation']['top_trials'] = 1
    for stage in ('stage1', 'stage2'):
        final_config.data['optimization'][stage]['trials_per_subject'] = 1
        with search_session(final_config, tmp_path, 'S1', stage) as session:
            fill_study(session)
            if stage == 'stage1':
                fill_reevaluation(session)
    result = selected_variants(final_config, tmp_path, 'S1')
    a, b, c = (result[v] for v in VARIANTS)
    assert a['learning_rate'] == 8.42e-5 and a['weight_decay'] == 1e-4
    assert b['learning_rate'] == c['learning_rate'] and b['weight_decay'] == c['weight_decay']
    assert a['architecture'] == b['architecture']
    assert c['source_trials'][1]['parameters'] == c['architecture']


def internal_fold(tiny_fold):
    """tiny_fold with its validation block named like the configured internal block (B5)."""
    blocks = {'B1': tiny_fold.blocks['B1'], 'B5': replace(tiny_fold.blocks['B2'], name='B5')}
    return replace(tiny_fold, name='val_B5', val_block='B5', block_names=('B1', 'B5'), blocks=blocks)


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
              load_fold=lambda *a, **kw: internal_fold(tiny_fold))
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
    # Disconnect after an epoch's internal trajectory is written but before its checkpoint:
    # the epoch is retrained and its trajectory replaced (freezing checks they reproduce).
    real_save = ns['save_training_checkpoint']
    saves = 0
    def lost_checkpoint(*a, **kw):
        nonlocal saves
        saves += 1
        if saves == 2:
            raise KeyboardInterrupt('synthetic disconnect before the checkpoint')
        return real_save(*a, **kw)
    ns['save_training_checkpoint'] = lost_checkpoint
    with pytest.raises(KeyboardInterrupt):
        run('selection', ns)
    assert (ns['output'] / 'S1' / VARIANTS[1] / 'selection_predictions' / 'epoch_001.npy').is_file()
    assert not (ns['output'] / 'S1' / VARIANTS[1] / 'selection.pt').exists()
    ns['save_training_checkpoint'] = real_save
    run('selection', ns)
    smoothing = io.read_json(ns['output'] / 'S1' / ENSEMBLE / 'selection.json')
    assert smoothing['members'] == ensemble_members(final_config)
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
    valid = np.load(saved / 'valid.npy')
    np.testing.assert_array_equal(valid, fv[xi] & yv[yi])
    np.testing.assert_allclose(np.load(saved / 'truth.npy'), expected_y, equal_nan=True)
    # D: the frozen sigma applied to the mean of its members (C is member 0), all seeds reported.
    members = ensemble_members(final_config)
    assert len(inputs) == len(VARIANTS) + len(members) - 1
    mean = np.mean([np.load(saved / f"prediction_{name.replace('/', '_')}.npy") for name in members], axis=0)
    np.testing.assert_array_equal(np.load(saved / 'prediction_ensemble_mean_D.npy'), mean)
    np.testing.assert_array_equal(np.load(saved / f'prediction_{ENSEMBLE}.npy'),
                                  smooth_trajectory(mean, valid, smoothing['selected_sigma_samples'], 4.0))
    report = io.read_json(saved / 'metrics.json')
    assert list(report['variants']) == [*VARIANTS, ENSEMBLE] and list(report['ensemble']['members']) == members
    runs = final_runs(final_config, plans()['S1'], 'S1')
    assert [report['ensemble']['members'][name]['seed'] for name in members] == [runs[n]['seed'] for n in members]
    assert report['ensemble']['sigma_samples'] == smoothing['selected_sigma_samples']
    table, summary, ensemble_detail = ns['report_final_evaluation'](final_config, ns['output'])
    assert len(table) == 20 and len(summary) == 4
    assert list(ensemble_detail.index.get_level_values('model')) == [*members, 'mean_before_smoothing']
    assert table.groupby('variant')['n_samples'].first().nunique() == 1
    manifest = io.verify_manifest(ns['output'], 'artifact_manifest.json')
    assert manifest['pipeline_id'] == pipeline.name
    for name, run_plan in runs.items():
        folder = ns['output'] / 'S1' / name
        selected = io.read_json(folder / 'selection.json')['selected_epoch']
        state = torch.load(folder / 'last.pt', weights_only=True)
        assert state['epoch'] == selected and state['parameters']['seed'] == run_plan['seed']
        assert io.load_selected_prediction(folder)[0] == selected
        assert not (folder / 'selection_predictions').exists()
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
