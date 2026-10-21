from copy import deepcopy

import pytest
import yaml

from tt_ecog.config import ConfigError, inside, load_config
from conftest import ROOT


def write(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_pending_scientific_values_preserved(config):
    assert config.data["training"]["search_epochs"] is None
    assert config.data["training"]["final_max_epochs"] is None
    assert config.data["paths"]["test_labels_dir"] is None
    with pytest.raises(ConfigError):
        config.path(ROOT, "test_labels_dir")


def test_channel_exclusion_regression_lock(config):
    assert {s: config.excluded_channels(s) for s in ("S1", "S2", "S3")} == {"S1": (54,), "S2": (20, 37), "S3": ()}
    for subject in ("S1", "S2"):
        for entry in config.channel_exclusions(subject):
            assert entry["reason"].strip()
            assert entry["evidence_figure"] == f"fig/s02_{subject}_c{entry['index']:03d}.png"


def test_quality_control_block_is_gone(config, tmp_path):
    assert "quality_control" not in config.data
    data = deepcopy(config.data)
    data["quality_control"] = {"thresholds": {"robust_z": None}}
    with pytest.raises(ConfigError, match="quality_control was removed"):
        load_config(write(tmp_path, data))


def test_malformed_yaml_has_explicit_diagnostic(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("project: [invalid", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


@pytest.mark.parametrize("path", ["../outside", "C:/private", "/absolute", "..\\outside"])
def test_relative_paths_cannot_escape(tmp_path, path):
    with pytest.raises(ConfigError):
        inside(tmp_path, path)


@pytest.mark.parametrize("subject,entries", [
    ("S1", [{"index": 62, "reason": "x", "evidence_figure": "fig/a.png"}]),
    ("S1", [{"index": -1, "reason": "x", "evidence_figure": "fig/a.png"}]),
    ("S1", [{"index": True, "reason": "x", "evidence_figure": "fig/a.png"}]),
    ("S2", [{"index": 20, "reason": " ", "evidence_figure": "fig/a.png"}]),
    ("S2", [{"index": 20, "reason": "x", "evidence_figure": "../a.png"}]),
    ("S2", [{"index": 37, "reason": "x", "evidence_figure": "fig/a.png"},
            {"index": 20, "reason": "x", "evidence_figure": "fig/b.png"}]),
    ("S2", [{"index": 20, "reason": "x", "evidence_figure": "fig/a.png"},
            {"index": 20, "reason": "x", "evidence_figure": "fig/b.png"}]),
    ("S3", [{"index": 1, "reason": "x"}]),
])
def test_invalid_exclusions_rejected(config, tmp_path, subject, entries):
    data = deepcopy(config.data)
    data["channels"]["excluded"][subject] = entries
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data))


def test_every_channel_cannot_be_excluded(config, tmp_path):
    data = deepcopy(config.data)
    data["channels"]["excluded"]["S2"] = [{"index": i, "reason": "x", "evidence_figure": f"fig/{i}.png"}
                                         for i in range(48)]
    with pytest.raises(ConfigError, match="every channel"):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize("section,key,value", [
    ("morlet", "n_cycles", 5), ("morlet", "zero_mean", False), ("morlet", "n_frequencies", 30),
    ("temporal_alignment", "delay_samples_at_target_fs", 25),
    ("temporal_alignment", "direction", "flexion_t_to_ecog_t_plus_delay"),
    ("glove", "storage_factor", 10), ("glove", "extrapolate", True),
    ("filtering", "phase", "minimum"), ("filtering", "notch_trans_hz", 0),
])
def test_signal_invariants(config, tmp_path, section, key, value):
    data = deepcopy(config.data)
    data["signal"][section][key] = value
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data))


def test_antialias_must_respect_target_nyquist(config, tmp_path):
    data = deepcopy(config.data)
    data["signal"]["resample"]["antialias_fir"]["stopband_hz"] = 60
    with pytest.raises(ConfigError, match="Nyquist"):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize("change", [
    lambda o: o["stage1"]["reevaluation"].update(validation_blocks=["B1", "B2", "B5"]),  # overlaps stage 1
    lambda o: o["stage1"]["reevaluation"].update(validation_blocks=["B1", "B3"]),        # leaves B5 out
    lambda o: o["stage1"]["reevaluation"].update(top_trials=21),
    lambda o: o["stage1"]["reevaluation"].update(top_trials=0),
    lambda o: o["budgets"].update(combined_search_fits=270)])
def test_stage1_reevaluation_completes_five_folds_within_budget(config, tmp_path, change):
    reevaluation = config.data["optimization"]["stage1"]["reevaluation"]
    assert reevaluation == {"top_trials": 3, "validation_blocks": ["B1", "B3", "B5"]}
    assert config.data["optimization"]["budgets"]["combined_search_fits"] == 297
    data = deepcopy(config.data)
    change(data["optimization"])
    with pytest.raises(ConfigError, match="optimization"):
        load_config(write(tmp_path, data))


def test_selected_pipeline_id(config, tmp_path):
    assert "reference_run" not in config.data["channels"]
    data = deepcopy(config.data)
    for bad in ("../outside", "a" * 16, 42, "z" * 64):
        data["paths"]["processed_pipeline_id"] = bad
        with pytest.raises(ConfigError, match="processed_pipeline_id"):
            load_config(write(tmp_path, data))
    data["paths"]["processed_pipeline_id"] = "a" * 64
    assert load_config(write(tmp_path, data)).data["paths"]["processed_pipeline_id"] == "a" * 64


def test_approved_purge_regression_lock(config):
    # Approved 2026-10-19 from real V04/V09 (docs/experimental_log.md); equal to the
    # candidate, so the published pipeline_id is preserved.
    validation = config.data["validation"]
    assert validation["purge_samples_at_100hz"] == validation["purge_candidate_for_tests"] == 256
    assert config.purge_for_build() == (256, "approved") and config.approved_purge() == 256


def test_decided_batch_regression_lock(config):
    # Decided 2026-10-21 from the T4/L4/A100 measurements (docs/experimental_log.md).
    assert config.data["training"]["batch_size"] == config.data["training"]["batch_size_initial_candidate"] == 128


def test_purge_is_pending_and_only_ever_a_labelled_candidate(config, tmp_path):
    data = deepcopy(config.data)
    data["validation"]["purge_samples_at_100hz"] = None
    pending = load_config(write(tmp_path, data))
    assert pending.purge_for_build() == (256, "candidate_pending_V04_V09")
    with pytest.raises(ConfigError, match="pending"):
        pending.approved_purge()
    data["validation"]["purge_samples_at_100hz"] = 300
    approved = load_config(write(tmp_path, data))
    assert approved.purge_for_build() == (300, "approved") and approved.approved_purge() == 300


@pytest.mark.parametrize("key,value", [
    ("purge_samples_at_100hz", -1), ("purge_samples_at_100hz", True), ("purge_samples_at_100hz", 2.5),
    ("purge_candidate_for_tests", None), ("block_names", ["B1", "B2", "B3", "B4"]),
    ("stage1_validation_blocks", ["B3"]), ("inference_stride_samples", 300),
    ("inference_window_samples", 128), ("allow_windows_across_gaps", True), ("train_stride_samples", 0)])
def test_validation_block_validated(config, tmp_path, key, value):
    data = deepcopy(config.data)
    data["validation"][key] = value
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize("section,key,value", [
    ("ecog", "quantile_range_percent", [0.1, 0.9]), ("ecog", "unit_variance", False),
    ("ecog", "fit_only_on", "development"), ("targets", "feature_range", [-1, 1])])
def test_scaling_block_validated(config, tmp_path, section, key, value):
    data = deepcopy(config.data)
    data["scaling"][section][key] = value
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data))


@pytest.mark.parametrize("key,value", [("development_labels_use", "tuning"), ("psd_overlap_fraction", 1.0),
                                       ("lag_range_seconds", [0.0, 1.0]), ("std_ratio_levels", [20, 10]),
                                       ("figure_dpi", 0)])
def test_eda_block_validated(config, tmp_path, key, value):
    data = deepcopy(config.data)
    data["eda"][key] = value
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, data))
