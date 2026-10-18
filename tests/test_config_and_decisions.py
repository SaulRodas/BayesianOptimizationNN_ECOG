from copy import deepcopy
import json

import pytest
import yaml

from tt_ecog.config import ConfigError, load_config, inside
from tt_ecog.decisions import validate_decisions
from tt_ecog.io import sha256_file
from conftest import ROOT


def test_pending_scientific_values_preserved(config):
    assert config.data["training"]["search_epochs"] is None
    assert config.data["training"]["batch_size"] is None
    assert config.data["validation"]["purge_samples_at_100hz"] is None
    assert config.data["paths"]["test_labels_dir"] is None
    with pytest.raises(ConfigError):
        config.path(ROOT, "test_labels_dir")


def test_malformed_yaml_has_explicit_diagnostic(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("project: [invalid", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML"):
        load_config(path)


@pytest.mark.parametrize("path", ["../outside", "C:/private", "/absolute", "..\\outside"])
def test_relative_paths_cannot_escape(tmp_path, path):
    with pytest.raises(ConfigError):
        inside(tmp_path, path)


@pytest.mark.parametrize("key,value", [("robust_z", -1), ("jump_z", True),
                                      ("flatline_fraction", 1.1), ("spectral_z", float("nan"))])
def test_invalid_candidate_settings(config, tmp_path, key, value):
    data = deepcopy(config.data)
    data["quality_control"]["thresholds"][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_pending_template_valid_but_cannot_self_approve(tmp_path):
    document = json.loads((ROOT / "configs/qc_decisions.template.json").read_text(encoding="utf-8"))
    schema = ROOT / "schemas/qc_decisions.schema.json"
    validate_decisions(document, schema, tmp_path, {})
    document["status"] = "approved"
    with pytest.raises(ValueError, match="reviewer"):
        validate_decisions(document, schema, tmp_path, {})


@pytest.fixture
def decision_example(tmp_path):
    document = json.loads((ROOT / "configs/qc_decisions.template.json").read_text(encoding="utf-8"))
    source = {"n_ecog_channels_original": 62, "sha256": "a" * 64, "duration_s": 2.0}
    (tmp_path / "evidence.csv").write_text("observed", encoding="utf-8")
    document["subjects"]["S1"].update(original_channel_count=62, raw_data_sha256="a" * 64,
                                        evidence_paths=["evidence.csv"])
    return document, {"S1": source}


@pytest.mark.parametrize("change", ["bounds", "hash", "interval", "evidence", "reason"])
def test_semantic_decisions_reject_bad_evidence(decision_example, tmp_path, change):
    document, source = decision_example
    subject = document["subjects"]["S1"]
    if change == "bounds":
        subject["excluded_channel_indices_0based"] = [62]
    elif change == "hash":
        subject["raw_data_sha256"] = "b" * 64
    elif change == "interval":
        subject["intervals"] = [{"start_s": 1.0, "end_s": 3.0, "action": "mask", "reason": "synthetic"}]
    elif change == "evidence":
        subject["evidence_paths"] = ["missing.csv"]
    elif change == "reason":
        subject["excluded_channel_indices_0based"] = [1]
    with pytest.raises(ValueError):
        validate_decisions(document, ROOT / "schemas/qc_decisions.schema.json", tmp_path, source)


def test_approval_requires_intact_evidence(decision_example, tmp_path):
    document, source = decision_example
    snapshot = tmp_path / "experiment_config.yaml"
    snapshot.write_text("reviewed: true\n", encoding="utf-8")
    document.update(reviewed_by="Synthetic reviewer", approval_at_utc="2026-10-18T12:00:00Z",
                    qc_config_sha256=sha256_file(snapshot))
    document["subjects"]["S1"]["status"] = "approved"
    manifest = {"sha256": {"evidence.csv": sha256_file(tmp_path / "evidence.csv")}}
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    schema = ROOT / "schemas/qc_decisions.schema.json"
    validate_decisions(document, schema, tmp_path, source)
    (tmp_path / "evidence.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        validate_decisions(document, schema, tmp_path, source)


def test_configuration_hash_must_match_run(decision_example, tmp_path):
    document, source = decision_example
    document["qc_config_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="Configuration hash"):
        validate_decisions(document, ROOT / "schemas/qc_decisions.schema.json", tmp_path, source, "b" * 64)
