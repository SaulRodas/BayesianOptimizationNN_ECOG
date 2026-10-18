from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from tt_ecog.config import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/experiment_config.yaml")


@pytest.fixture
def candidate_qc(config):
    result = deepcopy(config.data["quality_control"])
    # Synthetic test settings only: never written into experiment_config.yaml.
    result["thresholds"].update(robust_z=8.0, jump_z=8.0, flatline_fraction=.01,
                                max_bad_fraction=.2, spectral_z=3.0, spectral_min_duration_s=.5)
    result["analysis"].update(window_seconds=.25, psd_window_seconds=.25,
                               event_merge_gap_seconds=.005)
    return result


@pytest.fixture
def data_root(tmp_path, config):
    raw = config.path(tmp_path, "raw_dir")
    raw.mkdir(parents=True)
    return tmp_path


def write_subject(root, config, subject="S1", *, n=2000, labels=None, values=None):
    channels = config.data["raw_data"]["expected_channels_nominal"][subject]
    rng = np.random.default_rng(123)
    if values is None:
        values = rng.normal(size=(n, channels))
    if labels is None:
        labels = rng.normal(size=(n, 5))
    path = config.path(root, "raw_dir") / f"sub{subject[1:]}_comp.mat"
    savemat(path, {"train_data": values, "train_dg": labels,
                   "test_data": np.full((17, channels), np.nan),
                   "test_dg": np.full((17, 5), np.nan)})
    return path


@pytest.fixture
def config_copy(tmp_path, config):
    """A standalone project shell for integration tests; no real dataset."""
    import shutil
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "schemas").mkdir()
    for name in ("experiment_config.yaml", "qc_decisions.template.json"):
        shutil.copyfile(ROOT / "configs" / name, project / "configs" / name)
    shutil.copyfile(ROOT / "schemas/qc_decisions.schema.json", project / "schemas/qc_decisions.schema.json")
    return project / "configs/experiment_config.yaml"
