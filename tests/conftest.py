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
def data_root(tmp_path, config):
    raw = config.path(tmp_path, "raw_dir")
    raw.mkdir(parents=True)
    return tmp_path


def synthetic_glove(n: int, rng: np.random.Generator, factor: int = 40, phase: int = 13) -> np.ndarray:
    """Smooth 25 Hz flexion stored on the 1 kHz grid as a staircase (time, 5)."""
    n25 = n // factor + 2
    t = np.arange(n25) / 25
    bursts = np.zeros((5, n25))
    for finger in range(5):
        for centre in rng.uniform(0, t[-1], size=max(2, n25 // 60)):
            bursts[finger] += np.exp(-0.5 * ((t - centre) / 0.25) ** 2)
    stored = np.repeat(bursts, factor, axis=1)[:, factor - phase:factor - phase + n]
    return stored.T


def synthetic_ecog(n: int, channels: int, glove: np.ndarray, rng: np.random.Generator,
                   artifact_channels: tuple[int, ...] = ()) -> np.ndarray:
    """Noise + 60 Hz line + high-gamma bursts that lead the glove by 200 ms (time, channels)."""
    fs = 1000
    t = np.arange(n) / fs
    values = rng.normal(scale=20.0, size=(n, channels))
    values += 15 * np.sin(2 * np.pi * 60 * t)[:, None]
    drive = np.roll(glove.mean(axis=1), -200)
    carrier = rng.normal(size=(n, 1))
    values[:, : channels // 2] += 40 * drive[:, None] * carrier
    for k, channel in enumerate(artifact_channels):
        start = int((0.3 + 0.2 * k) * n)
        values[start:start + 1500, channel] += 2e4 * rng.normal(size=1500)
    return values


def write_subject(root, config, subject="S1", *, n=2000, labels=None, values=None, artifacts=False):
    channels = config.data["raw_data"]["expected_channels_nominal"][subject]
    rng = np.random.default_rng(123)
    if labels is None:
        labels = synthetic_glove(n, rng)
    if values is None:
        excluded = config.excluded_channels(subject) if artifacts else ()
        # Deliberately invalid label shapes must not break the ECoG generator.
        drive = labels if labels.ndim == 2 and labels.shape[0] == n else synthetic_glove(n, rng)
        values = synthetic_ecog(n, channels, drive, rng, excluded)
    path = config.path(root, "raw_dir") / f"sub{subject[1:]}_comp.mat"
    savemat(path, {"train_data": values, "train_dg": labels,
                   "test_data": np.full((17, channels), np.nan),
                   "test_dg": np.full((17, 5), np.nan)})
    return path


@pytest.fixture
def config_copy(tmp_path):
    """A standalone project shell for integration tests; no real dataset."""
    import shutil
    project = tmp_path / "p"
    (project / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/experiment_config.yaml", project / "configs/experiment_config.yaml")
    return project / "configs/experiment_config.yaml"
