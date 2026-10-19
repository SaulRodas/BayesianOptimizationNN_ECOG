import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
import yaml

from tt_ecog.config import load_config


ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- V01-V14 evidence

def pytest_addoption(parser):
    parser.addoption("--verification-json", default=None,
                     help="Write the outcome of every test and its verification ids (V01-V14) to this JSON file")


class VerificationRecorder:
    """Outcome per test (passed/failed/skipped over all phases) and its ``verification`` marker ids."""

    def __init__(self, path: str):
        self.path, self.ids, self.outcomes = Path(path), {}, {}

    def pytest_collection_modifyitems(self, items):
        for item in items:
            self.ids[item.nodeid] = sorted({i for m in item.iter_markers("verification") for i in m.args})

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            if self.outcomes.get(report.nodeid, "passed") == "passed":
                self.outcomes[report.nodeid] = report.outcome

    def pytest_sessionfinish(self, session, exitstatus):
        tests = {node: {"ids": ids, "outcome": self.outcomes.get(node, "not_run")} for node, ids in self.ids.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"exitstatus": int(exitstatus), "tests": tests}, indent=1), encoding="utf-8")


def pytest_configure(config):
    path = config.getoption("--verification-json")
    if path:
        config.pluginmanager.register(VerificationRecorder(path), "tt-ecog-verification")


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


# --------------------------------------------------------------------------- step 2 on synthetic data

EDA_RUN_ID = "20260101T000000Z_abcdef"
SUBJECTS = ("S1", "S2", "S3")


def project(base, reference_run=EDA_RUN_ID, **validation):
    """Project shell whose YAML points to a synthetic EDA run (and optional validation overrides)."""
    (base / "p/configs").mkdir(parents=True)
    data = yaml.safe_load((ROOT / "configs/experiment_config.yaml").read_text(encoding="utf-8"))
    data["channels"]["reference_run"] = reference_run
    data["validation"].update(validation)
    path = base / "p/configs/experiment_config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def write_eda_evidence(config, drive, raw_paths, run_id=EDA_RUN_ID):
    """Minimal COMPLETE EDA run folder describing ``raw_paths``."""
    from tt_ecog.io import sha256_file
    from tt_ecog.reporting import write_json
    run = config.path(drive, "eda_dir") / run_id
    (run / "fig").mkdir(parents=True)
    for subject in SUBJECTS:
        for entry in config.channel_exclusions(subject):
            (run / entry["evidence_figure"]).write_bytes(b"evidence " + entry["evidence_figure"].encode())
    write_json(run / "run_summary.json", {"status": "COMPLETE", "test_accessed": False, "git_commit": "eda"})
    write_json(run / "data_manifest.json", {s: {"ecog": {"sha256": sha256_file(p)}} for s, p in raw_paths.items()})
    write_json(run / "effective_config.json", config.data)
    write_json(run / "artifact_manifest.json", {"sha256": {
        p.relative_to(run).as_posix(): sha256_file(p) for p in sorted(run.rglob("*")) if p.is_file()}})
    return run


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """Step 2 over three synthetic 60 s subjects (blocks of 12 s) with a synthetic EDA reference."""
    from tt_ecog.io import sha256_file
    from tt_ecog.prep import run_prep
    base = tmp_path_factory.mktemp("prep")
    config_path = project(base)
    config = load_config(config_path)
    drive = base / "d"
    config.path(drive, "raw_dir").mkdir(parents=True)
    raw = {s: write_subject(drive, config, s, n=60000) for s in SUBJECTS}
    originals = {s: sha256_file(p) for s, p in raw.items()}
    write_eda_evidence(config, drive, raw)
    output = run_prep(config_path, drive)
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    storage = config.storage_root(drive)
    return {"config": config, "config_path": config_path, "drive": drive, "raw": raw, "originals": originals,
            "output": output, "summary": summary, "base": base,
            "pipeline": storage / summary["pipeline_dir"] if summary["pipeline_dir"] else None,
            "folds": storage / summary["fold_set_dir"] if summary["fold_set_dir"] else None}
