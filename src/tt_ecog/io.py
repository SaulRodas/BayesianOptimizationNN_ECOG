"""Stage-specific MATLAB access and persistence; test requires frozen final models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import shutil

import h5py
import numpy as np
from scipy.io import loadmat

from .config import ExperimentConfig, inside


class DataContractError(ValueError):
    """The supplied data do not match the documented development contract."""


class MissingDevelopmentData(FileNotFoundError):
    """A development file is absent: the run is BLOCKED, not FAILED."""


class StorageUnavailable(FileNotFoundError):
    """The persistent storage root (mounted Drive) is absent: BLOCKED."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DevelopmentRecord:
    subject: str
    values: np.ndarray  # (channels, time), readonly; original numerical units
    metadata: dict[str, Any]


def _read_variables(path: Path, keys: tuple[str, ...]) -> tuple[dict[str, np.ndarray], str]:
    """Read only allowlisted development variables, in a single file opening."""
    if not keys or set(keys) - {"train_data", "train_dg"}:
        raise PermissionError(f"Development forbids MATLAB variables {keys!r}")
    if "testlabels" in path.name.lower() or "testlabels" in path.resolve().name.lower():
        raise PermissionError("Development forbids test-label files, including symlink targets")
    return _read_numeric_variables(path, keys)


def _read_numeric_variables(path: Path, keys: tuple[str, ...]):
    """MATLAB numeric adapter; callers enforce their stage-specific allowlists."""
    if h5py.is_hdf5(path):
        content = {}
        with h5py.File(path, "r") as handle:
            for key in keys:
                if key not in handle:
                    raise DataContractError(f"Missing development variable: {key}")
                if not isinstance(handle.get(key, getlink=True), h5py.HardLink):
                    raise DataContractError("Development datasets must not follow HDF5 soft/external links")
                dataset = handle[key]
                if not isinstance(dataset, h5py.Dataset) or "MATLAB_class" not in dataset.attrs:
                    raise DataContractError("HDF5 layout requires a MATLAB numeric dataset with MATLAB_class")
                if dataset.ndim != 2 or dataset.dtype.kind not in "fiu":
                    raise DataContractError(f"{key}: expected a two-dimensional real numeric matrix")
                content[key] = dataset[()].T
        return content, "matlab_v7.3_hdf5_reversed_axes"
    try:
        content = loadmat(path, variable_names=list(keys), squeeze_me=False)
    except (ValueError, TypeError, OSError) as exc:
        raise DataContractError(f"Cannot read MATLAB development variables: {exc}") from exc
    for key in keys:
        if key not in content:
            raise DataContractError(f"Missing development variable: {key}")
    return {key: content[key] for key in keys}, "matlab_classic_samples_by_channels"


def _read_variable(path: Path, key: str) -> tuple[np.ndarray, str]:
    content, adapter = _read_variables(path, (key,))
    return content[key], adapter


def development_path(config: ExperimentConfig, drive_root: Path, subject: str) -> Path:
    if subject not in config.data["project"]["subjects"]:
        raise DataContractError(f"Unknown subject: {subject}")
    filename = config.data["raw_data"]["train_file_template"].format(subject_number=subject[1:])
    path = inside(config.path(drive_root, "raw_dir"), filename)
    if "testlabels" in path.name.lower():
        raise PermissionError("Development path resolves to test labels")
    if not path.is_file():
        raise MissingDevelopmentData(f"Missing development file for {subject}: {filename}")
    return path


def _matrix(values: np.ndarray, columns: int, key: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 2 or values.dtype.kind not in "fiu":
        raise DataContractError(f"{key}: expected a two-dimensional real numeric matrix")
    if values.shape[1] != columns or values.shape[0] <= columns:
        raise DataContractError(f"{key}: expected (time,{columns}) with time>{columns}; got {values.shape}; no inferred transpose")
    finite = np.isfinite(values)
    if not finite.all():
        rows, channels = np.where(~finite)
        raise DataContractError(f"{key}: {len(rows)} nonfinite values; first at sample={rows[0]}, channel_0based={channels[0]}")
    result = np.ascontiguousarray(values.T)
    result.setflags(write=False)
    return result


def _ecog_record(config, subject, path, matrix, adapter, before):
    channels = config.data["raw_data"]["expected_channels_nominal"][subject]
    values = _matrix(matrix, channels, "train_data")
    fs = config.data["raw_data"]["ecog_fs_hz"]
    return DevelopmentRecord(subject, values, {
        "subject": subject, "dataset": config.data["project"]["dataset"],
        "path": (Path(config.data["paths"]["raw_dir"]) / path.name).as_posix(),
        "sha256": before, "loaded_variables": ["train_data"], "adapter": adapter,
        "original_shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "n_ecog_channels_original": channels, "n_train_samples_actual": values.shape[1],
        "sampling_hz_ecog": fs, "time_origin_s": 0.0,
        "last_sample_time_s": (values.shape[1] - 1) / fs,
        "duration_s": values.shape[1] / fs,
        "time_basis": "nominal sampling rate; origin assumed 0; no measured timestamps",
        "units": "original; physical calibration unconfirmed", "nonfinite_count": 0,
    })


def _glove_record(config, record, matrix, adapter):
    values = _matrix(matrix, config.data["raw_data"]["n_fingers"], "train_dg")
    n = values.shape[1]
    ecog_n = record.values.shape[1]
    fs = config.data["raw_data"]["ecog_fs_hz"]
    effective_fs = config.data["raw_data"]["glove_fs_effective_hz"]
    if n == ecog_n:
        storage_fs = fs
        representation = "stored_on_ecog_grid; upsampling pattern verified by preprocessing.inspect_glove_storage"
    elif ecog_n * effective_fs == n * fs:
        storage_fs = effective_fs
        representation = "effective_rate_grid"
    else:
        raise DataContractError(f"train_dg length {n} cannot be reconciled with {ecog_n} ECoG samples")
    return DevelopmentRecord(record.subject, values, {
        "subject": record.subject, "sha256": record.metadata["sha256"],
        "loaded_variables": ["train_dg"], "adapter": adapter,
        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "sampling_hz_glove_effective": effective_fs,
        "sampling_hz_glove_storage": storage_fs, "representation": representation,
        "duration_s": n / storage_fs, "nonfinite_count": 0,
        "use": "development only; descriptive EDA and fold training; never test labels"})



def read_development(config: ExperimentConfig, drive_root: Path, subject: str
                     ) -> tuple[DevelopmentRecord, DevelopmentRecord]:
    """Read ECoG and glove together, hashing the original once before and once after."""
    path = development_path(config, drive_root, subject)
    before = sha256_file(path)
    content, adapter = _read_variables(path, ("train_data", "train_dg"))
    ecog = _ecog_record(config, subject, path, content["train_data"], adapter, before)
    glove = _glove_record(config, ecog, content["train_dg"], adapter)
    if sha256_file(path) != before:
        raise DataContractError("Original file changed during reading")
    return ecog, glove


def read_ecog(config: ExperimentConfig, drive_root: Path, subject: str) -> DevelopmentRecord:
    path = development_path(config, drive_root, subject)
    before = sha256_file(path)
    matrix, adapter = _read_variable(path, "train_data")
    record = _ecog_record(config, subject, path, matrix, adapter, before)
    if sha256_file(path) != before:
        raise DataContractError("Original file changed during reading")
    return record


def read_development_glove(config: ExperimentConfig, drive_root: Path,
                           record: DevelopmentRecord) -> DevelopmentRecord:
    path = development_path(config, drive_root, record.subject)
    if sha256_file(path) != record.metadata["sha256"]:
        raise DataContractError("Source changed since ECoG load")
    matrix, adapter = _read_variable(path, "train_dg")
    glove = _glove_record(config, record, matrix, adapter)
    if sha256_file(path) != record.metadata["sha256"]:
        raise DataContractError("Original file changed during label validation")
    return glove


def validate_development_labels(config: ExperimentConfig, drive_root: Path,
                                record: DevelopmentRecord) -> dict[str, Any]:
    """Validate train_dg and return its metadata only."""
    return read_development_glove(config, drive_root, record).metadata


BLOCK_FILES = ("x.npy", "y.npy", "x_index.npy", "y_index.npy", "valid.npy", "block_manifest.json")
PROCESSED_MANIFEST = "manifests/processed_manifest.json"
FOLDS_MANIFEST = "folds_manifest.json"
FOLD_MANIFEST = "fold_manifest.json"


class StoreError(RuntimeError):
    """A published directory is incomplete, corrupt or inconsistent with its name."""


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def processing_contract(config: ExperimentConfig, raw_sha256: dict[str, str]) -> dict[str, Any]:
    """Effective data contract. Presentation, approval and training are not data identity."""
    data = config.data
    validation = data["validation"]
    purge, _ = config.purge_for_build()
    return {
        "schema_version": 2, "raw_sha256": raw_sha256,
        "raw_data": {k: data["raw_data"][k] for k in (
            "ecog_fs_hz", "glove_fs_effective_hz", "expected_channels_nominal", "n_fingers",
            "train_ecog_key", "train_glove_key")},
        "excluded_channels": {s: list(config.excluded_channels(s)) for s in data["project"]["subjects"]},
        "signal": data["signal"], "scaling": data["scaling"], "purge_samples": purge,
        "validation": {k: validation[k] for k in (
            "design", "n_base_blocks", "block_names", "stage1_validation_blocks", "stage2_validation_blocks",
            "train_window_samples", "train_stride_samples", "inference_window_samples",
            "inference_stride_samples", "allow_windows_across_gaps", "coverage_policy", "score_reconstruction")},
    }


def numerical_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*.npy"))}


def pipeline_identity(contract: dict, numerical: dict[str, str]) -> str:
    return canonical_hash({"contract": contract, "numerical_sha256": numerical})


def verify_pipeline(root: Path, config: ExperimentConfig | None = None,
                    raw_sha256: dict[str, str] | None = None) -> dict:
    """Verify stored identity, every artifact and (optionally) the effective contract."""
    manifest = verify_manifest(root, PROCESSED_MANIFEST)
    if manifest.get("schema_version") != 2:
        raise StoreError("Historical format: regenerate with notebook 02 (no automatic migration)")
    numerical = {name: digest for name, digest in manifest["files"].items() if name.endswith(".npy")}
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*.npy")}
    if set(numerical) != actual_names:
        raise StoreError("Numerical files differ from the registered inventory")
    digest = pipeline_identity(manifest["identity"], numerical)
    if digest != manifest["pipeline_id"] or digest != root.name:
        raise StoreError("Pipeline identity does not match its data contract and numerical artifacts")
    if config is not None:
        expected = processing_contract(config, raw_sha256 if raw_sha256 is not None
                                       else manifest["identity"]["raw_sha256"])
        if expected != manifest["identity"]:
            raise StoreError("Selected pipeline has a different effective contract; regenerate with notebook 02")
    folds = read_json(root / "folds" / FOLDS_MANIFEST)
    if folds.get("pipeline_id") != digest:
        raise StoreError("Folds belong to a different pipeline")
    return manifest


def save_arrays(folder: Path, **arrays: Any) -> None:
    """Persist named arrays without pickle; callers validate their numerical contract."""
    import numpy as np
    folder.mkdir(parents=True, exist_ok=False)
    for name, array in arrays.items():
        if not name.isidentifier():
            raise ValueError(f"Invalid array name: {name}")
        np.save(folder / f"{name}.npy", array, allow_pickle=False)


def block_dir(pipeline_dir: Path, subject: str, block: str) -> Path:
    return pipeline_dir / subject / block


def fold_dir(fold_set_dir: Path, subject: str, val_block: str) -> Path:
    return fold_set_dir / subject / f"val_{val_block}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(root: Path, relative_manifest: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Hash every file under ``root`` (read back from disk) and write the manifest last."""
    manifest_path = root / relative_manifest
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p != manifest_path}
    manifest = {**payload, "files": files, "manifest_is_commit_marker": True}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    from .reporting import write_json
    write_json(manifest_path, manifest)
    return manifest


def verify_manifest(root: Path, relative_manifest: str, prefix: str = "") -> dict[str, Any]:
    """Check the files listed in a manifest (optionally only those under ``prefix``)."""
    path = root / relative_manifest
    if not path.is_file():
        raise StoreError(f"{root}: manifest {relative_manifest} is missing (incomplete directory)")
    manifest = read_json(path)
    problems = []
    for relative, digest in manifest["files"].items():
        if not relative.startswith(prefix):
            continue
        target = root / relative
        if not target.is_file():
            problems.append(f"missing {relative}")
        elif sha256_file(target) != digest:
            problems.append(f"sha256 mismatch {relative}")
    if problems:
        raise StoreError(f"{root}: {len(problems)} problem(s): {problems[:5]}")
    return manifest


def publish(partial: Path, final: Path, relative_manifest: str) -> bool:
    """Rename a committed partial directory to its final name.

    Returns True when published now, False when an identical published
    directory already existed (it is verified and the partial is removed).
    """
    partial, final = partial.resolve(), final.resolve()
    if partial == final or partial.parent != final.parent or not partial.name.startswith(".partial_"):
        raise StoreError("Publication requires a distinct staging directory beside the final directory")
    if not (partial / relative_manifest).is_file():
        raise StoreError(f"{partial}: cannot publish without its manifest")
    if final.exists():
        existing = verify_manifest(final, relative_manifest)
        candidate = verify_manifest(partial, relative_manifest)
        if existing.get("pipeline_id") != candidate.get("pipeline_id"):
            raise StoreError("Cannot reuse a directory belonging to a different pipeline")
        shutil.rmtree(partial)
        return False
    partial.rename(final)
    return True


def approved_pipeline(config: ExperimentConfig, drive_root: Path) -> Path:
    """Use the existing 03 evidence and explicit YAML approval; no new approval files."""
    from .config import VERIFICATION_IDS
    verification = config.data["verification"]
    if verification["status"] != "approved" or not verification["approved_manifest_file"]:
        raise StoreError("First complete V01-V14 and record approval in verification.status/approved_manifest_file")
    config.approved_purge()
    selected = config.data["paths"]["processed_pipeline_id"]
    if selected is None:
        raise StoreError("Select paths.processed_pipeline_id from notebook 02")
    approval_path = inside(config.storage_root(drive_root), verification["approved_manifest_file"])
    approval = read_json(approval_path)
    report_path = approval_path.parent / "verification_report.json"
    report = read_json(report_path)
    if (approval.get("pipeline_id") != selected or report.get("pipeline_id") != selected
            or not report.get("all_pass")
            or any(report.get("tests", {}).get(vid, {}).get("status") != "PASS" for vid in VERIFICATION_IDS)):
        raise StoreError("The selected pipeline does not have V01-V14 PASS evidence")
    evidence = read_json(approval_path.parent / "artifact_manifest.json")["sha256"]
    for path in (approval_path, report_path):
        if evidence.get(path.name) != sha256_file(path):
            raise StoreError(f"Verification evidence changed: {path.name}")
    components = approval["methodological_components"]
    for key, current in (("model", config.data["model"]), ("loss", config.data["training"]["loss"]),
                         ("metrics", config.data["metrics"])):
        if components[key] != current:
            raise StoreError(f"Frozen {key} differs from verification; repeat 03")
    source_root = Path(__file__).resolve().parent
    for name in ("preprocessing.py", "splits.py", "datasets.py", "model.py", "losses.py", "metrics.py",
                 "validation.py", "prep/blocks.py", "prep/folds.py"):
        recorded = approval["component_source_sha256"].get(f"src/tt_ecog/{name}")
        if recorded != sha256_file(source_root / name):
            raise StoreError(f"Frozen source changed: {name}; repeat 03")
    root = config.path(drive_root, "processed_dir") / selected
    verify_pipeline(root, config)
    if approval["manifests_sha256"]["processed_manifest"] != sha256_file(root / PROCESSED_MANIFEST):
        raise StoreError("The approved processed manifest differs from the selected data")
    return root


def save_training_checkpoint(path, model, optimizer, epoch, parameters, loader, history=None):
    """Replace only the last complete checkpoint, including states needed to resume."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    torch.save({"epoch": epoch, "parameters": parameters, "history": history,
                "model": model.state_dict(),
                "loader_contract": {"subject": loader.dataset.fold.subject,
                                    "fold": loader.dataset.fold.name, "batch_size": loader.batch_size},
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "loader_rng": loader.generator.get_state()}, partial)
    partial.replace(path)


def load_training_checkpoint(path, model, optimizer, loader, pipeline_id):
    """Restore the last epoch; caller resumes at returned epoch + 1 on the same device."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["parameters"]["pipeline_id"] != pipeline_id:
        raise StoreError("Checkpoint belongs to another pipeline")
    expected = {"subject": loader.dataset.fold.subject, "fold": loader.dataset.fold.name,
                "batch_size": loader.batch_size}
    if payload["loader_contract"] != expected:
        raise StoreError("Checkpoint subject, fold or batch differs from the current loader")
    if payload["cuda_rng"] and not torch.cuda.is_available():
        raise StoreError("CUDA checkpoint needs CUDA for reproducible continuation")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    loader.generator.set_state(payload["loader_rng"])
    return payload


def atomic_json(path, value):
    """A small commit marker is either complete or absent/previous, never truncated."""
    from .reporting import write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    write_json(partial, value)
    partial.replace(path)


def open_final_run(config, drive_root, plans, device):
    """Persist the pre-test plan; resume only the run explicitly selected in YAML.

    The notebook admits approved data before calling this data-free persistence layer.
    """
    from datetime import datetime, timezone
    from uuid import uuid4
    from .config import RUN_ID
    from .final_evaluation import final_contract
    from .reporting import environment
    from .training import device_info
    contract = final_contract(config)
    selected = config.data["paths"].get("final_run_id")
    if selected is not None and (not isinstance(selected, str) or not RUN_ID.fullmatch(selected)):
        raise StoreError("paths.final_run_id must be the explicit run ID printed by 07")
    run_id = selected or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    output = config.path(drive_root, "evaluation_dir") / run_id
    if selected:
        recorded = read_json(output / "plan.json")
        if recorded["contract"] != contract or recorded["plans"] != plans:
            raise StoreError("Final run settings/selected trials changed; do not mix decisions")
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "config.json", config.data)
        atomic_json(output / "environment.json", {**environment(config.source.parent.parent), **device_info(device),
                                                   "gpu_used": str(device).startswith("cuda")})
        atomic_json(output / "plan.json", {"pipeline_id": contract["pipeline_id"], "run_id": run_id,
                                          "contract": contract, "plans": plans})
    return output


def save_final_scalers(folder, feature, target, channel_map):
    """One shared pair of development-fitted scalers per subject, used by A/B/C."""
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / "scalers.partial"
    with partial.open("wb") as stream:
        np.savez(stream, center=feature.center_, scale=feature.scale_,
                 data_min=target.data_min_, data_max=target.data_max_, kept_channels=channel_map)
    partial.replace(folder / "scalers.npz")


def load_final_scalers(config, folder):
    from . import preprocessing as pp
    with np.load(folder / "scalers.npz", allow_pickle=False) as a:
        return (pp.restore_feature_scaler(config, a["center"], a["scale"]),
                pp.restore_target_scaler(config, a["data_min"], a["data_max"]))


def freeze_final_run(config, output):
    """Freeze all subject/variant models BEFORE opening any test variable."""
    import torch
    from .final_evaluation import final_contract, VARIANTS, epoch_selection, final_policy
    from .reporting import utc_now
    if (output / "frozen.json").exists():
        return verify_final_run(config, output)
    plan = read_json(output / "plan.json")
    if plan["contract"] != final_contract(config):
        raise StoreError("Final configuration changed since the pre-test plan")
    files = {name: sha256_file(output / name) for name in ("plan.json", "config.json", "environment.json")}
    decisions = {}
    for subject in config.data["project"]["subjects"]:
        scaler_path = f"{subject}/scalers.npz"
        files[scaler_path] = sha256_file(output / scaler_path)
        decisions[subject] = {}
        for variant in VARIANTS:
            folder = output / subject / variant
            chosen = read_json(folder / "selection.json")
            selection = torch.load(folder / "selection.pt", map_location="cpu", weights_only=True)
            if chosen != epoch_selection(selection["history"], final_policy(config)) or not chosen["stop"]:
                raise StoreError("Incomplete internal epoch selection")
            checkpoint = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
            expected = plan["plans"][subject][variant]
            if (checkpoint["epoch"] != chosen["selected_epoch"]
                    or checkpoint["parameters"]["pipeline_id"] != plan["pipeline_id"]
                    or checkpoint["parameters"]["variant_parameters"] != expected
                    or checkpoint["loader_contract"]["subject"] != subject
                    or checkpoint["loader_contract"]["fold"] != "full_development"):
                raise StoreError("Refit checkpoint is incomplete or belongs to another decision")
            decisions[subject][variant] = chosen
            for name in ("selection.json", "selection.pt", "last.pt", "refit.csv", "selection.csv"):
                relative = f"{subject}/{variant}/{name}"
                files[relative] = sha256_file(output / relative)
    source = config.source.parent.parent
    code = {p.relative_to(source).as_posix(): sha256_file(p) for p in sorted((source / "src/tt_ecog").rglob("*.py"))}
    frozen = {**plan, "decisions": decisions, "files": files, "source_sha256": code,
              "frozen_at_utc": utc_now(), "test_accessed_at_freeze": False}
    atomic_json(output / "frozen.json", frozen)
    return frozen


def verify_final_run(config, output):
    from .final_evaluation import final_contract, VARIANTS
    if not (output / "frozen.json").is_file():
        raise StoreError("Freeze all final models and development decisions before test access")
    frozen = verify_manifest(output, "frozen.json")
    if frozen["contract"] != final_contract(config):
        raise StoreError("Frozen final settings differ from the current configuration")
    for subject in config.data["project"]["subjects"]:
        if set(frozen["decisions"].get(subject, {})) != set(VARIANTS):
            raise StoreError("Missing final subject/variant decisions")
    source = config.source.parent.parent
    for name, digest in frozen["source_sha256"].items():
        if sha256_file(inside(source, name)) != digest:
            raise StoreError(f"Code changed after final freezing: {name}")
    return frozen


def read_final_test(config, drive_root, output, subject):
    """07-only test access: approved pipeline and every final decision must be frozen."""
    from .reporting import utc_now
    pipeline_dir = approved_pipeline(config, drive_root)
    frozen = verify_final_run(config, output)
    if subject not in frozen["decisions"]:
        raise DataContractError("Unknown final subject")
    raw = config.data["raw_data"]
    if raw["test_ecog_key"] != "test_data" or raw["test_labels_key"] != "test_dg":
        raise DataContractError("Unsupported final MATLAB keys; document an adapter")
    # Resolve the test-label path only after the gate. Never expose it through config.path.
    labels_root = inside(config.storage_root(drive_root), config.data["paths"]["test_labels_dir"])
    ecog_path = development_path(config, drive_root, subject)
    labels_path = inside(labels_root, raw["test_labels_file_template"].format(subject_number=subject[1:]))
    # Once test opening begins, training remains forbidden even if reading/evaluation fails.
    atomic_json(output / "test_access.json", {"run_id": frozen["run_id"], "pipeline_id": frozen["pipeline_id"],
                                             "started_at_utc": utc_now(), "subject": subject})
    before = {"ecog": sha256_file(ecog_path), "labels": sha256_file(labels_path)}
    expected_source = read_json(pipeline_dir / PROCESSED_MANIFEST)["identity"]["raw_sha256"][subject]
    if before["ecog"] != expected_source:
        raise StoreError("Competition file differs from the source used by the selected pipeline")
    ecog, ecog_adapter = _read_numeric_variables(ecog_path, ("test_data",))
    labels, labels_adapter = _read_numeric_variables(labels_path, ("test_dg",))
    x = _matrix(ecog["test_data"], raw["expected_channels_nominal"][subject], "test_data")
    y = _matrix(labels["test_dg"], raw["n_fingers"], "test_dg")
    if y.shape[1] == x.shape[1]:
        storage_fs = raw["ecog_fs_hz"]
    elif y.shape[1] * raw["ecog_fs_hz"] == x.shape[1] * raw["glove_fs_effective_hz"]:
        storage_fs = raw["glove_fs_effective_hz"]
    else:
        raise DataContractError("test_dg length does not match test ECoG duration")
    if before != {"ecog": sha256_file(ecog_path), "labels": sha256_file(labels_path)}:
        raise DataContractError("Test source changed during reading")
    return x, y, {"sha256": before, "adapters": [ecog_adapter, labels_adapter],
                  "sampling_hz_glove_storage": storage_fs, "loaded_variables": ["test_data", "test_dg"]}


def save_test_result(folder, arrays, report):
    """Publish a complete subject evaluation atomically; never replace an old result."""
    if folder.exists():
        return verify_manifest(folder, "artifact_manifest.json")
    partial = folder.with_name(".partial_" + folder.name)
    partial.mkdir(parents=True, exist_ok=True)
    # A failed write can be retried only with the already frozen models.
    for name, array in arrays.items():
        np.save(partial / f"{name}.npy", array, allow_pickle=False)
    atomic_json(partial / "metrics.json", report)
    write_manifest(partial, "artifact_manifest.json", {"pipeline_id": report["pipeline_id"], "run_id": report["run_id"]})
    partial.rename(folder)
    return verify_manifest(folder, "artifact_manifest.json")


class StudyArchive:
    """Local SQLite workspace; publish DB and sampler together in one atomic archive.

    Use one writer/runtime per study. Only load archives produced by this project:
    the sampler uses pickle, as recommended by Optuna for exact RNG continuation.
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / "study.zip"

    def __enter__(self):
        import pickle
        import tempfile
        import zipfile
        import optuna
        self.folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="tt_optuna_")
        self.database = Path(self.temporary.name) / "study.db"
        self.sampler = None
        try:
            if self.path.exists():
                with zipfile.ZipFile(self.path) as archive:
                    version = json.loads(archive.read("version.json"))["optuna"]
                    if version != optuna.__version__:
                        raise StoreError(f"Resume with Optuna {version}, found {optuna.__version__}")
                    self.database.write_bytes(archive.read("study.db"))
                    self.sampler = pickle.loads(archive.read("sampler.pkl"))
        except BaseException:
            self.temporary.cleanup()
            raise
        return self

    def save(self, sampler):
        from contextlib import closing
        import pickle
        import sqlite3
        import zipfile
        import optuna
        snapshot = self.database.with_name("snapshot.db")
        with closing(sqlite3.connect(self.database)) as source, closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
        partial = self.path.with_suffix(".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(snapshot, "study.db")
            archive.writestr("sampler.pkl", pickle.dumps(sampler))
            archive.writestr("version.json", json.dumps({"optuna": optuna.__version__}))
        partial.replace(self.path)

    def __exit__(self, *exc):
        self.temporary.cleanup()
