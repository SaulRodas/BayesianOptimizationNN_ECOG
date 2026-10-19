"""Persistence of processed data: identity hashes, layout and verified manifests.

A directory is written under a partial name, its manifest is written LAST (the
commit marker, with the SHA-256 of every file read back from disk) and only then
it is renamed to its final, content-addressed name. A published directory is
reused only when every file of its manifest still matches; it is never
overwritten, and a corrupt one is a failure that needs human inspection.
"""

from importlib import metadata
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .io import sha256_file
from .reporting import write_json

HASH_LENGTH = 16
BLOCK_FILES = ("x.npy", "y.npy", "x_index.npy", "y_index.npy", "valid.npy", "block_manifest.json")
PROCESSED_MANIFEST = "manifests/processed_manifest.json"
FOLDS_MANIFEST = "folds_manifest.json"
FOLD_MANIFEST = "fold_manifest.json"


class StoreError(RuntimeError):
    """A published directory is incomplete, corrupt or inconsistent with its name."""


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def source_hashes(relative_names: tuple[str, ...]) -> dict[str, str]:
    """SHA-256 of package sources with normalised line endings (Windows checkouts
    use CRLF, Colab LF), so the same commit yields the same identity."""
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in relative_names}


def library_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def block_dir(pipeline_dir: Path, subject: str, block: str) -> Path:
    return pipeline_dir / subject / block


def fold_set_name(purge: int, fold_hash: str) -> str:
    return f"purge_{purge}_{fold_hash}"


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
    if not (partial / relative_manifest).is_file():
        raise StoreError(f"{partial}: cannot publish without its manifest")
    if final.exists():
        verify_manifest(final, relative_manifest)
        shutil.rmtree(partial)
        return False
    partial.rename(final)
    return True
