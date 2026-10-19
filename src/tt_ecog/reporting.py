"""Portable, timestamped evidence: environment, hashes and JSON output."""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import platform
import re
import subprocess

from .io import sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def environment(project_root: Path) -> dict:
    versions = {}
    for package in ("tt-ecog", "numpy", "scipy", "pandas", "h5py", "PyYAML", "matplotlib",
                    "mne", "scikit-learn", "IPython", "pytest", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    installed = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata['Name']})
    commit = None
    dirty = None
    remote = None
    branch = None
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                                text=True, capture_output=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
        state = subprocess.run(["git", "status", "--porcelain"], cwd=project_root,
                               text=True, capture_output=True, check=False)
        if state.returncode == 0:
            dirty = bool(state.stdout.strip())
        for name, args in (("remote", ["remote", "get-url", "origin"]),
                           ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
            result = subprocess.run(["git", *args], cwd=project_root, text=True,
                                    capture_output=True, check=False)
            if result.returncode == 0:
                if name == "remote":
                    # Never persist credentials embedded in a remote URL.
                    remote = re.sub(r"//[^/@]*@", "//", result.stdout.strip())
                else:
                    branch = result.stdout.strip()
    except FileNotFoundError:
        pass  # Git is optional for local synthetic runs; source hashes remain mandatory.
    files = []
    for directory, patterns in {"src": ["*.py"], "configs": ["*.yaml"], "scripts": ["*.py"],
                                 "notebooks": ["*.ipynb"], "tests": ["*.py"]}.items():
        for pattern in patterns:
            files.extend((project_root / directory).rglob(pattern))
    files.extend(p for p in (project_root / "pyproject.toml", project_root / "requirements-eda.txt") if p.exists())
    hashes = {p.relative_to(project_root).as_posix(): sha256_file(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"timestamp_utc": utc_now(), "python": platform.python_version(), "platform": platform.system(),
            "versions": versions, "installed_distributions": installed, "git_commit": commit, "git_dirty": dirty,
            "git_remote": remote, "git_branch": branch,
            "source_sha256": digest, "source_files": hashes,
            "device": "CPU", "gpu_used": False}
