"""Inspect common requirements, optionally install missing packages only.

Run with the notebook kernel's sys.executable. Never installs torch or CUDA.
"""

import argparse
from importlib import metadata
from pathlib import Path
import platform
import subprocess
import sys

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-missing", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(f"Python {platform.python_version()} — Common dependencies on CPU")
    missing, incompatible = [], []
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        requirement = Requirement(line)
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            print(f"MISSING {requirement.name}: {requirement.specifier}")
            missing.append(line)
        else:
            print(f"INSTALLED {requirement.name}=={version}")
            if version not in requirement.specifier:
                incompatible.append(f"{requirement.name}=={version}; requires {requirement.specifier}")
    if incompatible:
        print("Incompatible environment; resolve explicitly before executing:\n" + "\n".join(incompatible))
        return 1
    if missing and not args.install_missing:
        print("Use --install-missing to install the listed missing common requirements.")
        return 2
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        print("If any libraries were already imported, restart the kernel before analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
