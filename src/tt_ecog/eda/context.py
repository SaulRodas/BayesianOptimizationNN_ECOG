"""Run folder, artifact registry and per-section status for one EDA execution."""

from dataclasses import dataclass, field
import functools
from pathlib import Path
import traceback
from typing import Any, Callable

import pandas as pd

from ..config import ExperimentConfig
from ..reporting import write_json

# Windows MAX_PATH is 260; keep every artifact path comfortably below it.
MAX_ABSOLUTE_PATH = 240


@dataclass(frozen=True)
class Artifact:
    relative: str  # posix path relative to the run folder
    path: Path
    kind: str  # figure | table | json
    caption: str
    section: str


@dataclass
class SectionRecord:
    name: str
    title: str
    justifies: str
    status: str = "PENDING"
    artifacts: list[Artifact] = field(default_factory=list)
    error: str | None = None


class RunContext:
    """Everything a section needs to save evidence inside one run folder."""

    def __init__(self, config: ExperimentConfig, drive_root: Path, output: Path,
                 run_id: str, env: dict[str, Any]):
        self.config = config
        self.drive_root = drive_root
        self.output = output
        self.run_id = run_id
        self.env = env
        self.eda = config.data["eda"]
        self.sections: dict[str, SectionRecord] = {}
        self.subject_status: dict[str, dict[str, Any]] = {}
        self.data_manifest: dict[str, Any] = {}
        self._current: SectionRecord | None = None
        for name in ("fig", "tables", "errors"):
            (output / name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- artifacts
    def _target(self, folder: str, filename: str) -> tuple[Path, str]:
        path = self.output / folder / filename
        if len(str(path.resolve())) >= MAX_ABSOLUTE_PATH:
            raise OSError(f"Artifact path too long for Windows ({len(str(path))} chars): {path}")
        return path, f"{folder}/{filename}"

    def _register(self, relative: str, path: Path, kind: str, caption: str) -> Artifact:
        section = self._current.name if self._current else "run"
        artifact = Artifact(relative, path, kind, caption, section)
        if self._current is not None:
            self._current.artifacts.append(artifact)
        return artifact

    def save_fig(self, fig, name: str, caption: str) -> Artifact:
        path, relative = self._target("fig", f"{name}.png")
        fig.savefig(path, dpi=self.eda["figure_dpi"], facecolor=fig.get_facecolor())
        return self._register(relative, path, "figure", caption)

    def save_table(self, frame: pd.DataFrame, name: str, caption: str) -> Artifact:
        path, relative = self._target("tables", f"{name}.csv")
        frame.to_csv(path, index=False)
        return self._register(relative, path, "table", caption)

    def save_json(self, value: Any, name: str, caption: str) -> Artifact:
        path, relative = self._target("tables", f"{name}.json")
        write_json(path, value)
        return self._register(relative, path, "json", caption)

    # ------------------------------------------------------------- sections
    def execute(self, name: str, title: str, justifies: str,
                func: Callable[["RunContext", dict], None], subjects: dict) -> list[Artifact]:
        import matplotlib
        from .plotting import RC
        record = SectionRecord(name, title, justifies)
        self.sections[name] = record
        self._current = record
        try:
            with matplotlib.rc_context(RC):
                func(self, subjects)
        except Exception:
            record.status = "FAIL"
            record.error = traceback.format_exc()
            (self.output / "errors" / f"{name}.txt").write_text(record.error, encoding="utf-8")
            raise
        else:
            record.status = "COMPLETE"
        finally:
            self._current = None
        return record.artifacts


def section(name: str, title: str, justifies: str):
    """Mark a function ``f(ctx, subjects)`` as an EDA section with status tracking."""
    def decorate(func):
        @functools.wraps(func)
        def run(ctx: RunContext, subjects: dict) -> list[Artifact]:
            return ctx.execute(name, title, justifies, func, subjects)
        run.section_name = name
        run.title = title
        run.justifies = justifies
        return run
    return decorate
