"""Step 2: per-block preprocessing of the development data and fold artifacts.

Run it from the notebook (``start_run`` → ``load_subjects`` → ``steps`` →
``finish_run``) or from the CLI (``python -m tt_ecog.prep``); both use the same
functions. Test variables are never read.
"""

from . import steps
from .gate import PrepBlocked, discover_eda_runs
from .runner import PrepFailed, finish_run, load_subjects, main, run_prep, start_run

__all__ = ["steps", "start_run", "load_subjects", "finish_run", "run_prep", "main", "PrepBlocked",
           "PrepFailed", "discover_eda_runs"]
