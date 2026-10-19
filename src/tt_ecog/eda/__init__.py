"""Descriptive EDA of the development data that justifies each preprocessing decision.

Run it from the notebook (``start_run`` → ``load_subjects`` → sections → ``finish_run``)
or from the CLI (``python -m tt_ecog.eda``); both use the same functions.
"""

from . import sections
from .notebook import show
from .runner import EdaBlocked, EdaFailed, finish_run, load_subjects, main, run_eda, start_run

__all__ = ["sections", "show", "start_run", "load_subjects", "finish_run", "run_eda", "main",
           "EdaBlocked", "EdaFailed"]
