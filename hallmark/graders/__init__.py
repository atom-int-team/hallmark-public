"""Capability-aware graders. Importing registers the built-in graders."""
from __future__ import annotations

# Register built-ins as a side effect of import.
from . import (
    b3_select,  # noqa: E402,F401
    olympus_design,  # noqa: E402,F401
    oracle_eval,  # noqa: E402,F401
    select_hypervolume,  # noqa: E402,F401
)
from .base import GRADERS, Grader, register_grader

__all__ = ["GRADERS", "Grader", "register_grader"]
