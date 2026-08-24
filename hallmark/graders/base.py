"""Grader ABC + registry (see UNIVERSAL_BENCHMARK_SPEC §7.2).

A grader checks one aspect of one turn. It declares the ``signal`` it needs; if
the SUT cannot produce that signal the grader is ``N/A`` (not a failure) — the
runner skips it and excludes it from the turn score's denominator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..loader import GroundTruth
from ..models import Capabilities, Signal, TurnTranscript

# name -> Grader instance (graders are stateless singletons)
GRADERS: dict[str, Grader] = {}


def register_grader(cls: type[Grader]) -> type[Grader]:
    """Class decorator: instantiate and register the grader under its ``name``."""
    instance = cls()
    GRADERS[instance.name] = instance
    return cls


class Grader(ABC):
    name: str
    signal: Signal = Signal.TEXT

    def applicable(self, t: TurnTranscript, caps: Capabilities) -> bool:
        if self.signal == Signal.TEXT:
            return True
        return bool(getattr(caps, self.signal.value, False))

    @abstractmethod
    async def grade(
        self, t: TurnTranscript, params: dict, gt: GroundTruth
    ) -> tuple[float, str]:
        """Return ``(score in 0..1, detail)``. Called only when ``applicable``."""
        raise NotImplementedError
