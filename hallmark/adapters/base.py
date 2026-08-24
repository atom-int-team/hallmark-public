"""Adapter ABC + capabilities + registry (see UNIVERSAL_BENCHMARK_SPEC §7.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Attachment, Capabilities, Message, SystemSpec, TurnTranscript

# name -> Adapter subclass
ADAPTERS: dict[str, type[Adapter]] = {}


class GenerationSkipped(Exception):
    """Raised by an adapter to skip one (task×seed) without failing the run.

    Used by the ``manual`` adapter when no submission file exists yet, so the
    dataset can be filled in over time. ``runner.generate`` catches it.
    """


def price_cost(
    price_in: float | None, price_out: float | None, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """USD cost from per-1k token prices; None if no pricing configured."""
    if price_in is None and price_out is None:
        return None
    return prompt_tokens / 1000 * (price_in or 0.0) + completion_tokens / 1000 * (price_out or 0.0)


def register_adapter(name: str):
    """Class decorator registering an adapter under ``name`` in ``ADAPTERS``."""

    def deco(cls: type[Adapter]) -> type[Adapter]:
        ADAPTERS[name] = cls
        return cls

    return deco


class Adapter(ABC):
    """Drives one System Under Test for a single conversation turn.

    Subclasses set ``capabilities`` and implement ``run_turn``. The registry key
    (``ADAPTERS[...]``) names the adapter; ``system_id`` names the concrete SUT
    instance (model + provider), and is what appears in transcripts/leaderboard.
    """

    capabilities: Capabilities = Capabilities()

    def __init__(self, spec: SystemSpec) -> None:
        self.spec = spec
        self.system_id = spec.id
        self.config = spec.config
        self._task_id: str = ""
        self._seed: int = 0
        # Per-instance capabilities = class default, overridden by config["capabilities"].
        caps = self.capabilities.model_dump()
        caps.update(spec.config.get("capabilities", {}))
        self.capabilities = Capabilities(**caps)

    def set_context(self, task_id: str, seed: int) -> None:
        """Called by the runner before each task; adapters may use it (e.g. mock)."""
        self._task_id = task_id
        self._seed = seed

    @abstractmethod
    async def run_turn(
        self, history: list[Message], attachments: list[Attachment]
    ) -> TurnTranscript:
        """Send the conversation so far (+ attachments) and return one turn's transcript."""
        raise NotImplementedError
