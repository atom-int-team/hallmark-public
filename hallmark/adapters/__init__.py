"""SUT adapters: translate a conversation into one System Under Test and back.

Importing this package registers the built-in adapters into ``ADAPTERS``.
"""
from __future__ import annotations

# Register built-in adapters as a side effect of import.
from . import (
    cli_agent,  # noqa: E402,F401
    manual,  # noqa: E402,F401
    mock,  # noqa: E402,F401
    openai_chat,  # noqa: E402,F401
)
from .base import ADAPTERS, Adapter, GenerationSkipped, register_adapter

__all__ = ["ADAPTERS", "Adapter", "GenerationSkipped", "register_adapter"]
