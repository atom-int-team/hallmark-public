"""Shared test fixtures.

The suite is fully offline: every network seam (per-model LLM judge `_llm_judge_one`,
provider adapters) is monkeypatched in the tests that reach it. `parse_ids` has no seam
to patch — its LLM fallback was removed 2026-07-23 and id parsing is now fully
deterministic. The LLM-judge graders still *resolve* a key before dispatching to
the (patched) call, so a dummy key must merely be present — no real key, no `.env`, and
no network are needed. This fixture supplies that dummy key when the environment has none.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _offline_judge_key():
    """Ensure a (dummy) judge key exists so key-resolution never raises in offline runs."""
    if not (os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        os.environ["OPENAI_API_KEY"] = "test-offline-dummy-key"
    yield
