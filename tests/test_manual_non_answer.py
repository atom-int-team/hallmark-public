"""`_reject_non_answer` — what is a capture failure (skip) vs a system failure (graded 0).

The line is deliberately narrow and was re-confirmed 2026-07-24 on the matched-pair captures:
only the banners already listed in `_UI_ERROR_RE` are skipped. Everything else — refusals AND
the product's first-person soft-failure replies ("I encountered an error…") — is a real answer
slot the system left unusable, and scores 0 in the denominator rather than becoming a dash.

Offline; no keys.
"""
from pathlib import Path

import pytest

from hallmark.adapters.base import GenerationSkipped
from hallmark.adapters.manual import _reject_non_answer

_PATH = Path("submissions/x/seed0/t.md")


def _check(answer: str) -> None:
    _reject_non_answer(answer, _PATH, 0, has_artifacts=False)


# Verbatim from real captures. These are infrastructure banners rendered in the answer slot
# with no reply at all — skipped so a re-capture can fill the cell.
UI_ERRORS = [
    "Sorry, something went wrong. Please try your request again.",
    "An error occurred while processing your request.",
    "You've reached your daily message limit.",
]

# Must be GRADED (0), not skipped. The last three are gemini-31-pro's captures on
# e2e_oracle_point / e2e_multi_constraint / e2e_target_point (2026-07-24): the product
# answered in the first person, so the cell counts as a failure to deliver, not a lost capture.
GRADED = [
    "I can't help with that.",
    "I won't produce that recipe.",
    "I'm unable to assist with this request.",
    "I'm having a hard time fulfilling your request. Can I help you with something else instead?",
    "I seem to be encountering an error. Can I try something else for you?",
    "I encountered an error doing what you asked. Could you try again?",
]


@pytest.mark.parametrize("answer", UI_ERRORS)
def test_ui_error_banner_is_skipped(answer):
    with pytest.raises(GenerationSkipped):
        _check(answer)


@pytest.mark.parametrize("answer", GRADED)
def test_refusal_and_first_person_failure_are_graded(answer):
    _check(answer)  # must not raise


def test_long_answer_quoting_a_banner_is_still_graded():
    """The <=400-char guard: a real answer discussing error handling must be graded."""
    long_answer = (
        "When the pipeline fails the UI shows 'Sorry, something went wrong. Please try your "
        "request again.' Handling that is out of scope here. " + "Filler analysis. " * 30
        + "Final: Cr 12, Ni 6, Mo 3; heat_treatment solution_aged, route wrought"
    )
    assert len(long_answer) > 400
    _check(long_answer)  # must not raise


def test_empty_answer_without_artifacts_is_skipped():
    with pytest.raises(GenerationSkipped):
        _check("   \n  ")


def test_empty_answer_with_an_artifact_is_graded():
    _reject_non_answer("", _PATH, 0, has_artifacts=True)  # must not raise
