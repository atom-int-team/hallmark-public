"""Deterministic extraction must survive markdown emphasis around labels.

A model that typesets its recipe as a bolded markdown table (``| **Cr** | 18.0 |``)
used to defeat the regex safety net in ``graders/extract.py``: the composition came
back empty and ``oracle.assemble_recipe`` reported "no proposed composition", turning
a bad-but-real recipe into an indistinguishable extraction miss.
"""
from __future__ import annotations

import pytest

from hallmark.graders.extract import (
    _find_comp_value,
    _find_labeled_choice,
    _find_labeled_number,
    _regex_extract_design,
    _regex_extract_point,
)
from hallmark.textutils import strip_markdown_emphasis

SPEC = {
    "composition": ["C", "Cr", "Ni", "Mo"],
    "balance_element": "Fe",
    "comp_total": 100.0,
    "variables": {
        "C": {"domain": [0, 1.2]},
        "Cr": {"domain": [0, 22]},
        "Ni": {"domain": [0, 25]},
        "Mo": {"domain": [0, 6]},
        "Fe": {"domain": [30, 100]},
        "aging_temp": {"domain": [400, 800]},
    },
    "categorical": {"heat_treatment": ["annealed", "quenched", "solution_aged"]},
    "terms": [],
}

BOLD_TABLE = """### Optimal Recipe

| Parameter | Value |
|-----------|-------|
| **C** | 0.85 wt% |
| **Cr** | 18.00 wt% |
| **Ni** | 12.00 wt% |
| **Mo** | 4.50 wt% |
| **aging_temp** | 620 °C |
| **heat_treatment** | solution_aged |

**Predicted target:** `46.82 MPa`
"""


def test_strip_markdown_emphasis_keeps_single_underscore():
    # single ``_`` carries machine field names and must survive; ``__`` is bold
    assert strip_markdown_emphasis("**aging_temp**: 620") == "aging_temp: 620"
    assert strip_markdown_emphasis("__Cr__ = `18`") == "Cr = 18"
    assert strip_markdown_emphasis("*Ni* 12") == "Ni 12"
    assert strip_markdown_emphasis(None) == ""


def test_bolded_table_yields_full_recipe():
    got = _regex_extract_design(BOLD_TABLE, SPEC)
    # multi-letter symbols only — bare ``C`` is excluded by design (prose collisions)
    assert got["Cr"] == 18.0
    assert got["Ni"] == 12.0
    assert got["Mo"] == 4.5
    assert got["aging_temp"] == 620.0
    assert got["heat_treatment"] == "solution_aged"


@pytest.mark.parametrize(
    "text",
    ["| **Cr** | 18.0 |", "**Cr**: 18.0", "`Cr` = 18.0", "__Cr__ 18.0 wt%", "Cr 18.0"],
)
def test_find_comp_value_tolerates_emphasis(text):
    assert _find_comp_value(text, "Cr", (0, 22)) == 18.0


def test_find_labeled_number_and_choice_tolerate_emphasis():
    assert _find_labeled_number("| **aging_temp** | 620 |", "aging_temp") == 620.0
    assert (
        _find_labeled_choice("**heat_treatment**: **solution_aged**", "heat_treatment",
                             ["annealed", "solution_aged"]) == "solution_aged"
    )


def test_regex_extract_point_tolerates_bold_final_line():
    point, complete = _regex_extract_point("**Final:** Cr=18.0, Ni=12.0", ["Cr", "Ni"])
    assert point == {"Cr": 18.0, "Ni": 12.0}
    assert complete


def test_domain_bound_still_rejects_out_of_range_prose():
    # emphasis stripping must not widen what counts as a wt% value
    assert _find_comp_value("The **Cr** content in 1998 was 40.0 wt%", "Cr", (0, 22)) is None


# --- "human"-formatted processing fields (web-UI answers) must still extract -----------
# Regression for a verified web-vs-bare grading defect: ``_find_labeled_number`` used to
# require the field's suffix word ("temp"), US spelling, and an un-separated number, so a
# perfectly complete recipe stated as "solutionizing 1058 °C" / "Solutionising temperature
# 1 026 °C" was scored 0 as an incomplete recipe. All three cells were web SUTs, none bare.
_PROC = ["solutionizing_temp", "aging_temp", "aging_time", "cooling_rate"]


def test_first_word_only_field_recovered_when_unambiguous():
    # model dropped the "temp" suffix; "solutionizing" has no sibling sharing its first word
    line = "solutionizing 1058 °C · aging 574 °C · 7.2 h · cooling rate 12 °C/s"
    assert _find_labeled_number(line, "solutionizing_temp", _PROC) == 1058.0
    assert _find_labeled_number("cooling 15 C/s", "cooling_rate", _PROC) == 15.0


def test_first_word_fallback_blocked_for_colliding_siblings():
    # "aging" alone must NOT fill aging_temp — aging_time shares the first word.
    # (Parity with the old suffix-required behaviour: no silent wrong steal.)
    assert _find_labeled_number("aging 574 then 999", "aging_temp", _PROC) is None
    # the fully-qualified form still works, exactly as before
    assert _find_labeled_number("aging_temp = 493", "aging_temp", _PROC) == 493.0
    assert _find_labeled_number("Aging temperature 580", "aging_temp", _PROC) == 580.0


def test_first_word_fallback_disabled_without_siblings():
    # Fail-safe: an uninformed caller (no siblings) must never trigger the first-word
    # fallback — otherwise aging_temp could steal aging_time's value via the space form.
    assert _find_labeled_number("aging time 480 min", "aging_temp") is None
    assert _find_labeled_number("solutionizing 1058", "solutionizing_temp") is None
    # ...but a fully-qualified restatement is still recovered with no siblings.
    assert _find_labeled_number("solutionizing temp 1058", "solutionizing_temp") == 1058.0


def test_british_spelling_and_spaced_thousands():
    # British 's' for the American 'z' in the field name, plus a spaced 4-digit value.
    cell = "| Solutionising temperature | 1 026 °C |"
    assert _find_labeled_number(cell, "solutionizing_temp", _PROC) == 1026.0
    assert _find_labeled_number("solutionizing_temp 1 250", "solutionizing_temp", _PROC) == 1250.0


@pytest.mark.parametrize(
    "text,exp",
    [
        ("aging_temp 500 550", 500.0),        # un-dashed range — NOT a thousands number
        ("aging_temp 520 540 K", 520.0),      # range with unit
        ("cooling_rate 100 200", 100.0),
        ("solutionizing_temp 980 980", 980.0),  # duplicated space-aligned column
        ("aging_temp 560 1200", 560.0),       # must not grab "120" out of "1200"
        ("aging_temp 500 120min", 500.0),
        ("solutionizing_temp 1058 574", 1058.0),
    ],
)
def test_spaced_thousands_never_merges_two_values(text, exp):
    # Precision-over-recall: a 3-digit (or longer) leading run means two distinct values,
    # so the narrow thousands form (single leading digit only) must not fire.
    field = text.split()[0]
    assert _find_labeled_number(text, field, _PROC) == exp


def test_labeled_number_decimals_negatives_and_single_word():
    assert _find_labeled_number("cooling_rate: 7.2", "cooling_rate", _PROC) == 7.2
    assert _find_labeled_number("cooling_rate = -5", "cooling_rate", _PROC) == -5.0
    # a single-word field name (no underscore) has no fallback branch; still matches
    assert _find_labeled_number("temperature = 640", "temperature") == 640.0
