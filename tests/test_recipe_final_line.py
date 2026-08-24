"""The recipe path parses its ``Final:`` contract block WITHOUT the LLM.

The scalar path (``_regex_extract_point``) reads its ``Final:`` line deterministically;
these tests pin the recipe counterpart and, most importantly, that a compliant answer
never reaches the network seam at all.
"""
from __future__ import annotations

import asyncio

import pytest

from hallmark.graders.extract import _regex_extract_design_final, extract_design

# Same shape as tests/test_oracle_eval.py::RECIPE_SPEC: two alloying elements, a balance
# element, one processing knob (pure noise — in no term) and one ACTIVE categorical.
SPEC = {
    "sense": "max",
    "composition": ["Cr", "Ni"], "balance_element": "Fe", "comp_total": 100.0,
    "variables": {"Cr": {"domain": [0, 25]}, "Ni": {"domain": [0, 30]},
                  "Fe": {"domain": [45, 100]}, "aging_temp": {"domain": [450, 750]}},
    "categorical": {"heat_treatment": ["annealed", "quenched", "solution_aged"]},
    "terms": [
        {"coef": 50.0, "const": True},
        {"coef": 1.0, "categorical": {"heat_treatment": {"solution_aged": 0, "quenched": -6,
                                                         "annealed": -10}}},
    ],
    "f_star": 50.0, "train_best": 30.0,
}
# aging_temp active -> a compliant answer must state it for the parse to count as complete.
SPEC_ACTIVE_PROC = {**SPEC, "terms": [*SPEC["terms"],
                    {"coef": -1e-4, "monomial": {"aging_temp": 2},
                     "center": {"aging_temp": 560}}]}


def test_contract_block_parses_without_the_llm():
    answer = ("Reasoning ... the peak sits near 18 wt% Cr.\n\n"
              "Predicted yield strength: 1483 MPa\n\n"
              "Final: Cr=18.0, Ni=10.0, aging_temp=560, heat_treatment=solution_aged")
    design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is True
    assert design == {"Cr": 18.0, "Ni": 10.0, "aging_temp": 560.0,
                      "heat_treatment": "solution_aged"}


def test_stated_balance_element_is_ignored_not_a_stray_token():
    """``Fe=72`` on the contract line is bookkeeping: ``assemble_recipe`` recomputes the
    balance as ``comp_total - Σ`` and never reads it, so it must not force the fallback."""
    answer = "Final: Cr=18, Ni=10, Fe=72, aging_temp=560, heat_treatment=solution_aged"
    design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is True and "Fe" not in design


def test_prediction_token_on_the_block_stays_deterministic():
    answer = ("Final: Cr=18, Ni=10, aging_temp=560, heat_treatment=solution_aged, "
              "predicted_strength=1483")
    _design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is True


def test_noise_processing_knob_may_be_omitted():
    """``aging_temp`` is in no term (pure noise): ``assemble_recipe`` back-fills the domain
    midpoint, so demanding it here would send a sound answer to the LLM for nothing."""
    design, complete = _regex_extract_design_final(
        "Final: Cr=18, Ni=10, heat_treatment=solution_aged", SPEC)
    assert complete is True and "aging_temp" not in design


def test_missing_active_field_falls_back():
    answer = "Final: Cr=18, Ni=10, heat_treatment=solution_aged"
    _design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is False          # aging_temp is active here and was not stated


def test_unknown_token_falls_back():
    """A field the model renamed (``solutionizing=1050``) means the block is not the
    contract we can trust — hand it to the LLM rather than grade a partial recipe."""
    answer = ("Final: Cr=18, Ni=10, aging_temp=560, solutionizing=1050, "
              "heat_treatment=solution_aged")
    _design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is False


def test_prose_recipe_falls_back():
    """An old-style answer (no contract line, space-separated values) must still reach the
    extractor — that is what keeps every cached transcript graded exactly as before."""
    answer = ("The final recipe is Cr 18, Ni 10, aging temperature 560 °C, "
              "heat treatment solution aged.")
    _design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is False


def test_placeholder_template_is_not_a_proposal():
    """A model that echoes the prompt's template verbatim proposes nothing: the
    placeholders carry no numbers, so the parse is incomplete and the answer falls through
    to the LLM exactly as any unparseable answer did."""
    answer = ("Final: Cr=<wt%>, Ni=<wt%>, aging_temp=<°C>, "
              "heat_treatment=<annealed|quenched|solution_aged>")
    design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is False and not any(k in design for k in ("Cr", "Ni"))


def test_last_final_block_wins():
    answer = ("Final: Cr=5, Ni=5, aging_temp=500, heat_treatment=annealed\n\n"
              "On reflection:\n\n"
              "Final: Cr=18, Ni=10, aging_temp=560, heat_treatment=solution_aged")
    design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is True and design["Cr"] == 18.0


def test_markdown_emphasis_survives():
    answer = "**Final:** Cr=18, Ni=10, aging_temp=560, heat_treatment=solution_aged"
    _design, complete = _regex_extract_design_final(answer, SPEC_ACTIVE_PROC)
    assert complete is True


def test_extract_design_never_calls_the_seam_on_a_compliant_answer(monkeypatch):
    """The regression guard: the whole point of the change is that this path is offline."""
    import hallmark.graders.extract as ex

    async def boom(messages, params):
        raise AssertionError("the LLM extractor must not be called on a Final: block")

    monkeypatch.setattr(ex, "_extract_llm", boom)
    answer = "Final: Cr=18, Ni=10, aging_temp=560, heat_treatment=solution_aged"
    design = asyncio.run(extract_design(answer, SPEC_ACTIVE_PROC, {}))
    assert design["Cr"] == pytest.approx(18.0)


def test_extract_design_still_falls_back_for_prose(monkeypatch):
    import json

    import hallmark.graders.extract as ex
    calls = []

    async def fake(messages, params):
        calls.append(1)
        return json.dumps({"Cr": 18, "Ni": 10, "aging_temp": 560,
                           "heat_treatment": "solution_aged"})

    monkeypatch.setattr(ex, "_extract_llm", fake)
    design = asyncio.run(extract_design("Cr is 18 wt%, Ni 10 wt%.", SPEC_ACTIVE_PROC, {}))
    assert calls == [1] and design["Ni"] == pytest.approx(10.0)
