"""Offline tests for the objective inverse-design grader oracle_eval.

Fully offline: the analytic oracle evaluator needs no network, the regex extract
backend needs no keys, and the llm backend is exercised via the monkeypatched
_extract_llm seam.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hallmark import oracle as ora
from hallmark.graders.oracle_eval import OracleEvalGrader, _score, _sense
from hallmark.loader import GroundTruth
from hallmark.models import TurnTranscript

SPEC = {
    "sense": "max",
    "variables": {"Cr": {"domain": [0, 25]}, "Ni": {"domain": [0, 30]}},
    "terms": [
        {"coef": 10.0, "const": True},
        {"coef": 5.0, "gaussian": {"Cr": [12, 3], "Ni": [18, 4]}},
    ],
    "f_star": 15.0,
    "x_star": {"Cr": 12, "Ni": 18},
    "train_best": 11.0,
    "noise_std": 0.0,
}


# --- analytic evaluator -------------------------------------------------------

def test_evaluate_terms():
    assert ora.evaluate(SPEC, {"Cr": 12, "Ni": 18}) == pytest.approx(15.0)  # peak
    assert ora.evaluate(SPEC, {"Cr": 0, "Ni": 0}) == pytest.approx(10.0, abs=0.1)


def test_monomial_with_center():
    spec = {"variables": {"x": {"domain": [0, 10]}},
            "terms": [{"coef": 2.0, "monomial": {"x": 2}, "center": {"x": 3}}]}
    assert ora.evaluate(spec, {"x": 5}) == pytest.approx(2.0 * (5 - 3) ** 2)


def test_monomial_fractional_power():
    # solid-solution √c and Hall–Petch d^-½ (A1 mechanistic oracle)
    spec = {"variables": {"c": {"domain": [0, 4]}, "d": {"domain": [1, 100]}},
            "terms": [{"coef": 10.0, "monomial": {"c": 0.5}},
                      {"coef": 20.0, "monomial": {"d": -0.5}}]}
    assert ora.evaluate(spec, {"c": 4, "d": 4}) == pytest.approx(10 * 2 + 20 * 0.5)
    # a fractional power of a non-positive base is out of range -> contributes 0
    assert ora.evaluate(spec, {"c": 0, "d": 4}) == pytest.approx(20 * 0.5)


def test_hinge_term():
    # one-sided penalty: coef·max(0, Σ aᵢxᵢ − θ)^p  (negative coef subtracts)
    spec = {"variables": {"Cr": {"domain": [0, 30]}, "Mo": {"domain": [0, 6]}},
            "terms": [{"coef": -6.0, "hinge": {"vars": {"Cr": 1.0, "Mo": 1.6},
                                               "threshold": 24.0, "power": 2}}]}
    assert ora.evaluate(spec, {"Cr": 10, "Mo": 2}) == 0.0          # below threshold
    assert ora.evaluate(spec, {"Cr": 24, "Mo": 5}) == pytest.approx(-6.0 * (8.0) ** 2)
    assert "Cr" in ora.active_variables(spec) and "Mo" in ora.active_variables(spec)


def test_clamp_to_domain():
    x, notes = ora.clamp_to_domain(SPEC, {"Cr": 99, "Ni": -5})
    assert x == {"Cr": 25.0, "Ni": 0.0}
    assert len(notes) == 2


def test_evaluate_robust_penalizes_sharp_peak():
    # narrow peak at (5,5); worst-case over ±10% (of [0,10]) drops well below the apex
    spec = {"sense": "max", "variables": {"a": {"domain": [0, 10]}, "b": {"domain": [0, 10]}},
            "terms": [{"coef": 10.0, "gaussian": {"a": [5, 0.5], "b": [5, 0.5]}}]}
    apex = ora.evaluate(spec, {"a": 5, "b": 5})
    worst, _ = ora.evaluate_robust(spec, {"a": 5, "b": 5}, {"delta_frac": 0.1})
    assert worst < apex


# --- score normalization ------------------------------------------------------

def test_score_normalization_max():
    assert _score("max", 15.0, 15.0, 11.0) == pytest.approx(1.0)   # hits optimum
    assert _score("max", 11.0, 15.0, 11.0) == pytest.approx(0.0)   # only matches floor
    assert _score("max", 13.0, 15.0, 11.0) == pytest.approx(0.5)
    assert _score("max", 9.0, 15.0, 11.0) == 0.0                   # below floor -> clamp


def test_score_normalization_min():
    assert _score("min", 1.0, 1.0, 5.0) == pytest.approx(1.0)
    assert _score("min", 3.0, 1.0, 5.0) == pytest.approx(0.5)


def test_score_degenerate_span():
    assert _score("max", 15.0, 15.0, 15.0) == 1.0
    assert _score("max", 14.0, 15.0, 15.0) == 0.0


# --- sense resolution: two gold conventions, one dispatch ---------------------

def test_sense_defaults_to_max_and_reads_either_key():
    assert _sense({}) == "max"
    assert _sense({"sense": "target"}) == "target"
    assert _sense({"goal": "target"}) == "target"   # the selection-grader convention
    assert _sense({"goal": "min"}) == "min"
    assert _sense({"sense": "max", "goal": "max"}) == "max"


def test_sense_refuses_a_gold_that_declares_both_and_disagrees():
    with pytest.raises(ValueError, match="conflicting"):
        _sense({"sense": "max", "goal": "target"})


def test_target_gold_in_the_goal_convention_is_not_graded_as_maximization(tmp_path):
    """A target gold written with ``goal:`` used to score 0.909 on an answer worth 0.0.

    The floor resolver already honoured ``goal``; the sense dispatch did not, so the miss was
    normalized against ``f_star`` as if the task were a maximization.
    """
    spec = dict(SPEC, goal="target", f_star=13.0, target_tol=0.5)
    spec.pop("sense")
    d = tmp_path / "e2e_goal_target"
    d.mkdir()
    (d / "oracle.json").write_text(json.dumps(spec))
    gt = GroundTruth(root=tmp_path)
    params = {"oracle_ref": "e2e_goal_target/oracle", "extract_backend": "regex"}

    # f(12, 18) = 15.0, i.e. 2.0 above the target of 13.0 with a tolerance of 0.5 -> 0.
    t = TurnTranscript(prompt="x", answer_text="Final: Cr 12, Ni 18")
    assert asyncio.run(OracleEvalGrader().grade(t, params, gt))[0] == 0.0

    # ...and a point that lands inside the tolerance still scores.
    spec_hit = dict(spec, f_star=15.0)
    (d / "oracle.json").write_text(json.dumps(spec_hit))
    assert asyncio.run(OracleEvalGrader().grade(t, params, gt))[0] == pytest.approx(1.0)


def test_shipped_oracle_golds_declare_exactly_one_sense_key():
    """Guard the convention: mixing the two keys in one gold is now an error, so no gold may."""
    for gold in sorted(Path("ground_truth").glob("e2e_*/oracle.json")):
        spec = json.loads(gold.read_text())
        assert "goal" not in spec, f"{gold} mixes conventions; _sense would have to arbitrate"
        assert _sense(spec) == spec.get("sense", "max")


# --- grader (regex + llm backends) -------------------------------------------

def _gt(tmp_path) -> GroundTruth:
    d = tmp_path / "e2e_demo"
    d.mkdir()
    (d / "oracle.json").write_text(json.dumps(SPEC))
    return GroundTruth(root=tmp_path)


def test_grader_regex_backend_hits_optimum(tmp_path):
    gt = _gt(tmp_path)
    t = TurnTranscript(prompt="x", answer_text="Final: Cr 12 wt%, Ni 18 wt%. Predicted 15.")
    params = {"oracle_ref": "e2e_demo/oracle", "extract_backend": "regex"}
    score, detail = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert score == pytest.approx(1.0)
    assert "f*" in detail


def test_grader_missing_value_is_zero(tmp_path):
    gt = _gt(tmp_path)
    t = TurnTranscript(prompt="x", answer_text="Cr 12 only, no nickel proposed")
    params = {"oracle_ref": "e2e_demo/oracle", "extract_backend": "regex"}
    score, detail = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert score == 0.0
    assert "Ni" in detail


def test_grader_out_of_domain_clamped(tmp_path):
    gt = _gt(tmp_path)
    t = TurnTranscript(prompt="x", answer_text="Cr 999, Ni 18")
    params = {"oracle_ref": "e2e_demo/oracle", "extract_backend": "regex"}
    score, detail = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert "clamped" in detail
    assert 0.0 <= score <= 1.0


def test_grader_llm_backend_monkeypatched(tmp_path, monkeypatch):
    gt = _gt(tmp_path)
    import hallmark.graders.extract as ex

    async def fake(messages, params):
        return json.dumps({"Cr": 12, "Ni": 18})

    monkeypatch.setattr(ex, "_extract_llm", fake)
    t = TurnTranscript(prompt="x", answer_text="prose with no parseable pairs at all")
    score, _ = asyncio.run(OracleEvalGrader().grade(t, {"oracle_ref": "e2e_demo/oracle"}, gt))
    assert score == pytest.approx(1.0)


def test_grader_robust_param(tmp_path):
    gt = _gt(tmp_path)
    t = TurnTranscript(prompt="x", answer_text="Cr 12, Ni 18")
    params = {"oracle_ref": "e2e_demo/oracle", "extract_backend": "regex",
              "robust": {"delta_frac": 0.05, "agg": "worst"}}
    score, detail = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert "robust" in detail
    assert 0.0 <= score <= 1.0


# --- recipe schema (high-D mixed-type) ---------------------------------------

RECIPE_SPEC = {
    "sense": "max",
    "composition": ["Cr", "Ni"], "balance_element": "Fe", "comp_total": 100.0, "comp_tol": 0.5,
    "variables": {"Cr": {"domain": [0, 25]}, "Ni": {"domain": [0, 30]}, "Fe": {"domain": [45, 100]},
                  "aging_temp": {"domain": [450, 750]}},
    "categorical": {"heat_treatment": ["annealed", "quenched", "solution_aged"]},
    "terms": [
        {"coef": 50.0, "const": True},
        {"coef": -10 / 25 ** 2, "monomial": {"Cr": 2}, "center": {"Cr": 18}},
        {"coef": -10 / 30 ** 2, "monomial": {"Ni": 2}, "center": {"Ni": 10}},
        {"coef": 1.0, "categorical": {"heat_treatment": {"solution_aged": 0, "quenched": -6,
                                                         "annealed": -10}}},
    ],
    "f_star": 50.0, "train_best": 30.0,
}
GOOD_RECIPE = {"Cr": 18, "Ni": 10, "aging_temp": 560, "heat_treatment": "solution_aged"}


def test_assemble_recipe_validity():
    x, note = ora.assemble_recipe(RECIPE_SPEC, GOOD_RECIPE)
    assert x["Fe"] == pytest.approx(72.0) and note == "ok"
    assert ora.assemble_recipe(RECIPE_SPEC, {**GOOD_RECIPE, "Cr": -1})[0] is None  # negative
    assert ora.assemble_recipe(RECIPE_SPEC, {"Cr": 80, "Ni": 40, "aging_temp": 560,
                                             "heat_treatment": "annealed"})[0] is None  # Σ>100
    # heat_treatment IS referenced by a term (active) -> omitting it is rejected
    assert ora.assemble_recipe(RECIPE_SPEC, {"Cr": 18, "Ni": 10,
                                             "aging_temp": 560})[0] is None  # missing active cat
    assert ora.assemble_recipe(RECIPE_SPEC, {**GOOD_RECIPE, "heat_treatment": "x"})[0] is None


def test_assemble_recipe_noise_field_backfilled():
    # aging_temp is in `variables` but in NO term -> a pure noise knob. Omitting it
    # must NOT zero an otherwise-valid recipe (Fix A): it's back-filled to the
    # domain midpoint, which by construction leaves f(x) unchanged.
    x, note = ora.assemble_recipe(
        RECIPE_SPEC, {"Cr": 18, "Ni": 10, "heat_treatment": "solution_aged"})
    assert x is not None and "defaulted" in note
    assert x["aging_temp"] == pytest.approx(600.0)  # (450 + 750) / 2
    assert ora.evaluate(RECIPE_SPEC, x) == pytest.approx(50.0)


def test_assemble_recipe_missing_active_field_rejected():
    # Make aging_temp affect f -> it is no longer noise, so omitting it is rejected.
    spec = {**RECIPE_SPEC, "terms": [*RECIPE_SPEC["terms"],
            {"coef": -1e-4, "monomial": {"aging_temp": 2}, "center": {"aging_temp": 560}}]}
    assert "aging_temp" in ora.active_variables(spec)
    assert ora.assemble_recipe(
        spec, {"Cr": 18, "Ni": 10, "heat_treatment": "solution_aged"})[0] is None


def test_recipe_categorical_effect():
    x_best, _ = ora.assemble_recipe(RECIPE_SPEC, GOOD_RECIPE)
    x_bad, _ = ora.assemble_recipe(RECIPE_SPEC, {**GOOD_RECIPE, "heat_treatment": "annealed"})
    assert ora.evaluate(RECIPE_SPEC, x_best) == pytest.approx(50.0)
    assert ora.evaluate(RECIPE_SPEC, x_bad) == pytest.approx(40.0)  # -10 categorical penalty


def _recipe_gt(tmp_path):
    d = tmp_path / "e2e_recipe"
    d.mkdir()
    (d / "oracle.json").write_text(json.dumps(RECIPE_SPEC))
    return GroundTruth(root=tmp_path)


def test_oracle_eval_recipe_path(tmp_path, monkeypatch):
    gt = _recipe_gt(tmp_path)
    import hallmark.graders.extract as ex

    async def fake(messages, params):
        return json.dumps(fake.payload)

    monkeypatch.setattr(ex, "_extract_llm", fake)
    params = {"oracle_ref": "e2e_recipe/oracle"}

    # The answer text is a stub (extraction is faked) but must be NON-EMPTY: an empty
    # answer short-circuits before the extractor by design — see
    # test_extract_skips_llm_on_empty_answer below.
    stub = TurnTranscript(prompt="x", answer_text="proposed recipe: see fields")

    fake.payload = GOOD_RECIPE
    assert asyncio.run(OracleEvalGrader().grade(stub, params, gt))[0] == pytest.approx(1.0)
    fake.payload = {"Cr": 80, "Ni": 40, "aging_temp": 560, "heat_treatment": "solution_aged"}
    score, detail = asyncio.run(OracleEvalGrader().grade(stub, params, gt))
    assert score == 0.0 and "invalid" in detail


def test_extract_skips_llm_on_empty_answer(tmp_path, monkeypatch):
    """An empty answer must never reach the extractor.

    9 of Opus 5's 276 pilot cells were empty (the model burned its whole output budget on
    reasoning). Each one still made a network call whose only possible result was 'nothing
    extracted' — a wasted call and a needless source of grade-time nondeterminism.
    """
    gt = _recipe_gt(tmp_path)
    import hallmark.graders.extract as ex

    async def boom(messages, params):
        raise AssertionError("extractor must not be called on an empty answer")

    monkeypatch.setattr(ex, "_extract_llm", boom)
    params = {"oracle_ref": "e2e_recipe/oracle"}

    for empty in ("", "   \n\t "):
        score, _detail = asyncio.run(OracleEvalGrader().grade(
            TurnTranscript(prompt="x", answer_text=empty), params, gt))
        assert score == 0.0


def test_final_line_trailing_fractions_clause_stays_deterministic():
    """``Final: ... (all other element fractions = 0)`` must parse without the LLM.

    The trailing bookkeeping clause reads as a ``fractions=0`` pair; before the fix that
    unknown token marked the parse incomplete and sent a fully contract-compliant answer to
    the LLM extractor (observed on b3_supercon_maxelem seed 1).
    """
    from hallmark.graders.extract import _regex_extract_point

    answer = ("...reasoning...\n"
              "**Final: O=0.125, Cu=0.375, Ba=0.25, Ca=0.25** "
              "(all other element fractions = 0)")
    point, complete = _regex_extract_point(answer, ["O", "Cu", "Ba", "Ca", "Sr", "Y"])
    assert complete is True
    assert point == {"O": 0.125, "Cu": 0.375, "Ba": 0.25, "Ca": 0.25}


# --- Fix B: deterministic regex back-fill of dropped processing fields ---------

def test_find_labeled_number_tolerant():
    from hallmark.graders.extract import _find_labeled_number

    assert _find_labeled_number("Solutionizing temp ≈ 1017 °C", "solutionizing_temp") == 1017.0
    assert _find_labeled_number("solutionizing_temp | 954.38", "solutionizing_temp") == 954.38
    assert _find_labeled_number("no mention at all here", "solutionizing_temp") is None


def test_find_labeled_number_skips_prose_mention():
    # Fix 5: when a field is restated in prose AFTER the recipe, with a *word*
    # between its label and a number that belongs to a different field, the recipe
    # value must win — not the trailing prose number (the opus hidden_optimum bug,
    # where "aging_temp low, set Cr ~16" made aging_temp read 16 instead of 493).
    from hallmark.graders.extract import _find_labeled_number

    ans = ("...solutionizing_temp 1095, aging_temp 493, aging_time 1.65...\n"
           "The levers that matter: drive aging_time and aging_temp low, set Cr ~16.")
    assert _find_labeled_number(ans, "aging_temp") == 493.0
    # a clean separator/unit gap is still fine (not treated as prose)
    assert _find_labeled_number("aging_temp = 560 °C", "aging_temp") == 560.0
    # a lone wordy mention still resolves (fallback to last hit, no regression)
    assert _find_labeled_number("set the aging temp to 530", "aging_temp") == 530.0


def test_extract_design_backfills_dropped_proc_field(monkeypatch):
    import hallmark.graders.extract as ex

    async def fake(messages, params):  # the LLM extractor drops aging_temp
        return json.dumps({"Cr": 18, "Ni": 10, "heat_treatment": "solution_aged"})

    monkeypatch.setattr(ex, "_extract_llm", fake)
    answer = "Final recipe: Cr 18, Ni 10; Aging temp ≈ 560 °C; heat_treatment solution_aged"
    data = asyncio.run(ex.extract_design(answer, RECIPE_SPEC, {}))
    assert data["aging_temp"] == pytest.approx(560.0)  # recovered from the raw text

    # a processing field genuinely absent from the text stays absent (no invention)
    answer2 = "Cr 18, Ni 10; heat_treatment solution_aged"
    assert "aging_temp" not in asyncio.run(ex.extract_design(answer2, RECIPE_SPEC, {}))


# --- Fix 1/2/3: composition recovery + extraction-failure guard ----------------
# Regression for the bug where the LLM extractor silently dropped a stated alloy
# composition, collapsing the recipe to the balance element (pure Fe) and scoring
# a complete, good answer as 0 (curve_match opus/sonnet, oracle_eval opus).

def test_find_comp_value_domain_bounded():
    from hallmark.graders.extract import _find_comp_value

    # plain / separator / table-cell / tilde forms are all recovered
    assert _find_comp_value("Cr=14.825, Ni 18.057", "Cr", (0, 25)) == 14.825
    assert _find_comp_value("| Cr | 16.0 | | Mn | 1.6 |", "Cr", (0, 25)) == 16.0
    assert _find_comp_value("set Cr ~ 12 wt%", "Cr", (0, 25)) == 12.0
    # an out-of-domain prose number is rejected (a duration in years, not a wt%)
    assert _find_comp_value("we annealed Cr for 30 years", "Cr", (0, 25)) is None
    # a formula subscript is NOT read as a wt% (Cr2O3 -> 2 is skipped, recipe 16 wins)
    assert _find_comp_value("oxide Cr2O3 forms; final Cr 16", "Cr", (0, 25)) == 16.0
    # the symbol inside a word is not matched
    assert _find_comp_value("the Crystal structure", "Cr", (0, 25)) is None
    assert _find_comp_value("no chromium stated", "Cr", (0, 25)) is None


def test_regex_extract_design_recovers_full_recipe():
    import hallmark.graders.extract as ex

    answer = ("Proposed recipe: Cr = 17.4, Ni = 9.8; aging_temp 560 °C; "
              "heat_treatment: solution_aged")
    d = ex._regex_extract_design(answer, RECIPE_SPEC)
    assert d["Cr"] == 17.4 and d["Ni"] == 9.8
    assert d["aging_temp"] == 560.0
    assert d["heat_treatment"] == "solution_aged"


def test_extract_design_regex_backend_is_offline():
    # extract_backend="regex" must NOT call the network seam (deterministic grading).
    import hallmark.graders.extract as ex

    async def boom(messages, params):  # would fire if the LLM seam were hit
        raise AssertionError("regex backend must not call _extract_llm")

    ex_answer = "Cr 17.4, Ni 9.8, aging_temp 560, heat_treatment solution_aged"
    import unittest.mock as mock
    with mock.patch.object(ex, "_extract_llm", boom):
        d = asyncio.run(ex.extract_design(ex_answer, RECIPE_SPEC, {"extract_backend": "regex"}))
    assert d["Cr"] == 17.4 and d["Ni"] == 9.8 and d["heat_treatment"] == "solution_aged"


def test_extract_design_backfills_dropped_composition(monkeypatch):
    # The opus/sonnet bug: the LLM extractor returns processing but DROPS the whole
    # composition. The deterministic back-fill must recover the stated alloy wt%.
    import hallmark.graders.extract as ex

    async def fake(messages, params):
        return json.dumps({"aging_temp": 560, "heat_treatment": "solution_aged"})

    monkeypatch.setattr(ex, "_extract_llm", fake)
    answer = "Recipe: Cr = 17.4 wt%, Ni = 9.8 wt%; aging_temp 560 C; solution_aged"
    data = asyncio.run(ex.extract_design(answer, RECIPE_SPEC, {}))
    assert data["Cr"] == pytest.approx(17.4) and data["Ni"] == pytest.approx(9.8)


# NOTE: five further checks in the private repository assert mechanistic invariants of
# the synthetic oracles behind tasks that are NOT part of this release (the benchmark's
# private half), so they are absent here.

def test_assemble_recipe_empty_composition_is_no_proposal():
    # Fix 3: a recipe with NO alloying element stated is "no proposal", not pure Fe.
    x, note = ora.assemble_recipe(
        RECIPE_SPEC, {"aging_temp": 560, "heat_treatment": "solution_aged"})
    assert x is None and "no proposed composition" in note
    # but a single stated element is enough to be a real (if lean) recipe
    x2, _ = ora.assemble_recipe(
        RECIPE_SPEC, {"Cr": 18, "aging_temp": 560, "heat_treatment": "solution_aged"})
    assert x2 is not None and x2["Ni"] == 0.0


# --- target sense (fixed 2026-07-23; the branch was dead until then) -----------

def test_score_target_uses_tolerance():
    # Before the fix `oracle_eval` called `_score` without `tol`, so the target branch fell
    # through to a float equality test and could only return 0.0 or 1.0.
    assert _score("target", 100.0, 100.0, 0.0, tol=10.0) == pytest.approx(1.0)
    assert _score("target", 105.0, 100.0, 0.0, tol=10.0) == pytest.approx(0.5)
    assert _score("target", 95.0, 100.0, 0.0, tol=10.0) == pytest.approx(0.5)
    assert _score("target", 120.0, 100.0, 0.0, tol=10.0) == 0.0        # beyond tolerance
    assert _score("target", 100.5, 100.0, 0.0) == 0.0                  # no tol -> exact only


def test_grader_forwards_target_tol(tmp_path):
    spec = dict(SPEC, sense="target", f_star=13.0, target_tol=2.0)
    d = tmp_path / "e2e_target_demo"
    d.mkdir()
    (d / "oracle.json").write_text(json.dumps(spec))
    gt = GroundTruth(root=tmp_path)
    g = OracleEvalGrader()
    # f(Cr=12, Ni=18) = 15.0 -> miss of 2.0 == tol -> exactly 0
    s, _ = asyncio.run(g.grade(
        TurnTranscript(index=0, prompt="", answer_text="Cr 12, Ni 18"),
        {"oracle_ref": "e2e_target_demo/oracle", "extract_backend": "regex"}, gt))
    assert s == pytest.approx(0.0, abs=1e-9)
    # a point whose f misses T by less than the tolerance scores partial credit
    s2, _ = asyncio.run(g.grade(
        TurnTranscript(index=0, prompt="", answer_text="Cr 12, Ni 15.6"),
        {"oracle_ref": "e2e_target_demo/oracle", "extract_backend": "regex"}, gt))
    assert 0.0 < s2 < 1.0


def test_recipe_target_path_scores_the_shipped_gold(monkeypatch):
    # The ONLY shipped `sense: target` gold is a RECIPE task, so it grades through
    # `_grade_recipe` — the second call site. The point-path test above would stay
    # green if only that one site had been fixed, so pin the recipe path against the
    # real gold. Extraction is stubbed: what is under test is scoring, not the parser.
    from hallmark.graders import oracle_eval as oe

    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "ground_truth" / "e2e_target_point" / "oracle.json").read_text())
    assert spec["sense"] == "target" and spec["target_tol"] > 0

    def _stub(design):
        async def _f(answer, s, params):
            return dict(design)
        return _f

    gt = GroundTruth(root=root / "ground_truth")
    params = {"oracle_ref": "e2e_target_point/oracle"}
    t = TurnTranscript(index=0, prompt="", answer_text="(stubbed)")
    x_star = {k: v for k, v in spec["x_star"].items() if k != "Fe"}  # Fe is the balance

    monkeypatch.setattr(oe, "extract_design", _stub(x_star))
    s_hit, detail = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert s_hit == pytest.approx(1.0, abs=0.01)   # T is reachable: f(x*) ≈ T
    assert "f(recipe)=" in detail                  # ...and it went through _grade_recipe

    # A near miss (|f − T| ≈ 71 < tol ≈ 81) must earn partial credit. This is the
    # assertion the pre-2026-07-23 code failed: without `tol` it returned a flat 0.0.
    monkeypatch.setattr(oe, "extract_design", _stub(dict(x_star, Ni=x_star["Ni"] + 20)))
    s_near, _ = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert 0.0 < s_near < 0.5


def test_stray_target_tol_does_not_touch_a_max_gold(tmp_path):
    # Regression guard for the tol forwarding: `tol` is read on the target branch only,
    # so a `sense: max` gold carrying a stray `target_tol` must score exactly as before.
    d = tmp_path / "e2e_max_demo"
    d.mkdir()
    (d / "oracle.json").write_text(json.dumps(dict(SPEC, target_tol=2.0)))
    gt = GroundTruth(root=tmp_path)
    t = TurnTranscript(index=0, prompt="", answer_text="Cr 12, Ni 18")
    params = {"oracle_ref": "e2e_max_demo/oracle", "extract_backend": "regex"}
    s, _ = asyncio.run(OracleEvalGrader().grade(t, params, gt))
    assert s == pytest.approx(1.0)  # f = f* = 15.0 on the max path, tol ignored
