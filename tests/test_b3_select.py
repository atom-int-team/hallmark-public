"""B3 selection: parse_ids (deterministic-first id parse) + b3_select grader.

Fully offline: a committed ``select_gold.json`` under tmp_path is resolved by the grader;
the real held-out labels live only there, never in the answer. ``parse_ids`` has no network
seam at all since 2026-07-23 — an answer with no deterministically readable id scores 0.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hallmark.graders.b3_select import B3SelectGrader
from hallmark.graders.parse_ids import parse_ids
from hallmark.loader import GroundTruth
from hallmark.models import TurnTranscript

VALID = {f"c{i:03d}" for i in range(10)}
P: dict = {}


def _ids(text, k=5, valid=VALID, params=P):
    return asyncio.run(parse_ids(text, set(valid), k, params))


# --------------------------------------------------------------------------- #
# parse_ids — deterministic paths                                             #
# --------------------------------------------------------------------------- #
def test_final_line_basic():
    assert _ids("reasoning...\nFinal: c003, c007, c001") == ["c003", "c007", "c001"]


def test_final_line_truncates_to_k():
    assert _ids("Final: c000 c001 c002 c003 c004 c005 c006", k=3) == ["c000", "c001", "c002"]


def test_fewer_than_k_ok():
    assert _ids("Final: c009, c008", k=5) == ["c009", "c008"]


def test_dedup_and_filter_junk():
    # duplicate c001 collapsed; "zzz" / "c999" not in VALID are dropped
    assert _ids("Final: c001, zzz, c001, c999, c002") == ["c001", "c002"]


def test_last_final_wins():
    # a draft list then a corrected final restatement -> the last one is authoritative
    assert _ids("Final: c000, c001\n...revised...\nFinal: c005, c006") == ["c005", "c006"]


def test_json_array_fallback():
    # no "Final:" line -> parse an embedded JSON array
    assert _ids('My picks: ["c004", "c002"]') == ["c004", "c002"]


def test_markdown_bold_final():
    assert _ids("**Final:** c008; c003") == ["c008", "c003"]


def test_markdown_bold_wraps_whole_final_line():
    # the bold closes AFTER the last id -> "c003**" must not be dropped as invalid
    assert _ids("**Final: c008, c003**") == ["c008", "c003"]


def test_ids_wrapped_in_punctuation():
    assert _ids("Final: `c008`, (c003), c001.") == ["c008", "c003", "c001"]


def test_prose_without_a_final_line_yields_nothing():
    """No LLM rescue: naming candidates in prose is not a deliverable (see parse_ids docstring)."""
    assert _ids("I think the sixth and second candidates look strongest.") == []


def test_echo_of_the_v1_format_example_yields_nothing():
    """The v1 example uses unreachable ids, so echoing it parses to an empty selection."""
    assert _ids("Final: c901, c902, c903, c904, c905") == []


def test_empty_when_nothing_valid():
    assert _ids("no ids here at all") == []


# --------------------------------------------------------------------------- #
# b3_select grader — normalized regret vs real held-out labels                #
# --------------------------------------------------------------------------- #
# candidates c000..c009 with y = 0,10,...,90.  top5_mean=70, floor(mean)=45.
def _gold(goal="max"):
    return {
        "goal": goal, "target": "yield strength (MPa)", "k": 5,
        "candidates": {f"c{i:03d}": {"y": float(i * 10)} for i in range(10)},
        "top_k_mean": 70.0, "floor": 45.0,
    }


def _gt(tmp_path: Path, gold: dict) -> GroundTruth:
    d = tmp_path / "_b3_snapshot" / "steels"
    d.mkdir(parents=True, exist_ok=True)
    (d / "select_gold.json").write_text(json.dumps(gold), encoding="utf-8")
    return GroundTruth(tmp_path)


def _grade(gt, text, params=None):
    p = {"oracle_ref": "_b3_snapshot/steels/select_gold", **(params or {})}
    return asyncio.run(B3SelectGrader().grade(TurnTranscript(prompt="p", answer_text=text), p, gt))


def test_top5_scores_one(tmp_path):
    gt = _gt(tmp_path, _gold())
    assert _grade(gt, "Final: c009, c008, c007, c006, c005")[0] == 1.0


def test_worst5_scores_zero(tmp_path):
    gt = _gt(tmp_path, _gold())
    assert _grade(gt, "Final: c000, c001, c002, c003, c004")[0] == 0.0


def test_mid_pick_between(tmp_path):
    gt = _gt(tmp_path, _gold())
    # picks 40,50,60,70,80 -> mean 60 -> (60-45)/(70-45)=0.6
    score, detail = _grade(gt, "Final: c004, c005, c006, c007, c008")
    assert abs(score - 0.6) < 1e-9 and "picked 5/5" in detail


def test_no_valid_pick_zero(tmp_path):
    gt = _gt(tmp_path, _gold())
    assert _grade(gt, "Final: zzz, nope")[0] == (0.0)


def test_min_goal(tmp_path):
    # minimize: best picks are the 5 lowest -> mean 20 == top_k_mean (min) -> 1.0
    gold = {**_gold("min"), "top_k_mean": 20.0}  # 5 lowest = 0,10,20,30,40 -> mean 20
    gt = _gt(tmp_path, gold)
    assert _grade(gt, "Final: c000, c001, c002, c003, c004")[0] == 1.0
    assert _grade(gt, "Final: c009, c008, c007, c006, c005")[0] == 0.0


# --- under-picking: unfilled slots are anchor-valued, never "fewer slots" ------------- #
# Regression for the 2026-07-21 fix. Before it, the mean was over len(picks), so naming ONE
# top candidate beat an honest 5-pick; one shipped run cell named 2/5 and scored 1.0.
def test_single_best_pick_cannot_beat_full_house(tmp_path):
    gt = _gt(tmp_path, _gold())
    lone, _ = _grade(gt, "Final: c009")            # the single best candidate, y=90
    full, _ = _grade(gt, "Final: c009, c008, c007, c006, c005")  # the true top 5 -> 1.0
    assert lone < full == 1.0
    # 1 slot of 5 filled with the best pick: anchor + (1/5)*(90-45) over the 25-wide span
    assert abs(lone - (1 / 5) * (90.0 - 45.0) / (70.0 - 45.0)) < 1e-9


def test_under_pick_is_proportional_partial_credit(tmp_path):
    gt = _gt(tmp_path, _gold())
    # the same picks, 2 of 5 slots: exactly 2/5 of the score those picks earn on their own
    part, detail = _grade(gt, "Final: c009, c008")
    assert "picked 2/5" in detail
    assert abs(part - (2 / 5) * ((90.0 + 80.0) / 2 - 45.0) / (70.0 - 45.0)) < 1e-9


def test_under_pick_target_mode_also_padded(tmp_path):
    gt = _gt(tmp_path, _gold_target(45.0))
    # one perfect pick (y=40, dist 5) out of 5 slots; the rest count as a blind pick (25)
    score, detail = _grade(gt, "Final: c004")
    got = (5.0 + 4 * 25.0) / 5
    assert "picked 1/5" in detail
    assert abs(score - (25.0 - got) / (25.0 - 13.0)) < 1e-9


def test_full_pick_unchanged_by_the_fix(tmp_path):
    # n == k must be byte-identical to the old sum/len path (no silent rescaling)
    gt = _gt(tmp_path, _gold())
    assert abs(_grade(gt, "Final: c004, c005, c006, c007, c008")[0] - 0.6) < 1e-9


def test_floor_override(tmp_path):
    gt = _gt(tmp_path, _gold())
    # override floor to 0 -> picks 40..80 mean 60 -> (60-0)/(70-0) ~ 0.857
    score, _ = _grade(gt, "Final: c004, c005, c006, c007, c008", {"floor": 0.0})
    assert abs(score - 60.0 / 70.0) < 1e-9


# --------------------------------------------------------------------------- #
# b3_select target mode — pick the K candidates closest to a target value     #
# --------------------------------------------------------------------------- #
# T=45: |y-45| over y=0..90 = 45,35,25,15,5,5,15,25,35,45. sorted 5-closest mean=13,
# blind (all) mean=25 -> regret span 12. Picks nearest 45 (y=40,50,30,60,20) -> mean_dist 13 -> 1.0.
def _gold_target(T=45.0):
    return {"goal": "target", "target": "band gap (eV)", "target_value": T, "k": 5,
            "candidates": {f"c{i:03d}": {"y": float(i * 10)} for i in range(10)}}


def test_target_closest_scores_one(tmp_path):
    gt = _gt(tmp_path, _gold_target(45.0))
    score, detail = _grade(gt, "Final: c004, c005, c003, c006, c002")
    assert score == 1.0 and "mean|y-T|" in detail


def test_target_far_scores_zero(tmp_path):
    gt = _gt(tmp_path, _gold_target(45.0))
    # the extremes (y=0,90,10,80,20) are as far from 45 as a blind pick -> 0
    assert _grade(gt, "Final: c000, c009, c001, c008, c002")[0] == 0.0


def test_target_partial(tmp_path):
    gt = _gt(tmp_path, _gold_target(45.0))
    # picks y=40,50,30,60,90 -> dists 5,5,15,15,45 -> mean 17; (25-17)/(25-13)=0.667
    score, _ = _grade(gt, "Final: c004, c005, c003, c006, c009")
    assert abs(score - (25.0 - 17.0) / (25.0 - 13.0)) < 1e-9
