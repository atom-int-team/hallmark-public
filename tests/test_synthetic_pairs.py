"""Offline invariants of the three matched synthetic pairs (`scripts/build_synthetic_pairs.py`).

These assert properties of the SHIPPED artifacts, so they fail if a rebuild silently changes
the scale — which is the failure mode that produced the four understated `f*` values on the
constrained B3 tasks. Everything here is pure numpy/json: no network, no keys, no rebuild.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hallmark import oracle as ora

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "ground_truth"
FIX = ROOT / "fixtures"
TASKS = ROOT / "tasks" / "end_to_end"

PAIRS = [  # (point task, select task, fixture stem)
    ("e2e_oracle_point", "e2e_oracle_select", "alloy_pair_max"),
    ("e2e_multi_constraint", "e2e_multi_constraint_select", "alloy_pair_constrained"),
    ("e2e_target_point", "e2e_target_select", "alloy_pair_target"),
]
POINTS = [p for p, _, _ in PAIRS]
SELECTS = [s for _, s, _ in PAIRS]


def _spec(task_id: str) -> dict:
    return json.loads((GT / task_id / "oracle.json").read_text())


def _gold(task_id: str) -> dict:
    return json.loads((GT / task_id / "select_gold.json").read_text())


@pytest.mark.parametrize("task_id", POINTS)
def test_point_gold_anchors_at_the_median(task_id):
    spec = _spec(task_id)
    assert "train_median" in spec, "new tasks must anchor at the median, not the best row"
    assert "train_best" not in spec, "two anchors in one gold would be read inconsistently"


@pytest.mark.parametrize("task_id", POINTS)
def test_recorded_optimum_is_reachable_and_consistent(task_id):
    """f at the recorded x* must equal the recorded f* — for the target task, equal T.

    This is the guard against the defect that understated f* on four shipped tasks: an optimum
    search that never saw the real sampled points can return a value the data itself beats.
    """
    spec = _spec(task_id)
    x, note = ora.assemble_recipe(spec, dict(spec["x_star"]))
    assert x is not None, note
    f = ora.evaluate(spec, x)
    if spec.get("sense") == "target":
        assert f == pytest.approx(float(spec["f_star"]), abs=0.5)   # x* SOLVES the task
        assert float(spec["target_tol"]) > 0
    else:
        assert f == pytest.approx(float(spec["f_star"]), rel=1e-6)
        assert float(spec["f_star"]) > float(spec["train_median"])


@pytest.mark.parametrize("task_id", POINTS)
def test_no_training_row_beats_the_optimum(task_id):
    """The scale is only meaningful if f* really is the ceiling of what the table contains."""
    spec = _spec(task_id)
    stem = next(s for p, _, s in PAIRS if p == task_id)
    rows = (FIX / f"{stem}_train.csv").read_text().splitlines()
    header = rows[0].split(",")
    fs = []
    for line in rows[1:]:
        vals = line.split(",")
        rec = {}
        for k, v in zip(header, vals, strict=True):
            if k == "target":
                continue
            rec[k] = v if k in spec.get("categorical", {}) else float(v)
        fs.append(ora.evaluate(spec, rec))
    if spec.get("sense") == "target":       # T must sit ABOVE the table, else copying wins
        assert float(spec["f_star"]) > max(fs) + float(spec["target_tol"])
    else:
        assert max(fs) < float(spec["f_star"])


@pytest.mark.parametrize("task_id", SELECTS)
def test_select_gold_shape(task_id):
    gold = _gold(task_id)
    assert gold["k"] == 5
    assert len(gold["candidates"]) == 100
    assert set(gold["candidates"]) == {f"c{i:03d}" for i in range(100)}
    ys = [c["y"] for c in gold["candidates"].values()]
    assert gold["floor"] == pytest.approx(sum(ys) / len(ys), abs=1e-3)   # blind-pick expectation
    if gold["goal"] == "target":
        assert "target_value" in gold
    else:
        assert gold["top_k_mean"] > gold["floor"]


def test_constrained_pool_contains_violators_and_they_are_worthless():
    """The constraint only bites in the select form if violating candidates are in the pool
    AND are folded into the label as zero — the b3/nlo/htsteel convention."""
    gold = _gold("e2e_multi_constraint_select")
    zeros = [i for i, c in gold["candidates"].items() if c["y"] == 0.0]
    assert 15 <= len(zeros) <= 45, f"{len(zeros)} violators — outside the declared window"


def test_target_pool_brackets_the_target():
    """Candidates must sit on BOTH sides of T, else "closest to T" degenerates to "largest"."""
    gold = _gold("e2e_target_select")
    T = float(gold["target_value"])
    ys = [c["y"] for c in gold["candidates"].values()]
    assert sum(1 for y in ys if y > T) >= 10
    assert sum(1 for y in ys if y < T) >= 10


@pytest.mark.parametrize("task_id", POINTS + SELECTS)
def test_task_yaml_declares_the_right_grader(task_id):
    text = (TASKS / f"{task_id}.yaml").read_text()
    assert ("grader: b3_select" if task_id in SELECTS else "grader: oracle_eval") in text
    assert "checklist_judge" not in text          # no LLM in the grading chain
    assert "Final: c901" in text or task_id in POINTS   # impossible ids in the format example


def test_the_three_oracles_are_distinct():
    """A shared optimum would fuse the blocks into one atomic block and deflate the bootstrap."""
    stars = []
    for task_id in POINTS:
        spec = _spec(task_id)
        stars.append(tuple(round(float(v), 3) for k, v in sorted(spec["x_star"].items())
                           if not isinstance(v, str)))
    assert len(set(stars)) == 3
    f_stars = {float(_spec(t)["f_star"]) for t in POINTS}
    assert len(f_stars) == 3
