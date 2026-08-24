"""NLO selection family: the `select_hypervolume` grader + the committed gold files.

Fully offline. The grader half uses a hand-written gold under ``tmp_path``; the consistency half
reads the committed ``ground_truth/_nlo_snapshot/*.json`` and re-derives every scalar in it from
the candidate labels, so a bad rebuild of `scripts/build_nlo_tasks.py` cannot ship silently.
No model is fitted anywhere -- the judge is the real DFT label of a held-out candidate.
"""
from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import pytest

from hallmark.graders.select_hypervolume import SelectHypervolumeGrader, _hv2d
from hallmark.loader import GroundTruth, load_task
from hallmark.models import TurnTranscript

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "ground_truth" / "_nlo_snapshot"
TASKS = REPO / "tasks" / "end_to_end"
K = 5


# --------------------------------------------------------------------------- #
# select_hypervolume grader                                                    #
# --------------------------------------------------------------------------- #
# A synthetic front: c000..c004 trade off along x + y = 1.2; c005..c009 are strictly dominated.
# The K=3 HV-optimal subset spans the front; 3 points bunched on one axis lose.
def _pareto_gold(k: int = 3) -> dict:
    front = {f"c{i:03d}": [0.2 + 0.2 * i, 1.0 - 0.2 * i] for i in range(5)}  # (.2,1) .. (1,.2)
    inner = {f"c{i:03d}": [0.1, 0.1] for i in range(5, 10)}  # strictly dominated
    pts = {**front, **inner}
    ref = [0.0, 0.0]
    hv_star = max(
        _hv2d([tuple(pts[i]) for i in c], ref) for c in itertools.combinations(front, k)
    )
    return {
        "goal": "hypervolume",
        "objectives": ["shg", "eg_hse"],
        "k": k,
        "ref_point": ref,
        "candidates": {i: {"p": p, "y": [p[0] * 100, p[1] * 6]} for i, p in pts.items()},
        "front_ids": list(front),
        "hv_star": hv_star,
        "floor": 0.02,
    }


def _hv_optimal(gold: dict) -> tuple[str, ...]:
    """The K-subset of the front that attains `hv_star` (which subset it is depends on the ref
    point, so the test derives it instead of hard-coding a guess)."""
    c, ref, k = gold["candidates"], gold["ref_point"], gold["k"]
    return max(
        itertools.combinations(gold["front_ids"], k),
        key=lambda s: _hv2d([tuple(c[i]["p"]) for i in s], ref),
    )


def _gt(tmp_path: Path, gold: dict) -> GroundTruth:
    d = tmp_path / "_nlo_snapshot"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pareto_gold.json").write_text(json.dumps(gold), encoding="utf-8")
    return GroundTruth(tmp_path)


def _grade(gt: GroundTruth, text: str, params: dict | None = None):
    p = {"oracle_ref": "_nlo_snapshot/pareto_gold", **(params or {})}
    return asyncio.run(
        SelectHypervolumeGrader().grade(TurnTranscript(prompt="p", answer_text=text), p, gt)
    )


def test_hv_optimal_front_subset_scores_one(tmp_path):
    gold = _pareto_gold()
    gt = _gt(tmp_path, gold)
    assert _grade(gt, "Final: " + ", ".join(_hv_optimal(gold)))[0] == pytest.approx(1.0)


def test_spanning_front_beats_single_axis(tmp_path):
    """The whole point of the task: 3 mutually non-dominated picks beat 3 stacked on one axis."""
    gold = _pareto_gold()
    gt = _gt(tmp_path, gold)
    spread = _grade(gt, "Final: " + ", ".join(_hv_optimal(gold)))[0]
    one_axis = _grade(gt, "Final: c004, c005, c006")[0]  # c004 is the x-extreme; rest dominated
    assert spread > one_axis


def test_dominated_picks_score_zero(tmp_path):
    """All-dominated picks land below the blind-pick floor -> clamped to 0."""
    gt = _gt(tmp_path, _pareto_gold())
    assert _grade(gt, "Final: c005, c006, c007")[0] == 0.0


def test_no_valid_ids_scores_zero(tmp_path):
    gt = _gt(tmp_path, _pareto_gold())
    score, detail = _grade(gt, "I decline to choose.")
    assert score == 0.0 and "no valid" in detail


def test_extra_picks_truncated_to_k(tmp_path):
    """parse_ids caps at k, so a SUT cannot inflate its hypervolume by listing the whole pool."""
    gt = _gt(tmp_path, _pareto_gold())
    everything = "Final: " + ", ".join(f"c{i:03d}" for i in range(10))
    assert _grade(gt, everything)[0] == _grade(gt, "Final: c000, c001, c002")[0]


def test_floor_param_overrides_gold(tmp_path):
    """Raising the 0-score anchor above a mediocre set's hypervolume drops it to 0."""
    gt = _gt(tmp_path, _pareto_gold())
    picks = "Final: c000, c005, c006"  # HV = 0.2 * 1.0, the rest dominated
    assert 0.0 < _grade(gt, picks)[0] < 1.0
    assert _grade(gt, picks, {"floor": 0.4})[0] == 0.0


# --------------------------------------------------------------------------- #
# committed gold files — every scalar re-derived from the real labels          #
# --------------------------------------------------------------------------- #
def _gold(name: str) -> dict:
    return json.loads((SNAP / f"{name}.json").read_text())


@pytest.mark.parametrize("name", ["dn_select_gold", "shg_select_gold", "transparent_gold"])
def test_max_gold_scalars_match_labels(name):
    g = _gold(name)
    ys = sorted(v["y"] for v in g["candidates"].values())
    assert g["goal"] == "max"
    assert len(ys) == 100 and g["k"] == K
    assert g["top_k_mean"] == pytest.approx(sum(ys[-K:]) / K)
    assert g["floor"] == pytest.approx(sum(ys) / len(ys))
    assert g["top_k_mean"] > g["floor"]  # non-degenerate span


@pytest.mark.parametrize("name", ["dn_target_gold", "shg_target_gold"])
def test_target_gold_scalars_match_labels(name):
    g = _gold(name)
    T = g["target_value"]
    d = sorted(abs(v["y"] - T) for v in g["candidates"].values())
    assert g["goal"] == "target"
    assert g["top_k_mean"] == pytest.approx(sum(d[:K]) / K)  # best achievable mean |y - T|
    assert g["floor"] == pytest.approx(sum(d) / len(d))  # blind-pick anchor
    assert g["floor"] > g["top_k_mean"]


def test_transparent_gold_zeroes_infeasible_only():
    """A violating candidate must carry y == 0; a feasible one must carry its real |d_ij|."""
    g = _gold("transparent_gold")
    zeros = [i for i, v in g["candidates"].items() if v["y"] == 0.0]
    assert g["constraint"] == {"column": "eg_hse", "op": ">=", "value": 3.0}
    assert len(zeros) == 100 - g["n_feasible"]
    assert g["n_feasible"] >= 12


def test_pareto_gold_front_and_ceiling():
    g = _gold("pareto_gold")
    c, ref = g["candidates"], g["ref_point"]
    assert len(g["front_ids"]) > K, "front must exceed K or the task is 'return the whole front'"

    # front_ids really are the non-dominated points
    P = {i: v["p"] for i, v in c.items()}
    for i, p in P.items():
        dominated = any(
            q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]) for q in P.values()
        )
        assert dominated != (i in g["front_ids"])

    # hv_star is the exact best-K subset of the front, and it is reachable
    best = max(
        _hv2d([tuple(P[i]) for i in s], ref)
        for s in itertools.combinations(g["front_ids"], K)
    )
    assert g["hv_star"] == pytest.approx(best)
    assert g["hv_star"] > g["floor"]


def test_normalized_points_agree_with_raw_labels():
    """`p` must be the min-max normalization of `y` under the frozen `norm` ranges."""
    g = _gold("pareto_gold")
    for axis, obj in enumerate(g["objectives"]):
        lo, hi = g["norm"][obj]
        for v in g["candidates"].values():
            assert v["p"][axis] == pytest.approx((v["y"][axis] - lo) / (hi - lo))
        assert min(v["p"][axis] for v in g["candidates"].values()) == pytest.approx(0.0)
        assert max(v["p"][axis] for v in g["candidates"].values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# task wiring                                                                  #
# --------------------------------------------------------------------------- #
NLO_TASKS = sorted(TASKS.glob("nlo_*.yaml"))


def test_twelve_dev_tasks():
    assert len(NLO_TASKS) == 12
    assert sum("_zeroshot" in p.name for p in NLO_TASKS) == 6


@pytest.mark.parametrize("path", NLO_TASKS, ids=lambda p: p.stem)
def test_task_loads_and_points_at_a_real_gold(path):
    t = load_task(path)
    g = t.turns[0].grading[0]
    assert g.grader in ("b3_select", "select_hypervolume")
    GroundTruth().resolve(g.params["oracle_ref"])  # raises if the gold is missing
    assert t.inputs.files[-1] == "nlo_pool.csv"  # every task sees the same 100 candidates
    assert "Final:" in t.turns[0].prompt


@pytest.mark.parametrize("path", [p for p in NLO_TASKS if "_zeroshot" in p.name], ids=lambda p: p.stem)
def test_zeroshot_twin_shares_gold_and_sees_pool_only(path):
    """The ablation is only interpretable if the twins differ *solely* by the training table."""
    zs = load_task(path)
    trained = load_task(path.with_name(path.name.replace("_zeroshot", "")))
    assert zs.inputs.files == ["nlo_pool.csv"]
    assert len(trained.inputs.files) == 2
    assert zs.turns[0].grading[0].params == trained.turns[0].grading[0].params


def test_pool_csv_carries_no_labels():
    header = (REPO / "fixtures" / "nlo_pool.csv").read_text().split("\n", 1)[0]
    assert header == "candidate_id,formula,poscar"


def test_dn_train_does_not_leak_shg():
    """A +0.40 correlation with dn would be a free hint, so the dn tasks must not see `shg`."""
    header = (REPO / "fixtures" / "nlo_dn_train.csv").read_text().split("\n", 1)[0]
    assert header == "formula,poscar,dn"
