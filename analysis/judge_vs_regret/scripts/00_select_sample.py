"""Step 0 — build the stratified sample: N_TASKS tasks x PER_TASK cached answers.

Blocked by task, not a flat random draw: judge scores are only comparable *within*
a task (each task has its own regret scale and its own pool), and within-task
pairwise accuracy is a headline metric.

Eligibility (a task): >= PER_TASK scored cells, score spread >= 0.4, at least one
cell <= 0.25 and one >= 0.60.  Within a task: low / mid / high by objective score,
each from a *different* system, so the style effect is not confounded with one
system's house style.

olympus_design cells are kept only when the answer's ``Final:`` line parses
completely offline (``_regex_extract_point`` -> complete), otherwise the grader
falls back to the LLM extractor and the "objective moves by exactly 0" claim
would not hold.

Writes data/sample.json.  Deterministic given SEED.
"""
from __future__ import annotations

import collections
import json
import random
import statistics
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA,
    N_TASKS,
    PER_TASK,
    SEED,
    STRATUM_D,
    answer_text,
    ensure_dirs,
    family,
    gold,
    ledger,
    public_tasks,
)

# quota by grader, proportional to the stratum (29 b3_select / 8 olympus / 4 hv)
QUOTA = {"b3_select": 9, "olympus_design": 3, "select_hypervolume": 2}
MIN_SPREAD, LOW_MAX, HIGH_MIN = 0.4, 0.25, 0.60


def _grader_of(result: dict) -> str | None:
    gs = {g["grader"] for t in result["turns"] for g in t["grader_results"]}
    return next(iter(gs)) if len(gs) == 1 and gs <= STRATUM_D else None


def _deterministic(task_id: str, grader: str, answer: str) -> bool:
    """True when the objective grader reads this answer without any LLM call."""
    if grader != "olympus_design":
        return True  # parse_ids has no LLM fallback at all
    from hallmark import oracle as ora
    from hallmark.graders.extract import _regex_extract_point

    spec = gold(task_id)
    _, complete = _regex_extract_point(answer, ora.variable_names(spec))
    return complete


def collect_cells() -> list[dict]:
    pub = public_tasks()
    out = []
    for r in ledger():
        if r["task_id"] not in pub or r.get("task_score") is None:
            continue
        grader = _grader_of(r)
        if grader is None:
            continue
        ans = answer_text(r["system_id"], r["task_id"], r["seed_idx"])
        if not ans or not _deterministic(r["task_id"], grader, ans):
            continue
        out.append(
            dict(
                task_id=r["task_id"],
                grader=grader,
                family=family(r["task_id"]),
                system_id=r["system_id"],
                seed_idx=int(r["seed_idx"]),
                objective=float(r["task_score"]),
                answer=ans,
            )
        )
    return out


def eligible(by_task: dict[str, list[dict]]) -> dict[str, list[dict]]:
    keep = {}
    for tid, cells in by_task.items():
        s = [c["objective"] for c in cells]
        if len(cells) < PER_TASK or max(s) - min(s) < MIN_SPREAD:
            continue
        if min(s) > LOW_MAX or max(s) < HIGH_MIN:
            continue
        if len({c["system_id"] for c in cells}) < PER_TASK:
            continue
        keep[tid] = cells
    return keep


def pick_tasks(elig: dict[str, list[dict]], rng: random.Random) -> list[str]:
    """Fill each grader's quota, round-robin over families for spread."""
    by_grader: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for tid, cells in elig.items():
        by_grader[cells[0]["grader"]][cells[0]["family"]].append(tid)

    chosen: list[str] = []
    deficit = 0
    for grader, quota in QUOTA.items():
        fams = {f: sorted(ts) for f, ts in by_grader.get(grader, {}).items()}
        for ts in fams.values():
            rng.shuffle(ts)
        picked: list[str] = []
        while len(picked) < quota and any(fams.values()):
            for f in sorted(fams):
                if fams[f] and len(picked) < quota:
                    picked.append(fams[f].pop())
        deficit += quota - len(picked)
        chosen += picked

    # spill any unmet quota into whatever eligible tasks are left (largest pool first)
    if deficit:
        rest = sorted(set(elig) - set(chosen))
        rng.shuffle(rest)
        chosen += rest[:deficit]
    return sorted(chosen)


def pick_cells(cells: list[dict]) -> list[dict]:
    """low / high / mid by objective score, each from a different system."""
    ordered = sorted(cells, key=lambda c: c["objective"])
    low = ordered[0]
    high = next(c for c in reversed(ordered) if c["system_id"] != low["system_id"])
    med = statistics.median([c["objective"] for c in ordered])
    used = {low["system_id"], high["system_id"]}
    mid = min(
        (c for c in ordered if c["system_id"] not in used),
        key=lambda c: abs(c["objective"] - med),
    )
    return [dict(low, level="low"), dict(mid, level="mid"), dict(high, level="high")]


def main() -> None:
    ensure_dirs()
    rng = random.Random(SEED)

    cells = collect_cells()
    by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cells:
        by_task[c["task_id"]].append(c)

    elig = eligible(by_task)
    tasks = pick_tasks(elig, rng)[:N_TASKS]

    items = []
    for tid in tasks:
        for c in pick_cells(elig[tid]):
            c["uid"] = f"{c['task_id']}|{c['system_id']}|{c['seed_idx']}"
            items.append(c)

    payload = {
        "seed": SEED,
        "n_tasks": len(tasks),
        "per_task": PER_TASK,
        "criteria": {"min_spread": MIN_SPREAD, "low_max": LOW_MAX, "high_min": HIGH_MIN},
        "pool": {
            "cells_stratum_d": len(cells),
            "tasks_stratum_d": len(by_task),
            "tasks_eligible": len(elig),
        },
        "items": items,
    }
    (DATA / "sample.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    print(f"stratum-D public cells: {len(cells)} over {len(by_task)} tasks")
    print(f"eligible tasks: {len(elig)} -> chosen {len(tasks)}, answers {len(items)}")
    for tid in tasks:
        got = [i for i in items if i["task_id"] == tid]
        scores = ", ".join(f"{i['level']}={i['objective']:.2f}" for i in got)
        print(f"  {tid:42s} {got[0]['grader']:19s} {scores}")


if __name__ == "__main__":
    main()
