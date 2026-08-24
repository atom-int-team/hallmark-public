"""Step 4 — score every variant with three judges, in two conditions.

ref_free   — the judge sees the task prompt and the answer. The realistic setting:
             if the optimum were known, nobody would need a judge.
ref_based  — the judge additionally sees the ground truth (true top-K and their
             held-out labels, or x*/f*). Strictly more information than the SUT had.

Contract copied from hallmark/graders/judge.py: 0-10 integer in a JSON object,
temperature 0, parsed by the same ``parse_score``. Resumable — (uid, style, judge,
condition) already in data/judge_scores.jsonl is never re-billed.

Run:  python analysis/judge_vs_regret/scripts/04_run_judges.py [--limit N] [--judges a,b]
                                                          [--conditions ref_free,ref_based]
                                                          [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    CONDITIONS,
    DATA,
    JUDGES,
    append_jsonl,
    ensure_dirs,
    gold,
    grading_spec,
    latest_variants,
    read_jsonl,
    task_prompt,
)

CONCURRENCY = 8

SYSTEM = (
    "You are a strict grader of an AI system's answer in a materials-science design task. "
    "Return EXACTLY one JSON object and nothing else: "
    '{"score": <integer 0-10>, "rationale": "<one or two sentences>"}. '
    "Do not play the assistant's role, do not continue the dialogue, no text outside the JSON. "
    "score: 0 - failure, 5 - acceptable, 10 - exemplary."
)

RUBRIC = (
    "Rate the answer 0-10 on:\n"
    "- correctness of the proposed material choice for the stated objective;\n"
    "- quality of the reasoning from the data the task provided;\n"
    "- whether the recommendation is well-supported rather than asserted.\n"
    "0 - failure, 5 - acceptable, 10 - exemplary."
)

USER = """## Task given to the system
{prompt}

## System answer (to grade)
{answer}
{reference}
## Evaluation criteria
{rubric}
"""


def reference_block(task_id: str) -> str:
    """Ground truth for the ref_based condition, rendered from the public gold."""
    grader = grading_spec(task_id)["grader"]
    spec = gold(task_id)

    if grader == "b3_select":
        k = int(spec["k"])
        cands = spec["candidates"]
        if spec.get("goal") == "target":
            t = float(spec["target_value"])
            best = sorted(cands.items(), key=lambda kv: abs(float(kv[1]["y"]) - t))[:k]
            body = ", ".join(f"{i} ({v['y']})" for i, v in best)
            return (
                f"\n## Ground truth (held-out measured labels)\nTarget: {spec.get('target')} "
                f"= {t}. The K={k} candidates closest to the target are: {body}.\n"
            )
        rev = spec.get("goal") != "min"
        best = sorted(cands.items(), key=lambda kv: float(kv[1]["y"]), reverse=rev)[:k]
        body = ", ".join(f"{i} ({v['y']})" for i, v in best)
        return (
            f"\n## Ground truth (held-out measured labels)\nProperty: {spec.get('target')}. "
            f"Goal: {spec.get('goal', 'max')}. The true best K={k} candidates are: {body}. "
            f"Mean label of that ideal selection: {spec['top_k_mean']}; "
            f"blind-pick baseline: {spec['floor']}.\n"
        )

    if grader == "select_hypervolume":
        front = ", ".join(str(i) for i in spec.get("front_ids", [])[:20])
        return (
            f"\n## Ground truth (held-out measured labels)\nObjectives: {spec.get('target')}. "
            f"Candidates on the true Pareto front: {front}. Best achievable hypervolume with "
            f"K={spec['k']} picks: {spec['hv_star']:.6g}; blind-pick baseline: "
            f"{spec['floor']:.6g}.\n"
        )

    # olympus_design
    x_star = ", ".join(
        f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
        for k, v in spec["x_star"].items()
    )
    tail = ""
    if spec.get("goal") == "target":
        tail = f" The task asks to hit {spec['f_star']} (tolerance {spec.get('target_tol')})."
    return (
        f"\n## Ground truth (frozen surrogate of the real lab data)\nObjective: "
        f"{spec.get('target')}, goal {spec.get('goal', 'max')}. The optimum is f* = "
        f"{spec['f_star']:.6g} at: {x_star}. Best row in the training data: "
        f"{spec.get('train_best')}.{tail}\n"
    )


def build_messages(task_id: str, answer: str, condition: str) -> list[dict]:
    ref = reference_block(task_id) if condition == "ref_based" else ""
    user = USER.format(prompt=task_prompt(task_id), answer=answer, reference=ref, rubric=RUBRIC)
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N variant records")
    ap.add_argument("--judges", default=",".join(JUDGES))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    judges = [j for j in args.judges.split(",") if j.strip()]
    conditions = [c for c in args.conditions.split(",") if c.strip()]

    variants = latest_variants()
    if args.limit:
        variants = variants[: args.limit]
    if not variants:
        print("no gated variants — run 01_make_variants.py first", file=sys.stderr)
        sys.exit(1)

    out = DATA / "judge_scores.jsonl"
    done = {
        (r["uid"], r["style"], r["judge"], r["condition"])
        for r in read_jsonl(out)
        if r.get("score") is not None
    }
    todo = [
        (v, j, c)
        for v in variants
        for j in judges
        for c in conditions
        if (v["uid"], v["style"], j, c) not in done
    ]
    print(
        f"{len(todo)} judge calls to make ({len(variants)} variants x {len(judges)} judges "
        f"x {len(conditions)} conditions, {len(done)} cached)",
        file=sys.stderr,
    )
    if args.dry_run or not todo:
        return

    from hallmark.graders.judge import _llm_judge_one, parse_score, resolve_judge_keys

    api_key, base_url = resolve_judge_keys({}, "judge-vs-regret")
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    ok = fail = 0

    async def one(v: dict, judge: str, condition: str) -> None:
        nonlocal ok, fail
        messages = build_messages(v["task_id"], v["text"], condition)
        async with sem:
            try:
                raw = await _llm_judge_one(judge, api_key, base_url, messages)
                score10, rationale = parse_score(raw)
                rec = dict(
                    uid=v["uid"], task_id=v["task_id"], style=v["style"], judge=judge,
                    condition=condition, score=score10 / 10.0, score10=score10,
                    rationale=rationale[:600],
                )
            except Exception as exc:
                rec = dict(
                    uid=v["uid"], task_id=v["task_id"], style=v["style"], judge=judge,
                    condition=condition, score=None, error=f"{type(exc).__name__}: {exc}",
                )
        async with lock:
            append_jsonl(out, rec)
            if rec.get("score") is None:
                fail += 1
                print(f"  ERR  {v['uid']} [{v['style']}] {judge} {condition}: "
                      f"{rec['error'][:120]}", file=sys.stderr)
            else:
                ok += 1
                if ok % 25 == 0:
                    print(f"  ... {ok}/{len(todo)} done", file=sys.stderr)

    await asyncio.gather(*(one(v, j, c) for v, j, c in todo))
    print(f"\n{ok} scored, {fail} failed (re-run to retry failures)", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
