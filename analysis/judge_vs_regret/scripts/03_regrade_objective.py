"""Step 3 — re-score every variant with the real grader, offline, for $0.

This is the H4 measurement ("a style-only rewrite moves the deterministic grader by
exactly 0") and, just as importantly, the safety net that runs BEFORE any judge money
is spent: if the gate let a content change through, the objective score moves and we
find out here.

The LLM extractor seam is monkeypatched to raise. Stratum D is supposed to grade with
no network at all; if some cell reaches for the extractor we want a loud failure, not
a silent API charge.

Writes data/objective.jsonl. Exit code 1 if any variant moved.

Run:  python analysis/judge_vs_regret/scripts/03_regrade_objective.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import DATA, ensure_dirs, grader_params, grading_spec, latest_variants  # noqa: E402

TOL = 1e-12


def _block_llm() -> None:
    from hallmark.graders import extract

    async def _boom(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("stratum D must grade offline — the LLM extractor was called")

    extract._extract_llm = _boom


async def score_text(task_id: str, text: str) -> tuple[float, str]:
    from hallmark.graders import GRADERS
    from hallmark.loader import GroundTruth
    from hallmark.models import TurnTranscript

    g = grading_spec(task_id)
    turn = TurnTranscript(prompt="", answer_text=text)
    return await GRADERS[g["grader"]].grade(turn, grader_params(task_id), GroundTruth())


async def main() -> None:
    ensure_dirs()
    _block_llm()

    sample = json.loads((DATA / "sample.json").read_text())
    ledger_score = {it["uid"]: it["objective"] for it in sample["items"]}

    variants = latest_variants()
    if not variants:
        print("no gated variants in data/variants.jsonl — run 01_make_variants.py first")
        sys.exit(1)

    out = DATA / "objective.jsonl"
    rows, drift, moved = [], [], []
    for r in variants:
        score, detail = await score_text(r["task_id"], r["text"])
        base = ledger_score[r["uid"]]
        rows.append(
            dict(uid=r["uid"], task_id=r["task_id"], style=r["style"], score=score,
                 ledger_score=base, delta=score - base, detail=detail)
        )
        if r["style"] == "orig" and abs(score - base) > TOL:
            drift.append(rows[-1])
        if r["style"] != "orig" and abs(score - base) > TOL:
            moved.append(rows[-1])

    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))

    n_orig = sum(1 for r in rows if r["style"] == "orig")
    print(f"regraded {len(rows)} texts ({n_orig} originals, {len(rows) - n_orig} rewrites)")
    print(f"ledger reproduction on originals: {n_orig - len(drift)}/{n_orig} exact")
    for d in drift:
        print(f"  DRIFT {d['uid']}: ledger {d['ledger_score']:.6f} -> regrade {d['score']:.6f}")
    print(f"H4 — rewrites with |delta| > 0: {len(moved)}")
    for m in moved:
        print(f"  MOVED {m['uid']} [{m['style']}]: delta {m['delta']:+.6f} | {m['detail']}")
    sys.exit(1 if (moved or drift) else 0)


if __name__ == "__main__":
    asyncio.run(main())
