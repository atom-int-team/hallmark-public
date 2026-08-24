"""Step 1 — generate the two style variants of every sampled answer.

The manipulation: voice, length and structure change; the material choice and the
``Final:`` line do not. Every generated variant must clear the step-2 gate before it
is written; up to MAX_ATTEMPTS retries feed the violations back into the prompt.

Resumable: (uid, style) pairs already present in data/variants.jsonl are skipped, so
a crashed or rate-limited run costs nothing to repeat.

Run:  python analysis/judge_vs_regret/scripts/01_make_variants.py [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    DATA,
    WRITER,
    append_jsonl,
    ensure_dirs,
    latest_variants,
    openrouter,
    read_jsonl,
)

_spec = importlib.util.spec_from_file_location(
    "gate_mod", pathlib.Path(__file__).with_name("02_verify_invariance.py")
)
_gate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gate_mod)
gate = _gate_mod.gate

MAX_ATTEMPTS = 3
CONCURRENCY = 6

SYSTEM = (
    "You rewrite an AI assistant's answer to a materials-design task in a different STYLE. "
    "Hard constraints, in order of priority:\n"
    "1. Every line that starts with 'Final:' must be reproduced BYTE-FOR-BYTE, once, as the "
    "last line of your output. If the original contains SEVERAL such lines, reproduce ONLY "
    "THE LAST ONE — that is the proposal the grader reads; the earlier ones are drafts.\n"
    "2. Do not change, add or remove any numeric value, candidate id (like c012), element "
    "symbol, material name or unit. You may drop details; you may never invent them.\n"
    "3. Do not add new technical claims, mechanisms, references or data. Only the voice, the "
    "length and the structure change.\n"
    "4. The word 'final' must not appear anywhere except in that one 'Final:' line, and the "
    "proposal must not be restated as a second list or summary line — the grader anchors on "
    "'final' and an extra occurrence changes what it reads.\n"
    "Return the rewritten answer as plain text and nothing else — no preamble, no commentary "
    "about the rewrite."
)

STYLE = {
    "terse": (
        "Rewrite it much SHORTER and HESITANT: no headings, no bullet lists, no jargon, "
        "plain flat sentences, under half the original length. Sound unsure — hedged wording "
        "such as 'probably', 'a rough guess', 'hard to say'. State no confidence in the choice."
    ),
    "verbose": (
        "Rewrite it much LONGER and CONFIDENT: use headings, a structured rationale, domain "
        "terminology, and assertive wording ('clearly', 'the decisive factor is'). At least "
        "twice the original length. Expand ONLY on reasoning already present in the original — "
        "restate it more thoroughly, do not introduce new facts, numbers or ids."
    ),
}


async def rewrite(client, orig: str, style: str, task_id: str) -> tuple[str, int, list[str]]:
    """Return (text, attempts, last_violations). Empty violations = accepted."""
    feedback = ""
    text, bad = "", ["not attempted"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        user = f"{STYLE[style]}{feedback}\n\n## Answer to rewrite\n{orig}"
        resp = await client.chat.completions.create(
            model=WRITER,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        bad = gate(orig, text, style, task_id)
        if not bad:
            return text, attempt, []
        feedback = (
            "\n\nYour previous attempt was REJECTED for: "
            + "; ".join(bad)
            + ". Fix exactly these problems and keep every other constraint."
        )
    return text, MAX_ATTEMPTS, bad


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N sampled answers")
    ap.add_argument("--dry-run", action="store_true", help="list the work, call nothing")
    args = ap.parse_args()

    ensure_dirs()
    sample = json.loads((DATA / "sample.json").read_text())
    items = sample["items"][: args.limit] if args.limit else sample["items"]
    out = DATA / "variants.jsonl"
    by_uid = {it["uid"]: it for it in items}

    # Re-run the gate over what is already stored instead of trusting the stored flag:
    # when the gate itself is corrected (it was, for markdown emphasis), a record rejected
    # under the old rule must be revalidated rather than regenerated at full price.
    done: set[tuple[str, str]] = set()
    for r in latest_variants(accepted_only=False):
        it = by_uid.get(r["uid"])
        if it is None:
            continue
        if r["style"] == "orig":
            done.add((r["uid"], "orig"))
            continue
        bad = gate(it["answer"], r["text"], r["style"], r.get("task_id"))
        if not bad:
            done.add((r["uid"], r["style"]))
            if not r.get("gate_ok"):
                append_jsonl(out, dict(r, gate_ok=True, violations=[], revalidated=True))

    # the untouched original is a variant too — recorded, never generated
    for it in items:
        if (it["uid"], "orig") not in done:
            append_jsonl(
                out,
                dict(uid=it["uid"], task_id=it["task_id"], style="orig", text=it["answer"],
                     gate_ok=True, attempts=0, violations=[], len_ratio=1.0),
            )
            done.add((it["uid"], "orig"))

    todo = [(it, s) for it in items for s in ("terse", "verbose") if (it["uid"], s) not in done]
    print(f"{len(todo)} variants to generate ({len(items)} answers)", file=sys.stderr)
    if args.dry_run or not todo:
        return

    client = openrouter()
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()

    async def one(it: dict, style: str) -> None:
        async with sem:
            try:
                text, attempts, bad = await rewrite(client, it["answer"], style, it["task_id"])
            except Exception as exc:  # provider error — leave it undone, resume later
                print(f"  ERROR {it['uid']} [{style}]: {type(exc).__name__}: {exc}", file=sys.stderr)
                return
        rec = dict(
            uid=it["uid"], task_id=it["task_id"], style=style, text=text,
            gate_ok=not bad, attempts=attempts, violations=bad,
            len_ratio=round(len(text) / max(len(it["answer"]), 1), 3),
        )
        async with lock:
            append_jsonl(out, rec)
        flag = "ok " if not bad else "GATE"
        print(f"  {flag} {it['uid']} [{style}] x{attempts} ratio={rec['len_ratio']}", file=sys.stderr)

    await asyncio.gather(*(one(it, s) for it, s in todo))

    recs = read_jsonl(out)
    failed = [r for r in recs if not r.get("gate_ok")]
    print(f"\n{len(recs)} records, {len(failed)} still failing the gate", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
