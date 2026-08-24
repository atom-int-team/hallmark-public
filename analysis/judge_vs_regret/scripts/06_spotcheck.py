"""Step 6 — render 10 rewrite triples for manual sign-off.

The gate (step 2) and the regrade (step 3) prove the *machine-readable* content is
untouched. They cannot prove the prose did not smuggle in a new technical claim that
carries no number and no id. That is what a human reads for.

Writes results/spotcheck.md with a verdict table to fill in by hand.

Run:  python analysis/judge_vs_regret/scripts/06_spotcheck.py [--n 10]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import DATA, RES, SEED, ensure_dirs, latest_variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    ensure_dirs()
    meta = {it["uid"]: it for it in json.loads((DATA / "sample.json").read_text())["items"]}
    variants: dict[tuple[str, str], dict] = {
        (r["uid"], r["style"]): r for r in latest_variants()
    }
    complete = sorted({u for (u, s) in variants if s == "verbose"}
                      & {u for (u, s) in variants if s == "terse"})
    if not complete:
        print("no complete triples yet — run 01_make_variants.py first")
        sys.exit(1)

    # one per task first (spread), then fill up at random, deterministically
    rng = random.Random(SEED)
    by_task: dict[str, list[str]] = {}
    for uid in complete:
        by_task.setdefault(meta[uid]["task_id"], []).append(uid)
    picked = [rng.choice(v) for _, v in sorted(by_task.items())]
    rng.shuffle(picked)
    picked = picked[: args.n]
    if len(picked) < args.n:
        rest = [u for u in complete if u not in picked]
        rng.shuffle(rest)
        picked += rest[: args.n - len(picked)]

    out = [
        "# Manual spot-check — did the rewrite change the content?",
        "",
        "Read each triple. `orig` is the cached system answer; `terse` and `verbose` are the",
        "style rewrites. The rewrite is ACCEPTABLE when it changes voice/length/structure only:",
        "no new material, no new mechanism, no new claim about a property, and the same",
        "recommendation. Fill the verdict column by hand.",
        "",
        "| # | uid | content preserved? (yes/no) | note |",
        "|---|---|---|---|",
    ]
    out += [f"| {i} | `{u}` |  |  |" for i, u in enumerate(picked, 1)]
    out.append("")

    for i, uid in enumerate(picked, 1):
        m = meta[uid]
        out += [
            "---", "",
            f"## {i}. `{uid}`",
            f"task `{m['task_id']}` · level **{m['level']}** · objective **{m['objective']:.3f}**",
            "",
        ]
        for style in ("orig", "terse", "verbose"):
            r = variants[(uid, style)]
            out += [f"### {style}  ({len(r['text'])} chars, ratio {r.get('len_ratio', 1.0)})",
                    "", "```text", r["text"].strip(), "```", ""]

    (RES / "spotcheck.md").write_text("\n".join(out))
    print(f"wrote {RES / 'spotcheck.md'} with {len(picked)} triples")


if __name__ == "__main__":
    main()
