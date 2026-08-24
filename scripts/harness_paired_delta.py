#!/usr/bin/env python3
"""What an agent scaffold adds: paired deltas of each harness configuration against the
same model's bare single-turn run.

A harness configuration (`pi/...`, `opencode/...`) runs one repeat (seed 0) of all 92
tasks. Its control is the SAME model's bare seed-0 cell on the SAME task, so the
comparison is paired cell by cell and the per-cell noise of a single repeat cancels.

Reports, per configuration: mean score under the harness, the bare control's mean, the
paired difference with a 95% cluster-bootstrap CI over the 29 atomic blocks (the same
resampling as scripts/pairwise_holm.py), and task-level wins/losses.

Deterministic, offline. Run:  python scripts/harness_paired_delta.py [--json out.json]
"""
from __future__ import annotations

import collections
import json
import random
import statistics as st
import sys

NBOOT, BSEED = 10_000, 20260818
HARNESS = "results/harness.json"
BARE = "results/final.json"
BLOCKS = "blocks.json"

# harness configuration -> bare control. Not derived from the id: the Kimi K3 control is
# the `-medium` reasoning configuration, and that mismatch of names should stay visible.
PAIRS = {
    "pi/x-ai-grok-4.6": "x-ai/grok-4.6",
    "pi/anthropic-claude-opus-5": "anthropic/claude-opus-5",
    "pi/google-gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "pi/moonshotai-kimi-k3": "moonshotai/kimi-k3-medium",
    "pi/openai-gpt-5.6-sol": "openai/gpt-5.6-sol",
    "pi/openai-gpt-6-astra": "openai/gpt-6-astra",
    "pi/openai-gpt-6.1-sol": "openai/gpt-6.1-sol",
    "opencode/x-ai-grok-4.6": "x-ai/grok-4.6",
    "opencode/google-gemini-3.1-pro-preview": "google/gemini-3.1-pro-preview",
    "opencode/openai-gpt-5.6-sol": "openai/gpt-5.6-sol",
    "opencode/openai-gpt-6-astra": "openai/gpt-6-astra",
}


def cells(path: str) -> dict[tuple[str, str, int], float]:
    return {(r["system_id"], r["task_id"], r["seed_idx"]): r["task_score"]
            for r in json.load(open(path))["results"]}


H = cells(HARNESS)
B = cells(BARE)
BLK = {t: b["key"] for b in json.load(open(BLOCKS))["blocks"] for t in b["tasks"]}


def control(bare_id: str, task: str) -> float:
    return B[(bare_id, task, 0)]


def boot(per_block: dict[str, list[float]], n: int = NBOOT, seed: int = BSEED):
    """Percentile 95% CI of the mean, resampling atomic blocks with replacement."""
    rnd = random.Random(seed)
    ks = list(per_block)
    draws = []
    for _ in range(n):
        vals = [x for k in (rnd.choice(ks) for _ in ks) for x in per_block[k]]
        draws.append(sum(vals) / len(vals))
    draws.sort()
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


rows = []
for hid, bid in PAIRS.items():
    tasks = sorted(t for s, t, _ in H if s == hid)
    d = {t: H[(hid, t, 0)] - control(bid, t) for t in tasks}
    per_block = collections.defaultdict(list)
    for t, x in d.items():
        per_block[BLK[t]].append(x)
    lo, hi = boot(per_block)
    rows.append(dict(
        harness=hid, bare=bid, n=len(tasks),
        mean_harness=st.mean(H[(hid, t, 0)] for t in tasks),
        mean_bare=st.mean(control(bid, t) for t in tasks),
        delta=st.mean(d.values()), lo=lo, hi=hi,
        wins=sum(x > 0 for x in d.values()), losses=sum(x < 0 for x in d.values()),
    ))

print(f"{len(rows)} harness configurations, seed 0, paired with the same model's bare seed 0; "
      f"cluster bootstrap on {len(set(BLK.values()))} atomic blocks, {NBOOT} resamples\n")
print(f"{'configuration':<42}{'n':>4}{'harness':>9}{'bare':>8}{'delta':>9}   95% CI           W-L")
for r in rows:
    sig = "*" if r["lo"] > 0 or r["hi"] < 0 else " "
    print(f"{r['harness']:<42}{r['n']:>4}{r['mean_harness']:9.4f}{r['mean_bare']:8.4f}"
          f"{r['delta']:+9.4f}  [{r['lo']:+.3f},{r['hi']:+.3f}]{sig}  {r['wins']}-{r['losses']}")
print("\n* = CI excludes 0")

if "--json" in sys.argv:
    out = sys.argv[sys.argv.index("--json") + 1]
    json.dump(rows, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
