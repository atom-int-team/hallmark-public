#!/usr/bin/env python3
"""All 91 pairwise system comparisons with a multiplicity correction.

Section 6.1 groups the leaderboard from paired comparisons. With 14 systems that is
91 simultaneous tests, so the uncorrected 95% intervals overstate how much the board
resolves. This recomputes each pair with the SAME cluster bootstrap that
the paper uses (10,000 resamples of the 29 atomic task blocks, fixed seed), adds a two-sided bootstrap p-value, and applies Holm-Bonferroni
across the 91 pairs.

Reports, for every pair: mean paired difference, uncorrected 95% CI, taskwise win/loss,
raw p, Holm-adjusted p, and whether the pair survives at each level.

Deterministic, offline. Run:  python scripts/pairwise_holm.py [--json out.json]
"""
from __future__ import annotations

import collections
import json
import random
import statistics as st
import sys

NBOOT, BSEED = 10_000, 20260818
LEDGER = "results/final.json"
BLOCKS = "blocks.json"

SHORT = {
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "anthropic/claude-opus-5": "Opus 5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic/claude-fable-5": "Fable 5",
    "openai/gpt-5.5": "GPT-5.5",
    "moonshotai/kimi-k3-medium": "Kimi K3 (med)",
    "anthropic/claude-sonnet-4.6": "Sonnet 4.6",
    "anthropic/claude-sonnet-5": "Sonnet 5",
    "anthropic/claude-opus-4.8": "Opus 4.8",
    "x-ai/grok-4.6": "Grok 4.6",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro",
    "qwen/qwen3.8-max-high": "Qwen3.8 Max (high)",
    "openai/gpt-6-astra": "GPT-6 Astra",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
}

# --------------------------------------------------------------------------- #
rows = json.load(open(LEDGER))["results"]
SC: dict[tuple[str, str], dict[int, float]] = collections.defaultdict(dict)
for r in rows:
    SC[(r["system_id"], r["task_id"])][r["seed_idx"]] = r["task_score"]

SYS = sorted({s for s, _ in SC})
TASKS = sorted({t for _, t in SC})
M3 = {k: st.mean(v.values()) for k, v in SC.items()}          # cell = mean of repeats
MEAN = {s: st.mean([M3[(s, t)] for t in TASKS]) for s in SYS}
ORDER = sorted(SYS, key=lambda s: -MEAN[s])

BLOCKS = json.load(open(BLOCKS))["blocks"]
BLK = {t: b["key"] for b in BLOCKS for t in b["tasks"]}
BY_BLOCK = collections.defaultdict(list)
for t in TASKS:
    BY_BLOCK[BLK[t]].append(t)
KEYS = sorted(BY_BLOCK)


def boot(per_block: dict, n: int = NBOOT, seed: int = BSEED):
    """Percentile CI and two-sided p, resampling atomic blocks with replacement.

    The p-value is the bootstrap analogue of the interval test: twice the smaller
    tail mass on either side of zero, floored at 1/n (the resolution of n resamples).
    """
    rnd = random.Random(seed)
    ks = list(per_block)
    draws = []
    for _ in range(n):
        vals = [x for k in (rnd.choice(ks) for _ in ks) for x in per_block[k]]
        draws.append(sum(vals) / len(vals))
    draws.sort()
    lo, hi = draws[int(0.025 * n)], draws[int(0.975 * n)]
    below = sum(1 for d in draws if d <= 0)
    p = 2 * min(below, n - below) / n
    return lo, hi, max(p, 1 / n)


pairs = []
for i, a in enumerate(ORDER):
    for b in ORDER[i + 1:]:
        per_block = {k: [M3[(a, t)] - M3[(b, t)] for t in BY_BLOCK[k]] for k in KEYS}
        lo, hi, p = boot(per_block)
        d = st.mean([M3[(a, t)] - M3[(b, t)] for t in TASKS])
        w = sum(1 for t in TASKS if M3[(a, t)] > M3[(b, t)])
        ties = sum(1 for t in TASKS if M3[(a, t)] == M3[(b, t)])
        pairs.append(dict(a=a, b=b, d=d, lo=lo, hi=hi, p=p, raw_sig=(lo > 0 or hi < 0),
                          wins=w, losses=len(TASKS) - w - ties, ties=ties))

# ---- Holm-Bonferroni over the 36 simultaneous tests ------------------------ #
m = len(pairs)
for rank, pr in enumerate(sorted(pairs, key=lambda x: x["p"])):
    pr["p_holm"] = min(1.0, (m - rank) * pr["p"])
running = 0.0
for pr in sorted(pairs, key=lambda x: x["p"]):                # enforce monotonicity
    running = max(running, pr["p_holm"])
    pr["p_holm"] = running
    pr["holm_sig"] = running < 0.05

# --------------------------------------------------------------------------- #
n_raw = sum(p["raw_sig"] for p in pairs)
n_holm = sum(p["holm_sig"] for p in pairs)

print(f"{m} pairwise comparisons over {len(SYS)} systems, "
      f"cluster bootstrap on {len(KEYS)} atomic blocks, {NBOOT} resamples")
print(f"resolved by uncorrected 95% CI : {n_raw}/{m}")
print(f"resolved after Holm (alpha=.05): {n_holm}/{m}")
print()
hdr = f"{'row':15s} {'col':15s} {'diff':>7s} {'95% CI':>18s} {'W-L':>7s} {'p':>8s} {'p_Holm':>8s}  sig"
print(hdr)
print("-" * len(hdr))
for pr in pairs:
    flag = "**" if pr["holm_sig"] else ("*" if pr["raw_sig"] else "")
    print(f"{SHORT[pr['a']]:15s} {SHORT[pr['b']]:15s} {pr['d']:+7.3f} "
          f"[{pr['lo']:+6.3f},{pr['hi']:+6.3f}] {pr['wins']:3d}-{pr['losses']:<3d} "
          f"{pr['p']:8.4f} {pr['p_holm']:8.4f}  {flag}")
print()
print("* = uncorrected CI excludes 0   ** = survives Holm")

# ---- what the leading cluster looks like under each rule ------------------- #
# The cluster is READ OFF the comparisons rather than fixed at a size: it is the longest prefix
# of the ranking inside which no pair resolves even by the uncorrected interval. Fixing it at
# five hid the fact that the sixth system joined it on 2026-08-21.
def _lead_prefix() -> list:
    P = {(p["a"], p["b"]): p for p in pairs}
    k = 1
    while k < len(ORDER):
        cand = ORDER[:k + 1]
        if any(P[(a, b)]["raw_sig"] for i, a in enumerate(cand) for b in cand[i + 1:]):
            break
        k += 1
    return ORDER[:k]


lead = _lead_prefix()
inside = [p for p in pairs if p["a"] in lead and p["b"] in lead]
trail = [s for s in ORDER if s not in lead]
print()
print(f"leading cluster, {len(lead)} systems ({', '.join(SHORT[s] for s in lead)}):")
print(f"  pairs inside the cluster: {len(inside)}, resolved raw {sum(p['raw_sig'] for p in inside)}, "
      f"Holm {sum(p['holm_sig'] for p in inside)}")
if inside:
    print(f"  widest difference inside: {max(inside, key=lambda p: abs(p['d']))['d']:+.3f}")
beaten = [b for b in trail
          if all(any(p["holm_sig"] for p in pairs if {p["a"], p["b"]} == {a, b}) for a in lead)]
print(f"  systems every one of them beats after Holm: {len(beaten)}"
      + (f" ({', '.join(SHORT[s] for s in beaten)})" if beaten else ""))

if "--json" in sys.argv:
    out = sys.argv[sys.argv.index("--json") + 1]
    json.dump({"n_pairs": m, "n_raw": n_raw, "n_holm": n_holm,
               "pairs": pairs, "order": ORDER, "mean": MEAN},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")
