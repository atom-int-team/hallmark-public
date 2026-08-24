#!/usr/bin/env python3
"""Controls on the second-condition gap: is it an artefact of response form or of chain?

Recomputes the condition-free vs condition-bearing contrast three ways:
  (1) as reported in the paper: within family, both response forms pooled;
  (2) FORM-CONTROLLED: selection-form tasks only (b3_select / select_hypervolume);
  (3) CHAIN-CONTROLLED: chain A only (real measured hidden labels).
Plus the form effect itself, on the three matched pairs (one hidden objective,
both response forms) and on the whole slate, and the per-chain means.

Reads the released ledger, the atomic blocks and the task registry; needs nothing else.
Cluster bootstrap over the 29 atomic blocks, fixed seed. Offline, deterministic.

Run: python scripts/s5_controls.py
"""
from __future__ import annotations

import collections
import json
import random
import statistics as st

NBOOT, BSEED = 10_000, 20260818
REG = {t["task_id"]: t for t in json.load(open("task_registry.json"))["tasks"]}
FAMS = ["Synthetic", "Olympus", "B3-real", "htsteel", "Thermoelec", "MOF", "NLO",
        "Crystal-DFT", "Perovskite"]
raw = json.load(open("results/final.json"))["results"]
SC = collections.defaultdict(dict)
for r in raw:
    if r.get("task_score") is not None:
        SC[(r["system_id"], r["task_id"])][r["seed_idx"]] = r["task_score"]
SYS = sorted({s for s, _ in SC}); TASKS = sorted({t for _, t in SC})
M3 = {k: st.mean(v.values()) for k, v in SC.items()}
BLK = {t: b["key"] for b in json.load(open("blocks.json"))["blocks"] for t in b["tasks"]}
GRADER = {t: REG[t]["grader"] for t in TASKS}
SELECT = {t for t in TASKS if GRADER[t] in ("b3_select", "select_hypervolume")}
def chain(t):
    return REG[t]["chain"]
def family(t):
    return REG[t]["family"]
TT = {t: REG[t]["construction"] for t in TASKS}
FREE = {"Select (maximize)", "Robust"}
COND = {"Constrained", "Extrapolation / zero-shot", "Target-hit", "Pareto (multi-obj)"}

def boot(per_block, n=NBOOT, seed=BSEED):
    rnd = random.Random(seed); ks = list(per_block)
    xs = sorted(st.mean([v for k in [ks[rnd.randrange(len(ks))] for _ in ks]
                         for v in per_block[k]]) for _ in range(n))
    return xs[int(.025*n)], xs[int(.975*n)]

def contrast(pool, tag):
    """Mean(free) - mean(cond) over `pool`, per family and overall, with a block CI."""
    a = [t for t in pool if TT[t] in FREE]; b = [t for t in pool if TT[t] in COND]
    ma = st.mean([M3[(s, t)] for s in SYS for t in a])
    mb = st.mean([M3[(s, t)] for s in SYS for t in b])
    # CI: block-bootstrap the family-balanced difference by resampling blocks
    pa, pb = collections.defaultdict(list), collections.defaultdict(list)
    for s in SYS:
        for t in a: pa[BLK[t]].append(M3[(s, t)])
        for t in b: pb[BLK[t]].append(M3[(s, t)])
    rnd = random.Random(BSEED); ka, kb = list(pa), list(pb); ds = []
    for _ in range(NBOOT):
        sa = [ka[rnd.randrange(len(ka))] for _ in ka]; sb = [kb[rnd.randrange(len(kb))] for _ in kb]
        ds.append(st.mean([v for k in sa for v in pa[k]]) - st.mean([v for k in sb for v in pb[k]]))
    ds.sort()
    npos = sum(1 for s in SYS
               if st.mean([M3[(s, t)] for t in a]) > st.mean([M3[(s, t)] for t in b]))
    fams = {}
    for f in FAMS:
        fa = [t for t in a if family(t) == f]; fb = [t for t in b if family(t) == f]
        if fa and fb:
            fams[f] = (st.mean([M3[(s, t)] for s in SYS for t in fa])
                       - st.mean([M3[(s, t)] for s in SYS for t in fb]))
    print(f"\n{tag}: {len(a)} free / {len(b)} conditioned tasks")
    print(f"  free {ma:.3f}  cond {mb:.3f}  delta {ma-mb:+.3f} "
          f"[{ds[int(.025*NBOOT)]:+.3f}, {ds[int(.975*NBOOT)]:+.3f}]  "
          f"positive for {npos}/{len(SYS)} systems, {sum(1 for v in fams.values() if v>0)}/{len(fams)} families")
    for f, v in fams.items(): print(f"    {f:14s}{v:+.3f}")
    return ma - mb

contrast(TASKS, "(1) as reported (both forms)")
contrast([t for t in TASKS if t in SELECT], "(2) FORM-CONTROLLED: selection form only")
contrast([t for t in TASKS if chain(t) == "A"], "(3) CHAIN-CONTROLLED: chain A only")
contrast([t for t in TASKS if t in SELECT and chain(t) == "A"], "(4) both controls")

print("\nFORM EFFECT")
PAIRS = [("e2e_oracle_point", "e2e_oracle_select"),
         ("e2e_multi_constraint", "e2e_multi_constraint_select"),
         ("e2e_target_point", "e2e_target_select")]
for a, b in PAIRS:
    ma = st.mean([M3[(s, a)] for s in SYS]); mb = st.mean([M3[(s, b)] for s in SYS])
    print(f"  {a:24s} {ma:.3f} -> {b:30s} {mb:.3f}  "
          f"select wins {sum(1 for s in SYS if M3[(s,b)]>M3[(s,a)])}/{len(SYS)}")
pp = st.mean([M3[(s, a)] for s in SYS for a, _ in PAIRS])
ps = st.mean([M3[(s, b)] for s in SYS for _, b in PAIRS])
print(f"  pooled point {pp:.3f} vs select {ps:.3f}  delta {ps-pp:+.3f}")
sel = st.mean([M3[(s, t)] for s in SYS for t in TASKS if t in SELECT])
con = st.mean([M3[(s, t)] for s in SYS for t in TASKS if t not in SELECT])
cx = st.mean([M3[(s, t)] for s in SYS for t in TASKS if t not in SELECT and family(t) != "Synthetic"])
print(f"  whole slate: select {sel:.3f} ({len(SELECT)} tasks) vs construct {con:.3f} "
      f"({len(TASKS)-len(SELECT)}), ex-synthetic construct {cx:.3f}")

print("\nCHAIN MEANS")
for c, nm in (("A", "real hidden labels"), ("B", "analytic"), ("C", "frozen surrogate")):
    tl = [t for t in TASKS if chain(t) == c]
    print(f"  chain {c} ({nm}): {len(tl)} tasks, mean {st.mean([M3[(s,t)] for s in SYS for t in tl]):.3f}")
    for x in sorted({TT[t] for t in tl}):
        n = [t for t in tl if TT[t] == x]
        print(f"      {x:28s} {len(n):3d}  {st.mean([M3[(s,t)] for s in SYS for t in n]):.3f}")
