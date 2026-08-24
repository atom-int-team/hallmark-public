#!/usr/bin/env python3
"""results/final.json -> results/final_mean.json: one cell per (system, task), score = mean over repeats.

The aggregation rule of the paper: average WITHIN the cell first, then let the leaderboard
bootstrap over cells. Feeding the raw (task x repeat) rows to `hallmark leaderboard` would
instead bootstrap over pseudo-independent draws and shrink every interval by about sqrt(3).
Repeats are averaged and never maximized: best-of-N is selection on the outcome and inflates
a system by exactly as much as it is noisy.

A system with a single repeat passes through unchanged.

Note on the shape of the output: only `task_score` and `usage` are means. The `turns` block is
copied from repeat 0, since per-grader detail cannot be averaged meaningfully.

Usage: python scripts/mean_of_seeds.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics

SRC = pathlib.Path("results/final.json")
DST = pathlib.Path("results/final_mean.json")
# Averaged across repeats; token counts too, so the cost/latency columns stay per-run figures.
USAGE_KEYS = ("cost_usd", "latency_ms", "prompt_tokens", "completion_tokens")
# `Usage` declares these three as int, so a fractional mean would fail pydantic validation
# when `hallmark leaderboard` reads the file back. cost_usd stays fractional on purpose.
INT_KEYS = ("prompt_tokens", "completion_tokens", "latency_ms")

src = json.loads(SRC.read_text())

by: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
for r in src["results"]:
    by[(r["system_id"], r["task_id"])].append(r)

out = []
for (_sysid, _tid), rs in by.items():
    rs.sort(key=lambda r: r["seed_idx"])
    scores = [r["task_score"] for r in rs if r["task_score"] is not None]
    cell = dict(rs[0])                      # repeat 0 carries the metadata
    cell["seed_idx"] = 0
    cell["task_score"] = statistics.fmean(scores) if scores else None
    usage = dict(cell["usage"])
    for k in USAGE_KEYS:
        vals = [v for v in ((r.get("usage") or {}).get(k) for r in rs) if v is not None]
        usage[k] = statistics.fmean(vals) if vals else None
        if k in INT_KEYS and usage[k] is not None:
            usage[k] = round(usage[k])
    cell["usage"] = usage
    out.append(cell)

dst = dict(src)
dst["run_name"] = "final_mean"
dst["results"] = out
DST.write_text(json.dumps(dst, ensure_ascii=False, indent=1) + "\n")

per_system = collections.Counter(r["system_id"] for r in out)
repeats_per_cell = collections.Counter(len(rs) for rs in by.values())
print(f"{len(out)} cells -> {DST}")
print("  cells per system:", dict(per_system))
print("  repeats per cell:", dict(repeats_per_cell))
