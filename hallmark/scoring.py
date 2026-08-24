"""Aggregate RunResults into a per-track leaderboard structure.

Ported/adapted from the old ``benchmark/metrics.py`` (the ``_agg_turns`` reduce).
Differences: scores are already 0..1 (no 1–5→0–10 rescale), aggregation is split
by ``track`` (reasoning | execution), and each cell carries a bootstrap 95% CI
(spec §10). Rendering lives in ``hallmark.reporters.leaderboard``.
"""
from __future__ import annotations

import random
import statistics
from collections import defaultdict

from .models import RunResult, Track

# Canonical column order for domains in the leaderboard.
_DOMAIN_ORDER = ["general_qa", "lit_review", "materials_databases", "end_to_end",
                 "code", "structure", "characterization"]
_BOOT_SEED = 12345  # deterministic CI (no wall-clock/random entropy)
_BOOT_ROUNDS = 2000


def _bootstrap_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. lo==hi==mean when n<2."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(_BOOT_SEED)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(_BOOT_ROUNDS)
    )
    lo = means[int((alpha / 2) * _BOOT_ROUNDS)]
    hi = means[min(_BOOT_ROUNDS - 1, int((1 - alpha / 2) * _BOOT_ROUNDS))]
    return round(lo, 4), round(hi, 4)


def _cell(values: list[float]) -> dict | None:
    if not values:
        return None
    lo, hi = _bootstrap_ci(values)
    return {"mean": round(statistics.fmean(values), 4), "lo": lo, "hi": hi, "n": len(values)}


def _mean_or_none(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 4) if xs else None


def summarize(results: list[RunResult]) -> dict:
    """Reduce per-(task×system×seed) results to a per-track leaderboard structure."""
    # (track, system, domain) -> [task_score per run]
    by_cell: dict[tuple, list[float]] = defaultdict(list)
    cost: dict[str, list[float]] = defaultdict(list)
    latency: dict[str, list[float]] = defaultdict(list)
    systems: list[str] = []
    domains_present: set[str] = set()

    for r in results:
        if r.system_id not in systems:
            systems.append(r.system_id)
        domains_present.add(r.domain.value)
        if r.task_score is not None:
            by_cell[(r.track.value, r.system_id, r.domain.value)].append(r.task_score)
        if r.usage.cost_usd is not None:
            cost[r.system_id].append(r.usage.cost_usd)
        latency[r.system_id].append(r.usage.latency_ms)

    domains = [d for d in _DOMAIN_ORDER if d in domains_present]

    tracks: dict[str, dict] = {}
    for track in (Track.REASONING.value, Track.EXECUTION.value):
        sys_rows: dict[str, dict] = {}
        for sysid in systems:
            per_domain = {d: _cell(by_cell.get((track, sysid, d), [])) for d in domains}
            all_vals = [v for d in domains for v in by_cell.get((track, sysid, d), [])]
            if not all_vals:
                continue  # system has no tasks on this track
            sys_rows[sysid] = {
                "domains": per_domain,
                "overall": _cell(all_vals),
                "cost_usd": _mean_or_none(cost.get(sysid, [])),
                "latency_ms": _mean_or_none(latency.get(sysid, [])),
            }
        if sys_rows:
            tracks[track] = {"systems": sys_rows}

    return {"systems": systems, "domains": domains, "tracks": tracks}
