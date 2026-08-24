"""select_hypervolume — pick the K pool candidates spanning the best 2D trade-off front.

Instead of evaluating *proposed* designs through two oracles, it looks up the K chosen candidates' REAL
held-out labels — frozen in the gold, never shown — maps them into the normalized objective square
via the pool min/max frozen at build time, and scores the dominated hypervolume by normalized
regret against the exact best-K front. No surrogate anywhere: the judge is the real measurement.

Used by ``nlo_pareto_select``, where the two objectives are the SHG coefficient max|d_ij| and the
HSE band gap — the canonical nonlinear-optics conflict (large nonlinearity vs wide transparency
window). Picking 5 candidates that are all strong on one axis loses to 5 that span the frontier.

gold shape (``pareto_gold.json``):
  {goal: "hypervolume", objectives: [o1, o2], k, ref_point: [r1, r2],
   norm: {o1: [lo, hi], o2: [lo, hi]},
   candidates: {id: {p: [p1, p2], y: [y1, y2]}},   # p = normalized, both maximized
   front_ids: [...], hv_star: float, floor: float}

params:
  oracle_ref (str)  ref to the ``pareto_gold.json``
  floor      (num)  optional override of the 0-score anchor (default: gold ``floor``)
"""
from __future__ import annotations

from ..loader import GroundTruth
from ..models import Signal, TurnTranscript
from .base import Grader, register_grader
from .oracle_eval import _score  # single source of truth for normalized regret
from .parse_ids import parse_ids


def _hv2d(points: list[tuple[float, float]], ref: list[float]) -> float:
    """Hypervolume dominated by ``points`` above ``ref`` (both objectives maximized)."""
    nd = []
    for p in points:
        if p[0] < ref[0] or p[1] < ref[1]:
            continue
        if any(q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]) for q in points):
            continue  # strictly dominated
        nd.append((p[0], p[1]))
    nd = sorted(set(nd), key=lambda p: -p[0])  # x descending => y ascending along the front
    hv, last_y = 0.0, float(ref[1])
    for x, y in nd:
        if y > last_y:
            hv += (x - ref[0]) * (y - last_y)
            last_y = y
    return hv


@register_grader
class SelectHypervolumeGrader(Grader):
    name = "select_hypervolume"
    signal = Signal.TEXT  # any text SUT qualifies; no capability gating

    async def grade(self, t: TurnTranscript, params: dict, gt: GroundTruth):
        spec = gt.resolve(params["oracle_ref"])
        cands = spec["candidates"]
        k = int(spec["k"])

        picks = await parse_ids(t.answer_text, set(cands), k, params)
        if not picks:
            return 0.0, "no valid candidates chosen"

        pts = [tuple(cands[i]["p"]) for i in picks]
        hv = _hv2d(pts, spec["ref_point"])
        hv_star = float(spec["hv_star"])
        floor = float(params.get("floor", spec["floor"]))

        score = _score("max", hv, hv_star, floor)
        return score, (
            f"picked {len(picks)}/{k}: HV={hv:.4g} vs HV*={hv_star:.4g} "
            f"floor={floor:.4g} -> {score:.2f}"
        )
