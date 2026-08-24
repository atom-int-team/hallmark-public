"""b3_select — pick the top-K real held-out candidates by a target property.

The B3 "selection" framing: the SUT sees a
training table (composition -> measured target) and a pool of held-out candidates
(composition only), and ranks the pool. The grader looks up the candidates' REAL measured
labels — frozen in ``select_gold.json`` and NEVER shown to the SUT — and scores the mean of
the picks by normalized regret vs the top-K achievable. There is NO surrogate/oracle: the
judge is the real measurement, so the "oracle is a fitted model" critique does not apply.

Mirrors ``combinatorial_select`` (discrete pick, normalized regret) but the value of each
item is a real held-out label rather than a brute-forced optimum, and id parsing is
deterministic-first (``parse_ids``) so grading stays offline.

**Unfilled slots count as the 0-score anchor, never as "fewer slots".** The task asks for
exactly K picks, so the mean is over K, not over however many ids the SUT happened to name:
a short answer is padded to K with the anchor value (the pool's random-pick expectation).
Dividing by ``len(picks)`` instead — as this grader did until 2026-07-21 — made naming ONE
good candidate beat an honest 5-pick, because a real 5-pick almost always includes weaker
candidates that drag the mean down. That was not hypothetical: in the shipped run one cell
named 2 of 5 and scored a perfect 1.0. Padding makes the score exactly ``n/K`` times the
score of the picks actually made — proportional partial credit, with under-picking never
profitable.

params:
  oracle_ref (str)  ref to ``select_gold.json``:
      {goal, target, k, candidates:{id:{y}}, top_k_mean, floor}
  floor      (num)  optional override of the 0-score anchor (default: gold ``floor``)
"""
from __future__ import annotations

from ..loader import GroundTruth
from ..models import Signal, TurnTranscript
from .base import Grader, register_grader
from .oracle_eval import _score  # single source of truth for normalized regret
from .parse_ids import parse_ids


def _mean_over_k(picked_sum: float, n: int, k: int, anchor: float) -> float:
    """Mean over the K slots the task asked for, unfilled slots valued at ``anchor``.

    ``n == k`` (the normal case) is exactly the old ``sum/n``, so a complete answer is scored
    byte-identically. ``n < k`` yields ``anchor + (n/k) * (mean(picks) - anchor)``: the score
    of the picks made, scaled by the fraction of slots filled. ``n == 0`` never reaches here
    (the caller returns 0 for an empty selection).
    """
    if n >= k:
        return picked_sum / n
    return (picked_sum + (k - n) * anchor) / k


@register_grader
class B3SelectGrader(Grader):
    name = "b3_select"
    signal = Signal.TEXT  # any text SUT qualifies; no capability gating

    async def grade(self, t: TurnTranscript, params: dict, gt: GroundTruth):
        spec = gt.resolve(params["oracle_ref"])
        cands = spec["candidates"]
        k = int(spec["k"])
        goal = spec.get("goal", "max")

        picks = await parse_ids(t.answer_text, set(cands), k, params)
        if not picks:
            return 0.0, "no valid candidates chosen"

        if goal == "target":  # pick the K candidates whose REAL label is closest to T
            T = float(spec["target_value"])
            dist = sorted(abs(float(c["y"]) - T) for c in cands.values())
            best = sum(dist[:k]) / k  # smallest achievable mean |y-T| (the ideal K picks)
            blind = sum(dist) / len(dist)  # random-pick expectation (0-score anchor)
            picked = sum(abs(float(cands[i]["y"]) - T) for i in picks)
            got = _mean_over_k(picked, len(picks), k, blind)
            score = _score("min", got, best, blind)  # regret: closer picks -> higher score
            return score, (
                f"picked {len(picks)}/{k}: mean|y-T|={got:.4g} vs best={best:.4g} "
                f"blind={blind:.4g} T={T:g} -> {score:.2f}"
            )

        f_star = float(spec["top_k_mean"])
        floor = float(params.get("floor", spec["floor"]))
        # mean of the REAL labels of the chosen candidates, over K slots (see module docstring)
        got = _mean_over_k(sum(float(cands[i]["y"]) for i in picks), len(picks), k, floor)

        score = _score(goal, got, f_star, floor)
        return score, (
            f"picked {len(picks)}/{k}: mean_y={got:.4g} vs top{k}={f_star:.4g} "
            f"floor={floor:.4g} -> {score:.2f}"
        )
