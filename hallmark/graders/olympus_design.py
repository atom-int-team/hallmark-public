"""olympus_design — propose a process setting that maximizes a REAL experimental response.

The SUT proposes a design point (values for the continuous knobs + a categorical choice
where the dataset has one); the grader extracts it, clamps to the domain, runs the pinned
Olympus emulator at that point and scores it by normalized stability regret — 1.0 = the
emulator's global optimum, 0.0 = the gold's anchor / an incomplete proposal. Un-gameable
by narration: only ``f(proposal)`` counts, never the claimed number. Mirrors
``mlip_design`` (which uses the frozen MLIP snapshot) but the "judge" is an
experiment-derived emulator run live.

**The 0 anchor here is ``train_median`` — the MEDIAN of the surrogate's predictions over
the training points, i.e. a typical known design, NOT the best one.** Scoring 1.0 means
"reached the surrogate's optimum", and a mid-range score means "well above a typical
known design"; neither licenses the claim "beat the best known design". ``oracle_eval``'s
module docstring explains why a best-known anchor is not usable for these surrogate
families (the prediction is clipped to the observed response range, so the attainable
optimum sits at the best observed label and the span nearly vanishes). Golds written
before the rename carry ``train_best`` with the same median value and still grade
identically.

params:
  oracle_ref (str)   ref to ``olympus.json`` {dataset, goal, variables, categorical,
                     emulator_ref, f_star, train_median, ...}
  floor      (num)   optional override of the gold's 0-score anchor
  robust     (dict)  optional ``{delta_frac: 0.05, agg: "worst"|"mean"}`` — score the worst
                     emulator value in a ±delta neighbourhood (manufacturing tolerance)
"""
from __future__ import annotations

from .. import oracle as ora
from ..loader import GroundTruth
from ..models import Signal, TurnTranscript
from ..olympusdb import emulate, emulate_robust
from .base import Grader, register_grader
from .extract import _find_labeled_choice, extract_point
from .oracle_eval import _floor_value, _score  # single source of truth for regret + anchor

_EPS = 1e-4  # a design variable at/below this fraction counts as "absent"


def _apply_constraint(spec: dict, point: dict, x: dict, variables: list[str]) -> tuple[bool, str]:
    """Enforce ``spec["constraint"]`` on the SUT's proposal (B3 constrained-design variants).
    ``forbid`` — none of the listed elements may be present; ``max_elements`` — at most N
    variables may be nonzero. Feasibility is judged on the raw ``point`` (so an omitted/zero
    element the domain-clamp would raise to its lower bound is still treated as absent), and
    the corresponding entries are pinned to 0 in ``x`` so the emulator evaluates the exact
    composition that was judged feasible. Returns (ok, detail); ok=False ⇒ score 0."""
    c = spec.get("constraint")
    if not c:
        return True, ""
    if "forbid" in c:
        bad = [e for e in c["forbid"] if point.get(e, 0.0) > _EPS]
        if bad:
            return False, f"infeasible: forbidden element(s) present: {', '.join(bad)}"
        for e in c["forbid"]:  # emulator's feature_order still includes them -> pin to 0
            x[e] = 0.0
    if "max_elements" in c:
        for v in variables:  # omitted / near-zero knobs stay exactly 0 (no clamp-up)
            if point.get(v, 0.0) <= _EPS:
                x[v] = 0.0
        active = [v for v in variables if x.get(v, 0.0) > _EPS]
        if len(active) > int(c["max_elements"]):
            return False, (f"infeasible: {len(active)} elements used "
                           f"(≤ {c['max_elements']} allowed)")
    return True, ""


@register_grader
class OlympusDesignGrader(Grader):
    name = "olympus_design"
    signal = Signal.TEXT

    async def grade(self, t: TurnTranscript, params: dict, gt: GroundTruth):
        spec = gt.resolve(params["oracle_ref"])
        variables = ora.variable_names(spec)
        cats = spec.get("categorical", {})

        point = await extract_point(t.answer_text, variables, params)
        if "max_elements" in spec.get("constraint", {}):
            # omitting an element IS how you use fewer of them; require some proposal, then fill
            # the rest with 0 so clamp/emulate have every feature (_apply_constraint pins them back).
            if not any(v in point for v in variables):
                return 0.0, "no composition proposed"
            for v in variables:
                point.setdefault(v, 0.0)
        missing = [v for v in variables if v not in point]
        if missing:
            return 0.0, f"no proposed value for {', '.join(missing)}"

        # Categorical knob (e.g. suzuki's ligand): recovered deterministically from the
        # answer against the allowed levels; a missing/unknown choice is "no proposal" -> 0.
        for c, levels in cats.items():
            ch = _find_labeled_choice(t.answer_text, c, list(levels))
            if ch is None:
                return 0.0, f"no valid {c} proposed (one of: {', '.join(levels)})"
            point[c] = ch

        x, oob = ora.clamp_to_domain(spec, point)  # clamps the numeric domains only
        # Integer-valued knobs (e.g. crossed_barrel's strut count): round after clamping so
        # the emulator sees a physically valid setting, not a fractional one.
        for v, meta in spec["variables"].items():
            if meta.get("integer"):
                x[v] = float(round(x[v]))
        for c in cats:  # categoricals pass through unchanged
            x[c] = point[c]

        # Constrained design (B3 maxelem / nontoxic): enforce feasibility on the PROPOSED
        # point, pinning off/forbidden elements to exactly 0 so the emulator sees the same
        # composition the check saw. A violation scores 0 — un-gameable by narration.
        ok, cdetail = _apply_constraint(spec, point, x, variables)
        if not ok:
            return 0.0, cdetail

        goal = spec.get("goal", "max")
        floor = _floor_value(spec, params.get("floor"))
        f_star = float(spec["f_star"])
        if params.get("robust"):
            f_val, rdetail = emulate_robust(spec, x, gt.root, params["robust"])
        else:
            f_val, rdetail = emulate(spec, x, gt.root), ""

        score = _score(goal, f_val, f_star, floor, tol=spec.get("target_tol"))
        detail = f"f(proposal)={f_val:.4g} vs f*={f_star:.4g} floor={floor:.4g} -> {score:.2f}"
        if rdetail:
            detail += f" [{rdetail}]"
        if oob:
            detail += " | clamped: " + "; ".join(oob)
        r2 = spec.get("emulator_r2")
        if r2 is not None:  # judge fidelity: how much of the real-data variance the surrogate explains
            detail += f" | judge R²={r2:.2f}"
        return score, detail
