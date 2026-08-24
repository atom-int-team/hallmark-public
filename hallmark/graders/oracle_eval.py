"""oracle_eval — objective inverse-design grader.

Evaluate the SUT's PROPOSED design point through a hidden analytic oracle and
score it by normalized regret. Un-gameable by narration: the *claimed* metric is
ignored; only ``f(x_proposed)`` matters, normalized so 1 = the true global optimum
and 0 = the gold's anchor. One grader serves the hidden-optimum, extrapolation-trap,
spurious-feature and robust-design tasks — they differ only in their fixture/oracle,
not this code.

**What the 0 anchor means differs by family, and the gold key says which** — do not
assume "best known composition":

* ``train_best`` — the **best** objective value reachable by reusing a training
  recipe. Only the synthetic analytic family (``build_alloy_tasks.py``) writes this,
  where the oracle is known in closed form, so the maximum over the clean training
  values is exact and beating it is a real claim.
* ``train_median`` — the **median** of the surrogate's predictions over the training
  points, i.e. a *typical* training design, not the best one. The GP-surrogate
  families (B3 continuum, Olympus) write this. A high score there means "much better
  than a typical known design", **never** "better than the best known design".

The distinction is not cosmetic: those surrogates clip their prediction to the
observed response range, so the attainable optimum sits at the best observed label
and a best-known anchor would leave almost no span (measured: 0.3–10% of the current
range, negative for two constrained variants). The median anchor is what keeps those
tasks scorable; the honest reporting duty is to call it a median.

A gold with ``sense: "target"`` holds the target value T in ``f_star`` and its tolerance
in ``target_tol``; the score then falls off linearly with |f − T| and no anchor is used.
``target_tol`` is forwarded to ``_score`` exactly as ``olympus_design`` already does. Until
2026-07-23 it was not, so the target branch compared two floats for exact equality — dead
code, since no shipped gold declared that sense.

params:
  oracle_ref      (str)     ground-truth ref to ``oracle.json``
  floor           (str|num) <number> overrides the anchor; any string (e.g. the legacy
                            ``"train_best"``) means "use the gold's own anchor"
  robust          (dict)    optional ``{delta_frac: 0.05, agg: "worst"|"mean"}``
  extract_backend (str)     ``"llm"`` (default) | ``"regex"`` (offline/testing)
"""
from __future__ import annotations

from .. import oracle as ora
from ..loader import GroundTruth
from ..models import Signal, TurnTranscript
from .base import Grader, register_grader
from .extract import extract_design, extract_point


def _sense(spec) -> str:
    """Resolve the optimization sense, tolerating both gold conventions.

    Golds in this repo declare it two ways: ``sense`` (oracle_eval's own key) and ``goal``
    (the selection-grader convention). Until 2026-07-24 the two call sites below read only
    ``sense`` while ``_floor_value`` also honoured ``goal``, so a target gold written in the
    other convention was silently graded as a *maximization* with floor 0 — measured on a copy
    of ``e2e_target_point``'s gold, a recipe 95 MPa outside the ±81 tolerance (true score 0.0)
    scored 0.909. No shipped gold under ``ground_truth/`` declares ``goal`` at all, so reading
    it here changes no score; the point is that the next one cannot mis-score in silence.
    """
    sense, goal = spec.get("sense"), spec.get("goal")
    if sense and goal and sense != goal:
        raise ValueError(f"gold declares conflicting sense={sense!r} and goal={goal!r}")
    return sense or goal or "max"


def _floor_value(spec, floor) -> float:
    """Resolve the 0-score anchor: an explicit number wins, otherwise the gold's own key.

    ``train_median`` (surrogate families) and ``train_best`` (synthetic family) are both
    accepted — see the module docstring for why they are NOT interchangeable in meaning.
    Both are read so a gold written before the rename still grades identically. Target
    tasks carry no anchor (``_score`` never uses one on that path), hence the 0.0.
    """
    if isinstance(floor, (int, float)) and not isinstance(floor, bool):
        return float(floor)
    for key in ("train_median", "train_best"):
        if key in spec:
            return float(spec[key])
    if _sense(spec) == "target":
        return 0.0  # unused: the target branch of _score returns before reading the floor
    raise KeyError(
        "gold defines no 0-score anchor (expected 'train_median' or 'train_best')"
    )


def _score(sense: str, f_val: float, f_star: float, floor: float,
           tol: float | None = None) -> float:
    if sense == "target":  # ``f_star`` holds the target value T; score falls off within ``tol``
        if not tol or tol <= 0:
            return 1.0 if f_val == f_star else 0.0
        return max(0.0, 1.0 - abs(f_val - f_star) / tol)
    span = (f_star - floor) if sense == "max" else (floor - f_star)
    if span <= 0:  # degenerate oracle; only an exact hit scores
        return 1.0 if f_val == f_star else 0.0
    raw = (f_val - floor) if sense == "max" else (floor - f_val)
    return max(0.0, min(1.0, raw / span))


@register_grader
class OracleEvalGrader(Grader):
    name = "oracle_eval"
    signal = Signal.TEXT

    async def grade(self, t: TurnTranscript, params: dict, gt: GroundTruth):
        spec = gt.resolve(params["oracle_ref"])
        if "composition" in spec:  # high-dimensional mixed-type recipe schema
            return await self._grade_recipe(t, params, spec)

        variables = ora.variable_names(spec)

        point = await extract_point(t.answer_text, variables, params)
        missing = [v for v in variables if v not in point]
        if missing:
            return 0.0, f"no proposed value for {', '.join(missing)}"

        x, oob = ora.clamp_to_domain(spec, point)
        sense = _sense(spec)
        floor = _floor_value(spec, params.get("floor"))

        if params.get("robust"):
            f_val, rdetail = ora.evaluate_robust(spec, x, params["robust"])
        else:
            f_val, rdetail = ora.evaluate(spec, x), ""

        f_star = float(spec["f_star"])
        score = _score(sense, f_val, f_star, floor, tol=spec.get("target_tol"))
        detail = f"f(proposal)={f_val:.4g} vs f*={f_star:.4g} floor={floor:.4g} -> {score:.2f}"
        if rdetail:
            detail += f" [{rdetail}]"
        if oob:
            detail += " | clamped: " + "; ".join(oob)
        return score, detail

    async def _grade_recipe(self, t: TurnTranscript, params: dict, spec: dict):
        design = await extract_design(t.answer_text, spec, params)
        x, note = ora.assemble_recipe(spec, design)
        if x is None:
            return 0.0, note  # invalid/incomplete recipe scores 0 (strict)

        sense = _sense(spec)
        floor = _floor_value(spec, params.get("floor"))
        if params.get("robust"):
            f_val, rdetail = ora.evaluate_robust(spec, x, params["robust"])
        else:
            f_val, rdetail = ora.evaluate(spec, x), ""

        f_star = float(spec["f_star"])
        score = _score(sense, f_val, f_star, floor, tol=spec.get("target_tol"))
        detail = f"f(recipe)={f_val:.4g} vs f*={f_star:.4g} floor={floor:.4g} -> {score:.2f}"
        if rdetail:
            detail += f" [{rdetail}]"
        if note and note != "ok":
            detail += f" [{note}]"
        return score, detail
