"""Analytic hidden-oracle evaluator for objective end-to-end design tasks.

``f(x)`` is a closed-form function over named variables, stored as JSON so it
round-trips through ``GroundTruth.resolve`` and needs no pickle / sklearn at grade
time (graders stay offline & light). A task's fixture CSV is SAMPLED from this
same ``f`` (+ noise) by ``scripts/build_alloy_tasks.py``, which also records
``f_star``/``x_star`` (the global optimum) and ``train_best`` (the best objective
value reachable by reusing a training recipe — here it genuinely IS the maximum
over the clean training values, since this family's oracle is known in closed form)
so the grader can normalize a proposal to ``[0, 1]`` without re-fitting. **The
GP-surrogate families do not have that luxury and anchor at ``train_median``
instead — see ``graders/oracle_eval`` for why the two must not be conflated.**
The dataset-driven tasks use a
high-dimensional, mixed-type **recipe** schema (composition + processing +
categorical) — see ``recipe_fields``/``assemble_recipe`` below.

Spec shape (see ``ground_truth/<task>/oracle.json``)::

    {
      "sense": "max",                       # or "min"
      "variables": {"Cr": {"domain": [0, 25]}, ...},
      "terms": [
        {"coef": 30.0, "const": true},
        {"coef": -0.5, "monomial": {"Cr": 2}, "center": {"Cr": 12}},
        {"coef": 12.0, "gaussian": {"Cr": [12, 4], "Ni": [18, 6]}}
      ],
      "f_star": 41.7, "x_star": {...}, "train_best": 35.2, "noise_std": 1.5
    }

Term kinds: ``const`` / ``monomial`` (integer *or* fractional powers) / ``gaussian`` /
``categorical`` / ``hinge`` (one-sided threshold penalty, for the mechanistic oracle).
The evaluator is a tiny interpreter (NOT ``eval``) so an oracle file is safe to load.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def variable_names(spec: Mapping) -> list[str]:
    return list(spec["variables"].keys())


def domain(spec: Mapping, var: str) -> tuple[float, float]:
    lo, hi = spec["variables"][var]["domain"]
    return float(lo), float(hi)


def _term_value(term: Mapping, x: Mapping[str, float]) -> float:
    """coef · (monomial factor) · (gaussian factor). A term may carry either or both;
    a ``const`` term is just ``coef``; a ``categorical`` term sums per-level effects
    (additive); a ``hinge`` term is a one-sided penalty ``coef·max(0, Σ aᵢxᵢ − θ)^p``
    (used by the mechanistic oracle for phase-stability / embrittlement / over-alloying
    thresholds — pass a negative ``coef`` to subtract). Combining monomial·gaussian lets
    a term express a composition-dependent amplitude times a localized bump (precipitation
    peak-aging, curve_match). Monomial powers may be fractional (e.g. solid-solution
    ``c^½``, Hall–Petch ``d^-½``); a fractional/negative power of a non-positive base is
    physically out of range and contributes 0."""
    coef = float(term.get("coef", 1.0))
    if term.get("const"):
        return coef
    if "categorical" in term:
        s = sum(
            float(levels.get(str(x.get(var)), 0.0))
            for var, levels in term["categorical"].items()
        )
        return coef * s
    if "hinge" in term:
        h = term["hinge"]
        s = sum(float(a) * float(x[v]) for v, a in h["vars"].items())
        excess = s - float(h.get("threshold", 0.0))
        if excess <= 0.0:
            return 0.0
        return coef * excess ** float(h.get("power", 2))
    val = coef
    has_factor = False
    if "monomial" in term:
        center = term.get("center", {})
        for var, power in term["monomial"].items():
            base = float(x[var]) - float(center.get(var, 0.0))
            p = float(power)
            if p.is_integer():
                val *= base ** int(p)
            elif base <= 0.0:
                val *= 0.0  # fractional power of a non-positive base — out of range
            else:
                val *= base ** p
        has_factor = True
    if "gaussian" in term:
        s = sum(
            ((float(x[var]) - float(mu)) / float(sigma)) ** 2
            for var, (mu, sigma) in term["gaussian"].items()
        )
        val *= math.exp(-s)
        has_factor = True
    if not has_factor:
        raise ValueError(f"unknown oracle term: {term}")
    return val


def evaluate(spec: Mapping, x: Mapping[str, float]) -> float:
    """``f(x)``. ``x`` must provide every oracle variable (caller validates/clamps)."""
    return sum(_term_value(t, x) for t in spec["terms"])


def active_variables(spec: Mapping) -> set[str]:
    """Variables that actually appear in some oracle term (i.e. affect ``f``).

    A variable listed in ``variables`` but absent from every term is a pure
    *noise* knob — by construction it cannot change ``f(x)``. ``assemble_recipe``
    uses this to avoid rejecting an otherwise-valid recipe just because such a
    knob wasn't restated (these are exactly the spurious/noise features the
    design tasks ask the agent to identify and ignore)."""
    out: set[str] = set()
    for term in spec.get("terms", []):
        for key in ("monomial", "gaussian", "categorical"):
            if key in term:
                out |= set(term[key].keys())
        if "hinge" in term:
            out |= set(term["hinge"]["vars"].keys())
    return out


def clamp_to_domain(spec: Mapping, x: Mapping[str, float]) -> tuple[dict, list[str]]:
    """Clamp each variable into its domain; return ``(clamped, out_of_domain_notes)``."""
    out, notes = {}, []
    for var in variable_names(spec):
        lo, hi = domain(spec, var)
        v = float(x[var])
        if v < lo or v > hi:
            notes.append(f"{var}={v:g}∉[{lo:g},{hi:g}]")
            v = min(max(v, lo), hi)
        out[var] = v
    return out, notes


def evaluate_robust(spec: Mapping, x: Mapping[str, float], cfg: Mapping) -> tuple[float, str]:
    """Aggregate ``f`` over a ``±delta_frac`` axis-box around ``x``.

    Samples the center plus two axis perturbations per variable (cheap and
    deterministic). ``agg="worst"`` (default) returns the worst value for the
    objective sense — i.e. ``min`` for maximization, ``max`` for minimization — so
    a sharp peak that collapses under perturbation scores below a flat optimum.
    """
    delta = float(cfg.get("delta_frac", 0.05))
    agg = cfg.get("agg", "worst")
    pts = [dict(x)]
    for v in variable_names(spec):
        lo, hi = domain(spec, v)
        step = delta * (hi - lo)
        for s in (step, -step):
            p = dict(x)
            p[v] = min(max(x[v] + s, lo), hi)
            pts.append(p)
    vals = [evaluate(spec, p) for p in pts]
    sense = spec.get("sense", "max")
    if agg == "mean":
        val = sum(vals) / len(vals)
    else:  # worst-case w.r.t. the objective sense
        val = min(vals) if sense == "max" else max(vals)
    return val, f"robust {agg} over ±{delta:.0%} ({len(pts)} pts)"


# --------------------------------------------------------------------------- #
# Recipe schema (high-dimensional, mixed-type design tasks)                   #
# --------------------------------------------------------------------------- #
# A "recipe" oracle carries, beyond ``variables``/``terms``:
#   composition       list[str]  alloying elements the agent designs (wt%)
#   balance_element   str        element filling the remainder (e.g. "Fe")
#   comp_total        float      composition sum target (default 100)
#   categorical       {var: [levels]}  categorical design variables
# The continuous ``variables`` cover composition (incl. the balance element) and
# the processing parameters; categorical vars live only in ``categorical``.


def recipe_fields(spec: Mapping) -> tuple[list[str], list[str], list[str]]:
    """(composition_elements, processing_vars, categorical_vars) for a recipe spec."""
    comp = list(spec.get("composition", []))
    balance = spec.get("balance_element")
    cats = list(spec.get("categorical", {}).keys())
    proc = [v for v in variable_names(spec) if v not in comp and v != balance]
    return comp, proc, cats


def assemble_recipe(spec: Mapping, design: Mapping) -> tuple[dict | None, str]:
    """Validate a proposed recipe and assemble the mixed evaluation point.

    Returns ``(x, note)`` where ``x`` maps every continuous variable (composition
    incl. the balance element, plus processing) to a float and every categorical
    variable to its chosen level string — or ``(None, reason)`` if the recipe is
    invalid. STRICT on composition (the user's choice): any negative wt% or an
    alloying sum exceeding ``comp_total`` (so the balance element would go negative)
    is rejected outright. Missing alloying elements default to 0 (absent); missing
    processing/categorical fields make the recipe incomplete (rejected); processing
    out of domain is clamped (with a note); an unknown categorical level is rejected.
    """
    comp, proc, _ = recipe_fields(spec)
    balance = spec.get("balance_element")
    total = float(spec.get("comp_total", 100.0))
    tol = float(spec.get("comp_tol", 0.5))
    cats = spec.get("categorical", {})

    # Extraction failure guard: a recipe with NOT A SINGLE alloying element stated
    # is "no proposal", not a pure-balance (e.g. pure-Fe) design. Scoring it as the
    # latter silently turns a dropped extraction into a fabricated worst-case point
    # (and makes distinct failed answers collapse to an identical bogus score).
    # Report it explicitly so the leaderboard shows an extraction miss, not a 0 design.
    if comp and not any(e in design for e in comp):
        return None, "no proposed composition (extraction empty)"

    alloy: dict[str, float] = {}
    for e in comp:
        try:
            alloy[e] = float(design.get(e, 0.0))
        except (TypeError, ValueError):
            return None, f"invalid recipe: non-numeric {e}={design.get(e)!r}"
    if any(v < 0 for v in alloy.values()):
        return None, "invalid recipe: negative wt%"
    ssum = sum(alloy.values())
    if ssum > total + tol:
        return None, f"invalid recipe: Σalloying={ssum:.1f} > {total:g}"

    x: dict = dict(alloy)
    if balance:
        x[balance] = total - ssum

    notes: list[str] = []
    active = active_variables(spec)
    for p in proc:
        if p not in design:
            # A processing knob absent from every oracle term is noise: it cannot
            # change f(x), so don't reject the recipe — backfill the domain
            # midpoint (any in-domain value gives the same f) and move on.
            if p not in active:
                lo, hi = domain(spec, p)
                x[p] = 0.5 * (lo + hi)
                notes.append(f"{p} defaulted (noise)")
                continue
            return None, f"incomplete recipe: missing {p}"
        try:
            val = float(design[p])
        except (TypeError, ValueError):
            return None, f"invalid recipe: non-numeric {p}={design.get(p)!r}"
        lo, hi = domain(spec, p)
        if val < lo or val > hi:
            notes.append(f"{p} clamped")
            val = min(max(val, lo), hi)
        x[p] = val

    for c, levels in cats.items():
        choice = design.get(c)
        if choice is None:
            # Same rule for a categorical that no term references (pure noise).
            if c not in active:
                x[c] = str(levels[0])
                notes.append(f"{c} defaulted (noise)")
                continue
            return None, f"incomplete recipe: missing {c}"
        if str(choice) not in levels:
            return None, f"invalid recipe: unknown {c}={choice!r}"
        x[c] = str(choice)

    return x, ("; ".join(notes) if notes else "ok")
