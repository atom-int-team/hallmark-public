"""Run a pinned Olympus experiment-emulator (used by the ``olympus_design`` grader).

Mirrors ``mlipdb.py`` in spirit, but the "judge" here is an emulator trained on a REAL
experimental campaign (the Olympus framework, Häse et al. 2021): a smooth surrogate that
maps any in-domain parameter point to the predicted experimental response. The B1 design
tasks score a proposed process setting by its emulated response, normalized to regret.

Unlike the MLIP snapshot (a frozen lookup, fully offline), the real Olympus emulator is run
LIVE at grade time — so real grading needs the ``.[olympus]`` extra. ``_run_emulator`` is the
single heavy seam: tests monkeypatch it to stay offline. A committed *frozen* emulator
(``emulator.json`` with ``kind="oracle"``) short-circuits the Olympus import entirely — this
is what ``scripts/build_olympus_tasks.py --mock`` writes (a deterministic analytic
placeholder) so the dev pipeline and CI run with no heavy deps; a real run regenerates the
gold and drops the placeholder.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from . import oracle as ora
from .config import settings


def _gt(gt_root: Path | None) -> Path:
    return gt_root or settings.ground_truth_dir


def _artifact_dir(spec: Mapping, gt_root: Path | None) -> Path:
    return _gt(gt_root) / spec["emulator_ref"]


def _apply_clip(spec: Mapping, val: float) -> float:
    """Clamp the surrogate output to the task's *physically* admissible range
    (``spec["value_clip"] = [lo, hi]``, either side ``None`` = unbounded). The frozen GP can
    extrapolate past the real-data envelope into nonsense at domain corners (e.g. a negative
    E-factor); a hard physical bound keeps ``f`` meaningful without capping legitimate interior
    optima (which sit inside the bound). No clip ⇒ value passed through unchanged."""
    clip = spec.get("value_clip")
    if not clip:
        return val
    lo, hi = clip
    if lo is not None:
        val = max(val, float(lo))
    if hi is not None:
        val = min(val, float(hi))
    return val


def emulate(spec: Mapping, x: Mapping, gt_root: Path | None = None) -> float:
    """``f(x)`` from the pinned emulator at point ``x`` (continuous + categorical knobs
    already assembled by the grader). Delegates to the ``_run_emulator`` seam, then clamps to
    the physically admissible range (``_apply_clip``)."""
    raw = float(_run_emulator(spec["dataset"], _artifact_dir(spec, gt_root), dict(x), spec))
    return _apply_clip(spec, raw)


def emulate_robust(
    spec: Mapping, x: Mapping, gt_root: Path | None, cfg: Mapping
) -> tuple[float, str]:
    """Aggregate the emulator over a ``±delta_frac`` axis-box around ``x`` (mirrors
    ``oracle.evaluate_robust``): center plus two axis perturbations per *continuous*
    variable (categoricals held fixed). ``agg="worst"`` (default) returns the worst value
    for the objective sense, so a sharp optimum that collapses under perturbation scores
    below a flat one — modeling real experimental irreproducibility."""
    delta = float(cfg.get("delta_frac", 0.05))
    agg = cfg.get("agg", "worst")
    pts = [dict(x)]
    for v in ora.variable_names(spec):
        lo, hi = ora.domain(spec, v)
        step = delta * (hi - lo)
        for s in (step, -step):
            p = dict(x)
            p[v] = min(max(float(x[v]) + s, lo), hi)
            pts.append(p)
    vals = [emulate(spec, p, gt_root) for p in pts]
    goal = spec.get("goal", "max")
    if agg == "mean":
        val = sum(vals) / len(vals)
    else:  # worst-case w.r.t. the objective sense
        val = min(vals) if goal == "max" else max(vals)
    return val, f"robust {agg} over ±{delta:.0%} ({len(pts)} pts)"


def _gp_predict(gp: Mapping, x: Mapping) -> float:
    """Pure-numpy mean prediction of a frozen Gaussian-process surrogate (anisotropic
    ``ConstantKernel*RBF`` with ``normalize_y``), exported by ``scripts/build_olympus_tasks.py``
    from a GP fit on the REAL Olympus experimental data. Offline + deterministic, no sklearn at
    grade time. ``mean = (k(x*, X_train) @ alpha) * y_std + y_mean``.

    Feature layout matches the fit: standardized continuous knobs (``feature_order``) followed by
    a one-hot block per categorical knob (``gp["categorical"]``, levels in order)."""
    import numpy as np

    cont = (np.array([float(x[n]) for n in gp["feature_order"]], dtype=float)
            - np.array(gp["x_mean"])) / np.array(gp["x_std"])
    onehot: list[float] = []
    for name, levels in gp.get("categorical", {}).items():
        onehot.extend(1.0 if lv == x.get(name) else 0.0 for lv in levels)
    xs = np.concatenate([cont, np.array(onehot)]) if onehot else cont
    diff = (xs - np.array(gp["X_train"])) / np.array(gp["length_scale"])
    k = float(gp["const"]) * np.exp(-0.5 * (diff ** 2).sum(axis=1))
    mean_norm = float(k @ np.array(gp["alpha"]))
    return mean_norm * float(gp["y_std"]) + float(gp["y_mean"])


def _run_emulator(dataset: str, art_dir: Path, x: dict, spec: Mapping) -> float:
    """SEAM. Evaluate the proposed point through the pinned emulator.

    Resolution order:
      1. A committed *frozen* emulator ``<art_dir>/emulator.json`` with ``kind="oracle"``
         is evaluated with the analytic ``oracle`` interpreter (offline, deterministic) —
         the ``--mock`` placeholder, and the numpy-fallback hook (plan §11).
      2. Otherwise import Olympus and run the real emulator (needs ``.[olympus]``).

    Tests monkeypatch this whole function, so neither branch runs in CI.
    """
    emu_file = art_dir / "emulator.json"
    if emu_file.exists():
        meta = json.loads(emu_file.read_text(encoding="utf-8"))
        kind = meta.get("kind")
        if kind == "oracle":  # analytic placeholder (--mock)
            return float(ora.evaluate(meta["spec"], x))
        if kind == "gp":  # frozen Gaussian-process surrogate fit on the real Olympus data
            return _gp_predict(meta["gp"], x)

    from olympus import Emulator  # heavy import — only here, only for real grading

    emu = Emulator(dataset=dataset, model="BayesNeuralNet")
    # Olympus param order = continuous knobs (in param_space order) then categoricals.
    # The build script writes ``variables`` / ``categorical`` in that exact order, so the
    # dict iteration below matches the emulator's expected feature vector.
    params = [x[v] for v in spec["variables"]] + [x[c] for c in spec.get("categorical", {})]
    value, _ = emu.run(params)
    return float(value[0] if hasattr(value, "__len__") else value)
