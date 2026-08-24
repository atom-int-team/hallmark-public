#!/usr/bin/env python3
"""Shortcut-vulnerability audit of every selection task in the suite.

A *free ranker* is any ordering of the candidate pool that uses only what the prompt
displays and no domain knowledge: sort one shown numeric column, sort a quantity read
straight off a shown structure (atom count, cell volume), prefer the most common option
in a categorical field, or just take the pool in the order it was printed. It needs no
training table, no chemistry and no hidden label, so anyone holding the released pool can
run it. If such a ranker beats the systems, the task is not measuring materials decisions.

The audit reports three quantities per task, because the obvious one is misleading on its
own:

  free_max   the best free ranker, chosen by maximizing ON THE POOL IT SCORES. This is the
             number a hostile reader computes, and it is biased upward: maximizing over N
             orderings on the sample they are scored on is selection on the test set.
  null_max   the same maximization applied to N RANDOM K-picks, averaged over MC draws.
             This is what free_max reaches by chance alone, and on a pool with 50 numeric
             columns it is already ~0.6. free_max is only interpretable against it.
  gate_free  the honest counterpart: the ranker is chosen on the OPEN TRAINING TABLE by
             Spearman correlation with the target, keeping only correlations that clear the
             5% critical value, and only then applied to the pool. This is what the build
             gate of `build_crystal_dft_tasks.py` does, and it is what a system could
             actually have discovered from the data it was given.

Each is then compared with the nine systems' scores on the same task. The criterion that
matters is not "free_max above some threshold" but whether the shortcut BEATS what is being
measured.

Deterministic and fully offline. Scores replicate `hallmark/graders/b3_select` and
`hallmark/graders/select_hypervolume` exactly.

Run:  python scripts/shortcut_audit.py [--json shortcut_audit.json]
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import statistics as st
import sys

import numpy as np
import yaml
from scipy.stats import spearmanr

K_DEFAULT = 5
MC_DRAWS = 200
SEED = 20260819
LEDGER = "results/final.json"

# Atomic masses only -- a periodic table, not materials knowledge.
MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Ne": 20.18, "Na": 22.99, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098,
    "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.63, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.468,
    "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95, "Tc": 98.0,
    "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868, "Cd": 112.414, "In": 114.818,
    "Sn": 118.71, "Sb": 121.76, "Te": 127.6, "I": 126.904, "Xe": 131.293, "Cs": 132.905,
    "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Pm": 145.0,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.5, "Ho": 164.93,
    "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967, "Hf": 178.49, "Ta": 180.948,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967,
    "Hg": 200.592, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.038, "Pa": 231.036,
    "U": 238.029, "Np": 237.0, "Pu": 244.0,
}


# --------------------------------------------------------------------------- #
# scoring -- mirrors the two graders
# --------------------------------------------------------------------------- #
def _score(goal, got, f_star, floor):
    if f_star == floor:
        return 1.0 if got == f_star else 0.0
    v = (floor - got) / (floor - f_star) if goal == "min" else (got - floor) / (f_star - floor)
    return float(min(max(v, 0.0), 1.0))


def _hv2d(points, ref):
    nd = []
    for p in points:
        if p[0] < ref[0] or p[1] < ref[1]:
            continue
        if any(q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]) for q in points):
            continue
        nd.append((p[0], p[1]))
    nd = sorted(set(nd), key=lambda p: -p[0])
    hv, last_y = 0.0, float(ref[1])
    for x, y in nd:
        if y > last_y:
            hv += (x - ref[0]) * (y - last_y)
            last_y = y
    return hv


def make_scorer(gold, grader):
    cands = gold["candidates"]
    k = int(gold.get("k", K_DEFAULT))

    if grader == "select_hypervolume":
        ref, hv_star, floor = gold["ref_point"], float(gold["hv_star"]), float(gold["floor"])

        def f(ids):
            return _score("max", _hv2d([tuple(cands[i]["p"]) for i in ids], ref), hv_star, floor)
        return f, k

    goal = gold.get("goal", "max")
    if goal == "target":
        T = float(gold["target_value"])
        dist = sorted(abs(float(c["y"]) - T) for c in cands.values())
        best, blind = sum(dist[:k]) / k, sum(dist) / len(dist)

        def f(ids):
            got = sum(abs(float(cands[i]["y"]) - T) for i in ids) / len(ids)
            return _score("min", got, best, blind)
        return f, k

    f_star, floor = float(gold["top_k_mean"]), float(gold["floor"])

    def f(ids):
        return _score(goal, sum(float(cands[i]["y"]) for i in ids) / len(ids), f_star, floor)
    return f, k


# --------------------------------------------------------------------------- #
# free features -- everything derivable from the displayed pool with no chemistry
# --------------------------------------------------------------------------- #
def poscar_features(poscar: str):
    """Geometry read straight off the structure block: counts, volume, density."""
    lines = poscar.splitlines()
    try:
        scale = float(lines[1])
        cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)]) * scale
        vol = abs(float(np.linalg.det(cell)))
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
    except (IndexError, ValueError):
        return None
    n = sum(counts)
    mass = sum(MASS.get(s, 0.0) * c for s, c in zip(species, counts))
    return {"n_atoms": float(n), "n_elements": float(len(species)),
            "cell_volume": vol, "volume_per_atom": vol / n if n else 0.0,
            "mass_density": mass / vol if vol else 0.0}


def free_features(pool: list[dict]) -> dict[str, np.ndarray]:
    """Every chemistry-free ordering key the displayed pool supports."""
    feats: dict[str, np.ndarray] = {}
    cols = [c for c in pool[0] if c not in ("candidate_id", "poscar")]

    for c in cols:                                              # shown numeric columns
        try:
            v = np.array([float(r[c]) for r in pool], dtype=float)
        except (ValueError, TypeError):
            continue
        if len(set(v.tolist())) > K_DEFAULT:
            feats[f"col:{c}"] = v

    if "poscar" in pool[0]:                                     # structure geometry
        rows = [poscar_features(r["poscar"]) for r in pool]
        if all(rows):
            for key in rows[0]:
                v = np.array([r[key] for r in rows], dtype=float)
                if len(set(v.tolist())) > K_DEFAULT:
                    feats[f"struct:{key}"] = v

    for c in cols:                                              # categorical: prefer the mode
        vals = [r[c] for r in pool]
        if f"col:{c}" in feats:
            continue
        freq = collections.Counter(vals)
        if 1 < len(freq) < len(pool):
            feats[f"mode:{c}"] = np.array([freq[v] for v in vals], dtype=float)

    feats["order:printed"] = np.arange(len(pool), dtype=float)  # does pool order leak?
    return feats


# --------------------------------------------------------------------------- #
def find_file(name):
    for c in (f"fixtures/{name}", name):
        if os.path.exists(c):
            return c
    hits = glob.glob(f"fixtures/**/{name}", recursive=True)
    return hits[0] if hits else None


def load_systems():
    rows = json.load(open(LEDGER))["results"]
    sc = collections.defaultdict(dict)
    for r in rows:
        sc[(r["system_id"], r["task_id"])][r["seed_idx"]] = r["task_score"]
    m3 = {k: st.mean(v.values()) for k, v in sc.items()}
    systems = sorted({s for s, _ in m3})
    by_task = collections.defaultdict(dict)
    for (s, t), v in m3.items():
        by_task[t][s] = v
    return systems, by_task


SYSTEMS, BY_TASK = load_systems()
rng = np.random.default_rng(SEED)
results = []
skipped = []

for path in sorted(glob.glob("tasks/end_to_end/*.yaml")):
    t = yaml.safe_load(open(path))
    graders = [g for tn in t["turns"] for g in (tn.get("grading") or [])]
    if len(graders) != 1 or graders[0]["grader"] not in ("b3_select", "select_hypervolume"):
        continue
    tid, grader = t["id"], graders[0]["grader"]

    gp = f"ground_truth/{graders[0]['params']['oracle_ref']}.json"
    if not os.path.exists(gp):
        skipped.append((tid, "no gold")); continue
    gold = json.load(open(gp))

    files = (t.get("inputs") or {}).get("files") or []
    pool_name = next((x for x in files if "pool" in x), None)
    pp = find_file(pool_name) if pool_name else None
    if pp is None:
        skipped.append((tid, "no pool file")); continue
    with open(pp) as fh:
        pool = list(csv.DictReader(fh))
    ids = [r["candidate_id"] for r in pool]
    if not set(ids) <= set(gold["candidates"]):
        skipped.append((tid, "pool ids not in gold")); continue

    score, K = make_scorer(gold, grader)
    feats = free_features(pool)

    # ---- free_max: best ordering, maximized on the pool it scores ---------- #
    free = {}
    for name, v in feats.items():
        order = np.argsort(v, kind="stable")
        free[f"{name}|asc"] = score([ids[i] for i in order[:K]])
        free[f"{name}|desc"] = score([ids[i] for i in order[::-1][:K]])
    n_rankers = len(free)
    best_name = max(free, key=free.get)
    free_max = free[best_name]

    # ---- null_max: the same maximization over random picks ----------------- #
    null_max, singles = [], []
    for _ in range(MC_DRAWS):
        draws = [score(list(rng.choice(ids, K, replace=False))) for _ in range(n_rankers)]
        null_max.append(max(draws)); singles.append(draws[0])

    # ---- gate_free: ranker selected on the OPEN TRAINING TABLE ------------- #
    gate_free, gate_name, gate_rho = None, None, None
    train_name = next((x for x in files if "train" in x), None)
    tp = find_file(train_name) if train_name else None
    if tp:
        with open(tp) as fh:
            train = list(csv.DictReader(fh))
        if train and "target" in train[0]:
            try:
                y = np.array([float(r["target"]) for r in train], dtype=float)
            except (ValueError, TypeError):
                y = None
            if y is not None:
                crit = 1.96 / np.sqrt(max(len(y) - 1, 2))   # 5% critical value for rho
                cand = []
                for name, v in feats.items():
                    key = name.split(":", 1)[1] if ":" in name else None
                    if not key or key not in train[0]:
                        continue
                    try:
                        xv = np.array([float(r[key]) for r in train], dtype=float)
                    except (ValueError, TypeError):
                        continue
                    rho = spearmanr(xv, y).statistic
                    if np.isfinite(rho) and abs(rho) >= crit:
                        cand.append((abs(rho), name, rho))
                if cand:
                    _, gate_name, gate_rho = max(cand)
                    order = np.argsort(feats[gate_name], kind="stable")
                    sel = order[::-1] if gate_rho > 0 else order
                    if gold.get("goal") in ("min", "target"):
                        sel = order if gate_rho > 0 else order[::-1]
                    gate_free = score([ids[i] for i in sel[:K]])

    sys_scores = BY_TASK.get(tid, {})
    sv = [sys_scores[s] for s in SYSTEMS if s in sys_scores]
    results.append(dict(
        task=tid, grader=grader, n_rankers=n_rankers,
        free_max=round(free_max, 4), best_ranker=best_name,
        null_max=round(st.mean(null_max), 4), rand_p50=round(st.median(singles), 4),
        gate_free=None if gate_free is None else round(gate_free, 4),
        gate_ranker=gate_name, gate_rho=None if gate_rho is None else round(float(gate_rho), 3),
        sys_mean=round(st.mean(sv), 4) if sv else None,
        sys_max=round(max(sv), 4) if sv else None,
        sys_min=round(min(sv), 4) if sv else None,
    ))

# --------------------------------------------------------------------------- #
def klass(r):
    """Three-way, and the chance null is what separates the first two.

    A free ranker can beat every system while sitting BELOW the score that maximizing over
    the same number of random picks reaches. That is not a shortcut in the task -- it says
    the systems are performing below chance on it, which is a finding about the systems.
    Only a ranker that clears its own null is exploiting something real.
    """
    if r["sys_max"] is None:
        return "no systems"
    if r["free_max"] > r["sys_max"]:
        return "shortcut" if r["free_max"] > r["null_max"] else "systems below chance"
    if r["free_max"] > r["sys_mean"]:
        return "beats the mean"
    return "clean"


for r in results:
    r["class"] = klass(r)

results.sort(key=lambda r: -(r["free_max"] - (r["sys_max"] or 0)))

print(f"{len(results)} selection tasks audited "
      f"({sum(1 for r in results if r['grader']=='b3_select')} b3_select, "
      f"{sum(1 for r in results if r['grader']=='select_hypervolume')} select_hypervolume)")
if skipped:
    print(f"skipped: {len(skipped)} -> " + ", ".join(f"{a} ({b})" for a, b in skipped))
print()
hdr = (f"{'task':40s} {'free':>6s} {'null':>6s} {'gate':>6s} {'sysmu':>6s} {'sysmax':>6s} "
       f"{'N':>4s}  {'class':18s} best free ranker")
print(hdr); print("-" * len(hdr))
for r in results:
    g = "  --  " if r["gate_free"] is None else f"{r['gate_free']:6.3f}"
    print(f"{r['task']:40s} {r['free_max']:6.3f} {r['null_max']:6.3f} {g} "
          f"{r['sys_mean']:6.3f} {r['sys_max']:6.3f} {r['n_rankers']:4d}  "
          f"{r['class']:18s} {r['best_ranker']}")

print()
c = collections.Counter(r["class"] for r in results)
for k in ("shortcut", "systems below chance", "beats the mean", "clean"):
    print(f"{k:20s}: {c[k]:3d}/{len(results)}")
print()
print(f"mean free_max {st.mean(r['free_max'] for r in results):.3f} "
      f"vs mean null_max {st.mean(r['null_max'] for r in results):.3f} "
      f"vs mean single random pick {st.mean(r['rand_p50'] for r in results):.3f}")
gated = [r for r in results if r["gate_free"] is not None]
if gated:
    print(f"train-selected ranker (gate_free) available on {len(gated)} tasks: "
          f"mean {st.mean(r['gate_free'] for r in gated):.3f}, "
          f"beats every system on {sum(1 for r in gated if r['gate_free'] > r['sys_max'])}")
print(f"free_max at or below its own chance null: "
      f"{sum(1 for r in results if r['free_max'] <= r['null_max'])}/{len(results)}")
below = [r for r in results if r["sys_max"] is not None and r["sys_max"] < r["null_max"]]
print(f"tasks where the BEST system is below the chance null: {len(below)}/{len(results)}"
      + (" -> " + ", ".join(r["task"] for r in below) if below else ""))

if "--json" in sys.argv:
    out = sys.argv[sys.argv.index("--json") + 1]
    json.dump({"tasks": results, "skipped": skipped, "systems": SYSTEMS,
               "mc_draws": MC_DRAWS, "seed": SEED}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")
