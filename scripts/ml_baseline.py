#!/usr/bin/env python3
"""Supervised tabular baseline on the 48 selection tasks that ship a training table.

The question this answers is the first one a reviewer asks: the systems are given a table of
measurements and a pool of anonymized candidates, so what does an ordinary supervised
regression get on exactly the same two files? A benchmark whose tasks are solved by
`fit(train); predict(pool); take the top K` is not measuring materials decisions, it is
measuring tabular regression, and the only way to know which is to run it.

Information parity is the whole point, so it is enforced literally:

  * the model reads exactly the two CSVs inlined into the task prompt and nothing else --
    no source dataset, no other task's files, no held-out gold;
  * hyperparameters are fixed in this file, identical across all 48 tasks, chosen before
    any score was computed. Nothing is selected by how it scores on a pool;
  * where a choice is unavoidable it is made by cross-validation INSIDE the training table
    (`RidgeCV` over a fixed alpha grid);
  * the private half of the suite is not touched;
  * scoring goes through the SAME graders that scored the systems' answers, reached by
    synthesizing the same `Final: c001, ...` line a system would emit.

The headline model is CatBoost, pre-registered on 2026-08-19 before the first run: it is
the default a practitioner reaches for on tabular data, and the only one of the three that
consumes categorical columns natively, so on MOF and perovskite tasks it sees the features
in the same form the system does. RidgeCV and RandomForest are reported alongside as the
spread across model families, never as alternatives to select from.

MOFid is dropped (decision D1). It is shown to the systems but it is a free-text
fingerprint, and hand-featurizing it is a modeling choice this baseline does not make. The
consequence is stated rather than hidden: on the 14 MOF tasks the baseline sees strictly
less than the system, so its MOF number is a lower bound.

Deterministic and fully offline -- `parse_ids` resolves a well-formed `Final:` line without
an LLM, so no network call happens anywhere in this script.

Run:  python scripts/ml_baseline.py [--json ml_baseline.json]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import glob
import json
import os
import random
import re
import statistics as st
import sys
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallmark.chem import parse_formula  # noqa: E402

LEDGER = "results/final.json"
BLOCKS = "blocks.json"
SEEDS = (0, 1, 2, 3, 4)
HEADLINE = "catboost"  # pre-registered 2026-08-19, before any score was computed
DROP = {"candidate_id", "mofid"}  # mofid: free-text fingerprint, decision D1
MC_DRAWS = 200
RNG_SEED = 20260819
NBOOT, BSEED = 10_000, 20260818
MISMATCH: list[dict] = []   # tasks where the prompt's T and the gold's T differ by rounding

SHORT = {
    "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "anthropic/claude-opus-5": "Opus 5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic/claude-fable-5": "Fable 5",
    "openai/gpt-5.5": "GPT-5.5",
    "moonshotai/kimi-k3-medium": "Kimi K3 (med)",
    "anthropic/claude-sonnet-4.6": "Sonnet 4.6",
    "anthropic/claude-sonnet-5": "Sonnet 5",
    "anthropic/claude-opus-4.8": "Opus 4.8",
    "x-ai/grok-4.6": "Grok 4.6",
    "deepseek/deepseek-v4-pro-0813": "DeepSeek V4 Pro",
    "qwen/qwen3.8-max-high": "Qwen3.8 Max (high)",
    "openai/gpt-6-astra": "GPT-6 Astra",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
}

FAMILY = [
    ("htsteel", lambda t: t.startswith("b3_htsteel_")),
    ("MOF", lambda t: t.startswith("b3_mof_")),
    ("thermoelectric", lambda t: t.startswith("b3_te_")),
    ("perovskite", lambda t: t.startswith("b3_pvk_")),
    ("synthetic", lambda t: t.startswith("e2e_")),
    ("B3-real", lambda t: t.startswith("b3_")),
]


def family_of(task: str) -> str:
    for name, pred in FAMILY:
        if pred(task):
            return name
    return "other"


# --------------------------------------------------------------------------- #
# task discovery -- same walk as scripts/shortcut_audit.py
# --------------------------------------------------------------------------- #
def find_file(name):
    if not name:
        return None
    for c in (f"fixtures/{name}", name):
        if os.path.isfile(c):
            return c
    hits = glob.glob(f"fixtures/**/{name}", recursive=True)
    return hits[0] if hits else None


def load_tasks():
    """Every selection task that ships a tabular training table (no POSCAR column)."""
    out = []
    for path in sorted(glob.glob("tasks/end_to_end/*.yaml")):
        t = yaml.safe_load(open(path))
        graders = [g for tn in t["turns"] for g in (tn.get("grading") or [])]
        if len(graders) != 1 or graders[0]["grader"] not in ("b3_select", "select_hypervolume"):
            continue
        files = (t.get("inputs") or {}).get("files") or []
        tp = find_file(next((x for x in files if "train" in x), None))
        pp = find_file(next((x for x in files if "pool" in x), None))
        if not tp or not pp:
            continue
        with open(tp) as fh:
            train = list(csv.DictReader(fh))
        if not train or "poscar" in train[0]:
            continue
        with open(pp) as fh:
            pool = list(csv.DictReader(fh))
        gold_path = f"ground_truth/{graders[0]['params']['oracle_ref']}.json"
        prompt = "\n".join(tn.get("prompt", "") for tn in t["turns"])
        out.append(dict(
            id=t["id"], grader=graders[0]["grader"], params=graders[0]["params"],
            train=train, pool=pool, gold=json.load(open(gold_path)), prompt=prompt,
            train_path=tp, pool_path=pp,
        ))
    return out


# --------------------------------------------------------------------------- #
# featurization -- one rule for all 48
# --------------------------------------------------------------------------- #
def _is_num(vals):
    seen = False
    for v in vals:
        if v is None or v == "":
            continue
        try:
            float(v)
        except (TypeError, ValueError):
            return False
        seen = True
    return seen


def featurize(train_rows, pool_rows, y_cols):
    """Return (Xtr, Xpool, cat_idx) over exactly the columns the prompt displays.

    A lone `composition` string column is expanded into element fractions over the union of
    elements in train+pool -- the same parse `build_b3_tasks` uses to fit its GP -- because a
    single opaque formula string is not a feature any regressor can use. Everything else is
    taken verbatim: numeric columns as floats, the rest as categories.
    """
    feats = [c for c in pool_rows[0] if c not in DROP and c not in y_cols]
    for c in feats:
        if c not in train_rows[0]:
            raise SystemExit(f"column parity broken: {c!r} in pool, not in train")

    # composition-only tasks: expand the formula string into an element-fraction vector
    if feats == ["composition"] and all(parse_formula(r["composition"]) for r in pool_rows[:5]):
        frac_tr = [parse_formula(r["composition"]) or {} for r in train_rows]
        frac_pl = [parse_formula(r["composition"]) or {} for r in pool_rows]
        elems = sorted({e for d in frac_tr + frac_pl for e in d})
        Xtr = [[float(d.get(e, 0.0)) for e in elems] for d in frac_tr]
        Xpl = [[float(d.get(e, 0.0)) for e in elems] for d in frac_pl]
        return np.array(Xtr, float), np.array(Xpl, float), [], elems

    num = [c for c in feats if _is_num([r[c] for r in pool_rows] + [r[c] for r in train_rows])]
    cat = [c for c in feats if c not in num]
    order = num + cat
    cat_idx = list(range(len(num), len(order)))

    def med(col):
        vals = [float(r[col]) for r in train_rows if r[col] not in (None, "")]
        return st.median(vals) if vals else 0.0

    fill = {c: med(c) for c in num}

    def row(r):
        out = []
        for c in num:
            out.append(float(r[c]) if r[c] not in (None, "") else fill[c])
        for c in cat:
            out.append(str(r[c]) if r[c] not in (None, "") else "__missing__")
        return out

    Xtr = [row(r) for r in train_rows]
    Xpl = [row(r) for r in pool_rows]
    return Xtr, Xpl, cat_idx, order


# --------------------------------------------------------------------------- #
# models -- fixed hyperparameters, identical on all 48 tasks
# --------------------------------------------------------------------------- #
def fit_predict(kind, Xtr, ytr, Xpl, cat_idx, seed):
    if kind == "catboost":
        from catboost import CatBoostRegressor, Pool
        m = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                              random_seed=seed, verbose=0, allow_writing_files=False)
        m.fit(Pool(Xtr, ytr, cat_features=cat_idx))
        return np.asarray(m.predict(Pool(Xpl, cat_features=cat_idx)), float)

    # sklearn families need the categoricals one-hot encoded first
    Xtr_n, Xpl_n = _onehot(Xtr, Xpl, cat_idx)
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr_n)
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(Xtr_n), ytr)
        return np.asarray(m.predict(sc.transform(Xpl_n)), float)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
        m.fit(Xtr_n, ytr)
        return np.asarray(m.predict(Xpl_n), float)
    raise ValueError(kind)


def _onehot(Xtr, Xpl, cat_idx):
    if not cat_idx:
        return np.asarray(Xtr, float), np.asarray(Xpl, float)
    ncols = len(Xtr[0])
    num_idx = [i for i in range(ncols) if i not in cat_idx]
    out_tr = [[float(r[i]) for i in num_idx] for r in Xtr]
    out_pl = [[float(r[i]) for i in num_idx] for r in Xpl]
    for i in cat_idx:
        levels = sorted({str(r[i]) for r in Xtr})  # fitted on train only
        for lv in levels:
            for src, dst in ((Xtr, out_tr), (Xpl, out_pl)):
                for j, r in enumerate(src):
                    dst[j].append(1.0 if str(r[i]) == lv else 0.0)
    return np.asarray(out_tr, float), np.asarray(out_pl, float)


# --------------------------------------------------------------------------- #
# the second condition -- constraints that live in the prompt, not in the table
# --------------------------------------------------------------------------- #
# Six of the 48 tasks attach a hard constraint to the objective ("pick the highest X among
# candidates that also satisfy C"). The gold folds C into the label -- an infeasible pick is
# worth 0 -- but the TRAINING TABLE contains no counterexample: five of the six have zero
# rows with target 0, so C is present nowhere except the sentence in the prompt. A regressor
# fitted on that table is therefore structurally blind to C, which is not a defect of the
# baseline but a property of the tasks worth measuring.
#
# To separate "the regression is weak" from "the regression never read the prompt", each
# machine-checkable constraint is transcribed here from its prompt sentence and applied as a
# hard filter before ranking. This is a STRONGER baseline than ordinary supervised
# regression: a human transcribed the constraint. Both numbers are reported; neither
# replaces the other.
#
# `b3_mof_co2_lowheat_select` is deliberately absent. Its constraint is an isosteric heat of
# adsorption <= 30 kJ/mol, and no column of the pool encodes binding strength -- it cannot be
# filtered from what the prompt displays at all, by any tabular pipeline.
def _nel_above(row, cols, thr):
    return sum(1 for c in cols if c != "Fe" and float(row[c]) > thr)


_LEAN_COLS = ["Fe", "Cr", "Ni", "Mo", "Mn", "C", "Ti", "Si", "Nb", "V", "P", "B",
              "Al", "Cu", "W", "S"]
_MOF_SCARCE = {"Ag", "Cd", "Hf", "Hg", "Mo", "Nb", "Pb", "Pd", "Rh", "Ru", "V", "W", "Zr"}
_TE_TOXIC = ["Pb", "Te", "Cd", "Hg", "As"]

CONSTRAINTS = {
    "b3_htsteel_creep_leanalloy_select":
        ("at most 4 alloying elements above 0.3 wt%",
         lambda r: _nel_above(r, _LEAN_COLS, 0.3) <= 4),
    "b3_htsteel_hot_nickel_lean_select":
        ("Ni at most 2 wt%", lambda r: float(r["Ni"]) <= 2.0),
    "b3_htsteel_creep_austenitic_select":
        ("Class == A", lambda r: r["Class"] == "A"),
    "b3_te_nontoxic_select":
        ("no Pb, Te, Cd, Hg, As",
         lambda r: all(float(r.get(e, 0) or 0) == 0 for e in _TE_TOXIC)),
    "b3_mof_co2_earthabundant_select":
        ("metal node earth-abundant", lambda r: r["metal"] not in _MOF_SCARCE),
}
UNFILTERABLE = {"b3_mof_co2_lowheat_select": "isosteric heat is not a displayed column"}


def check_constraint(task):
    """A transcribed constraint must zero out every candidate the gold zeroes for it.

    The check is one-directional, and that direction is the load-bearing one: infeasible
    must imply a zero label, because the gold folds the constraint in. The converse does not
    hold -- a feasible candidate can carry a genuine measured zero, as `c043` of
    `b3_mof_co2_earthabundant_select` does (metal Zn, allowed, uptake 0.0). Requiring
    equivalence flagged that legitimate row as a transcription error.

    A predicate that is too permissive would silently flatter the baseline, so a violation
    in the load-bearing direction is fatal rather than a warning.
    """
    _desc, pred = CONSTRAINTS[task["id"]]
    cands = task["gold"]["candidates"]
    bad = [r["candidate_id"] for r in task["pool"]
           if not pred(r) and float(cands[r["candidate_id"]]["y"]) != 0.0]
    if bad:
        raise SystemExit(f"{task['id']}: constraint predicate calls {len(bad)} candidates "
                         f"infeasible that the gold scores above zero, e.g. {bad[:5]}")
    feasible = [i for i, r in enumerate(task["pool"]) if pred(r)]
    if not feasible:
        raise SystemExit(f"{task['id']}: constraint leaves no feasible candidate")
    return feasible


# --------------------------------------------------------------------------- #
# ranking rules -- exactly what the task asks for, no access to true labels
# --------------------------------------------------------------------------- #
def rank_scalar(pred, goal, T, k, allow=None):
    if goal == "target":
        order = np.argsort(np.abs(pred - T), kind="stable")
    elif goal == "min":
        order = np.argsort(pred, kind="stable")
    else:
        order = np.argsort(-pred, kind="stable")
    if allow is not None:
        ok = set(allow)
        order = [i for i in order if i in ok] + [i for i in order if i not in ok]
    return list(order[:k])


def rank_pareto(p1, p2, k):
    """Non-dominated K by PREDICTED (y1, y2), both maximized; ties broken by normalized sum.

    The grader rewards dominated hypervolume, so a set spanning the predicted front beats a
    set stacked on one axis. Selection uses predictions only.
    """
    def norm(v):
        lo, hi = float(np.min(v)), float(np.max(v))
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    n1, n2 = norm(p1), norm(p2)
    n = len(p1)
    picked, remaining = [], set(range(n))
    while len(picked) < k and remaining:
        front = [i for i in remaining
                 if not any(j != i and p1[j] >= p1[i] and p2[j] >= p2[i]
                            and (p1[j] > p1[i] or p2[j] > p2[i]) for j in remaining)]
        front.sort(key=lambda i: -(n1[i] + n2[i]))
        for i in front:
            if len(picked) == k:
                break
            picked.append(i)
            remaining.discard(i)
    return picked[:k]


# --------------------------------------------------------------------------- #
# scoring -- the real graders, reached through a synthesized `Final:` line
# --------------------------------------------------------------------------- #
def make_grader():
    from hallmark.graders.b3_select import B3SelectGrader
    from hallmark.graders.select_hypervolume import SelectHypervolumeGrader
    from hallmark.loader import GroundTruth
    from hallmark.models import TurnTranscript
    return B3SelectGrader(), SelectHypervolumeGrader(), GroundTruth, TurnTranscript


_G = None


def grade(task, ids):
    """Score a pick list with the SAME grader path that scored the systems' answers."""
    global _G
    if _G is None:
        _G = make_grader()
    b3, hv, GroundTruth, TurnTranscript = _G
    grader = b3 if task["grader"] == "b3_select" else hv
    turn = TurnTranscript(prompt="", answer_text="Final: " + ", ".join(ids))
    score, _detail = asyncio.run(grader.grade(turn, task["params"], GroundTruth()))
    return float(score)


def score_replica(gold, grader, ids):
    """Independent re-implementation of the same formula, as a cross-check on `grade`."""
    cands, k = gold["candidates"], int(gold["k"])

    def norm(goal, got, star, floor):
        if star == floor:
            return 1.0 if got == star else 0.0
        v = (floor - got) / (floor - star) if goal == "min" else (got - floor) / (star - floor)
        return float(min(max(v, 0.0), 1.0))

    if grader == "select_hypervolume":
        from hallmark.graders.pareto_hypervolume import _hv2d
        hv = _hv2d([tuple(cands[i]["p"]) for i in ids], gold["ref_point"])
        return norm("max", hv, float(gold["hv_star"]), float(gold["floor"]))

    goal = gold.get("goal", "max")
    if goal == "target":
        T = float(gold["target_value"])
        dist = sorted(abs(float(c["y"]) - T) for c in cands.values())
        best, blind = sum(dist[:k]) / k, sum(dist) / len(dist)
        got = sum(abs(float(cands[i]["y"]) - T) for i in ids) / len(ids)
        return norm("min", got, best, blind)

    got = sum(float(cands[i]["y"]) for i in ids) / len(ids)
    return norm(goal, got, float(gold["top_k_mean"]), float(gold["floor"]))


# --------------------------------------------------------------------------- #
def target_from_prompt(task):
    """Use the T the PROMPT prints, not the T the gold stores -- parity, not convenience.

    The two are not always the same number: `e2e_target_select` asks for the candidates
    closest to 1051 MPa while its gold scores against 1050.8861, the prompt having rounded at
    build time. A system optimizing what it was told aims at 1051, so the baseline does too.
    A mismatch beyond rounding would be a real defect, so anything past 0.1% relative is a
    hard failure rather than a silent substitution.
    """
    T_gold = float(task["gold"]["target_value"])
    nums = [float(m) for m in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", task["prompt"])]
    near = [v for v in nums if abs(v - T_gold) <= max(1e-3 * abs(T_gold), 1e-9)]
    if not near:
        raise SystemExit(f"{task['id']}: gold target_value {T_gold} has no counterpart in the prompt")
    T = min(near, key=lambda v: abs(v - T_gold))
    if T != T_gold:
        MISMATCH.append(dict(task=task["id"], prompt=T, gold=T_gold))
    return T


def cluster_boot(per_block: dict, n=NBOOT, seed=BSEED):
    keys = sorted(per_block)
    rnd = random.Random(seed)
    means = []
    for _ in range(n):
        draw = [per_block[keys[rnd.randrange(len(keys))]] for _ in range(len(keys))]
        flat = [v for d in draw for v in d]
        means.append(sum(flat) / len(flat))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--models", default="catboost,ridge,rf")
    args = ap.parse_args()
    models = args.models.split(",")

    tasks = load_tasks()
    print(f"{len(tasks)} tabular selection tasks "
          f"({sum(1 for t in tasks if t['grader'] == 'b3_select')} b3_select, "
          f"{sum(1 for t in tasks if t['grader'] == 'select_hypervolume')} select_hypervolume)")

    rng = np.random.default_rng(RNG_SEED)
    per_task = {}

    for task in tasks:
        gold, grader = task["gold"], task["grader"]
        k = int(gold["k"])
        ids = [r["candidate_id"] for r in task["pool"]]
        if not set(ids) <= set(gold["candidates"]):
            raise SystemExit(f"{task['id']}: pool ids not in gold")

        if grader == "select_hypervolume":
            y_cols = [c for c in task["train"][0] if c not in task["pool"][0]]
            if len(y_cols) != 2:
                raise SystemExit(f"{task['id']}: expected 2 objective columns, got {y_cols}")
        else:
            y_cols = ["target"]

        Xtr, Xpl, cat_idx, cols = featurize(task["train"], task["pool"], set(y_cols))
        T = target_from_prompt(task) if gold.get("goal") == "target" else None
        allow = check_constraint(task) if task["id"] in CONSTRAINTS else None

        rec = dict(task=task["id"], family=family_of(task["id"]), grader=grader,
                   goal=gold.get("goal", "max"), n_train=len(task["train"]),
                   n_pool=len(ids), n_features=len(cols), k=k, models={})

        aware: list[float] = []
        for kind in models:
            per_seed = []
            for seed in SEEDS:
                if grader == "select_hypervolume":
                    preds = []
                    for col in y_cols:
                        y = np.array([float(r[col]) for r in task["train"]], float)
                        preds.append(fit_predict(kind, Xtr, y, Xpl, cat_idx, seed))
                    idx = rank_pareto(preds[0], preds[1], k)
                else:
                    y = np.array([float(r["target"]) for r in task["train"]], float)
                    pred = fit_predict(kind, Xtr, y, Xpl, cat_idx, seed)
                    idx = rank_scalar(pred, gold.get("goal", "max"), T, k)
                    if kind == HEADLINE and allow is not None:
                        filtered = [ids[i] for i in rank_scalar(
                            pred, gold.get("goal", "max"), T, k, allow=allow)]
                        aware.append(grade(task, filtered))
                picks = [ids[i] for i in idx]
                s = grade(task, picks)
                s_rep = score_replica(gold, grader, picks)
                if abs(s - s_rep) > 1e-9:
                    raise SystemExit(f"{task['id']}/{kind}/{seed}: grader {s} != replica {s_rep}")
                per_seed.append(s)
            rec["models"][kind] = dict(
                mean=round(st.mean(per_seed), 4),
                sd=round(st.pstdev(per_seed), 4),
                lo=round(min(per_seed), 4), hi=round(max(per_seed), 4),
                seeds=[round(v, 4) for v in per_seed],
            )

        if aware:
            rec["constraint_aware"] = dict(
                constraint=CONSTRAINTS[task["id"]][0],
                mean=round(st.mean(aware), 4), sd=round(st.pstdev(aware), 4),
                seeds=[round(v, 4) for v in aware])
        elif task["id"] in UNFILTERABLE:
            rec["constraint_unfilterable"] = UNFILTERABLE[task["id"]]

        # controls: random K-picks
        draws = [grade(task, list(rng.choice(ids, k, replace=False))) for _ in range(MC_DRAWS)]
        draws.sort()
        rec["random_p50"] = round(draws[MC_DRAWS // 2], 4)
        rec["random_p95"] = round(draws[int(0.95 * MC_DRAWS)], 4)

        per_task[task["id"]] = rec
        head = rec["models"].get(HEADLINE) or list(rec["models"].values())[0]
        print(f"  {task['id']:42s} {HEADLINE}={head['mean']:.3f}"
              f" (sd {head['sd']:.3f})  rand50={rec['random_p50']:.3f}")

    # ---------------- compare with the systems on the SAME 48 tasks ---------- #
    rows = json.load(open(LEDGER))["results"]
    SC = collections.defaultdict(dict)
    for r in rows:
        SC[(r["system_id"], r["task_id"])][r["seed_idx"]] = r["task_score"]
    M3 = {kk: st.mean(v.values()) for kk, v in SC.items()}
    SYS = sorted({s for s, _ in M3})
    TIDS = sorted(per_task)
    ALLT = sorted({t for _, t in M3})

    blocks = json.load(open(BLOCKS))["blocks"]
    BLK = {t: b["key"] for b in blocks for t in b["tasks"]}
    by_block = collections.defaultdict(list)
    for t in TIDS:
        by_block[BLK[t]].append(t)

    # free ranker chosen on the training table, from the shortcut audit
    gate = {}
    ap_json = "shortcut_audit.json"
    if os.path.exists(ap_json):
        for r in json.load(open(ap_json))["tasks"]:
            if r.get("gate_free") is not None:
                gate[r["task"]] = r["gate_free"]

    summary = {"n_tasks": len(TIDS), "headline": HEADLINE, "models": {}, "systems": {},
               "controls": {}, "by_family": {}, "vs_systems": [],
               "prompt_gold_target_mismatch": MISMATCH}

    for kind in models:
        vals = [per_task[t]["models"][kind]["mean"] for t in TIDS]
        summary["models"][kind] = round(st.mean(vals), 4)

    # the same headline model, with each prompt-only constraint transcribed and enforced
    aware_by_task = {t: (per_task[t].get("constraint_aware") or {}).get(
        "mean", per_task[t]["models"][HEADLINE]["mean"]) for t in TIDS}
    ctasks = [t for t in TIDS if "constraint_aware" in per_task[t]]
    summary["constraint_aware"] = dict(
        all48=round(st.mean([aware_by_task[t] for t in TIDS]), 4),
        n_constrained=len(ctasks),
        n_unfilterable=len([t for t in TIDS if "constraint_unfilterable" in per_task[t]]),
        on_constrained_naive=round(st.mean(
            [per_task[t]["models"][HEADLINE]["mean"] for t in ctasks]), 4),
        on_constrained_aware=round(st.mean([aware_by_task[t] for t in ctasks]), 4),
        per_task={t: dict(naive=per_task[t]["models"][HEADLINE]["mean"],
                          aware=aware_by_task[t],
                          constraint=per_task[t]["constraint_aware"]["constraint"])
                  for t in ctasks})
    def mean_ci(vals_by_task: dict, tids: list[str], blockmap: dict) -> tuple:
        """Mean with the same percentile cluster bootstrap the leaderboard uses."""
        pb = collections.defaultdict(list)
        for t in tids:
            pb[blockmap[t]].append(vals_by_task[t])
        lo, hi = cluster_boot(dict(pb))
        return round(st.mean([vals_by_task[t] for t in tids]), 4), [round(lo, 4), round(hi, 4)]

    all_blocks = collections.defaultdict(list)
    for t in ALLT:
        all_blocks[BLK[t]].append(t)
    for s in SYS:
        m48, ci48 = mean_ci({t: M3[(s, t)] for t in TIDS}, TIDS, BLK)
        m92, ci92 = mean_ci({t: M3[(s, t)] for t in ALLT}, ALLT, BLK)
        summary["systems"][s] = dict(tab=m48, tab_ci=ci48, all=m92, all_ci=ci92)
    summary["controls"]["random_p50"] = round(st.mean([per_task[t]["random_p50"] for t in TIDS]), 4)
    summary["controls"]["random_p95"] = round(st.mean([per_task[t]["random_p95"] for t in TIDS]), 4)
    if gate:
        gv = [gate[t] for t in TIDS if t in gate]
        summary["controls"]["gate_free"] = dict(mean=round(st.mean(gv), 4), n=len(gv))

    for fam in sorted({per_task[t]["family"] for t in TIDS}):
        ft = [t for t in TIDS if per_task[t]["family"] == fam]
        summary["by_family"][fam] = dict(
            n=len(ft),
            baseline=round(st.mean([per_task[t]["models"][HEADLINE]["mean"] for t in ft]), 4),
            systems=round(st.mean([M3[(s, t)] for s in SYS for t in ft]), 4),
            best_system=round(max(st.mean([M3[(s, t)] for t in ft]) for s in SYS), 4))

    def compare(base: dict) -> list[dict]:
        """One pick-vector against every system on the same tasks, Holm-corrected."""
        pvals, recs = [], []
        for sysid in SYS:
            diff = {t: base[t] - M3[(sysid, t)] for t in TIDS}
            pb = {b: [diff[t] for t in ts] for b, ts in by_block.items()}
            lo, hi = cluster_boot(pb)
            # two-sided bootstrap p: resamples on the wrong side of zero, doubled
            keys = sorted(pb)
            rnd = random.Random(BSEED)
            cnt = 0
            for _ in range(NBOOT):
                dr = [pb[keys[rnd.randrange(len(keys))]] for _ in range(len(keys))]
                flat = [v for d in dr for v in d]
                if (sum(flat) / len(flat)) <= 0:
                    cnt += 1
            pvals.append(min(1.0, 2 * min(cnt, NBOOT - cnt) / NBOOT))
            recs.append(dict(
                system=sysid, short=SHORT.get(sysid, sysid),
                sys_mean=round(st.mean([M3[(sysid, t)] for t in TIDS]), 4),
                diff=round(st.mean(diff.values()), 4),
                ci=[round(lo, 4), round(hi, 4)], p=round(pvals[-1], 4),
                wins=sum(1 for t in TIDS if base[t] > M3[(sysid, t)]),
                losses=sum(1 for t in TIDS if base[t] < M3[(sysid, t)]),
                ties=sum(1 for t in TIDS if base[t] == M3[(sysid, t)])))
        for r, ph in zip(recs, holm(pvals)):
            r["p_holm"] = round(ph, 4)
            r["sig_holm"] = bool(ph < 0.05)
        return sorted(recs, key=lambda r: -r["sys_mean"])

    base = {t: per_task[t]["models"][HEADLINE]["mean"] for t in TIDS}
    summary["vs_systems_by_variant"] = {"naive": compare(base),
                                        "constraint_aware": compare(aware_by_task)}
    summary["vs_systems"] = summary["vs_systems_by_variant"]["naive"]

    for vname, vec in (("naive", base), ("constraint_aware", aware_by_task)):
        m, ci = mean_ci(vec, TIDS, BLK)
        summary.setdefault("baseline_ci", {})[vname] = dict(mean=m, ci=ci)
    for kind in models:
        if kind == HEADLINE:
            continue
        m, ci = mean_ci({t: per_task[t]["models"][kind]["mean"] for t in TIDS}, TIDS, BLK)
        summary["baseline_ci"][kind] = dict(mean=m, ci=ci)
    for name, key in (("random_p50", "random_p50"), ("random_p95", "random_p95")):
        m, ci = mean_ci({t: per_task[t][key] for t in TIDS}, TIDS, BLK)
        summary["baseline_ci"][name] = dict(mean=m, ci=ci)

    beats_all = [t for t in TIDS if base[t] > max(M3[(s, t)] for s in SYS)]
    beats_none = [t for t in TIDS if base[t] < min(M3[(s, t)] for s in SYS)]
    summary["beats_every_system"] = beats_all
    summary["below_every_system"] = beats_none

    # ---------------- report ------------------------------------------------ #
    if MISMATCH:
        print("\nprompt/gold target mismatch (baseline used the prompt value):")
        for m in MISMATCH:
            print(f"  {m['task']:42s} prompt={m['prompt']:g}  gold={m['gold']:g}")

    print(f"\n=== baseline on {len(TIDS)} tabular selection tasks ===")
    for kind in models:
        tag = "  <- headline" if kind == HEADLINE else ""
        print(f"  {kind:10s} {summary['models'][kind]:.4f}{tag}")
    print(f"  {'random p50':10s} {summary['controls']['random_p50']:.4f}")
    print(f"  {'random p95':10s} {summary['controls']['random_p95']:.4f}")
    if "gate_free" in summary["controls"]:
        g = summary["controls"]["gate_free"]
        print(f"  {'free rank':10s} {g['mean']:.4f}  (n={g['n']}, chosen on the train table)")

    ca = summary["constraint_aware"]
    print(f"\n=== the second condition: {ca['n_constrained']} tasks whose constraint is "
          f"transcribed from the prompt ({ca['n_unfilterable']} cannot be filtered at all) ===")
    print(f"{'task':42s} {'naive':>7s} {'aware':>7s}  constraint")
    for t, d in ca["per_task"].items():
        print(f"{t:42s} {d['naive']:7.3f} {d['aware']:7.3f}  {d['constraint']}")
    print(f"{'  on those tasks':42s} {ca['on_constrained_naive']:7.3f} "
          f"{ca['on_constrained_aware']:7.3f}")
    print(f"{'  over all 48':42s} {summary['models'][HEADLINE]:7.3f} {ca['all48']:7.3f}")

    print(f"\n=== {HEADLINE} vs each system, same {len(TIDS)} tasks "
          f"(cluster bootstrap over {len(by_block)} blocks, Holm over {len(SYS)}) ===")
    print(f"{'system':16s} {'sys':>6s} {'base-sys':>9s} {'95% CI':>18s} "
          f"{'W/L/T':>10s} {'p_holm':>7s}")
    for vname in ("naive", "constraint_aware"):
        print(f"-- {vname}")
        for r in summary["vs_systems_by_variant"][vname]:
            print(f"{r['short']:16s} {r['sys_mean']:6.4f} {r['diff']:+9.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
                  f"{r['wins']:>3d}/{r['losses']:<3d}/{r['ties']:<2d} {r['p_holm']:7.4f}"
                  f"{'  *' if r['sig_holm'] else ''}")

    print("\n=== by family ===")
    print(f"{'family':16s} {'n':>3s} {'baseline':>9s} {'sys mean':>9s} {'best sys':>9s}")
    for fam, d in sorted(summary["by_family"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"{fam:16s} {d['n']:3d} {d['baseline']:9.4f} {d['systems']:9.4f} "
              f"{d['best_system']:9.4f}")

    print(f"\nbaseline beats every system on {len(beats_all)}/{len(TIDS)} tasks: "
          + (", ".join(beats_all) if beats_all else "none"))
    print(f"baseline below every system on {len(beats_none)}/{len(TIDS)} tasks: "
          + (", ".join(beats_none) if beats_none else "none"))

    if args.json:
        json.dump({"summary": summary, "tasks": [per_task[t] for t in TIDS]},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
