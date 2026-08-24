"""Step 5 — metrics, tables and figures for H1-H5 (see PLAN.md §1.5).

Unit of analysis for the style effect is the ANSWER (n<=42), with judges averaged
inside a condition; per-judge numbers are reported separately. Treating each
(answer x judge) as an independent observation would triple the apparent n and
inflate every interval — the three judges score the same text.

Writes results/metrics.json, results/tables/*.md, results/figures/*.pdf.

Run:  python analysis/judge_vs_regret/scripts/05_analyze.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import DATA, RES, SEED, ensure_dirs, family, read_jsonl  # noqa: E402

BOOT = 10000


# ------------------------------------------------------------------ helpers

def boot_ci(xs: list[float], stat=np.mean, alpha: float = 0.05) -> tuple[float, float]:
    """Deterministic percentile bootstrap CI (same convention as hallmark/scoring.py)."""
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    arr = np.asarray(xs, dtype=float)
    draws = stat(rng.choice(arr, size=(BOOT, len(arr)), replace=True), axis=1)
    return tuple(float(v) for v in np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def corr(x: list[float], y: list[float]) -> dict:
    if len(x) < 3:
        return dict(n=len(x), pearson=None, pearson_p=None, spearman=None, spearman_p=None)
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return dict(n=len(x), pearson=float(pr.statistic), pearson_p=float(pr.pvalue),
                spearman=float(sr.statistic), spearman_p=float(sr.pvalue))


def fmt(v, nd=3) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


# ------------------------------------------------------------------ load

def load() -> tuple[dict, dict, dict]:
    sample = json.loads((DATA / "sample.json").read_text())
    meta = {it["uid"]: it for it in sample["items"]}

    obj = {(r["uid"], r["style"]): r["score"] for r in read_jsonl(DATA / "objective.jsonl")}

    judge: dict[tuple[str, str, str, str], float] = {}
    for r in read_jsonl(DATA / "judge_scores.jsonl"):
        if r.get("score") is not None:  # last write wins (a retry supersedes an error)
            judge[(r["uid"], r["style"], r["judge"], r["condition"])] = float(r["score"])
    return meta, obj, judge


def judges_conditions(judge: dict) -> tuple[list[str], list[str]]:
    return (sorted({k[2] for k in judge}), sorted({k[3] for k in judge}))


def mean_over_judges(judge: dict, uid: str, style: str, cond: str, js: list[str]) -> float | None:
    vals = [judge[(uid, style, j, cond)] for j in js if (uid, style, j, cond) in judge]
    return statistics.mean(vals) if vals else None


# ------------------------------------------------------------------ H1

def h1(meta, obj, judge, js, cs) -> dict:
    out = {}
    for c in cs:
        for j in js + ["POOLED"]:
            xs, ys = [], []
            for uid in meta:
                o = obj.get((uid, "orig"))
                v = (mean_over_judges(judge, uid, "orig", c, js) if j == "POOLED"
                     else judge.get((uid, "orig", j, c)))
                if o is not None and v is not None:
                    xs.append(o), ys.append(v)
            r = corr(xs, ys)
            r["judge_mean"] = statistics.mean(ys) if ys else None
            r["objective_mean"] = statistics.mean(xs) if xs else None
            zeros = [y for x, y in zip(xs, ys) if x == 0.0]
            r["n_objective_zero"] = len(zeros)
            r["judge_mean_on_zero"] = statistics.mean(zeros) if zeros else None
            r["share_judge_ge_0.9_on_zero"] = (
                sum(1 for y in zeros if y >= 0.9) / len(zeros) if zeros else None
            )
            out[f"{c}|{j}"] = r
    return out


# ------------------------------------------------------------------ H2

def h2(meta, obj, judge, js, cs) -> dict:
    by_task = collections.defaultdict(list)
    for uid, m in meta.items():
        by_task[m["task_id"]].append(uid)

    out = {}
    for c in cs:
        for j in js + ["POOLED"]:
            for scope in ("low_high", "all_pairs"):
                hits = ties = total = 0
                for _tid, uids in by_task.items():
                    pairs = []
                    if scope == "low_high":
                        lo = [u for u in uids if meta[u]["level"] == "low"]
                        hi = [u for u in uids if meta[u]["level"] == "high"]
                        pairs = [(a, b) for a in lo for b in hi]
                    else:
                        pairs = [(a, b) for i, a in enumerate(uids) for b in uids[i + 1:]]
                    for a, b in pairs:
                        oa, ob = obj.get((a, "orig")), obj.get((b, "orig"))
                        va = (mean_over_judges(judge, a, "orig", c, js) if j == "POOLED"
                              else judge.get((a, "orig", j, c)))
                        vb = (mean_over_judges(judge, b, "orig", c, js) if j == "POOLED"
                              else judge.get((b, "orig", j, c)))
                        if None in (oa, ob, va, vb) or oa == ob:
                            continue
                        total += 1
                        if va == vb:
                            ties += 1
                        elif (va > vb) == (oa > ob):
                            hits += 1
                # ties count as half a hit — a judge that cannot separate is not correct
                acc = (hits + 0.5 * ties) / total if total else None
                ci = (stats.binomtest(int(round(hits + 0.5 * ties)), total).proportion_ci()
                      if total else None)
                out[f"{c}|{j}|{scope}"] = dict(
                    n_pairs=total, hits=hits, ties=ties, accuracy=acc,
                    ci_low=float(ci.low) if ci else None, ci_high=float(ci.high) if ci else None,
                )
    return out


# ------------------------------------------------------------------ H3

def h3(meta, judge, js, cs) -> dict:
    out = {}
    for c in cs:
        for j in js + ["POOLED"]:
            deltas, per_family = [], collections.defaultdict(list)
            for uid, m in meta.items():
                get = (lambda s: mean_over_judges(judge, uid, s, c, js)) if j == "POOLED" \
                    else (lambda s: judge.get((uid, s, j, c)))
                t, v = get("terse"), get("verbose")
                if t is None or v is None:
                    continue
                deltas.append(v - t)
                per_family[family(m["task_id"])].append(v - t)
            if not deltas:
                continue
            lo, hi = boot_ci(deltas)
            try:
                w = stats.wilcoxon(deltas)
                wp = float(w.pvalue)
            except ValueError:  # all-zero differences
                wp = None
            out[f"{c}|{j}"] = dict(
                n=len(deltas), mean_delta=float(np.mean(deltas)),
                median_delta=float(np.median(deltas)), ci_low=lo, ci_high=hi,
                wilcoxon_p=wp, share_positive=float(np.mean([d > 0 for d in deltas])),
                by_family={f: float(np.mean(v)) for f, v in sorted(per_family.items())},
            )
    return out


def h3_vs_orig(meta, judge, js, cs) -> dict:
    out = {}
    for c in cs:
        for style in ("terse", "verbose"):
            deltas = []
            for uid in meta:
                o = mean_over_judges(judge, uid, "orig", c, js)
                s = mean_over_judges(judge, uid, style, c, js)
                if o is not None and s is not None:
                    deltas.append(s - o)
            if deltas:
                lo, hi = boot_ci(deltas)
                out[f"{c}|{style}-orig"] = dict(n=len(deltas), mean_delta=float(np.mean(deltas)),
                                                ci_low=lo, ci_high=hi)
    return out


# ------------------------------------------------------------------ self-preference

def self_preference(meta, judge, js, cs) -> dict:
    """Does a judge score answers written by its own model higher?

    Three of the sampled systems are also judges (10 of 42 answers), so this confound
    has to be measured rather than assumed away. It cannot touch H3 — that effect is
    paired within one answer — but it can bias H1/H2.
    """
    out = {}
    for c in cs:
        for j in js:
            own, other = [], []
            for uid, m in meta.items():
                v = judge.get((uid, "orig", j, c))
                if v is None:
                    continue
                (own if m["system_id"] == j else other).append(v)
            out[f"{c}|{j}"] = dict(
                n_own=len(own), n_other=len(other),
                mean_own=float(np.mean(own)) if own else None,
                mean_other=float(np.mean(other)) if other else None,
                delta=(float(np.mean(own) - np.mean(other)) if own and other else None),
            )
    return out


# ------------------------------------------------------------------ H4

def h4(obj, meta) -> dict:
    moved = [(uid, st, s - obj[(uid, "orig")])
             for (uid, st), s in obj.items() if st != "orig"]
    worst = max((abs(d) for _, _, d in moved), default=0.0)
    return dict(n_rewrites=len(moved), max_abs_delta=float(worst),
                n_moved=sum(1 for _, _, d in moved if abs(d) > 1e-12),
                n_answers=len(meta))


# ------------------------------------------------------------------ tables

def write_tables(m: dict, js: list[str], cs: list[str]) -> None:
    t = RES / "tables"

    rows = ["| condition | judge | n | Pearson | p | Spearman | p | mean judge | mean objective |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in m["h1"].items():
        c, j = k.split("|")
        rows.append(f"| {c} | {j} | {r['n']} | {fmt(r['pearson'])} | {fmt(r['pearson_p'])} | "
                    f"{fmt(r['spearman'])} | {fmt(r['spearman_p'])} | {fmt(r['judge_mean'])} | "
                    f"{fmt(r['objective_mean'])} |")
    (t / "h1_correlation.md").write_text(
        "# H1 — judge score vs objective score (original answers)\n\n" + "\n".join(rows) + "\n")

    rows = ["| condition | judge | scope | pairs | accuracy | 95% CI |", "|---|---|---|---:|---:|---|"]
    for k, r in m["h2"].items():
        c, j, sc = k.split("|")
        ci = f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] is not None else "—"
        rows.append(f"| {c} | {j} | {sc} | {r['n_pairs']} | {fmt(r['accuracy'])} | {ci} |")
    (t / "h2_pairwise.md").write_text(
        "# H2 — within-task pairwise accuracy (chance = 0.5)\n\n" + "\n".join(rows) + "\n")

    rows = ["| condition | judge | n | mean Δ(verbose−terse) | 95% CI | median | Wilcoxon p | share Δ>0 |",
            "|---|---|---:|---:|---|---:|---:|---:|"]
    for k, r in m["h3"].items():
        c, j = k.split("|")
        rows.append(f"| {c} | {j} | {r['n']} | {fmt(r['mean_delta'])} | "
                    f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] | {fmt(r['median_delta'])} | "
                    f"{fmt(r['wilcoxon_p'])} | {fmt(r['share_positive'], 2)} |")
    rows += ["", "Objective score, same texts: "
             f"max |Δ| = {m['h4']['max_abs_delta']:.1e} over {m['h4']['n_rewrites']} rewrites."]
    (t / "h3_style.md").write_text(
        "# H3/H4 — style effect on the judge, with the objective grader as control\n\n"
        + "\n".join(rows) + "\n")

    rows = ["| condition | metric | value |", "|---|---|---:|"]
    for c in cs:
        r1, r3 = m["h1"].get(f"{c}|POOLED"), m["h3"].get(f"{c}|POOLED")
        if r1:
            rows.append(f"| {c} | Spearman(judge, objective) | {fmt(r1['spearman'])} |")
            rows.append(f"| {c} | judge mean on objective=0 | {fmt(r1['judge_mean_on_zero'])} |")
            rows.append(f"| {c} | share judge≥0.9 on objective=0 | "
                        f"{fmt(r1['share_judge_ge_0.9_on_zero'], 2)} |")
        if r3:
            rows.append(f"| {c} | mean Δ(verbose−terse) | {fmt(r3['mean_delta'])} |")
    (t / "h5_reference.md").write_text(
        "# H5 — does giving the judge the ground truth rescue it?\n\n" + "\n".join(rows) + "\n")

    rows = ["| condition | judge | n own | mean own | n other | mean other | Δ |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for k, r in m["self_preference"].items():
        c, j = k.split("|")
        rows.append(f"| {c} | {j} | {r['n_own']} | {fmt(r['mean_own'])} | {r['n_other']} | "
                    f"{fmt(r['mean_other'])} | {fmt(r['delta'])} |")
    (t / "self_preference.md").write_text(
        "# Confound check — judges scoring answers written by their own model\n\n"
        + "\n".join(rows) + "\n\nAffects H1/H2 only; H3 is paired within a single answer.\n")


# ------------------------------------------------------------------ figures

def write_figures(meta, obj, judge, js, cs) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(js, ["#4C6EF5", "#E8590C", "#2B8A3E"]))

    fig, axes = plt.subplots(1, len(cs), figsize=(5.2 * len(cs), 4.4), squeeze=False)
    for ax, c in zip(axes[0], cs):
        for j in js:
            xs = [obj[(u, "orig")] for u in meta if (u, "orig", j, c) in judge]
            ys = [judge[(u, "orig", j, c)] for u in meta if (u, "orig", j, c) in judge]
            ax.scatter(xs, ys, s=34, alpha=0.75, label=j.split("/")[-1], color=colors.get(j))
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#adb5bd")
        ax.set_xlabel("objective score (regret vs known optimum)")
        ax.set_ylabel("judge score")
        ax.set_title(c)
        ax.set_xlim(-0.03, 1.03), ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "figures" / "fig1_judge_vs_objective.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(cs), figsize=(4.6 * len(cs), 4.4), squeeze=False)
    for ax, c in zip(axes[0], cs):
        for uid in meta:
            t = mean_over_judges(judge, uid, "terse", c, js)
            v = mean_over_judges(judge, uid, "verbose", c, js)
            if t is None or v is None:
                continue
            ax.plot([0, 1], [t, v], color="#4C6EF5", alpha=0.45, lw=1.2, marker="o", ms=3)
        o = [obj[(uid, "orig")] for uid in meta if (uid, "orig") in obj]
        ax.plot([0, 1], [np.mean(o)] * 2, color="#E8590C", lw=2.4,
                label="objective (unchanged by construction)")
        ax.set_xticks([0, 1]), ax.set_xticklabels(["terse / hesitant", "verbose / confident"])
        ax.set_ylabel("judge score")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(c)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "figures" / "fig2_style_effect.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ main

def main() -> None:
    ensure_dirs()
    meta, obj, judge = load()
    if not judge:
        print("no judge scores yet — run 04_run_judges.py first")
        sys.exit(1)
    js, cs = judges_conditions(judge)

    m = dict(
        n_answers=len(meta), n_tasks=len({v["task_id"] for v in meta.values()}),
        judges=js, conditions=cs,
        h1=h1(meta, obj, judge, js, cs),
        h2=h2(meta, obj, judge, js, cs),
        h3=h3(meta, judge, js, cs),
        h3_vs_orig=h3_vs_orig(meta, judge, js, cs),
        h4=h4(obj, meta),
        self_preference=self_preference(meta, judge, js, cs),
    )
    (RES / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    write_tables(m, js, cs)
    write_figures(meta, obj, judge, js, cs)

    print(f"answers {m['n_answers']} over {m['n_tasks']} tasks; judges {js}; conditions {cs}")
    for c in cs:
        r1, r2, r3 = m["h1"].get(f"{c}|POOLED"), m["h2"].get(f"{c}|POOLED|all_pairs"), \
            m["h3"].get(f"{c}|POOLED")
        print(f"\n[{c}]")
        if r1:
            print(f"  H1 Spearman {fmt(r1['spearman'])} (p={fmt(r1['spearman_p'])}), "
                  f"judge mean {fmt(r1['judge_mean'])} vs objective {fmt(r1['objective_mean'])}")
            print(f"     on objective=0 (n={r1['n_objective_zero']}): judge mean "
                  f"{fmt(r1['judge_mean_on_zero'])}, share ≥0.9 "
                  f"{fmt(r1['share_judge_ge_0.9_on_zero'], 2)}")
        if r2:
            print(f"  H2 pairwise accuracy {fmt(r2['accuracy'])} "
                  f"[{fmt(r2['ci_low'])}, {fmt(r2['ci_high'])}] over {r2['n_pairs']} pairs")
        if r3:
            print(f"  H3 Δ(verbose−terse) {fmt(r3['mean_delta'])} "
                  f"[{fmt(r3['ci_low'])}, {fmt(r3['ci_high'])}], Wilcoxon p={fmt(r3['wilcoxon_p'])}")
    print(f"\n  H4 objective max |Δ| = {m['h4']['max_abs_delta']:.1e} "
          f"over {m['h4']['n_rewrites']} rewrites ({m['h4']['n_moved']} moved)")
    print(f"\nwrote {RES/'metrics.json'}, tables/ and figures/")


if __name__ == "__main__":
    main()
