# LLM judge vs regret — a controlled experiment

Does an LLM judge score the *material* an AI system proposed, or the *prose* it wrote
around it? This directory holds the experiment that answers it, and the data behind the
paper's judge section.

**The manipulation.** Take a cached answer, rewrite it in two styles — short and
hesitant, long and confident — while keeping the material choice and the `Final:`
contract line byte-identical. The deterministic grader must then move by exactly 0.
Whatever the judge does instead is a measurement of style sensitivity, not noise.

## Scope

* **Stratum D only** — tasks whose grader (`b3_select`, `select_hypervolume`,
  `olympus_design`) reads the answer with no LLM in the chain. `oracle_eval` tasks are
  excluded: their extractor calls a model, so "moves by exactly 0" would not be provable.
* 14 tasks x 3 cached answers x 3 styles = 126 texts, scored by three judges under two
  conditions (with and without the ground truth).
* Nothing outside this directory is modified. `tasks/`, `ground_truth/`, `results/` and
  `transcripts/` are read-only inputs.

## What is shipped, and what is withheld

The experiment was run over 14 tasks. Five of them are in the benchmark's **private half**,
which is not released, so the **raw answer texts for those five are withheld here**: the shipped
`data/*.jsonl`, `data/sample.json` and `results/spotcheck.md` hold the public-task subset —
27 of 42 answers, 9 of 14 tasks, 6 of 10 spot-check triples.

Everything in `results/` is the **full** analysis over all 42 answers: `metrics.json`, the
tables and the figures are exactly the numbers the paper quotes. So the published result is
here in full; only part of the raw material behind it is. Re-running step 05 on the shipped
subset will therefore not reproduce `metrics.json` exactly — it recomputes the same statistics
over 27 answers instead of 42.

## Pipeline

```bash
python analysis/judge_vs_regret/scripts/00_select_sample.py       # $0  — 14 tasks x 3 answers
python analysis/judge_vs_regret/scripts/01_make_variants.py       # ~$3 — terse + verbose, gated
python analysis/judge_vs_regret/scripts/02_verify_invariance.py   # $0  — gate report
python analysis/judge_vs_regret/scripts/03_regrade_objective.py   # $0  — H4, and the safety net
python analysis/judge_vs_regret/scripts/04_run_judges.py          # ~$20 — 3 judges x 2 conditions
python analysis/judge_vs_regret/scripts/05_analyze.py             # $0  — metrics, tables, figures
python analysis/judge_vs_regret/scripts/06_spotcheck.py           # $0  — 10 triples to read by hand
```

Run them in order. **Step 03 before step 04 is not optional**: it re-scores every rewrite
with the real grader offline, so a content leak the gate missed surfaces before any judge
call is billed. Steps 01 and 04 are resumable — anything already in `data/*.jsonl` is
never regenerated or re-billed, so a rate-limited run just needs re-invoking.

Steps 02, 03, 05 and 06 are **offline and need no key**: they read the shipped
`data/*.jsonl` and reproduce the published analysis exactly. Steps 01 and 04 call models
and re-bill. Step 00 is the one step that does **not** reproduce: the published sample was
drawn from a single-repeat ledger that included product-UI systems and is not part of this
release, so re-running it draws a different sample against `results/final.json`. The sample
that was actually used is shipped verbatim in `data/sample.json`.

API keys come from the repo's `.env` (`JUDGE_API_KEY`, `JUDGE_API_BASE_URL` — OpenRouter),
resolved through `hallmark.graders.judge.resolve_judge_keys`.

## Layout

| path | what |
|---|---|
| `scripts/_common.py` | paths, seed, model lists, repo readers |
| `data/sample.json` | the chosen 14 tasks x 3 answers, with the cached answer text |
| `data/variants.jsonl` | orig / terse / verbose per answer, with gate verdicts |
| `data/objective.jsonl` | offline re-grade of every variant |
| `data/judge_scores.jsonl` | one record per (answer, style, judge, condition) |
| `results/metrics.json` | every number the paper's judge section quotes |
| `results/tables/*.md` | correlation, pairwise agreement, style effect, reference condition |
| `results/figures/*.pdf` | judge-vs-objective scatter, terse→verbose slope chart |
| `results/spotcheck.md` | 10 triples + the hand-filled verdict table |

## Reading the result honestly

Two of the pre-registered hypotheses failed and are reported as failures: the judge is not
noise (rho = 0.529 against the objective score), and handing it the answer key largely
repairs it (rho = 0.794) — though that condition does not exist in practice, since a judge
that must be given the optimum is no substitute for computing regret against it. The claim
that survives is the narrow one: about a quarter of a reference-free judge's scale is
recoverable by rewriting prose around an unchanged design.
