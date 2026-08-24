# Hallmark

**A benchmark that scores AI materials decisions by regret against a known optimum.**

Prediction benchmarks ask how well a system knows a property surface. Hallmark asks the
question a laboratory asks first: *given what is known, which candidate should we commit to,
and how much do we lose by committing to it?* Every task hands a system a candidate pool or a
declared design space, asks it to commit, and a deterministic grader measures how far that
commitment fell short of an optimum that is already known — by construction, by exhaustive
search over a declared set, or from measured labels withheld from the prompt.

Only the deliverable is scored. Reasoning length, citations and stated confidence neither help
nor hurt, and an answer that cannot be read scores zero by construction rather than by an
adjudicator's discretion.

```
                 prompt + attached CSVs
                          |
                          v
       [ any text-in / text-out system, one turn ]
                          |
                          v
        "... Final: c017, c003, c061, c044, c008"
                          |
                          v
     deterministic parse -> hidden labels -> score in [0, 1]
        0 = no better than doing nothing useful
        1 = the known optimum
```

## What is in this repository

This is the **public half** of the benchmark: **46 of its 92 tasks**, with every artifact needed
to run them, grade them, and reproduce the published numbers.

| | |
|---|---|
| `tasks/end_to_end/` | 46 task YAMLs — a prompt, its attachments, one grader, one ground-truth ref |
| `fixtures/` | the CSVs each task attaches (training tables and anonymized candidate pools) |
| `ground_truth/` | the answer keys, plus the frozen surrogates used by the continuum tasks |
| `transcripts/` | 1,932 cached responses — 14 systems x 46 tasks x 3 independent repeats — plus 506 agent runs with their tool traces (11 configurations x 46 tasks) |
| `results/final.json` | the scored ledger of the paper: all 3,864 cells (14 x 92 x 3) |
| `results/harness.json` | the agent ledger: 11 scaffold configurations x 92 tasks, one repeat (see below) |
| `harness/` | the Docker images and scaffold configurations the agent runs used |
| `blocks.json` | the 29 atomic blocks — the correct resampling unit (see below) |
| `task_registry.json` | all 92 tasks: family, chain, response form, construction, block, scores |
| `hallmark/` | the harness: adapters, graders, scoring, leaderboard |
| `analysis/judge_vs_regret/` | the controlled experiment comparing an LLM judge with regret |
| `scripts/` | the published baselines, the pairwise tests and the repeat-averaging step |

The other 46 tasks are held back permanently as an uncontaminated instrument. See
[DATASHEET.md](DATASHEET.md) for what that means, what it does and does not protect, and the
provenance and licensing of every source dataset.

The public half spans eight of the nine families: NLO crystals (12), B3 real datasets (9),
crystal-DFT (6), synthetic analytic alloys (6), heat-resistant steels (4), perovskite devices (4),
thermoelectrics (4) and one Olympus laboratory campaign.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# optional: pip install -e ".[baselines]"   # to re-run the shortcut audit / ML baseline
```

## Quickstart

Generation (networked, cached) is separated from grading (cheap, offline), so iterating on a
grader never costs an API call.

```bash
# 1. offline dry run — no key, no network, no cost
hallmark generate --systems configs/systems.mock.yaml
hallmark grade    --systems configs/systems.mock.yaml --run demo
hallmark leaderboard --run demo

# 2. reproduce the published leaderboard from the cached transcripts (no key needed)
hallmark leaderboard --run final_mean
```

To evaluate your own system, add an entry to a systems YAML (adapter, model, `api_key_env`,
optional `base_url` and prices) and run `generate` then `grade`.
`configs/systems.api-bare.yaml` holds the fourteen systems of the paper, with the exact
configuration each was run under; `configs/systems.pi.yaml` and `configs/systems.opencode.yaml`
hold the agent configurations.

## The scale

Every score is in `[0, 1]` on one shared scale, and `1` means the same thing in all 92 tasks.

* **Selection** (35 of the 46 public tasks). The system sees `N` anonymized candidates with the
  target column physically removed and returns `K = 5` identifiers. With `floor` the pool mean and
  `top_K` the mean of the `K` best hidden labels, the score is
  `clip((got - floor) / (top_K - floor), 0, 1)`. So `0` is no discrimination from a blind draw and
  `1` is exactly the best `K`. A short answer earns proportional partial credit; under-picking is
  never profitable.
* **Constructed point** (11 public tasks). The system proposes a point in a declared design space,
  which is evaluated exactly — by a closed-form analytic objective, or by a Gaussian process fitted
  to real laboratory data and frozen to fixed numeric weights. The score is the clipped normalized
  gain over an anchor, which is one minus normalized regret.
* **Target-hitting** variants replace objective gain with normalized deviation from a requested
  value: `max(0, 1 - |f(x) - T| / tau)`. Not "move in the known direction" but "stop at the
  requested point".
* **Two-objective** variants are scored by normalized hypervolume against the exact best-`K`
  subset of the pool's Pareto front.

Constraints — a forbidden element, a required band gap, a cap on constituent elements — are folded
directly into the hidden label, so an infeasible candidate carries no reward and feasibility must be
inferred rather than enforced by the grader.

## Grading is deterministic

A selection is read off the answer by a fixed procedure with no adjudication in it. The parser takes
the **last** line matching `final` plus optional punctuation (case-insensitive, tolerating `**Final:**`,
`final -`, a missing colon), splits the tail on whitespace/commas/semicolons and strips non-alphanumeric
edges. If no such line yields a valid identifier it looks for a JSON array or object anywhere in the
answer; that is the whole fallback. There is no language-model recovery for a selection. Identifiers
not in the pool are dropped, duplicates collapsed, order preserved as the ranking, and missing slots
valued at the anchor.

**One honest exception.** The 11 constructed-point tasks parse their `Final:` line deterministically
too, but where that parse does not recover every declared variable, a language-model extractor reads
the proposed values out of the free text — so those tasks need `JUDGE_API_KEY` at grade time.
`extract_backend: regex` is a key-less deterministic fallback for them. The 35 selection tasks are
fully offline: **grading them from the shipped transcripts reproduces every published cell exactly**
(1,470 cells, zero mismatches, scores and grader detail strings alike).

## Reporting results honestly

Three things the published numbers depend on, all of which a reuser inherits:

1. **Average repeats, never maximize.** Each system runs the whole suite three times independently;
   the cell is the mean. Best-of-three is selection on the outcome and inflates a system by exactly
   as much as it is noisy. `scripts/mean_of_seeds.py` does the collapse.
2. **Resample blocks, not tasks.** Tasks are not independent: two framings can rank the same pool by
   the same hidden column or sit on one hidden optimum. `blocks.json` gives the atomic blocks — the
   transitive closure of tasks sharing an answer key, a pool or an optimum. Treating 92 tasks as
   independent when they close into 29 blocks overstates the resolution by roughly the square root
   of that ratio. Note that the built-in `hallmark leaderboard` bootstraps over tasks, so its
   intervals are narrower than the paper's; `scripts/pairwise_holm.py` does it correctly.
3. **The suite is not saturated and the top does not resolve.** The best system reaches 0.604.
   Its lead over every other system clears an uncorrected 95% interval, but after the Holm
   correction across all 91 pairs it separates from none of the next seven; reporting a strict
   ranking would invent resolution the measurement does not have.

## Agent scaffolds

Eleven configurations run the same 92 tasks through an off-the-shelf coding-agent scaffold —
pi (`@earendil-works/pi-coding-agent`) with seven models and opencode (`opencode-ai`) with
four — inside a Docker sandbox whose only network route is the model gateway. The agent
receives the task prompt and its CSVs as files, may write and run code, and is scored on the same
`Final:` line by the same graders. Each configuration runs one repeat (seed 0), so the comparison
is paired cell by cell against the same model's bare seed-0 run on the same task
(`scripts/harness_paired_delta.py`, block bootstrap as above):

| configuration | with scaffold | bare | difference | 95% CI |
|---|---|---|---|---|
| pi × Grok 4.6 | 0.585 | 0.538 | +0.047 | [+0.005, +0.088] |
| pi × Opus 5 | 0.555 | 0.532 | +0.023 | [-0.022, +0.074] |
| pi × Gemini 3.1 Pro | 0.478 | 0.528 | -0.050 | [-0.124, +0.027] |
| pi × Kimi K3 | 0.481 | 0.484 | -0.003 | [-0.068, +0.072] |
| pi × GPT-5.6 Sol | 0.574 | 0.534 | +0.041 | [-0.000, +0.092] |
| pi × GPT-6 Astra | 0.639 | 0.609 | +0.029 | [-0.031, +0.124] |
| pi × GPT-6.1 Sol | 0.646 | 0.556 | +0.090 | [+0.022, +0.184] |
| opencode × Grok 4.6 | 0.588 | 0.538 | +0.051 | [+0.012, +0.097] |
| opencode × Gemini 3.1 Pro | 0.461 | 0.528 | -0.067 | [-0.155, +0.002] |
| opencode × GPT-5.6 Sol | 0.582 | 0.534 | +0.048 | [+0.004, +0.102] |
| opencode × GPT-6 Astra | 0.604 | 0.609 | -0.005 | [-0.073, +0.092] |

Grok 4.6 gains under both scaffolds with intervals that exclude zero; no configuration loses
with an interval that excludes zero. Re-running a configuration needs Docker:
the images build from `harness/`, and the container runs on an internal network whose only
reachable host is a gateway to the model provider (`harness/gateway/nginx.conf.template`, with
the key substituted for `__OPENROUTER_API_KEY__`).

## Reproducing the published analysis

Six scripts recompute published numbers from the released ledgers, all offline:

```bash
python scripts/mean_of_seeds.py     # final.json -> final_mean.json (average the repeats)
python scripts/pairwise_holm.py     # all 91 pairwise comparisons, block bootstrap + Holm
python scripts/s5_controls.py       # the second-condition gap under form and chain controls
python scripts/harness_paired_delta.py  # each agent scaffold against the same model, bare
python scripts/shortcut_audit.py    # can a rule with no chemistry in it beat the systems?
python scripts/ml_baseline.py       # the supervised tabular baseline (needs ".[baselines]")
```

The first four read only the ledgers in `results/`, `blocks.json` and `task_registry.json`, so they
reproduce the paper's numbers exactly. The last two read the task files as well, and therefore see
46 of the 92 tasks — their output is a subset of the published audit, not a reproduction of it.

## Tests

```bash
pytest -q      # 208 tests, fully offline, no API keys required
```

Every network seam is monkeypatched, so the suite never makes a call.

## Citation

If you use this benchmark, please cite the paper and the source datasets your tasks draw on —
[NOTICE](NOTICE) lists them per family, with two that must be cited by name because they carry
no license of their own.
