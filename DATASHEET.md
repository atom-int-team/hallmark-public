# Datasheet — Hallmark

## What a task is, physically

A task is a YAML file naming a prompt, zero or more attached CSV files, and one grader with a
reference to a ground-truth JSON. Nothing else is needed to run it: a system that reads text
and writes text can be evaluated, and a harness that can drive such a system can reproduce a
cell. A *cell* is one system–task pair, the unit of the results ledger, and is the mean of
three independent repeats.

Every selection task is built the same way, so differences between families are differences of
chemistry rather than of construction. A dataset is split into a training part and a held-out
part; the training part is shown in the prompt, capped at 300 rows and sampled representatively
where the source is larger; the pool is N = 100 held-out candidates — fewer in the tasks whose
defining restriction exhausts the source data, down to 26 — with the target column physically
removed and identifiers replaced by the opaque scheme `c000`–`c099`; the system returns K = 5.
The hold-out fraction is 30% except where a family's size forced otherwise. Pool draws are made
once at build time under a fixed seed and frozen; nothing is resampled at grade time.

## What is never in a prompt

The hidden label of any pool candidate; the optimum or its location; the direction of any
individual knob; the identity of the source dataset or any field that would identify a record.
The format example in each prompt is neutral by construction — mid-range values for a
constructed point, and identifiers `c901`–`c905` for a selection, which are invalid in every
pool.

## The public and private halves

The 92 tasks of the benchmark are split into two halves of 46. **This repository is the public
half. The private half is not published — not the tasks, not the answer keys, not later.** That
is a standing part of the anti-contamination policy rather than an embargo.

Release is by **atomic block**, not by task, because tasks are not independent artifacts: two
framings can rank the same pool by the same hidden column, share a candidate pool drawn once, or
sit on one hidden optimum. A block is the transitive closure of tasks sharing an answer key, a
candidate pool or a hidden optimum. The 92 tasks close into 29 blocks; 15 of them, holding 46
tasks, are public and are listed in `blocks.json`. Publishing either member of a block publishes
the other, which is why the split cannot be made task by task.

Over the fourteen evaluated systems the two halves score 0.514 and 0.506, and per system the gap
never exceeds 0.051 with every interval covering zero — so no system does conspicuously better
on the half it could have seen. **That balance is the only contamination claim we make, and it
holds in aggregate only.** Cut by family it does not survive: three families lie entirely in one
half (MOF and NLO wholly private and wholly public respectively, perovskite wholly public) and
contribute no contrast at all, and among the six that do split the halves diverge by up to 0.56.
As a detector this instrument can see a system uniformly better on public tasks; it cannot see
one better on public tasks *within* a family whose halves already differ by more than the effect
being looked for.

## Anti-contamination, stated at its real strength

We do not claim that a system cannot know an individual record — a POSCAR or a MOFid is close to
a unique fingerprint of public labeled data. The claim is narrower: knowing the source dataset is
not sufficient to name the best K of N anonymized candidates ranked by a column physically absent
from the prompt. Target columns are omitted, identifiers anonymized, and strongly predictive
non-target columns withheld, so memorizing a database can at most supply a property model — which
is the capability under evaluation. Anti-contamination here is **operational, not structural**.

## Two differences between this release and the run

Both are recorded rather than glossed:

1. **Task YAMLs.** Two non-scoring routing fields were removed from the shipped YAMLs: `split:
   dev` (there is no dev/test split in this release) and `required_capabilities: [manual]` (a
   legacy of a product-UI tier that routed tasks to a human-capture adapter). Neither field is
   read by any grader; prompts, attachments and grading specs are byte-identical to what was run.
2. **The ledger and the two index files.** `results/final.json` carries all 3,864 scored cells of
   the paper, including the private half's, so every published table recomputes. For the private
   half's cells the grader's free-text `detail` string is removed, because it prints aggregates of
   the answer key (pool mean, top-K mean, f*). Scores, usage and cost are untouched; public cells
   keep their `detail`, which in any case regenerates exactly from the shipped answer keys. For the
   same reason `blocks.json` lists all 29 blocks (the bootstrap needs the grouping of every scored
   task) and `task_registry.json` all 92 tasks — but in the registry the four fields derived from a
   task's answer key (`anchor`, `optimum`, `naive_baseline_score` and the constraint text) are set
   to null for the private half and named under `redacted`.

## Provenance and licensing

Every source, with its terms and required citation, is listed in [NOTICE](NOTICE). Most are
CC BY 4.0 and redistribute cleanly. Two need saying plainly:

* The **NLO** records carry no open license and are released with the written permission of the
  source database's author, covering exactly the 300 structures the twelve NLO tasks draw on —
  labels included, source identifiers stripped — and not the database.
* The **htsteel** compilation is one author's own and is not distributable as a whole; the four
  derived tasks are released, the compilation is not.

The **MOF** family, whose source publishes no license at all, is entirely in the private half,
so nothing from it is redistributed here.

## Reproducing the numbers

Generation and grading are separate: generation drives a system and caches a transcript, grading
scores cached transcripts and never touches the network.

* `hallmark leaderboard --run final_mean` reproduces the published main table from the shipped
  ledger, with no API key and no cost. Its intervals bootstrap over tasks; the paper's bootstrap
  over the blocks in `blocks.json`, which is wider and is the correct unit.
* Re-grading the 35 fully-offline public tasks from the shipped transcripts reproduces every
  published cell exactly — 1,470 cells, zero mismatches in both the scores and the grader detail
  strings. The other 11 public tasks parse a constructed point and call a language-model
  extractor where the deterministic parse is incomplete, so they need `JUDGE_API_KEY`;
  `extract_backend: regex` is the key-less fallback.
* `scripts/pairwise_holm.py` recomputes all 91 pairwise system comparisons with the block
  bootstrap and the Holm correction, and `scripts/s5_controls.py` the second-condition contrast
  under the form and chain controls. Both read only the ledger, `blocks.json` and
  `task_registry.json`, and reproduce the published numbers exactly.
* `scripts/shortcut_audit.py` and `scripts/ml_baseline.py` re-run the two published baselines
  (`pip install -e ".[baselines]"`). These read the task files as well, so on the public half they
  see 46 of 92 tasks and their output is a subset of the published audit rather than a
  reproduction of it.
* `analysis/judge_vs_regret/` reproduces the judge experiment; its offline steps (02, 03, 05, 06)
  need no key.

**The build scripts that generated the tasks from their source datasets are not part of this
release**, because they are deterministic over the whole slate and would regenerate the private
half along with the public one for anyone holding the sources.

## Intended use, and a use it does not support

The suite measures how much objective value a system loses by committing to one proposed object,
under a single turn with no tools. It does **not** measure whether a proposed material can be
synthesized, whether it is safe, or whether it would behave as the label predicts outside the
conditions the source data covers. A high score is evidence about search and selection behavior
on these objectives and **is not a recommendation to make anything**.

The headline numbers characterize the bare single-turn regime and nothing else. We expect them to
move under a harness — and to move unevenly, since a fitted model helps most where the training
table already carries the signal and least where the condition is stated only in the prompt — so
the ranking should not be assumed to survive one.
