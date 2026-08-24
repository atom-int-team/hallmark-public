"""Shared constants, loaders and the OpenRouter client for the judge-vs-regret experiment.

Everything this package writes lives under ``analysis/judge_vs_regret/``. The repository
(``tasks/``, ``ground_truth/``, ``results/``, ``transcripts/``) is read-only here.
"""
from __future__ import annotations

import json
import pathlib
from functools import lru_cache

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP = ROOT / "analysis/judge_vs_regret"
DATA = EXP / "data"
RES = EXP / "results"

SEED = 20260724

# Stratum D — graders whose score is a deterministic function of the answer text
# (no LLM extractor in the chain), so a style-only rewrite must move them by exactly 0.
STRATUM_D = {"b3_select", "select_hypervolume", "olympus_design"}

JUDGES = [
    "openai/gpt-5.5",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.6",
]
WRITER = "anthropic/claude-opus-4.8"  # the paraphraser

N_TASKS, PER_TASK = 14, 3
CONDITIONS = ("ref_free", "ref_based")
STYLES = ("orig", "terse", "verbose")


def ensure_dirs() -> None:
    for d in (DATA, RES, RES / "tables", RES / "figures"):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- repo readers

@lru_cache(maxsize=1)
def public_tasks() -> frozenset[str]:
    """Every task shipped in this repository (the benchmark's public half)."""
    return frozenset(p.stem for p in (ROOT / "tasks/end_to_end").glob("*.yaml"))


@lru_cache(maxsize=1)
def ledger() -> tuple[dict, ...]:
    """The scored ledger the sample was drawn from.

    The published sample (``data/sample.json``) was drawn from a single-repeat ledger that
    included product-UI systems and is not part of this release, so re-running step 00 here
    draws a DIFFERENT sample against ``results/final.json``. Steps 02-06 read the shipped
    ``data/*.jsonl`` and reproduce the published analysis exactly.
    """
    return tuple(json.loads((ROOT / "results/final.json").read_text())["results"])


@lru_cache(maxsize=1)
def transcript_index() -> dict[tuple[str, str, int], pathlib.Path]:
    """(system_id, task_id, seed_idx) -> path, read from the transcripts themselves.

    Directory names are not a reliable function of ``system_id`` (slashes are
    rewritten), so the index is built from each file's own header.
    """
    idx: dict[tuple[str, str, int], pathlib.Path] = {}
    for p in sorted((ROOT / "transcripts").glob("*/*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        idx[(d["system_id"], d["task_id"], int(d.get("seed_idx", 0)))] = p
    return idx


def answer_text(system_id: str, task_id: str, seed_idx: int) -> str:
    p = transcript_index().get((system_id, task_id, seed_idx))
    if p is None:
        return ""
    d = json.loads(p.read_text())
    return (d["turns"][0].get("answer_text") or "").strip()


@lru_cache(maxsize=None)
def grading_spec(task_id: str) -> dict:
    """The single grading entry of a stratum-D task ({grader, params, ...})."""
    d = yaml.safe_load((ROOT / f"tasks/end_to_end/{task_id}.yaml").read_text())
    entries = d["turns"][0]["grading"]
    if len(entries) != 1:
        raise ValueError(f"{task_id}: expected exactly 1 grader, got {len(entries)}")
    return entries[0]


def grader_params(task_id: str) -> dict:
    return dict(grading_spec(task_id).get("params") or {})


@lru_cache(maxsize=None)
def gold(task_id: str) -> dict:
    from hallmark.loader import GroundTruth

    return GroundTruth().resolve(grader_params(task_id)["oracle_ref"])


def family(task_id: str) -> str:
    """Sub-family granularity: b3 datasets are separate families here.

    Coarser labels ("b3") would let the round-robin pick three framings of one
    dataset (fatigue design / robust / target) and call it spread.
    """
    for pre, fam in (
        ("b3_htsteel_", "htsteel"),
        ("b3_te_", "te"),
        ("cdft_", "cdft"),
        ("nlo_", "nlo"),
        ("e2e_olympus_", "olympus"),
    ):
        if task_id.startswith(pre):
            return fam
    if task_id.startswith("e2e_"):
        return "synthetic"
    if task_id.startswith("b3_"):
        return "b3_" + task_id.split("_")[1]  # b3_fatigue_design -> b3_fatigue
    return "other"


def task_prompt(task_id: str) -> str:
    d = yaml.safe_load((ROOT / f"tasks/end_to_end/{task_id}.yaml").read_text())
    return d["turns"][0]["prompt"]


# ---------------------------------------------------------------- jsonl helpers

def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def latest_variants(accepted_only: bool = True) -> list[dict]:
    """One record per (uid, style): a rejected attempt stays in the log, the accepted
    retry supersedes it. Downstream steps must never score a superseded rewrite."""
    latest: dict[tuple[str, str], dict] = {}
    for r in read_jsonl(DATA / "variants.jsonl"):
        key = (r["uid"], r["style"])
        if key not in latest or r.get("gate_ok") or not latest[key].get("gate_ok"):
            latest[key] = r
    return [r for r in latest.values() if r.get("gate_ok") or not accepted_only]


def append_jsonl(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- LLM client

def openrouter():
    """(AsyncOpenAI client, base_url) using the repo's JUDGE_API_KEY/.env resolution."""
    from openai import AsyncOpenAI

    from hallmark.graders.judge import resolve_judge_keys

    api_key, base_url = resolve_judge_keys({}, "judge-vs-regret")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=2)
