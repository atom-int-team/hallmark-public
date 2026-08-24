"""reference_judge — reference-based LLM-as-judge (ported from benchmark/judge.py).

Adds a reference answer to the judge prompt (the old judge was reference-free).
Score 0–10 from the judge -> 0..1.

Backends:
  llm      (default) — call an OpenAI-compatible judge model.
  overlap            — offline bag-of-words F1 vs the reference (no network/keys);
                       a cheap proxy for tests and key-less dry-runs.

Judge model/keys (llm backend), overridable via params:
  JUDGE_MODEL          (default "gpt-4o-mini")
  JUDGE_API_KEY        (fallback: OPENAI_API_KEY)
  JUDGE_API_BASE_URL   (optional, e.g. OpenRouter)

Judge board: with ``params['judge_board']`` (set by ``grade --judge-board``) the
answer is scored by every model in ``JUDGE_LIST`` (';'-separated, sharing the
JUDGE_API_KEY/base_url) and the per-judge scores are averaged. See ``judge_panel``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import statistics

from ..textutils import strip_think

DEFAULT_RUBRIC = """Rate the answer on a 0-10 scale across:
- correctness: factual accuracy (consistency with the reference)
- completeness: how fully the request was addressed
- grounding: claims are supported; no fabricated facts or numbers
0 — failure, 5 — acceptable, 10 — exemplary.
"""

# Fix the judge's role/format explicitly: some models otherwise "play the agent".
_SYSTEM = (
    "You are a strict grader of an AI system's answer in a materials-science task. "
    "Return EXACTLY one JSON object and nothing else: "
    '{"score": <integer 0-10>, "rationale": "<justification with a final verdict>"}. '
    "Do not play the agent's role, do not continue the dialogue, no text outside the JSON. "
    "score: 0 — failure, 5 — acceptable, 10 — exemplary."
)

_USER_TEMPLATE = """## User prompt
{prompt}

## Reference answer (ground truth)
{reference}

## System answer (to grade)
{answer}

## Evaluation criteria
{rubric}
"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_PARSE_FAIL = "FAILED TO PARSE judge score."


def parse_score(text: str) -> tuple[int, str]:
    """Extract (score 0-10, rationale). Ported from the old judge.

    Order: whole text as JSON, first {...} block as JSON, then an explicitly
    labeled "score" near a number. On failure return 0 (= failure on the scale).
    """
    text = strip_think(text or "").strip()
    candidates = [text]
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict) and "score" in data:
            try:
                score = max(0, min(10, int(data["score"])))
            except (ValueError, TypeError):
                continue
            return score, str(data.get("rationale", "")).strip() or text
    m = re.search(r"(?:score)\D{0,12}?(10|[0-9])", text, re.IGNORECASE)
    if m:
        return int(m.group(1)), text
    return 0, f"{_PARSE_FAIL} Full answer:\n{text[:500]}"


def _as_text(val) -> str:
    return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)


def resolve_judge_keys(params: dict, who: str) -> tuple[str, str | None]:
    """(api_key, base_url) for the LLM judge, shared by single + board modes.

    Resolution order per value: explicit ``params`` → env var → ``settings`` field
    (loaded from .env). Keys live in settings because pydantic-settings drops
    undeclared .env vars, so reading os.environ alone misses .env-only keys.
    """
    from ..config import settings

    api_key = (
        os.environ.get("JUDGE_API_KEY")
        or settings.judge_api_key
        or os.environ.get("OPENAI_API_KEY")
        or settings.openai_api_key
    )
    base_url = (
        params.get("base_url") or os.environ.get("JUDGE_API_BASE_URL") or settings.judge_api_base_url
    )
    if not api_key:
        raise RuntimeError(f"{who} (llm): no JUDGE_API_KEY / OPENAI_API_KEY")
    return api_key, base_url


def resolve_judge_args(params: dict, who: str) -> tuple[str, str, str | None]:
    """(model, api_key, base_url) for a single LLM judge, shared by both graders."""
    from ..config import settings

    model = (
        params.get("model")
        or os.environ.get("JUDGE_MODEL")
        or settings.judge_model
        or "gpt-4o-mini"
    )
    api_key, base_url = resolve_judge_keys(params, who)
    return model, api_key, base_url


def resolve_judge_models(params: dict) -> list[str]:
    """Models for the judge board: ``JUDGE_LIST`` split on ';' (env → settings).

    Returns ``[]`` when unset, which the caller treats as an error in board mode.
    A ``params['judge_list']`` override (also ';'-separated) takes precedence.
    """
    from ..config import settings

    raw = params.get("judge_list") or os.environ.get("JUDGE_LIST") or settings.judge_list
    if not raw:
        return []
    return [m.strip() for m in raw.split(";") if m.strip()]


def _client_opts() -> tuple[float, int]:
    """(timeout_s, max_retries) for the judge client (env → settings).

    Overrides the openai SDK defaults (600s read timeout, 2 retries) so a stalled
    provider can't freeze grading for tens of minutes.
    """
    from ..config import settings

    timeout = float(os.environ.get("JUDGE_TIMEOUT") or settings.judge_timeout)
    max_retries = int(os.environ.get("JUDGE_MAX_RETRIES") or settings.judge_max_retries)
    return timeout, max_retries


async def _llm_judge_one(model: str, api_key: str, base_url: str | None, messages: list[dict]) -> str:
    """Single network call to one judge model. The per-model seam for the board."""
    from openai import AsyncOpenAI

    timeout, max_retries = _client_opts()
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    return resp.choices[0].message.content or ""


async def judge_panel(messages: list[dict], params: dict, who: str) -> tuple[float, str]:
    """Score ``messages`` with a board of judges (``JUDGE_LIST``) and average.

    Each model is called concurrently; a model that errors or whose reply fails to
    parse is excluded from the mean (not counted as 0). Returns (score01, detail)
    with the per-judge breakdown and disagreement (min/max/stdev). Raises if the
    list is empty (``--judge-board`` set without ``JUDGE_LIST``) or all judges fail.
    """
    models = resolve_judge_models(params)
    if not models:
        raise RuntimeError(f"{who}: judge board requested but JUDGE_LIST is empty/unset")
    api_key, base_url = resolve_judge_keys(params, who)

    raws = await asyncio.gather(
        *(_llm_judge_one(m, api_key, base_url, messages) for m in models),
        return_exceptions=True,
    )

    valid: list[tuple[str, int]] = []  # (model, score 0-10)
    failed: list[str] = []
    for model, raw in zip(models, raws, strict=True):
        if isinstance(raw, Exception):
            failed.append(f"{model} ({type(raw).__name__})")
            continue
        score, rationale = parse_score(raw)
        if rationale.startswith(_PARSE_FAIL):
            failed.append(f"{model} (parse)")
            continue
        valid.append((model, score))

    if not valid:
        raise RuntimeError(f"{who}: all {len(models)} board judges failed ({'; '.join(failed)})")

    scores = [s for _, s in valid]
    mean = sum(scores) / len(scores)
    spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    per = ", ".join(f"{m}={s}/10" for m, s in valid)
    detail = (
        f"judge-board mean {mean:.1f}/10 over {len(valid)}/{len(models)} judges "
        f"[min {min(scores)}, max {max(scores)}, stdev {spread:.2f}] — {per}"
    )
    if failed:
        detail += f" | excluded: {'; '.join(failed)}"
    return mean / 10.0, detail
