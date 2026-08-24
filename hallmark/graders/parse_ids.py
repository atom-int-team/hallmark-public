"""Fully deterministic extraction of a ranked candidate-id list from a free-form answer.

The selection grader (`b3_select`) asks the SUT to end with ``Final: c012, c047, ...``.
We parse that contract line with a regex (offline, free — the common path) and fall back to
any JSON array in the answer. An answer that satisfies neither scores 0: there is **no LLM
extraction fallback**, so no language model appears anywhere in the grading chain.

The fallback was removed 2026-07-23. Two measurements over the cached v0 ledger killed it:

* It was **not** unused, contrary to what the paper's §3.3/A.5 said: 16 of 785 cached selection
  cells reached it, because they contain no deterministically readable id.
* On one of them it **fabricated a selection**. ``deepseek/deepseek-chat`` ×
  ``b3_te_zt_target_select`` answered with Python code that would print a ``Final:`` line but
  never named a single candidate (0 occurrences of ``c\\d+`` in 3.2 kB); the extractor returned
  five ids anyway and the cell scored 0.0839 instead of 0. Removing the fallback is the only
  change to the v0 regrade: that cell becomes 0.0 (its system's mean moves by ~0.001).

The v1 format-example repair would have made this worse: the old example used ids valid in
every pool (``c012, c047, …``), so an echo parsed as a real answer, while v1's unreachable
``c901…c905`` turn an echo into exactly the id-less input that reaches the last resort —
15 of the 18 cached echo cells. Grading is now free, offline and reproducible instead.
"""
from __future__ import annotations

import re

from .extract import _parse_any, coerce_list

# "Final: c012, c047" / "final - c1 c2" / "**Final:** c3" -> capture the tail
_FINAL_RE = re.compile(r"final\b[^\n:]*:?\s*(.+)$", re.I | re.M)
_SPLIT_RE = re.compile(r"[\s,;]+")
# Markdown/punctuation clinging to a token: "**Final: c1, c2**" splits to "c2**",
# and "`c2`." / "(c2)" likewise. Ids are alphanumeric, so trimming non-alphanumeric
# edges can only rescue a token, never rewrite one valid id into another.
_EDGE_RE = re.compile(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$")


def _dedup(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


def _filter(cands, valid: set[str]) -> list[str]:
    return _dedup([c for c in (_EDGE_RE.sub("", str(x)) for x in cands) if c in valid])


async def parse_ids(answer: str, valid: set[str], k: int, params: dict) -> list[str]:
    """Return up to ``k`` valid candidate ids, ranked best-first. Empty if none found.

    ``async`` and ``params`` are kept for call-site compatibility with the graders; nothing in
    here awaits or reaches the network any more.
    """
    answer = answer or ""

    # 1) the "Final: ..." contract line (last match wins — the final restatement)
    matches = _FINAL_RE.findall(answer)
    for tail in reversed(matches):
        picks = _filter(_SPLIT_RE.split(tail.strip()), valid)
        if picks:
            return picks[:k]

    # 2) any JSON array/object embedded in the answer
    data = _parse_any(answer)
    if data is not None:
        picks = _filter(coerce_list(data), valid)
        if picks:
            return picks[:k]

    return []  # no deterministic reading -> no selection; the grader scores 0
