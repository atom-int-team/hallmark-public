"""Inline data delivery for SUTs without file upload (spec §7.3).

Builds a compact textual view of a dataset — column schema, a head sample, and
per-numeric-column statistics — so a text-only model can still reason about it.
The runner marks this as a sample; adapters that *can* upload use the real file.
Stdlib only (no pandas) so the runner stays light.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .models import Attachment


def render_inline_attachments(attachments: list[Attachment]) -> str:
    """Inline textual attachments into a prompt block (for SUTs without file upload)."""
    if not attachments:
        return ""
    blocks = ["\n\n--- Attached files (inline sample; treat as the full dataset) ---"]
    for a in attachments:
        body = a.text if a.text is not None else "(binary content omitted)"
        blocks.append(f"\n[file: {a.name}]\n{body}")
    return "\n".join(blocks)


# Text-only SUTs get the WHOLE file when it fits this budget (all e2e CSV fixtures do —
# the largest is ~18k chars). Only genuinely oversized files fall back to a labelled sample,
# so data is not artificially withheld from bare models (spec §7.3; deliverable-zeros plan).
FULL_DELIVERY_MAX_CHARS = 480_000


def full_or_sample(path: str | Path, *, max_chars: int = FULL_DELIVERY_MAX_CHARS) -> str:
    """Full file content for text-only delivery when it fits the context budget; otherwise a
    labelled sample (and say so, so the truncation is never silent)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if len(raw) <= max_chars:
        return raw
    return (
        f"[NOTE: {path.name} is {len(raw)} chars — too large for full inline delivery; a "
        "representative sample follows. Treat it as representative, not complete.]\n"
        + csv_inline_sample(path)
    )


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def csv_inline_sample(path: str | Path, *, n_head: int = 10) -> str:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = list(reader)

    n = len(rows)
    lines = [
        f"dataset: {path.name}  ({n} rows × {len(header)} columns)",
        f"columns: {', '.join(header)}",
        "",
        "stats (numeric columns): column  min / mean / max",
    ]
    for j, col in enumerate(header):
        vals = [float(r[j]) for r in rows if j < len(r) and _is_number(r[j])]
        if vals:
            mean = sum(vals) / len(vals)
            lines.append(f"  {col}: {min(vals):g} / {mean:g} / {max(vals):g}")

    head = rows[:n_head]
    lines += ["", f"head ({len(head)} of {n} rows):", ",".join(header)]
    lines += [",".join(r) for r in head]
    return "\n".join(lines)
