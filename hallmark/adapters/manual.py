"""manual — ingest human-captured transcripts from a product UI (e.g. ChatGPT web).

For systems with no programmatic API (the actual ChatGPT web product), a human
runs each task in the UI and saves the answer under either

    submissions/<system_slug>/seed<n>/<task_id>.md     (preferred, nested)
    submissions/<system_slug>/<task_id>__seed<n>.md    (legacy, flat)

with YAML frontmatter + one ``## turn N`` section per turn::

    ---
    model: gpt-x (web, 2026-06-01)
    latency_ms: 42000            # optional -> else 0
    cost_usd: null               # optional -> else None
    tools_used: [code_interpreter, web_search]   # optional -> trajectory
    artifacts: [opt_composition.csv]             # optional, files sit next to the .md
    ---
    ## turn 1
    <final visible answer to prompt 1>
    ## turn 2
    <answer to prompt 2>

Capabilities mirror the product (file upload, file output, web, code exec) so the
SUT is gated and graded like the API path. A missing submission raises
``GenerationSkipped`` (the runner skips it without failing), so the dataset can be
filled in over time.

Config (``SystemSpec.config``):
    submissions_dir:  str   default "submissions" (relative to repo root)
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from ..config import settings
from ..models import (
    Artifact,
    Attachment,
    Capabilities,
    Message,
    ToolCall,
    TurnTranscript,
    Usage,
)
from ..textutils import extract_citations, slug
from .base import Adapter, GenerationSkipped, register_adapter

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_TURN_RE = re.compile(r"^##\s+turn\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)

# Infrastructure failures a product UI renders *in the answer slot* — the capture
# then freezes them into the transcript cache as a permanent zero, indistinguishable
# from a model that answered badly. These are skipped (GenerationSkipped) so a re-run
# retries, mirroring how ``openai_chat`` refuses to cache a provider-flake empty answer.
#
# Deliberately NOT listed: refusals ("I can't help with that"). A refusal is a real
# answer the SUT chose to give and must be graded 0, not skipped.
_UI_ERROR_RE = re.compile(
    r"sorry,\s*something went wrong"
    r"|something went wrong\W+(?:please\s+)?try"
    r"|please try your request again"
    r"|an error occurred while (?:processing|generating)"
    r"|error in message stream"
    r"|conversation not found"
    r"|you(?:'ve| have) reached (?:your|the)[^.]{0,40}limit",
    re.IGNORECASE,
)
# Only a SHORT answer is treated as a captured error. A long, real answer that happens
# to quote one of these phrases (e.g. discussing error handling) must still be graded.
_UI_ERROR_MAX_CHARS = 400


def _reject_non_answer(answer: str, path: Path, turn_idx: int, *, has_artifacts: bool) -> None:
    """Raise ``GenerationSkipped`` when a submission holds no answer at all.

    Two cases, both capture failures rather than model failures:
      * the turn section is blank AND the turn ships no artifact — the file was
        scaffolded but never filled in;
      * the turn holds a product-UI error banner instead of the model's reply.

    Caching either freezes a silent zero into ``transcripts/`` that no later grading
    pass can distinguish from a genuinely bad answer. Skipping leaves the cell empty
    (a dash on the leaderboard) and lets a re-capture fill it.

    ``has_artifacts`` is the exception that keeps the blank check honest: on the
    ``file_output`` tasks the deliverable IS the attached file (``predictions.csv``,
    read by ``held_out_score``), and a submission may legitimately carry no prose at
    all. Those must be graded, not skipped.
    """
    if not answer.strip() and not has_artifacts:
        raise GenerationSkipped(f"empty submission (turn {turn_idx + 1}): {path}")
    if len(answer) <= _UI_ERROR_MAX_CHARS and _UI_ERROR_RE.search(answer):
        snippet = " ".join(answer.split())[:80]
        raise GenerationSkipped(
            f"product-UI error captured instead of an answer (turn {turn_idx + 1}): "
            f"{path} — {snippet!r}"
        )


@register_adapter("manual")
class ManualAdapter(Adapter):
    capabilities = Capabilities(
        text=True,
        file_upload=True,
        file_output=True,
        tool_trajectory=True,
        web=True,
        code_exec=True,
        manual=True,
    )

    def __init__(self, spec) -> None:
        super().__init__(spec)
        self._dir = self.config.get("submissions_dir", "submissions")

    def _submission_path(self) -> Path:
        base = Path(self._dir)
        if not base.is_absolute():
            base = settings.repo_root / base
        sys_dir = base / slug(self.system_id)
        # New layout: submissions/<system_slug>/seed<n>/<task_id>.md
        nested = sys_dir / f"seed{self._seed}" / f"{self._task_id}.md"
        if nested.exists():
            return nested
        # Old (flat) layout: submissions/<system_slug>/<task_id>__seed<n>.md
        flat = sys_dir / f"{self._task_id}__seed{self._seed}.md"
        if flat.exists():
            return flat
        # Default to the new layout for the "missing submission" error message.
        return nested

    @staticmethod
    def _parse(path: Path) -> tuple[dict, dict[int, str]]:
        """Return (frontmatter_dict, {turn_index0: answer_text})."""
        import yaml

        raw = path.read_text(encoding="utf-8")
        m = _FRONTMATTER_RE.match(raw)
        if m:
            meta = yaml.safe_load(m.group(1)) or {}
            body = m.group(2)
        else:
            meta, body = {}, raw

        sections: dict[int, str] = {}
        marks = list(_TURN_RE.finditer(body))
        if not marks:  # single-turn, no "## turn N" headers
            sections[0] = body.strip()
        else:
            for i, mk in enumerate(marks):
                start = mk.end()
                end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
                turn_no = int(mk.group(1))  # 1-based in the file
                sections[turn_no - 1] = body[start:end].strip()
        return meta, sections

    async def run_turn(
        self, history: list[Message], attachments: list[Attachment]
    ) -> TurnTranscript:
        path = self._submission_path()
        if not path.exists():
            raise GenerationSkipped(f"no submission: {path}")
        meta, sections = self._parse(path)

        turn_idx = sum(1 for m in history if m.role == "assistant")  # 0-based
        answer = sections.get(turn_idx, "")

        # artifacts + trajectory are declared once, on the first turn by convention.
        # Built before the non-answer check: an attached deliverable is what makes a
        # prose-less submission legitimate on the file_output tasks.
        artifacts: list[Artifact] = []
        trajectory: list[ToolCall] = []
        if turn_idx == 0:
            for name in meta.get("artifacts", []) or []:
                fp = path.parent / name
                if fp.exists():
                    artifacts.append(
                        Artifact(
                            name=name,
                            b64=base64.b64encode(fp.read_bytes()).decode("ascii"),
                        )
                    )
            trajectory = [ToolCall(name=t) for t in (meta.get("tools_used") or [])]

        _reject_non_answer(answer, path, turn_idx, has_artifacts=bool(artifacts))

        usage = Usage(
            prompt_tokens=int(meta.get("prompt_tokens", 0) or 0),
            completion_tokens=int(meta.get("completion_tokens", 0) or 0),
            cost_usd=meta.get("cost_usd"),  # None unless filled
            latency_ms=int(meta.get("latency_ms", 0) or 0),
        )

        return TurnTranscript(
            prompt=history[-1].content if history else "",
            answer_text=answer,
            artifacts=artifacts,
            trajectory=trajectory,
            citations=extract_citations(answer),
            usage=usage,
        )
