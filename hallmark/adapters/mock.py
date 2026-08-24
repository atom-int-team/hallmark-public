"""mock — offline adapter for tests and dry-runs (no network).

Returns a canned answer. Config:
    answers:  dict[task_id -> str]   per-task canned answer
    default:  str                    fallback answer (default: echoes the prompt)
The answer may include "{seed}" which is formatted with the seed index.
"""
from __future__ import annotations

from ..models import Attachment, Capabilities, Message, TurnTranscript, Usage
from ..textutils import extract_citations
from .base import Adapter, register_adapter


@register_adapter("mock")
class MockAdapter(Adapter):
    capabilities = Capabilities(text=True)

    async def run_turn(
        self, history: list[Message], attachments: list[Attachment]
    ) -> TurnTranscript:
        prompt = history[-1].content if history else ""
        answers: dict = self.config.get("answers", {})
        answer = answers.get(self._task_id, self.config.get("default", f"echo: {prompt}"))
        answer = answer.replace("{seed}", str(self._seed))
        return TurnTranscript(
            prompt=prompt,
            answer_text=answer,
            citations=extract_citations(answer),
            usage=Usage(prompt_tokens=len(prompt.split()), completion_tokens=len(answer.split())),
        )
