"""OpenAI / OpenRouter chat-completions adapter.

Capabilities: text only (no file upload) — attachments are delivered inline in
the prompt (see §7.3). Set ``base_url`` to point at OpenRouter or any
OpenAI-compatible endpoint.

System config (``SystemSpec.config``):
    model:            str   (required) e.g. "gpt-4o-mini" / "openai/gpt-4o-mini"
    api_key_env:      str   env var holding the key (default "OPENAI_API_KEY")
    base_url:         str   optional, e.g. "https://openrouter.ai/api/v1"
    system_prompt:    str   optional
    temperature:      float optional
    max_tokens:       int   optional
    price_per_1k_in:  float optional — for cost_usd
    price_per_1k_out: float optional
    empty_answer_retries: int optional (default 2) — see ``run_turn``
    extra_body:       dict  optional — passed verbatim as the request's ``extra_body``, for
                            provider-specific params the OpenAI schema has no field for (e.g.
                            OpenRouter's ``{"reasoning": {"effort": "high"}}``). Inert when
                            unset, which is the case for every system that does not declare it.
"""
from __future__ import annotations

import time

from ..config import settings
from ..delivery import render_inline_attachments
from ..models import Attachment, Capabilities, Message, TurnTranscript, Usage
from ..textutils import extract_citations, strip_think
from .base import Adapter, register_adapter


@register_adapter("openai_chat")
class OpenAIChatAdapter(Adapter):
    capabilities = Capabilities(text=True, file_upload=False)

    def __init__(self, spec) -> None:
        super().__init__(spec)
        self.model: str = self.config["model"]
        self.api_key_env: str = self.config.get("api_key_env", "OPENAI_API_KEY")
        self.base_url: str | None = self.config.get("base_url")
        self.system_prompt: str | None = self.config.get("system_prompt")
        self.temperature = self.config.get("temperature")
        self.max_tokens = self.config.get("max_tokens")
        self.price_in = self.config.get("price_per_1k_in")
        self.price_out = self.config.get("price_per_1k_out")
        self.empty_answer_retries = int(self.config.get("empty_answer_retries", 2))
        self.extra_body = self.config.get("extra_body")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            key = settings.api_key(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.system_id}: API key not found (env {self.api_key_env})"
                )
            self._client = AsyncOpenAI(
                api_key=key,
                base_url=self.base_url,
                timeout=self.config.get("request_timeout", 300),  # per-request seconds
                max_retries=self.config.get("max_retries", 2),  # SDK retries on timeout/5xx
            )
        return self._client

    async def _create(self, **kwargs):
        return await self._get_client().chat.completions.create(**kwargs)

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.price_in is None and self.price_out is None:
            return None
        return (
            prompt_tokens / 1000 * (self.price_in or 0.0)
            + completion_tokens / 1000 * (self.price_out or 0.0)
        )

    async def run_turn(
        self, history: list[Message], attachments: list[Attachment]
    ) -> TurnTranscript:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        rendered = [m.model_dump() for m in history]
        if attachments and rendered and rendered[-1]["role"] == "user":
            rendered[-1]["content"] += render_inline_attachments(attachments)
        messages.extend(rendered)

        kwargs: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        answer = ""
        finish = None
        for _ in range(self.empty_answer_retries + 1):
            t0 = time.perf_counter()
            resp = await self._create(**kwargs)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            choices = getattr(resp, "choices", None)
            if not choices:  # OpenRouter returns provider errors in-band (200 + no choices)
                err = getattr(resp, "error", None) or getattr(resp, "model_extra", None)
                raise RuntimeError(f"{self.system_id}: API returned no choices ({err})")
            raw = choices[0].message.content or ""
            answer = strip_think(raw)
            finish = getattr(choices[0], "finish_reason", None)
            if answer or finish == "length":
                break

        # An empty answer has two unlike causes, split by finish_reason:
        #   "length" — the model spent its whole output budget reasoning and never spoke.
        #     A genuine failure, so it is graded (flat 0), not retried: the retry would hit
        #     the same cap and cost another full budget.
        #   "stop"   — a provider flake: `content` comes back null with the output stranded
        #     in `reasoning`. Raising (not caching "") stops the runner from freezing it into
        #     a permanent zero — it skips the task, and a re-run retries.
        if not answer and finish != "length":
            raise RuntimeError(
                f"{self.system_id}: empty answer after {self.empty_answer_retries + 1} "
                f"attempt(s) (finish_reason={finish})"
            )

        u = resp.usage
        prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
        completion_tokens = getattr(u, "completion_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._cost(prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
        )

        prompt_text = history[-1].content if history else ""
        return TurnTranscript(
            prompt=prompt_text,
            answer_text=answer,
            citations=extract_citations(answer),
            usage=usage,
        )
