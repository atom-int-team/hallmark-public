"""Core data models (see UNIVERSAL_BENCHMARK_SPEC §6, §7).

Pydantic models throughout so that Task YAML is validated on load and
Transcripts/Results round-trip to JSON for caching.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #


class Domain(str, Enum):
    GENERAL_QA = "general_qa"
    LIT_REVIEW = "lit_review"
    END_TO_END = "end_to_end"
    MATERIALS_DATABASES = "materials_databases"
    CODE = "code"
    STRUCTURE = "structure"
    CHARACTERIZATION = "characterization"


class Track(str, Enum):
    REASONING = "reasoning"
    EXECUTION = "execution"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Split(str, Enum):
    DEV = "dev"
    TEST = "test"


class Signal(str, Enum):
    """The kind of output a grader needs from the SUT."""

    TEXT = "text"
    FILE_OUTPUT = "file_output"
    TOOL_TRAJECTORY = "tool_trajectory"


# --------------------------------------------------------------------------- #
# Task schema (§6.1)                                                           #
# --------------------------------------------------------------------------- #


class GradingItem(BaseModel):
    """One grader invocation; produces exactly one GraderResult."""

    grader: str
    signal: Signal = Signal.TEXT
    weight: float = 1.0
    params: dict = Field(default_factory=dict)


class Turn(BaseModel):
    prompt: str
    grading: list[GradingItem] = Field(default_factory=list)


class TaskInputs(BaseModel):
    files: list[str] = Field(default_factory=list)


class Task(BaseModel):
    id: str
    domain: Domain
    track: Track
    difficulty: Difficulty = Difficulty.MEDIUM
    split: Split = Split.DEV
    time_gate: str | None = None  # ISO date; question about data after model cutoff
    required_capabilities: list[str] = Field(default_factory=list)
    inputs: TaskInputs = Field(default_factory=TaskInputs)
    turns: list[Turn]
    ground_truth_ref: str | None = None
    runs: int | None = None  # override of the global default
    canary: str | None = None  # unique marker to detect dataset leakage (§9)


# --------------------------------------------------------------------------- #
# Adapter I/O (§7.1)                                                           #
# --------------------------------------------------------------------------- #


class Capabilities(BaseModel):
    text: bool = True
    file_upload: bool = False  # accepts attached files
    file_output: bool = False  # emits generated files
    tool_trajectory: bool = False  # emits a list of tool calls
    web: bool = False
    code_exec: bool = False
    manual: bool = False  # answers are human-captured runs ingested from submissions/


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class Attachment(BaseModel):
    name: str
    mime: str = "application/octet-stream"
    text: str | None = None  # textual payload (CSV head, schema, …) for inline delivery
    path: str | None = None  # on-disk path for file_upload delivery


class Artifact(BaseModel):
    name: str
    mime: str = "application/octet-stream"
    path: str | None = None
    b64: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Transcript (§6.2)                                                            #
# --------------------------------------------------------------------------- #


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int = 0

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=cost,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class TurnTranscript(BaseModel):
    prompt: str
    answer_text: str  # final visible text (think-tags stripped)
    artifacts: list[Artifact] = Field(default_factory=list)
    trajectory: list[ToolCall] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)  # extracted DOIs/URLs
    # How the system reached its answer, step by step: messages, tool calls with the full
    # command text, (truncated) results. Only agentic adapters fill it — a one-turn API
    # system has no steps, and the field stays empty.
    trace: list[dict] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)  # per-turn; Transcript.usage is the sum


class Transcript(BaseModel):
    task_id: str
    system_id: str  # e.g. "anthropic/claude-opus-4-8"
    seed_idx: int
    turns: list[TurnTranscript]
    usage: Usage = Field(default_factory=Usage)
    adapter_capabilities: Capabilities = Field(default_factory=Capabilities)
    # What produced the cell: for an API system, the model and its parameters (already in the
    # systems YAML); for a harness, its version, argv, prompt hash and the files the agent
    # created. Optional and empty by default, so older transcripts validate unchanged.
    provenance: dict = Field(default_factory=dict)
    schema_version: int = 1


# --------------------------------------------------------------------------- #
# Results (§6.3)                                                               #
# --------------------------------------------------------------------------- #


class GraderResult(BaseModel):
    grader: str
    signal: Signal
    applicable: bool  # False -> N/A (signal absent for this SUT)
    score: float | None  # 0..1 when applicable
    weight: float
    detail: str = ""


class TurnResult(BaseModel):
    turn_idx: int
    grader_results: list[GraderResult] = Field(default_factory=list)
    turn_score: float | None = None  # weighted mean over applicable; None if all N/A


class RunResult(BaseModel):
    task_id: str
    domain: Domain
    track: Track
    difficulty: Difficulty
    system_id: str
    seed_idx: int
    turns: list[TurnResult] = Field(default_factory=list)
    task_score: float | None = None
    usage: Usage = Field(default_factory=Usage)


# --------------------------------------------------------------------------- #
# System spec (which SUTs to run)                                              #
# --------------------------------------------------------------------------- #


class SystemSpec(BaseModel):
    """One System Under Test, as declared in a systems YAML file.

    A SUT may be driven by **one or several** adapters. Per task the runner picks
    exactly one (see ``runner._select_adapter``): a task that requires the
    ``manual`` capability is served by the system's ``manual`` adapter (answers
    read from ``submissions/``), every other task by its first non-manual adapter.

    Three equivalent ways to declare the adapter(s):

        adapter: openai_chat                 # single (uses the shared ``config``)
        adapter: openai_chat + manual        # several, shared ``config``
        adapters:                            # several, per-adapter ``config``
          openai_chat: {model: gpt-5.5, api_key_env: OPENAI_API_KEY}
          manual:      {submissions_dir: submissions}
    """

    id: str  # display id, e.g. "openai/gpt-4o-mini"
    adapter: str | list[str] | None = None  # registry key(s); "a + b" or [a, b]
    adapters: dict[str, dict] | None = None  # name -> per-adapter config (overrides config)
    config: dict = Field(default_factory=dict)  # adapter-specific (model, base_url, …)

    @model_validator(mode="after")
    def _require_some_adapter(self) -> SystemSpec:
        if not self.adapter and not self.adapters:
            raise ValueError(f"system {self.id!r}: declare `adapter` or `adapters`")
        return self

    def adapter_components(self) -> list[tuple[str, dict]]:
        """Resolve to an ordered list of ``(adapter_name, merged_config)``.

        For the ``adapters`` map, each per-adapter config is layered over the
        shared ``config``; for ``adapter`` every name shares ``config`` verbatim.
        """
        if self.adapters:
            return [(name, {**self.config, **(sub or {})}) for name, sub in self.adapters.items()]
        return [(name, self.config) for name in self.adapter_names()]

    def adapter_names(self) -> list[str]:
        """The adapter registry keys, in declaration order."""
        if self.adapters:
            return list(self.adapters.keys())
        if isinstance(self.adapter, list):
            return [str(a).strip() for a in self.adapter if str(a).strip()]
        if isinstance(self.adapter, str):
            return [p.strip() for p in self.adapter.split("+") if p.strip()]
        return []
