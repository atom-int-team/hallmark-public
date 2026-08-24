"""Subprocess adapter: runs a local CLI coding agent (*harness*) on one task.

The harness axis: the control is the bare tier, where the model gets the data
inline and answers in a single turn; the treatment is the same model slug inside
an agent loop with tools and the data files on disk.

Three differences from the other adapters:

* **Data is NOT delivered inline.** Task files are copied into a fresh per-cell
  workspace, mounted into the container as ``/work``; the prompt only lists them.
  The repository (and so ``ground_truth/``) is not visible to the agent at all —
  isolation comes from docker, not from discipline.
* **The answer is the last assistant message** in the harness event stream;
  trajectory and usage come from the same events.
* **Failure is detected by ``stopReason``, not by an empty answer.** A provider
  drop in the middle of the agent loop can leave a non-empty final text that is a
  preamble, not an answer. An "answer is non-empty" check would cache a zero that
  no later regrade can tell apart from a genuinely bad answer.

System config (``SystemSpec.config``):
    command:        list[str] (required) argv template with ``{workspace}``/``{model}``/``{prompt}``
    model:          str  (required) model slug substituted into the template
    thinking:       str  harness reasoning level, substituted as {thinking}
    harness:        str  harness name for provenance (default "pi")
    harness_version:str  harness version (pinned, recorded in provenance)
    parser:         str  event parser key: "pi" (default) | "opencode"
    retries:        int  number of RETRIES after a transport fault (default 2)
    timeout:        float|None wall-clock timeout per attempt; None = no timeout
    price_per_1k_in/out: float fallback cost estimate when the harness reports none
    keep_events:    bool keep the raw event stream and agent-written files (default True)
    trace_result_chars: int truncate tool results in the trace to this many chars (default 1500)
    trace_args_chars:   int truncate agent commands in the trace to this many chars (default 20000)

What stays on disk after a cell:
    transcripts/<system>/<task>__seed<n>.json   answer + trace (steps) + provenance
    harness_runs/<system>/<task>__seed<n>/      events.jsonl.gz (raw stream, ~265 KB),
                                               stderr.txt, cmd.txt, agent_files/ —
                                               scripts the agent wrote itself
    harness_workspaces/<system>/<task>__seed<n>/  cell working directory (overwritten)
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from ..config import settings
from ..models import (
    Attachment,
    Capabilities,
    Message,
    ToolCall,
    TurnTranscript,
    Usage,
)
from ..textutils import slug as _slug
from ..textutils import strip_think
from .base import Adapter, GenerationSkipped, price_cost, register_adapter

# Agent-written files are kept in full up to this size.
_MAX_AGENT_FILE_BYTES = 2_000_000

# The only addition to the task prompt. VERSIONED: any change makes a new harness
# arm, otherwise the ledger would mix two prompt versions.
HARNESS_PREAMBLE_V1 = (
    "\n\nThe data files for this task are in your current working directory:\n"
    "{listing}\n"
    "Read them from disk. Give your answer as your final message — do not write it to a file."
)


def _trunc(text: str, limit: int) -> str:
    """Truncate with an explicit marker of how much was dropped (never silently)."""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated {len(text) - limit} chars]"


def _parse_pi_events(
    stdout: str, *, result_chars: int = 1500, args_chars: int = 20000, stderr: str = ""
) -> tuple[str, list[ToolCall], Usage, dict]:
    """Parse the JSONL stream of ``pi --mode json``.

    Usage is on every ``message_end``, tool calls are in ``tool_execution_start``,
    the final text is the last non-empty assistant message. ``thinking`` blocks are
    a separate content type and never enter the answer.

    Besides the answer, a **solution trace** (``meta["trace"]``) is collected —
    what the agent did step by step: messages, tool calls with the FULL command text
    (that is the code it wrote) and tool results truncated to ``result_chars``. This
    lets the harness axis compare not just the score but how the task was solved.

    Returns ``(answer, trajectory, usage, meta)``; ``meta`` carries what the adapter
    needs to decide "answer vs transport fault", the trace and provenance fields.
    """
    answer = ""
    tools: list[ToolCall] = []
    trace: list[dict] = []
    pt = ct = cache_read = cache_write = reasoning = 0
    cost: float | None = None
    stop_reasons: list[str] = []
    errors: list[str] = []
    models: set[str] = set()
    thinking_chars = 0
    tool_errors = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("type")
        if kind == "tool_execution_start":
            args = ev.get("args") or {}
            tools.append(ToolCall(name=ev.get("toolName", "?"), arguments=args))
            trace.append(
                {
                    "kind": "tool_call",
                    "tool": ev.get("toolName", "?"),
                    # Commands get a much higher limit: a bash command is the agent's code.
                    "args": {k: _trunc(str(v), args_chars) for k, v in args.items()},
                }
            )
        elif kind == "tool_execution_end":
            result = ev.get("result") or {}
            text = "".join(
                c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
            )
            if ev.get("isError"):
                tool_errors += 1
            trace.append(
                {
                    "kind": "tool_result",
                    "tool": ev.get("toolName", "?"),
                    "is_error": bool(ev.get("isError")),
                    "chars": len(text),
                    "text": _trunc(text, result_chars),
                }
            )
        elif kind == "message_end":
            msg = ev.get("message") or {}
            usage = msg.get("usage") or {}
            pt += usage.get("input", 0) or 0
            ct += usage.get("output", 0) or 0
            cache_read += usage.get("cacheRead", 0) or 0
            cache_write += usage.get("cacheWrite", 0) or 0
            reasoning += usage.get("reasoning", 0) or 0
            total = (usage.get("cost") or {}).get("total")
            if total is not None:
                cost = (cost or 0.0) + total
            if msg.get("role") != "assistant":
                continue
            if msg.get("model"):
                models.add(msg["model"])
            if msg.get("stopReason"):
                stop_reasons.append(msg["stopReason"])
            if msg.get("errorMessage"):
                errors.append(str(msg["errorMessage"])[:400])
            content = msg.get("content") or []
            text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
            think = "".join(c.get("thinking", "") for c in content if c.get("type") == "thinking")
            thinking_chars += len(think)
            if text.strip() or think.strip():
                trace.append(
                    {
                        "kind": "assistant",
                        "stop_reason": msg.get("stopReason"),
                        "text": _trunc(text, result_chars),
                        "thinking": _trunc(think, result_chars),
                        "error": msg.get("errorMessage"),
                    }
                )
            if text.strip():
                answer = text

    meta = {
        "stop_reasons": stop_reasons,
        "final_stop_reason": stop_reasons[-1] if stop_reasons else None,
        "errors": errors,
        "models_seen": sorted(models),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "n_tool_calls": len(tools),
        "n_tool_errors": tool_errors,
        "thinking_chars": thinking_chars,
        "harness_cost_usd": cost,
        "trace": trace,
    }
    return answer, tools, Usage(prompt_tokens=pt, completion_tokens=ct, cost_usd=cost), meta


def _parse_opencode_events(
    stdout: str, *, result_chars: int = 1500, args_chars: int = 20000, stderr: str = ""
) -> tuple[str, list[ToolCall], Usage, dict]:
    """Parse the JSONL stream of ``opencode run --format json``.

    The payload is in ``part``; ``text``/``reasoning`` are whole parts, not deltas;
    ``tool_use`` carries both the tool input and output; usage and cost are on every
    ``step_finish`` (``reason``: ``tool-calls`` / ``stop``).

    Two differences from pi, which is why this parser also receives ``stderr``:

    * a transport fault shows up in stdout as ONE ``{"type": "error"}`` event, and
      only after opencode itself has retried the call for ~60 s; until then stdout is
      empty and stderr (``--print-logs``) has ``stream error`` lines. If the loop
      never reached ``step_finish.reason == "stop"`` and such lines exist, it is a
      fault, not agent silence;
    * stdout never names the model — ``models_seen`` comes from stderr
      (``modelID=…`` on ``stream`` lines): a cheap check that the declared model was
      called and not another one (e.g. by the title agent).

    Answer = the last non-empty ``text`` event. A stop on ``tool-calls`` with no
    following step is agent behaviour (as with pi) and is cached as the answer.
    """
    answer = ""
    tools: list[ToolCall] = []
    trace: list[dict] = []
    pt = ct = cache_read = cache_write = reasoning = 0
    cost: float | None = None
    stop_reasons: list[str] = []
    errors: list[str] = []
    models: set[str] = set()
    thinking_chars = 0
    tool_errors = 0
    cur_text: list[str] = []
    cur_think: list[str] = []

    def flush(reason: str | None) -> None:
        text, think = "".join(cur_text), "".join(cur_think)
        if text.strip() or think.strip():
            trace.append(
                {
                    "kind": "assistant",
                    "stop_reason": reason,
                    "text": _trunc(text, result_chars),
                    "thinking": _trunc(think, result_chars),
                    "error": None,
                }
            )
        cur_text.clear()
        cur_think.clear()

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("type")
        part = ev.get("part") or {}
        if kind == "text":
            text = part.get("text") or ""
            cur_text.append(text)
            if text.strip():
                answer = text
        elif kind == "reasoning":
            think = part.get("text") or ""
            cur_think.append(think)
            thinking_chars += len(think)
        elif kind == "tool_use":
            state = part.get("state") or {}
            args = state.get("input") or {}
            name = part.get("tool") or "?"
            tools.append(ToolCall(name=name, arguments=args))
            trace.append(
                {
                    "kind": "tool_call",
                    "tool": name,
                    "args": {k: _trunc(str(v), args_chars) for k, v in args.items()},
                }
            )
            out = state.get("output")
            if out is None:
                out = state.get("error") or ""
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False)
            is_error = state.get("status") == "error"
            if is_error:
                tool_errors += 1
            trace.append(
                {
                    "kind": "tool_result",
                    "tool": name,
                    "is_error": is_error,
                    "chars": len(out),
                    "text": _trunc(out, result_chars),
                }
            )
        elif kind == "step_finish":
            tok = part.get("tokens") or {}
            pt += tok.get("input", 0) or 0
            ct += tok.get("output", 0) or 0
            reasoning += tok.get("reasoning", 0) or 0
            cache = tok.get("cache") or {}
            cache_read += cache.get("read", 0) or 0
            cache_write += cache.get("write", 0) or 0
            c = part.get("cost")
            if c is not None:
                cost = (cost or 0.0) + float(c)
            reason = part.get("reason")
            if reason:
                stop_reasons.append(str(reason))
            flush(reason)
        elif kind == "error":
            err = ev.get("error") or {}
            data = err.get("data") or {}
            msg = data.get("message") or json.dumps(err, ensure_ascii=False)
            errors.append(f"{err.get('name', '?')}: {msg}"[:400])
    flush(None)  # tail without a step_finish (stream cut off)

    stream_errors = 0
    for line in stderr.splitlines():
        if "message=stream " in line or 'message="stream error"' in line:
            m = re.search(r"modelID=(\S+)", line)
            if m and "stream error" not in line:
                models.add(m.group(1))
            # Title-agent (session title) failures do not break the main loop —
            # they are excluded from the fault rule, or a real answer would be retried.
            if "stream error" in line and "agent=title" not in line:
                stream_errors += 1

    # Subagents (`task`) run in CHILD sessions, and `--format json` prints only the
    # parent's events: their tokens and cost never reach step_finish. A cell with at
    # least one `task` call has a lower-bound cost, which provenance must show.
    n_task_calls = sum(1 for t in tools if t.name == "task")

    final = stop_reasons[-1] if stop_reasons else None
    if errors:
        final = "error"
    elif final != "stop" and stream_errors:
        # The loop did not end with an answer and the log shows transport failures.
        final = "error"
        errors.append(f"{stream_errors} × 'stream error' in stderr without a final stop")
    elif final is None and not answer.strip():
        final = "error"  # no step and no text at all — the attempt never happened

    meta = {
        "stop_reasons": stop_reasons,
        "final_stop_reason": final,
        "errors": errors,
        "models_seen": sorted(models),
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "n_tool_calls": len(tools),
        "n_tool_errors": tool_errors,
        "thinking_chars": thinking_chars,
        "harness_cost_usd": cost,
        "n_task_calls": n_task_calls,
        "cost_incomplete": n_task_calls > 0,
        "trace": trace,
    }
    return answer, tools, Usage(prompt_tokens=pt, completion_tokens=ct, cost_usd=cost), meta


PARSERS = {"pi": _parse_pi_events, "opencode": _parse_opencode_events}

# stopReasons meaning a transport/provider fault rather than a model answer. Such an
# attempt is retried and NOT cached: a fast crash is not a zero. `timeout` is added
# artificially (the harness never reports it — the attempt was killed from outside),
# but it is the same class: the attempt produced no answer.
_TRANSPORT_STOPS = frozenset({"error", "timeout"})


@register_adapter("cli_agent")
class CliAgentAdapter(Adapter):
    # manual=True — same trick as the bare tier: passes the capability gate of tasks
    # with ``required_capabilities: [manual]``. code_exec/file_upload are true for
    # this arm: the agent reads files from disk and executes code.
    capabilities = Capabilities(
        text=True,
        file_upload=True,
        code_exec=True,
        tool_trajectory=True,
        manual=True,
    )

    def __init__(self, spec) -> None:
        super().__init__(spec)
        self.command: list[str] = list(self.config["command"])
        self.model: str = self.config["model"]
        self.thinking: str = str(self.config.get("thinking", ""))
        self.harness: str = self.config.get("harness", "pi")
        self.harness_version: str = str(self.config.get("harness_version", ""))
        self.parser = PARSERS[self.config.get("parser", "pi")]
        self.retries: int = int(self.config.get("retries", 2))
        # An exhausted WATCHDOG is not the same as an exhausted transport. A gateway
        # 502 is an infrastructure failure and must not be charged to the system (the
        # cell stays empty). The time limit, however, is declared part of the arm — a
        # compute budget given to the system: no answer within it is the system's
        # behaviour, not an environment fault. Same class as the output-token cap in
        # the bare tier, which the ledger scores as zero, not as missing.
        self.timeout_exhausted_scores_zero: bool = bool(
            self.config.get("timeout_exhausted_scores_zero", False)
        )
        self.timeout: float | None = self.config.get("timeout")
        self.price_in = self.config.get("price_per_1k_in")
        self.price_out = self.config.get("price_per_1k_out")
        # Keep the raw event stream and the files the agent wrote itself.
        # ~265 KB gzipped per cell — the ONLY complete record of how the harness
        # solved the task; the trace in the transcript is truncated by design.
        self.keep_events: bool = bool(self.config.get("keep_events", True))
        self.trace_result_chars: int = int(self.config.get("trace_result_chars", 1500))
        # An agent command is its code, so it is truncated much later than results:
        # in a real cell 16 commands totalled 45 KB, and one alone hit 8 KB.
        self.trace_args_chars: int = int(self.config.get("trace_args_chars", 20000))

    # ------------------------------------------------------------------ #
    # workspace                                                          #
    # ------------------------------------------------------------------ #
    def _workspace(self) -> Path:
        """Fresh per-cell directory. It is the only thing the container sees."""
        ws = (
            settings.repo_root
            / "harness_workspaces"
            / _slug(self.system_id)
            / f"{self._task_id}__seed{self._seed}"
        )
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        return ws

    def _run_dir(self) -> Path:
        """Per-cell artifact directory: raw events, stderr, agent files.

        Separate from the workspace: the workspace is wiped on every run, while this
        is the run record and must survive a restart.
        """
        d = (
            settings.repo_root
            / "harness_runs"
            / _slug(self.system_id)
            / f"{self._task_id}__seed{self._seed}"
        )
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        return d

    @staticmethod
    def _stage(ws: Path, attachments: list[Attachment]) -> str:
        """Copy task files into the workspace; return their listing for the prompt."""
        names: list[str] = []
        for att in attachments:
            if not att.path:
                continue
            shutil.copy2(att.path, ws / Path(att.name).name)
            names.append(f"  - {Path(att.name).name}")
        return "\n".join(names)

    @staticmethod
    def _new_files(ws: Path, before: set[str]) -> list[str]:
        """Files the agent left in the workspace — an anti-contamination record."""
        after = {str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()}
        return sorted(after - before)

    # ------------------------------------------------------------------ #
    # execution                                                          #
    # ------------------------------------------------------------------ #
    async def _run_process(self, argv: list[str], cwd: Path) -> tuple[int, str, str]:
        """Process seam — monkeypatched in tests."""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            # Killing the `docker run` CLIENT is not enough: the container outlives its
            # client and keeps calling the LLM through the gateway — spending money on
            # an attempt whose result is already discarded and holding CPU, which pushes
            # neighbouring cells into the watchdog too. The container is found by its
            # mounted workspace, which is unique per cell, so argv stays unchanged (the
            # arm stays homogeneous).
            await self._kill_container_for(cwd)
            raise
        return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def _kill_container_for(self, ws: Path) -> None:
        """docker kill the container that has THIS workspace mounted."""
        if not self.command or self.command[0] != "docker":
            return
        try:
            ps = await asyncio.create_subprocess_exec(
                "docker", "ps", "-q",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            ids_out, _ = await ps.communicate()
            ids = ids_out.decode().split()
            if not ids:
                return
            insp = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f",
                "{{.Id}} {{range .Mounts}}{{.Source}} {{end}}", *ids,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            insp_out, _ = await insp.communicate()
            target = str(ws)
            doomed = [
                line.split()[0]
                for line in insp_out.decode().splitlines()
                if len(line.split()) > 1 and target in line.split()[1:]
            ]
            if doomed:
                kill = await asyncio.create_subprocess_exec(
                    "docker", "kill", *doomed,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await kill.wait()
        except Exception:  # cleanup must never fail the cell — the retry matters more
            pass

    async def run_turn(
        self, history: list[Message], attachments: list[Attachment]
    ) -> TurnTranscript:
        ws = self._workspace()
        listing = self._stage(ws, attachments)
        staged = {str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()}

        task_prompt = history[-1].content if history else ""
        prompt = task_prompt + HARNESS_PREAMBLE_V1.format(listing=listing)
        argv = [
            a.format(workspace=str(ws), model=self.model, prompt=prompt, thinking=self.thinking)
            for a in self.command
        ]

        t0 = time.perf_counter()
        attempts: list[dict] = []
        answer, tools, usage, meta = "", [], Usage(), {}
        for attempt in range(self.retries + 1):
            try:
                rc, out, err = await self._run_process(argv, ws)
            except TimeoutError:
                # A wall-clock timeout is the same class as a transport fault: the
                # attempt produced NO answer, so retry rather than cache a zero. The
                # process is already killed in _run_process; there are no events, so
                # only the fact is recorded.
                answer, tools, usage, meta = "", [], Usage(), {"final_stop_reason": "timeout"}
                rc, out, err = -1, "", f"timeout after {self.timeout}s"
            else:
                answer, tools, usage, meta = self.parser(
                    out,
                    result_chars=self.trace_result_chars,
                    args_chars=self.trace_args_chars,
                    stderr=err,
                )
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": rc,
                    "final_stop_reason": meta.get("final_stop_reason"),
                    "errors": meta.get("errors"),
                    "stderr_tail": err.strip()[-300:],
                    "_stdout": out,  # popped below: written to a file or dropped
                    "_stderr": err,  # in full — for opencode it is the only fault log
                }
            )
            transport_fault = meta.get("final_stop_reason") in _TRANSPORT_STOPS
            if answer.strip() and not transport_fault:
                break
        else:
            all_timeout = all(a.get("final_stop_reason") == "timeout" for a in attempts)
            if not (all_timeout and self.timeout_exhausted_scores_zero):
                # Every attempt was a transport fault: the cell stays empty (missing)
                # rather than cached as a zero indistinguishable from a bad answer.
                raise GenerationSkipped(
                    f"{self.harness}: transport fault after {self.retries + 1} "
                    f"attempts — {json.dumps(attempts[-1], ensure_ascii=False)[:300]}"
                )
            # The system never finished within its declared budget: no answer.
            # Cache an empty answer — the grader scores 0, as for an answer without `Final:`.
            answer, tools, usage = "", [], Usage()
            meta = {"final_stop_reason": "timeout_exhausted"}

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # Run artifacts: raw events of every attempt, stderr and the files the agent
        # wrote itself (its scripts are the most informative part of a "how it solved
        # it" analysis; the workspace will not survive).
        run_dir = None
        if self.keep_events:
            run_dir = self._run_dir()
            (run_dir / "cmd.txt").write_text("\n".join(argv), encoding="utf-8")
            for rec in attempts:
                tag = "" if len(attempts) == 1 else f".attempt{rec['attempt']}"
                with gzip.open(run_dir / f"events{tag}.jsonl.gz", "wt", encoding="utf-8") as fh:
                    fh.write(rec.pop("_stdout", ""))
                full_err = rec.pop("_stderr", "") or rec.get("stderr_tail", "")
                if full_err.strip():
                    (run_dir / f"stderr{tag}.txt").write_text(full_err, encoding="utf-8")
            created = self._new_files(ws, staged)
            if created:
                dst = run_dir / "agent_files"
                dst.mkdir()
                for rel in created:
                    src = ws / rel
                    out_path = dst / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if src.stat().st_size <= _MAX_AGENT_FILE_BYTES:
                        shutil.copy2(src, out_path)
                    else:  # a huge dump is not copied, but its existence is recorded
                        out_path.write_text(
                            f"[not saved: {src.stat().st_size} bytes > "
                            f"{_MAX_AGENT_FILE_BYTES}]",
                            encoding="utf-8",
                        )
        else:
            for rec in attempts:
                rec.pop("_stdout", None)
                rec.pop("_stderr", None)

        # Cost at LEDGER prices (the systems-YAML snapshot, same as the bare rows) is
        # ALWAYS computed and stored in provenance next to the harness's self-reported
        # cost: harness price catalogs drift on their own (opencode 1.18.29 prices
        # gpt-5.6-sol at 2/10 per 1M while the ledger used 5/30), and an arm-vs-bare
        # cost comparison is only fair on a single price snapshot.
        ledger_cost = price_cost(
            self.price_in, self.price_out, usage.prompt_tokens, usage.completion_tokens
        )
        cost = usage.cost_usd
        if cost is None:
            cost = ledger_cost
        usage = Usage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        self.last_provenance = {
            "harness": self.harness,
            "harness_version": self.harness_version,
            "model_requested": self.model,
            "thinking": self.thinking,
            "models_seen": meta.get("models_seen"),
            "argv": [a for a in argv if a != prompt],  # prompt not duplicated — it is in the transcript
            "preamble_version": "v1",
            "prompt_sha256": hashlib.sha256(task_prompt.encode("utf-8")).hexdigest(),
            "attempts": attempts,
            "workspace": str(ws),
            "files_created_by_agent": self._new_files(ws, staged),
            "cache_read_tokens": meta.get("cache_read_tokens"),
            "cache_write_tokens": meta.get("cache_write_tokens"),
            "reasoning_tokens": meta.get("reasoning_tokens"),
            "n_tool_calls": meta.get("n_tool_calls"),
            "n_tool_errors": meta.get("n_tool_errors"),
            "thinking_chars": meta.get("thinking_chars"),
            "final_stop_reason": meta.get("final_stop_reason"),
            # opencode: subagent calls and the "cost is a lower bound" flag.
            "n_task_calls": meta.get("n_task_calls"),
            "cost_incomplete": meta.get("cost_incomplete"),
            "harness_cost_usd": meta.get("harness_cost_usd"),
            "cost_at_ledger_prices_usd": ledger_cost,
            "run_dir": str(run_dir) if run_dir else None,
        }

        return TurnTranscript(
            prompt=prompt,
            answer_text=strip_think(answer),
            trajectory=tools if self.capabilities.tool_trajectory else [],
            usage=usage,
            # Step-by-step solution trace: messages, full commands, truncated results.
            # The transcript stays self-contained for analysis even without harness_runs/.
            trace=meta.get("trace") or [],
        )
