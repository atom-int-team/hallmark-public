"""Offline tests of the harness adapter (`cli_agent`) — no docker, no network.

They cover exactly what can break silently and spoil a whole run:
  1. task files really land in the workspace, while their CONTENT does not enter
     the prompt (otherwise delivery is no longer "from disk" and the matched pair
     with the bare tier breaks);
  2. usage / trajectory / provenance are parsed from the harness events;
  3. a transport fault (`stopReason: error`) is retried and NOT cached as a zero,
     even when the answer text is non-empty (a provider drop mid-loop);
  4. a real answer without `Final:` is cached and left to the grader.
"""
from __future__ import annotations

import json

import pytest

from hallmark.adapters.base import GenerationSkipped
from hallmark.adapters.cli_agent import CliAgentAdapter
from hallmark.config import settings
from hallmark.models import Attachment, Message, SystemSpec


def _spec(**cfg) -> SystemSpec:
    base = {
        "harness": "pi",
        "harness_version": "0.84.3",
        "model": "x-ai/grok-4.6",
        "command": ["echo", "{workspace}", "{model}", "{prompt}"],
        "retries": 1,
    }
    base.update(cfg)
    return SystemSpec(id="pi/test-model", adapter="cli_agent", config=base)


def _events(*, answer: str, stop: str = "stop", n_tools: int = 2,
            result: str = "stdout of echo") -> str:
    """A JSONL snippet in the `pi --mode json` schema."""
    lines = [{"type": "session", "id": "x", "cwd": "/work"}, {"type": "agent_start"}]
    for i in range(n_tools):
        lines.append(
            {
                "type": "tool_execution_start",
                "toolCallId": f"c{i}",
                "toolName": "bash",
                "args": {"command": f"echo {i}"},
            }
        )
        lines.append(
            {
                "type": "tool_execution_end",
                "toolCallId": f"c{i}",
                "toolName": "bash",
                "isError": False,
                "result": {"content": [{"type": "text", "text": result}]},
            }
        )
        lines.append(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "model": "x-ai/grok-4.6",
                    "stopReason": "toolUse",
                    "content": [{"type": "thinking", "thinking": "step think"},
                                {"type": "text", "text": "working on it"}],
                    "usage": {
                        "input": 100,
                        "output": 20,
                        "cacheRead": 7,
                        "cacheWrite": 0,
                        "reasoning": 5,
                        "cost": {"total": 0.01},
                    },
                },
            }
        )
    lines.append(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "model": "x-ai/grok-4.6",
                "stopReason": stop,
                "errorMessage": "402: out of credits" if stop == "error" else None,
                "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": answer}],
                "usage": {
                    "input": 50,
                    "output": 30,
                    "cacheRead": 3,
                    "cacheWrite": 0,
                    "reasoning": 1,
                    "cost": {"total": 0.02},
                },
            },
        }
    )
    lines.append({"type": "agent_end", "willRetry": False})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """Point the repo root at a tmp dir: the workspace is created there."""
    monkeypatch.setattr(settings, "repo_root", tmp_path)
    return tmp_path


@pytest.fixture()
def csv_attachment(tmp_path):
    src = tmp_path / "src" / "train.csv"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("a,b,target\n1,2,3.0\n", encoding="utf-8")
    return Attachment(name="train.csv", mime="text/csv", text=src.read_text(), path=str(src))


def _run(adapter, prompt, attachments, outputs):
    """Run run_turn with the process seam replaced by a sequence of outputs."""
    calls = []

    async def fake(argv, cwd):
        calls.append((argv, cwd))
        return 0, outputs[min(len(calls) - 1, len(outputs) - 1)], ""

    adapter._run_process = fake  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    turn = asyncio.run(adapter.run_turn([Message(role="user", content=prompt)], attachments))
    return turn, calls


def test_files_staged_and_csv_not_inlined(repo, csv_attachment):
    adapter = CliAgentAdapter(_spec())
    turn, calls = _run(adapter, "TASK PROMPT", [csv_attachment], [_events(answer="Final: c1")])

    ws = repo / "harness_workspaces" / "pi_test-model" / "t_task__seed0"
    assert (ws / "train.csv").read_text() == "a,b,target\n1,2,3.0\n"

    # The CSV content must not be in the prompt — only the file name.
    assert "1,2,3.0" not in turn.prompt
    assert "train.csv" in turn.prompt
    assert turn.prompt.startswith("TASK PROMPT")
    # ...nor in argv (the prompt goes there too).
    assert not any("1,2,3.0" in a for a in calls[0][0])


def test_usage_trajectory_and_provenance(repo, csv_attachment):
    adapter = CliAgentAdapter(_spec())
    turn, _ = _run(adapter, "P", [csv_attachment], [_events(answer="Final: c1, c2", n_tools=3)])

    assert turn.answer_text.strip() == "Final: c1, c2"
    assert [t.name for t in turn.trajectory] == ["bash", "bash", "bash"]
    assert turn.usage.prompt_tokens == 3 * 100 + 50
    assert turn.usage.completion_tokens == 3 * 20 + 30
    # Cost comes from the harness report, not recomputed from prices.
    assert turn.usage.cost_usd == pytest.approx(3 * 0.01 + 0.02)

    prov = adapter.last_provenance
    assert prov["harness"] == "pi" and prov["harness_version"] == "0.84.3"
    assert prov["models_seen"] == ["x-ai/grok-4.6"]
    assert prov["final_stop_reason"] == "stop"
    assert prov["n_tool_calls"] == 3
    assert prov["files_created_by_agent"] == []
    # The sha is of the TASK prompt, without the preamble — the invariant for matching bare.
    import hashlib

    assert prov["prompt_sha256"] == hashlib.sha256(b"P").hexdigest()


def test_transport_fault_retries_then_skips(repo, csv_attachment):
    """`stopReason: error` is a fault, not an answer, EVEN if the text is non-empty."""
    adapter = CliAgentAdapter(_spec(retries=1))
    partial = _events(answer="Low-pressure uptake should track small pores.", stop="error")
    with pytest.raises(GenerationSkipped) as exc:
        _run(adapter, "P", [csv_attachment], [partial])
    assert "transport fault" in str(exc.value)


def test_transport_fault_recovers_on_retry(repo, csv_attachment):
    adapter = CliAgentAdapter(_spec(retries=2))
    outputs = [_events(answer="preamble", stop="error"), _events(answer="Final: c9")]
    turn, calls = _run(adapter, "P", [csv_attachment], outputs)
    assert turn.answer_text.strip() == "Final: c9"
    assert len(calls) == 2
    assert [a["final_stop_reason"] for a in adapter.last_provenance["attempts"]] == ["error", "stop"]


def test_real_answer_without_final_is_kept(repo, csv_attachment):
    """A bad answer is still an answer: cache it and let the grader score it (0)."""
    adapter = CliAgentAdapter(_spec())
    turn, _ = _run(adapter, "P", [csv_attachment], [_events(answer="I think c1 is nice.")])
    assert turn.answer_text.strip() == "I think c1 is nice."


def test_agent_created_files_are_recorded(repo, csv_attachment):
    """Files the agent left in the workspace are recorded for anti-contamination."""
    adapter = CliAgentAdapter(_spec())

    async def fake(argv, cwd):
        (cwd / "downloaded.json").write_text("{}", encoding="utf-8")
        return 0, _events(answer="Final: c1"), ""

    adapter._run_process = fake  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    asyncio.run(adapter.run_turn([Message(role="user", content="P")], [csv_attachment]))
    assert adapter.last_provenance["files_created_by_agent"] == ["downloaded.json"]


def test_trace_records_steps_commands_and_results(repo, csv_attachment):
    """Solution trace: messages, the FULL command text, the (truncated) result."""
    adapter = CliAgentAdapter(_spec())
    turn, _ = _run(adapter, "P", [csv_attachment], [_events(answer="Final: c1", n_tools=2)])

    kinds = [s["kind"] for s in turn.trace]
    assert kinds.count("tool_call") == 2
    assert kinds.count("tool_result") == 2
    assert kinds.count("assistant") == 3  # 2 intermediate + final
    call = next(s for s in turn.trace if s["kind"] == "tool_call")
    assert call["tool"] == "bash" and call["args"]["command"] == "echo 0"
    res = next(s for s in turn.trace if s["kind"] == "tool_result")
    assert res["text"] == "stdout of echo" and res["is_error"] is False
    # Reasoning goes into the trace but NOT into the answer.
    assert any(s.get("thinking") for s in turn.trace)
    assert "thinking" not in turn.answer_text


def test_long_tool_result_is_truncated_with_a_marker(repo, csv_attachment):
    """Results are truncated only with an explicit marker, never silently."""
    adapter = CliAgentAdapter(_spec(trace_result_chars=50))
    turn, _ = _run(adapter, "P", [csv_attachment],
                   [_events(answer="Final: c1", n_tools=1, result="x" * 500)])
    res = next(s for s in turn.trace if s["kind"] == "tool_result")
    assert res["chars"] == 500 and "truncated 450 chars" in res["text"]


def test_run_artifacts_are_written(repo, csv_attachment):
    """The raw stream, the command and the agent's files outlive the cell."""
    import gzip

    adapter = CliAgentAdapter(_spec())

    async def fake(argv, cwd):
        (cwd / "fit.py").write_text("from sklearn import ensemble\n", encoding="utf-8")
        return 0, _events(answer="Final: c1"), "some warning"

    adapter._run_process = fake  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    asyncio.run(adapter.run_turn([Message(role="user", content="P")], [csv_attachment]))

    run_dir = repo / "harness_runs" / "pi_test-model" / "t_task__seed0"
    with gzip.open(run_dir / "events.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert '"tool_execution_start"' in fh.read()
    assert "docker" not in (run_dir / "cmd.txt").read_text()  # the test template is echo
    assert (run_dir / "agent_files" / "fit.py").read_text().startswith("from sklearn")
    assert adapter.last_provenance["run_dir"] == str(run_dir)
    # Raw stdout must not leak into provenance — it is large and already in a file.
    assert all("_stdout" not in a for a in adapter.last_provenance["attempts"])


def test_keep_events_false_writes_nothing(repo, csv_attachment):
    adapter = CliAgentAdapter(_spec(keep_events=False))
    _run(adapter, "P", [csv_attachment], [_events(answer="Final: c1")])
    assert not (repo / "harness_runs").exists()
    assert adapter.last_provenance["run_dir"] is None


def test_timeout_is_a_retryable_transport_fault(repo, csv_attachment):
    """A wall-clock timeout is a failed attempt: retry, not a cached zero."""
    adapter = CliAgentAdapter(_spec(retries=2, timeout=0.01))
    calls = []

    async def fake(argv, cwd):
        calls.append(argv)
        if len(calls) == 1:
            raise TimeoutError
        return 0, _events(answer="Final: c7"), ""

    adapter._run_process = fake  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    turn = asyncio.run(adapter.run_turn([Message(role="user", content="P")], [csv_attachment]))
    assert turn.answer_text.strip() == "Final: c7"
    stops = [a["final_stop_reason"] for a in adapter.last_provenance["attempts"]]
    assert stops == ["timeout", "stop"]


def test_persistent_timeout_skips_the_cell(repo, csv_attachment):
    adapter = CliAgentAdapter(_spec(retries=1, timeout=0.01))

    async def fake(argv, cwd):
        raise TimeoutError

    adapter._run_process = fake  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    with pytest.raises(GenerationSkipped):
        asyncio.run(adapter.run_turn([Message(role="user", content="P")], [csv_attachment]))


def test_watchdog_kills_the_leaked_container(repo, csv_attachment):
    """The watchdog must remove the container too, not only the `docker run` client.

    The container outlives its client: if not killed, it keeps calling the LLM
    (spending money on a discarded attempt) and holds CPU, pushing neighbouring cells
    into the watchdog. This checks the cleanup seam itself — that it is called with
    this cell's workspace.
    """
    import asyncio

    adapter = CliAgentAdapter(_spec(command=["sleep", "30"], timeout=0.2, retries=0))
    adapter.set_context("t_task", 0)
    killed: list[str] = []

    async def spy(ws):
        killed.append(str(ws))

    adapter._kill_container_for = spy  # type: ignore[method-assign]

    with pytest.raises(GenerationSkipped):
        asyncio.run(adapter.run_turn([Message(role="user", content="P")], [csv_attachment]))

    assert killed and killed[0].endswith("t_task__seed0")


def test_container_cleanup_is_a_noop_for_non_docker_harness(repo):
    """Cleanup is docker-specific: a non-docker harness must not raise."""
    import asyncio

    adapter = CliAgentAdapter(_spec(command=["echo", "{prompt}"]))
    asyncio.run(adapter._kill_container_for(repo))  # does not raise and does nothing


def test_exhausted_watchdog_can_score_zero_instead_of_skipping(repo, csv_attachment):
    """An exhausted watchdog scores 0 if the arm declares it; an exhausted transport does not.

    The time limit is declared part of the arm — a compute budget given to the
    system: no answer within it is a meaningful zero. A gateway 502 is an
    infrastructure failure and must stay missing, or it would be charged to the system.
    """
    adapter = CliAgentAdapter(_spec(retries=1, timeout_exhausted_scores_zero=True))
    calls = {"n": 0}

    async def only_timeouts(argv, ws):
        calls["n"] += 1
        raise TimeoutError

    adapter._run_process = only_timeouts  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    turn = asyncio.run(adapter.run_turn([Message(role="user", content="P")], [csv_attachment]))
    assert calls["n"] == 2  # both attempts used up
    assert turn.answer_text == ""  # empty answer -> the grader scores 0
    assert adapter.last_provenance["final_stop_reason"] == "timeout_exhausted"


def test_exhausted_transport_still_skips_even_with_zero_on(repo, csv_attachment):
    """The zero flag must not turn a gateway fault into a zero for the system."""
    adapter = CliAgentAdapter(_spec(retries=1, timeout_exhausted_scores_zero=True))
    with pytest.raises(GenerationSkipped):
        _run(adapter, "P", [csv_attachment],
             [_events(answer="", stop="error"), _events(answer="", stop="error")])


# --------------------------------------------------------------------------- #
# Second harness: opencode                                                    #
# --------------------------------------------------------------------------- #

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _oc_spec(**cfg) -> SystemSpec:
    base = {
        "harness": "opencode",
        "harness_version": "1.18.29",
        "parser": "opencode",
        "model": "x-ai/grok-4.6",
        "command": ["echo", "{workspace}", "{model}", "{thinking}", "{prompt}"],
        "retries": 1,
    }
    base.update(cfg)
    return SystemSpec(id="opencode/test-model", adapter="cli_agent", config=base)


def _oc_events(*, answer: str, n_tools: int = 2, final: bool = True,
               result: str = "stdout of echo", error: str | None = None) -> str:
    """A JSONL snippet in the `opencode run --format json` schema."""
    def step(reason, text=None, think=None, tool=None):
        out = [{"type": "step_start", "part": {"type": "step-start"}}]
        if think:
            out.append({"type": "reasoning", "part": {"type": "reasoning", "text": think}})
        if text:
            out.append({"type": "text", "part": {"type": "text", "text": text}})
        if tool:
            out.append({"type": "tool_use", "part": {
                "type": "tool", "tool": "bash", "callID": "c",
                "state": {"status": "completed", "input": {"command": tool, "timeout": 30000},
                          "output": result}}})
        out.append({"type": "step_finish", "part": {
            "type": "step-finish", "reason": reason,
            "tokens": {"total": 0, "input": 100, "output": 20, "reasoning": 5,
                       "cache": {"read": 7, "write": 0}},
            "cost": 0.01}})
        return out

    lines: list[dict] = []
    for i in range(n_tools):
        lines += step("tool-calls", text="working on it", think="step think", tool=f"echo {i}")
    if final:
        lines += step("stop", text=answer, think="...")
    if error:
        lines.append({"type": "error", "error": {"name": "APIError", "data": {
            "message": error, "isRetryable": True}}})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


_OC_STDERR_OK = (
    'timestamp=t level=INFO message=stream providerID=openrouter modelID=x-ai/grok-4.6 '
    'small=false agent=build mode=primary\n'
)
_OC_STDERR_FAIL = _OC_STDERR_OK + (
    'timestamp=t level=ERROR message="stream error" providerID=openrouter '
    'modelID=x-ai/grok-4.6 error.error="AI_APICallError: Cannot connect"\n'
) * 3


def _run_oc(adapter, prompt, attachments, outputs, stderr=_OC_STDERR_OK):
    calls = []

    async def fake(argv, cwd):
        calls.append((argv, cwd))
        return 0, outputs[min(len(calls) - 1, len(outputs) - 1)], stderr

    adapter._run_process = fake  # type: ignore[method-assign]
    adapter.set_context("t_task", 0)
    import asyncio

    turn = asyncio.run(adapter.run_turn([Message(role="user", content=prompt)], attachments))
    return turn, calls


def test_opencode_usage_trace_and_provenance(repo, csv_attachment):
    adapter = CliAgentAdapter(_oc_spec(thinking="high"))
    turn, calls = _run_oc(adapter, "P", [csv_attachment],
                          [_oc_events(answer="Final: c1, c2", n_tools=3)])

    assert turn.answer_text.strip() == "Final: c1, c2"
    assert [t.name for t in turn.trajectory] == ["bash", "bash", "bash"]
    assert turn.usage.prompt_tokens == 4 * 100
    assert turn.usage.completion_tokens == 4 * 20
    assert turn.usage.cost_usd == pytest.approx(4 * 0.01)  # from step_finish.cost, not prices
    # {thinking} is substituted into argv as the wrapper's --variant.
    assert calls[0][0][3] == "high"

    prov = adapter.last_provenance
    assert prov["harness"] == "opencode" and prov["harness_version"] == "1.18.29"
    assert prov["models_seen"] == ["x-ai/grok-4.6"]  # from stderr — stdout has no model
    assert prov["final_stop_reason"] == "stop"
    assert prov["n_tool_calls"] == 3 and prov["reasoning_tokens"] == 4 * 5
    assert prov["cache_read_tokens"] == 4 * 7

    kinds = [s["kind"] for s in turn.trace]
    assert kinds.count("tool_call") == 3 and kinds.count("tool_result") == 3
    assert kinds.count("assistant") == 4  # 3 intermediate + final
    call = next(s for s in turn.trace if s["kind"] == "tool_call")
    assert call["args"]["command"] == "echo 0"
    res = next(s for s in turn.trace if s["kind"] == "tool_result")
    assert res["text"] == "stdout of echo" and res["is_error"] is False
    assert any(s.get("thinking") for s in turn.trace)
    assert "step think" not in turn.answer_text


def test_opencode_error_event_is_a_transport_fault(repo, csv_attachment):
    """`{"type":"error"}` after opencode's own retries is a fault: adapter retry, not a zero."""
    adapter = CliAgentAdapter(_oc_spec(retries=2))
    outputs = [
        _oc_events(answer="preamble", n_tools=1, final=False, error="Cannot connect to API"),
        _oc_events(answer="Final: c9"),
    ]
    turn, calls = _run_oc(adapter, "P", [csv_attachment], outputs)
    assert turn.answer_text.strip() == "Final: c9"
    assert len(calls) == 2
    stops = [a["final_stop_reason"] for a in adapter.last_provenance["attempts"]]
    assert stops == ["error", "stop"]
    assert "APIError: Cannot connect" in adapter.last_provenance["attempts"][0]["errors"][0]


def test_opencode_stream_errors_without_stop_are_a_fault(repo, csv_attachment):
    """Loop cut on tool-calls with stderr full of 'stream error' — a fault, not an answer.

    On a transport failure opencode's stdout may hold NOTHING but a preamble (no
    error event yet — the watchdog killed it or the process died earlier).
    """
    adapter = CliAgentAdapter(_oc_spec(retries=1))
    partial = _oc_events(answer="", n_tools=2, final=False)
    with pytest.raises(GenerationSkipped) as exc:
        _run_oc(adapter, "P", [csv_attachment], [partial], stderr=_OC_STDERR_FAIL)
    assert "transport fault" in str(exc.value)


def test_opencode_cut_on_tool_calls_without_stream_errors_is_an_answer(repo, csv_attachment):
    """A stop on tool-calls WITHOUT transport failures is agent behaviour: cached."""
    adapter = CliAgentAdapter(_oc_spec())
    turn, _ = _run_oc(adapter, "P", [csv_attachment],
                      [_oc_events(answer="", n_tools=2, final=False)])
    assert turn.answer_text.strip() == "working on it"
    assert adapter.last_provenance["final_stop_reason"] == "tool-calls"


def test_opencode_full_stderr_is_kept_in_run_dir(repo, csv_attachment):
    """For opencode stderr is the only fault log: it is saved in full, not just the tail."""
    adapter = CliAgentAdapter(_oc_spec())
    long_err = "x" * 2000 + "\n" + _OC_STDERR_OK
    _run_oc(adapter, "P", [csv_attachment], [_oc_events(answer="Final: c1")], stderr=long_err)
    run_dir = repo / "harness_runs" / "opencode_test-model" / "t_task__seed0"
    assert (run_dir / "stderr.txt").read_text() == long_err
    assert all("_stderr" not in a for a in adapter.last_provenance["attempts"])


def test_opencode_lock_is_identical_to_pi():
    """The scientific stack belongs to the AXIS, not an arm: both images' lock files are byte-identical."""
    pi = (REPO_ROOT / "harness" / "pi" / "requirements.lock").read_bytes()
    oc = (REPO_ROOT / "harness" / "opencode" / "requirements.lock").read_bytes()
    assert pi == oc


def test_opencode_title_agent_errors_do_not_trigger_the_fault_rule(repo, csv_attachment):
    """A title-agent failure in stderr is not a main-loop fault: the answer is cached."""
    adapter = CliAgentAdapter(_oc_spec(retries=0))
    title_err = _OC_STDERR_OK + (
        'timestamp=t level=ERROR message="stream error" providerID=openrouter '
        'modelID=x-ai/grok-4.6 small=true agent=title mode=primary error.error="AI_RetryError"\n'
    )
    turn, _ = _run_oc(adapter, "P", [csv_attachment],
                      [_oc_events(answer="", n_tools=1, final=False)], stderr=title_err)
    assert turn.answer_text.strip() == "working on it"
    assert adapter.last_provenance["final_stop_reason"] == "tool-calls"


def test_opencode_subagent_calls_flag_cost_as_incomplete(repo, csv_attachment):
    """Subagents run in child sessions with no usage in the stream: cost is a lower bound."""
    from hallmark.adapters.cli_agent import _parse_opencode_events

    ev = _oc_events(answer="Final: c1", n_tools=1)
    ev += json.dumps({"type": "tool_use", "part": {
        "type": "tool", "tool": "task", "callID": "t",
        "state": {"status": "completed", "input": {"description": "explore", "prompt": "…"},
                  "output": "child summary"}}}) + "\n"
    _, tools, _, meta = _parse_opencode_events(ev)
    assert [t.name for t in tools] == ["bash", "task"]
    assert meta["n_task_calls"] == 1 and meta["cost_incomplete"] is True

    _, _, _, meta0 = _parse_opencode_events(_oc_events(answer="Final: c1"))
    assert meta0["n_task_calls"] == 0 and meta0["cost_incomplete"] is False


def test_ledger_price_cost_is_recorded_next_to_harness_cost(repo, csv_attachment):
    """Cost at the ledger price snapshot is ALWAYS in provenance, next to the harness's own.

    Harness price catalogs drift (opencode: gpt-5.6-sol at 2/10 per 1M vs 5/30 in the
    ledger) — an arm-vs-bare cost comparison is only fair on a single snapshot.
    """
    adapter = CliAgentAdapter(_oc_spec(price_per_1k_in=5.0, price_per_1k_out=30.0))
    turn, _ = _run_oc(adapter, "P", [csv_attachment], [_oc_events(answer="Final: c1", n_tools=1)])
    prov = adapter.last_provenance
    assert turn.usage.cost_usd == pytest.approx(0.02)  # self-reported: 2 × 0.01
    assert prov["harness_cost_usd"] == pytest.approx(0.02)
    # 200 input × 5/1k + 40 output × 30/1k
    assert prov["cost_at_ledger_prices_usd"] == pytest.approx(200 * 5 / 1000 + 40 * 30 / 1000)
