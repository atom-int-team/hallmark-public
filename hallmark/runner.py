"""Two-phase runner (see UNIVERSAL_BENCHMARK_SPEC §7.4).

Phase 1 ``generate`` — drive each SUT over each task×seed, cache the transcripts.
Phase 2 ``grade``    — score the cached transcripts with the graders.

Generation (expensive, networked) is separated from grading (cheap, repeatable):
transcripts are cached, so re-grading is free.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from .adapters import ADAPTERS
from .adapters.base import GenerationSkipped
from .config import settings
from .graders import GRADERS
from .loader import GroundTruth
from .models import (
    Attachment,
    Capabilities,
    GraderResult,
    Message,
    RunResult,
    Signal,
    SystemSpec,
    Task,
    Transcript,
    TurnResult,
    TurnTranscript,
    Usage,
)
from .textutils import slug as _slug


def _seeds(n: int, seed: int | None) -> list[int]:
    """Seed indices to run: just ``[seed]`` when one is pinned, else ``range(n)``."""
    return [seed] if seed is not None else list(range(n))


def _transcript_path(system_id: str, task_id: str, seed: int) -> Path:
    return settings.transcripts_dir / _slug(system_id) / f"{task_id}__seed{seed}.json"


def _cached(system_id: str, task_id: str, seed: int) -> bool:
    return _transcript_path(system_id, task_id, seed).exists()


def _save_transcript(tr: Transcript) -> Path:
    path = _transcript_path(tr.system_id, tr.task_id, tr.seed_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tr.model_dump_json(indent=2), encoding="utf-8")
    return path


def _load_transcript(system_id: str, task_id: str, seed: int) -> Transcript | None:
    path = _transcript_path(system_id, task_id, seed)
    if not path.exists():
        return None
    return Transcript.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Capability gating (§7.3)                                                     #
# --------------------------------------------------------------------------- #


def _caps_satisfy(caps: Capabilities, required: list[str]) -> bool:
    """Whether a SUT with ``caps`` can participate in a task needing ``required``.

    "file_io" is satisfied by file_upload or code_exec; other names map directly
    to a Capabilities flag. Text-only delivery (inline) is handled per-adapter.
    """
    for req in required:
        if req == "file_io":
            if not (caps.file_upload or caps.code_exec):
                return False
        elif not bool(getattr(caps, req, False)):
            return False
    return True


def _needs_unproducible_deliverable(caps: Capabilities, task: Task) -> bool:
    """True if the task requires a **deliverable** the SUT provably cannot produce.

    The deliverable rule (spec §7.3): a ``file_output`` grader needs the SUT to
    emit a generated file (e.g. a trained model's ``predictions.csv``). A SUT with
    no ``file_output`` capability cannot do it — that is a task *failure* (score 0,
    in the denominator), NOT ``N/A``. This is deliberately narrower than "any
    missing signal": ``tool_trajectory`` is *observability*, not a deliverable, so
    it stays ``N/A`` (a system may have done the work without emitting a trace).
    """
    if caps.file_output:
        return False
    for turn in task.turns:
        for item in turn.grading:
            grader = GRADERS.get(item.grader)
            if grader is not None and grader.signal == Signal.FILE_OUTPUT:
                return True
    return False


def _build_adapters(system: SystemSpec) -> list:
    """Instantiate every adapter declared by ``system`` (one or several).

    Each adapter gets its own ``SystemSpec`` view carrying that adapter's merged
    config, so existing single-config adapters need no change.
    """
    adapters = []
    for name, cfg in system.adapter_components():
        if name not in ADAPTERS:
            raise KeyError(f"unknown adapter {name!r} for system {system.id}")
        sub_spec = system.model_copy(update={"adapter": name, "adapters": None, "config": cfg})
        adapters.append(ADAPTERS[name](sub_spec))
    return adapters


def _select_adapter(adapters: list, task: Task):
    """Pick the single adapter that serves ``task`` for this system.

    A task requiring the ``manual`` capability is routed to the system's manual
    adapter (answers ingested from ``submissions/``); any other task goes to the
    first non-manual adapter. Returns None if no suitable adapter is declared.
    """
    if "manual" in task.required_capabilities:
        return next((a for a in adapters if a.capabilities.manual), None)
    primary = next((a for a in adapters if not a.capabilities.manual), None)
    return primary or (adapters[0] if adapters else None)


def _load_attachments(task: Task) -> list[Attachment]:
    """Build attachments for a task. ``text`` is the inline-delivery payload (a
    sample/schema for CSVs); ``path`` is the real file for upload-capable SUTs."""
    attachments: list[Attachment] = []
    for name in task.inputs.files:
        path = settings.fixtures_dir / name
        exists = path.exists()
        if exists and name.endswith(".csv"):
            from .delivery import full_or_sample

            text = full_or_sample(path)  # whole file when it fits; labelled sample otherwise
            mime = "text/csv"
        elif exists:
            text = path.read_text(encoding="utf-8")
            mime = "text/plain"
        else:
            text, mime = None, "application/octet-stream"
        attachments.append(
            Attachment(name=name, mime=mime, text=text, path=str(path) if exists else None)
        )
    return attachments


# --------------------------------------------------------------------------- #
# Phase 1: generate                                                           #
# --------------------------------------------------------------------------- #


async def _run_task(adapter, task: Task, seed: int) -> Transcript:
    attachments = _load_attachments(task)
    adapter.set_context(task.id, seed)

    history: list[Message] = []
    turns: list[TurnTranscript] = []
    total = Usage()
    for turn_idx, turn in enumerate(task.turns):
        history.append(Message(role="user", content=turn.prompt))
        # Attachments are delivered once, on the first turn.
        turn_attachments = attachments if turn_idx == 0 else []
        tt = await adapter.run_turn(history, turn_attachments)
        history.append(Message(role="assistant", content=tt.answer_text))
        turns.append(tt)
        total = total + tt.usage

    return Transcript(
        task_id=task.id,
        system_id=adapter.system_id,
        seed_idx=seed,
        turns=turns,
        usage=total,
        adapter_capabilities=adapter.capabilities,
        # An adapter may attach what produced the cell (harness version, argv, prompt
        # hash). One that does not leaves an empty dict.
        provenance=getattr(adapter, "last_provenance", {}) or {},
    )


def _noop(_: str) -> None:
    pass


async def generate(
    tasks: list[Task],
    systems: list[SystemSpec],
    runs: int | None = None,
    *,
    seed: int | None = None,
    use_cache: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Phase 1: collect and cache transcripts (system × task × seed).

    ``progress`` (if given) is called with a human-readable line per work item:
    capability skips, cache hits, and live "generating…" steps.

    ``seed`` (if given) restricts work to that single seed index instead of
    iterating ``range(n_seeds)``.
    """
    emit = progress or _noop
    default_runs = runs or settings.default_runs

    # Plan first so progress can show "[i/total]" and report gated tasks up front.
    plan: list[tuple] = []  # (adapter, task, seeds)
    for system in systems:
        adapters = _build_adapters(system)
        for task in tasks:
            adapter = _select_adapter(adapters, task)
            if adapter is None or not _caps_satisfy(adapter.capabilities, task.required_capabilities):
                emit(f"skip      {system.id} · {task.id} — needs {task.required_capabilities}")
                continue
            plan.append((adapter, task, _seeds(task.runs or default_runs, seed)))

    total = sum(len(seeds) for _, _, seeds in plan)
    written: list[Path] = []
    i = 0
    for adapter, task, seeds in plan:
        for seed in seeds:
            i += 1
            tag = f"[{i:>3}/{total}] {adapter.system_id} · {task.id} seed{seed}"
            if use_cache and _cached(adapter.system_id, task.id, seed):
                emit(f"{tag} — cached, skip")
                continue
            # Deliverable the SUT provably cannot produce (file_output, no cap):
            # write an empty stub transcript WITHOUT calling the API — grading
            # scores it a flat 0 (task failure, see _grade_transcript / §7.3).
            if _needs_unproducible_deliverable(adapter.capabilities, task):
                stub = Transcript(
                    task_id=task.id,
                    system_id=adapter.system_id,
                    seed_idx=seed,
                    turns=[],
                    usage=Usage(),
                    adapter_capabilities=adapter.capabilities,
                )
                written.append(_save_transcript(stub))
                emit(f"{tag} — deliverable-zero (unproducible file_output, no API call)")
                continue
            emit(f"{tag} — generating…")
            try:
                tr = await _run_task(adapter, task, seed)
            except GenerationSkipped as e:
                emit(f"{tag} — skipped ({e})")
                continue
            except Exception as e:  # one bad API call must not abort the whole batch
                emit(f"{tag} — ERROR ({type(e).__name__}: {e}) — skip")
                continue
            written.append(_save_transcript(tr))
            emit(f"{tag} — done ({len(tr.turns)} turn(s))")
    emit(f"generate: {len(written)} new transcript(s), {total - len(written)} cached/skipped")
    return written


# --------------------------------------------------------------------------- #
# Phase 2: grade                                                              #
# --------------------------------------------------------------------------- #


async def _grade_transcript(
    tr: Transcript, task: Task, gt: GroundTruth, extra_params: dict | None = None
) -> RunResult:
    caps = tr.adapter_capabilities

    # Deliverable rule (§7.3): the task needs a file_output artifact this SUT
    # cannot produce → task FAILURE, flat 0, no grader is run (not even the
    # text/rubric ones, and no judge API call). Kept in the denominator.
    if _needs_unproducible_deliverable(caps, task):
        turn_results = [
            TurnResult(
                turn_idx=turn_idx,
                grader_results=[
                    GraderResult(
                        grader=item.grader,
                        signal=item.signal,
                        applicable=True,
                        score=0.0,
                        weight=item.weight,
                        detail="0 (deliverable not produced: SUT lacks file_output)",
                    )
                    for item in turn_spec.grading
                ],
                turn_score=0.0,
            )
            for turn_idx, turn_spec in enumerate(task.turns)
        ]
        return RunResult(
            task_id=task.id,
            domain=task.domain,
            track=task.track,
            difficulty=task.difficulty,
            system_id=tr.system_id,
            seed_idx=tr.seed_idx,
            turns=turn_results,
            task_score=0.0,
            usage=tr.usage,
        )

    turn_results: list[TurnResult] = []

    for turn_idx, turn_spec in enumerate(task.turns):
        if turn_idx >= len(tr.turns):
            break
        turn_tr = tr.turns[turn_idx]
        grader_results: list[GraderResult] = []
        for item in turn_spec.grading:
            grader = GRADERS.get(item.grader)
            if grader is None:
                raise KeyError(f"unknown grader {item.grader!r} (task {task.id})")
            # Per-run overrides (e.g. {"judge_board": True}) merged over task params;
            # graders that don't know a key ignore it.
            params = {**item.params, **extra_params} if extra_params else item.params
            applicable = grader.applicable(turn_tr, caps)
            if applicable:
                try:
                    # Watchdog: a hung grader (e.g. a stalled judge provider) is
                    # cancelled and recorded as an error, not allowed to freeze the run.
                    score, detail = await asyncio.wait_for(
                        grader.grade(turn_tr, params, gt), timeout=settings.grader_timeout
                    )
                except TimeoutError:  # asyncio.wait_for cancellation (py3.11+ alias)
                    score, detail = None, f"ERROR: timed out after {settings.grader_timeout:.0f}s"
                except Exception as e:  # one grader's failure must not kill the run
                    score, detail = None, f"ERROR: {type(e).__name__}: {e}"
            else:
                score, detail = None, f"N/A (needs signal {grader.signal.value})"
            grader_results.append(
                GraderResult(
                    grader=item.grader,
                    signal=item.signal,
                    applicable=applicable,
                    score=score,
                    weight=item.weight,
                    detail=detail,
                )
            )
        turn_results.append(
            TurnResult(
                turn_idx=turn_idx,
                grader_results=grader_results,
                turn_score=_weighted_turn_score(grader_results),
            )
        )

    return RunResult(
        task_id=task.id,
        domain=task.domain,
        track=task.track,
        difficulty=task.difficulty,
        system_id=tr.system_id,
        seed_idx=tr.seed_idx,
        turns=turn_results,
        task_score=_mean_turn_score(turn_results),
        usage=tr.usage,
    )


def _weighted_turn_score(results: list[GraderResult]) -> float | None:
    """Weighted mean of score over applicable graders; None if all N/A."""
    num = den = 0.0
    for r in results:
        if r.applicable and r.score is not None:
            num += r.weight * r.score
            den += r.weight
    return (num / den) if den > 0 else None


def _mean_turn_score(turns: list[TurnResult]) -> float | None:
    scores = [t.turn_score for t in turns if t.turn_score is not None]
    return sum(scores) / len(scores) if scores else None


async def grade(
    tasks: list[Task],
    systems: list[SystemSpec],
    runs: int | None = None,
    *,
    seed: int | None = None,
    existing: list[RunResult] | None = None,
    progress: Callable[[str], None] | None = None,
    extra_params: dict | None = None,
    checkpoint: Callable[[list[RunResult]], None] | None = None,
) -> list[RunResult]:
    """Phase 2: score cached transcripts.

    ``seed`` (if given) restricts grading to that single seed index.

    ``existing`` (prior results, e.g. from ``results/<run>.json``) makes grading
    *incremental*: a ``(system, task, seed)`` cell already present is reused as-is
    instead of being re-graded, so only newly added systems/tasks are scored.
    Prior results that fall *outside* the current selection are passed through
    untouched, so the run file accumulates. Pass ``existing=None`` (the default)
    to force a full re-grade of everything in the selection.

    ``checkpoint`` (if given) is called after every newly-scored cell with the
    full result set so far (reused + freshly scored + out-of-selection kept). The
    caller persists it, so an interrupted run leaves a valid partial file and a
    re-run resumes from where it stopped (the saved cells become ``existing``).
    """
    emit = progress or _noop
    default_runs = runs or settings.default_runs
    gt = GroundTruth()
    tasks_by_id = {t.id: t for t in tasks}
    prior = {(r.system_id, r.task_id, r.seed_idx): r for r in (existing or [])}

    # Compute the selection up front so `kept` (prior results outside it) is known
    # from the start and can be included in every checkpoint.
    seeds_by_task = {t.id: list(_seeds(t.runs or default_runs, seed)) for t in tasks}
    selection = {
        (system.id, task.id, s)
        for system in systems
        for task in tasks
        for s in seeds_by_task[task.id]
    }
    kept = [r for key, r in prior.items() if key not in selection]

    results: list[RunResult] = []
    missing = reused = scored = 0
    for system in systems:
        for task in tasks:
            for s in seeds_by_task[task.id]:
                key = (system.id, task.id, s)
                if key in prior:
                    results.append(prior[key])
                    reused += 1
                    emit(f"reused    {system.id} · {task.id} seed{s} — already graded, skip")
                    continue
                tr = _load_transcript(system.id, task.id, s)
                if tr is None:
                    missing += 1
                    emit(f"missing   {system.id} · {task.id} seed{s} — no transcript, skip")
                    continue
                rr = await _grade_transcript(tr, tasks_by_id[task.id], gt, extra_params)
                score = "N/A" if rr.task_score is None else f"{rr.task_score:.3f}"
                emit(f"graded    {system.id} · {task.id} seed{s} — {score}")
                results.append(rr)
                scored += 1
                if checkpoint is not None:
                    checkpoint(results + kept)  # persist progress for resume
    results.extend(kept)
    emit(
        f"grade: {scored} newly scored, {reused} reused, {len(kept)} kept "
        f"(outside selection), {missing} missing transcript(s)"
    )
    return results
