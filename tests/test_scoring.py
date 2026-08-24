"""Scoring: bootstrap CI, per-track split."""
from __future__ import annotations

from hallmark import scoring
from hallmark.models import (
    Difficulty,
    Domain,
    RunResult,
    Track,
    TurnResult,
    Usage,
)


def _run(system, task, domain, track, score, seed=0):
    return RunResult(
        task_id=task, domain=domain, track=track, difficulty=Difficulty.EASY,
        system_id=system, seed_idx=seed, task_score=score,
        turns=[TurnResult(turn_idx=0, grader_results=[], turn_score=score)],
        usage=Usage(cost_usd=0.01, latency_ms=100),
    )


def test_bootstrap_ci_is_deterministic_and_bracketing():
    results = [
        _run("a", f"t{i}", Domain.GENERAL_QA, Track.REASONING, s, seed=i)
        for i, s in enumerate([0.2, 0.4, 0.6, 0.8, 1.0])
    ]
    s1 = scoring.summarize(results)
    s2 = scoring.summarize(results)
    cell = s1["tracks"]["reasoning"]["systems"]["a"]["overall"]
    assert cell["n"] == 5 and abs(cell["mean"] - 0.6) < 1e-9
    assert cell["lo"] <= cell["mean"] <= cell["hi"]
    assert s1 == s2  # deterministic CI (fixed bootstrap seed)


def test_tracks_are_split():
    results = [
        _run("a", "qa", Domain.GENERAL_QA, Track.REASONING, 1.0),
        _run("a", "e2e", Domain.END_TO_END, Track.EXECUTION, 0.8),
    ]
    s = scoring.summarize(results)
    assert "a" in s["tracks"]["reasoning"]["systems"]
    assert "a" in s["tracks"]["execution"]["systems"]
    assert s["tracks"]["execution"]["systems"]["a"]["overall"]["mean"] == 0.8
