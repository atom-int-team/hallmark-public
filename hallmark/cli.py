"""hallmark command-line interface.

    hallmark generate    --systems systems.yaml [--tasks tasks] [--no-cache|--force]
    hallmark grade       --systems systems.yaml [--tasks tasks] --run <name>
    hallmark leaderboard --run <name> [--out leaderboard.md]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import runner, scaffold, scoring
from .config import settings
from .loader import load_systems, load_tasks
from .models import RunResult
from .reporters import leaderboard as lb


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--systems", required=True, help="path to a systems YAML file")
    p.add_argument("--tasks", default=None, help="tasks dir or file (default: <repo>/tasks)")
    p.add_argument("--split", default=None, choices=["dev", "test"])
    p.add_argument("--domain", default=None)
    p.add_argument("--task-id", action="append", dest="task_ids", help="filter by id (repeatable)")
    p.add_argument("--runs", type=int, default=None, help="seeds per task (default from config)")
    p.add_argument("--seed", type=int, default=None,
                   help="evaluate only this single seed index (e.g. --seed 1)")
    p.add_argument("--quiet", action="store_true", help="suppress per-item progress on stderr")


def _results_path(run_name: str) -> Path:
    return settings.results_dir / f"{run_name}.json"


def _printer(quiet: bool):
    """Progress callback writing to stderr (None when quiet, keeping stdout clean)."""
    if quiet:
        return None

    def emit(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    return emit


def _load_selection(args):
    systems = load_systems(args.systems)
    tasks = load_tasks(
        args.tasks, split=args.split, domain=args.domain, task_ids=args.task_ids
    )
    return systems, tasks


def cmd_generate(args) -> int:
    systems, tasks = _load_selection(args)
    written = asyncio.run(
        runner.generate(
            tasks, systems, runs=args.runs, seed=args.seed,
            use_cache=not (args.no_cache or args.force),
            progress=_printer(args.quiet),
        )
    )
    print(f"generated {len(written)} transcript(s) for {len(systems)} system(s) "
          f"over {len(tasks)} task(s)")
    return 0


def cmd_grade(args) -> int:
    systems, tasks = _load_selection(args)
    path = _results_path(args.run)

    extra_params: dict | None = None

    # Incremental grading + crash resume: reuse prior results unless --force-grade
    # re-grades all.
    existing: list[RunResult] = []
    prior_tasks: list[dict] = []
    if path.exists() and not args.force_grade:
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            existing = [RunResult.model_validate(r) for r in prior.get("results", [])]
            prior_tasks = prior.get("tasks", [])
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"could not reuse {path} ({exc}); grading from scratch", file=sys.stderr)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Merge task metadata (prior + current selection, current wins) so the file
    # accumulates alongside accumulated results. Built before grading so the
    # incremental checkpoint can write a complete payload.
    tasks_meta = {t["id"]: t for t in prior_tasks}
    for t in tasks:
        tasks_meta[t.id] = {
            "id": t.id, "domain": t.domain.value, "track": t.track.value,
            "time_gate": t.time_gate, "split": t.split.value,
        }

    def _write(run_results: list[RunResult]) -> None:
        # Atomic: write a temp file then replace, so an interrupt mid-write can't
        # corrupt results/<run>.json and break resume.
        payload = {
            "run_name": args.run,
            "judge_board": False,  # no LLM-judge grader ships in this release
            "results": [r.model_dump(mode="json") for r in run_results],
            "tasks": list(tasks_meta.values()),
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    results: list[RunResult] = asyncio.run(
        runner.grade(tasks, systems, runs=args.runs, seed=args.seed,
                     existing=existing, progress=_printer(args.quiet),
                     extra_params=extra_params, checkpoint=_write)
    )
    _write(results)  # final write (also covers the no-new-cells / all-reused case)
    scored = sum(1 for r in results if r.task_score is not None)
    print(f"graded {len(results)} run-result(s) ({scored} scored) -> {path}")

    return 0


def cmd_scaffold(args) -> int:
    tasks = load_tasks(
        args.tasks, split=args.split, domain=args.domain, task_ids=args.task_ids
    )
    written = scaffold.scaffold_submissions(
        tasks,
        system_id=args.system,
        runs=args.runs,
        seed=args.seed,
        submissions_dir=args.out,
        force=args.force,
        progress=_printer(args.quiet),
    )
    n_tasks = len(scaffold.manual_tasks(tasks))
    print(f"scaffolded {len(written)} submission file(s) for '{args.system}' "
          f"over {n_tasks} manual task(s)")
    return 0


def _olympus_diagnostics(tasks: list[dict]) -> list[dict] | None:
    """Per-task judge fidelity (R²) + trivial-boundary baseline for the Olympus design tasks,
    read from their committed gold — surfaced as a leaderboard appendix (plan §15.4)."""
    diags = []
    for t in tasks:
        tid = t.get("id", "")
        if not tid.startswith("e2e_olympus"):
            continue
        gold_path = settings.ground_truth_dir / tid / "olympus.json"
        if not gold_path.exists():
            continue
        g = json.loads(gold_path.read_text(encoding="utf-8"))
        diags.append({"task": tid, "r2": g.get("emulator_r2"), "f_star": g.get("f_star"),
                      # 0-score anchor: `train_median` post-rename, `train_best` in older golds
                      "floor": g.get("train_median", g.get("train_best")),
                      "naive": g.get("naive_baseline_score")})
    return diags or None


def cmd_leaderboard(args) -> int:
    path = _results_path(args.run)
    if not path.exists():
        print(f"results not found: {path}", file=sys.stderr)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = [RunResult.model_validate(r) for r in payload["results"]]
    summary = scoring.summarize(results)
    gated = [t for t in payload.get("tasks", []) if t.get("time_gate")]
    notes = [
        f"time-gated: `{t['id']}` (data after {t['time_gate']})" for t in gated
    ] or None
    diagnostics = _olympus_diagnostics(payload.get("tasks", []))
    render = lb.to_html if args.format == "html" else lb.to_markdown
    text = render(summary, run_name=payload.get("run_name"), notes=notes,
                  diagnostics=diagnostics)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hallmark", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="phase 1: run SUTs and cache transcripts")
    _add_selection_args(g)
    g.add_argument("--no-cache", action="store_true", help="ignore cached transcripts")
    g.add_argument("--force", "-f", action="store_true",
                   help="regenerate even if a cached transcript exists")
    g.set_defaults(func=cmd_generate)

    gr = sub.add_parser("grade", help="phase 2: score cached transcripts")
    _add_selection_args(gr)
    gr.add_argument("--run", required=True, help="run name (results/<name>.json)")
    gr.add_argument("--force-grade", action="store_true",
                    help="re-grade everything in the selection instead of reusing "
                         "already-graded cells from results/<name>.json")
    gr.set_defaults(func=cmd_grade)

    sc = sub.add_parser(
        "scaffold",
        help="generate empty submission skeletons (manual adapter) from task YAMLs",
    )
    sc.add_argument("--system", required=True,
                    help="system id to scaffold submissions for (e.g. chatgpt)")
    sc.add_argument("--tasks", default=None, help="tasks dir or file (default: <repo>/tasks)")
    sc.add_argument("--split", default=None, choices=["dev", "test"])
    sc.add_argument("--domain", default=None)
    sc.add_argument("--task-id", action="append", dest="task_ids",
                    help="filter by id (repeatable)")
    sc.add_argument("--runs", type=int, default=None, help="seeds per task (default from config)")
    sc.add_argument("--seed", type=int, default=None,
                    help="scaffold only this single seed index (e.g. --seed 1)")
    sc.add_argument("--out", default="submissions",
                    help="submissions root dir (default: submissions)")
    sc.add_argument("--force", "-f", action="store_true",
                    help="overwrite existing submission files")
    sc.add_argument("--quiet", action="store_true",
                    help="suppress per-file progress on stderr")
    sc.set_defaults(func=cmd_scaffold)

    lbp = sub.add_parser("leaderboard", help="render the leaderboard for a graded run")
    lbp.add_argument("--run", required=True)
    lbp.add_argument("--out", default=None, help="write output to this path")
    lbp.add_argument("--format", default="md", choices=["md", "html"])
    lbp.set_defaults(func=cmd_leaderboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
