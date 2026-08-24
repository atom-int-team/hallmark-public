"""scaffold — generate empty submission skeletons for the ``manual`` adapter.

The instruction header (TASK / UPLOAD / PROMPT / RECORD), the frontmatter, and the
per-turn ``## turn N`` stubs are **fully derivable from a task YAML**, so there is
no reason to hand-copy templates between systems. ``scaffold_submissions`` writes
one ``submissions/<slug>/seed<n>/<task_id>.md`` per (manual task × seed) for a
single system; existing files are left untouched unless ``force=True`` — so you
only ever paste in the answers, never the boilerplate.

Used by ``hallmark scaffold`` (see cli.py).
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import settings
from .models import GradingItem, Task
from .textutils import slug

# A line that begins a list item ("- ", "* ", "• ", "1. ", "2) ", …). Such a
# line must stay on its own logical line when we un-wrap a prompt.
_LIST_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s")


def reflow_prompt(text: str) -> str:
    """Undo the hard line-wrapping of a task prompt so it copy-pastes cleanly.

    Task YAML wraps prompts at ~75 columns for readability, which means pasting
    a prompt into a chat brings along mid-sentence line breaks (and, formerly,
    leading indentation) that have to be cleaned up by hand. This joins the
    soft-wrapped lines of each paragraph into one line while preserving the
    structure that is meaningful: blank lines (paragraph breaks) and list-item
    boundaries (``- ``, ``* ``, ``1. ``). The result is flush-left with no
    mid-sentence breaks, so the human can select it with the mouse and paste it
    verbatim.
    """
    paragraphs: list[str] = []
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        logical: list[str] = []
        for raw in block.splitlines():
            line = raw.strip()
            if not line:
                continue
            if logical and not _LIST_RE.match(line):
                logical[-1] = f"{logical[-1]} {line}"
            else:
                logical.append(line)
        paragraphs.append("\n".join(logical))
    return "\n\n".join(paragraphs)


def _grader_hint(g: GradingItem) -> str:
    """A short, human-facing note on what the grader parses from the answer text."""
    p = g.params or {}
    if g.grader == "numeric_match" and p.get("regex"):
        return f"answer must contain a value matching /{p['regex']}/"
    if g.grader == "normalized_match":
        return "answer must contain the expected term/phrase (normalized match)"
    if g.grader == "composition_valid":
        return 'include "<Element> <wt%>" pairs: real elements, physical bounds'
    if g.grader == "frontier_match":
        prop = p.get("property", "the target property")
        return f"predicted {prop} must land near the achievable frontier"
    if g.grader == "set_f1":
        return "list the expected set of items (e.g. DOIs / phases / descriptors)"
    if g.grader == "doi_real":
        return "every cited DOI must resolve (real, not hallucinated)"
    if g.grader in ("rubric_judge", "reference_judge"):
        return "judged against a rubric — be thorough, specific, well-justified"
    if g.grader == "trajectory":
        return "fill the frontmatter `tools_used:` list (diagnostic)"
    # Unknown grader: surface its params so the human can see what's checked.
    return f"params: {p}" if p else ""


def _render_record(task: Task) -> str:
    lines: list[str] = []
    multiturn = len(task.turns) > 1
    for ti, turn in enumerate(task.turns, start=1):
        if multiturn:
            lines.append(f"  turn {ti}:")
        prefix = "    " if multiturn else "  "
        if not turn.grading:
            lines.append(f"{prefix}(no graders declared)")
            continue
        for g in turn.grading:
            hint = _grader_hint(g)
            tail = f"  -> {hint}" if hint else ""
            lines.append(f"{prefix}- {g.grader} (weight {g.weight}){tail}")
    return "\n".join(lines)


def render_template(task: Task, *, system_id: str, seed: int, runs: int) -> str:
    """Render the full skeleton .md (frontmatter + instruction block + turn stubs)."""
    multiturn = len(task.turns) > 1

    upload = ""
    if task.inputs.files:
        files = "\n".join(f"      fixtures/{f}" for f in task.inputs.files)
        upload = (
            "\nUPLOAD  (attach these into the chat, they live in this repo)\n"
            f"{files}\n"
        )

    if multiturn:
        prompt_block = (
            "\nPROMPT  (select each turn's lines below and paste verbatim — "
            "one per `## turn`)\n"
        )
        for ti, turn in enumerate(task.turns, start=1):
            body = reflow_prompt(turn.prompt)
            prompt_block += f"\nturn {ti}:\n{body}\n"
    else:
        body = reflow_prompt(task.turns[0].prompt)
        prompt_block = f"\nPROMPT  (select the lines below and paste verbatim)\n{body}\n"

    header = (
        f"TASK  {task.id}   ({task.domain.value} · {task.track.value} · "
        f"{task.difficulty.value})\n"
        f"SEED  {seed}   — run #{seed + 1} of {runs}. Use a fresh chat per seed.\n"
        f"{upload}{prompt_block}\n"
        "RECORD  (the answer text must literally contain what each grader parses)\n"
        f"{_render_record(task)}\n\n"
        "DOWNLOAD  (optional, helps the file_output signal)\n"
        "      If the system generates files, save them INTO THIS FOLDER and list\n"
        "      their names in the frontmatter `artifacts: [...]`.\n"
    )

    turn_stubs = "\n\n".join(f"## turn {i + 1}\n" for i in range(len(task.turns)))

    return (
        "---\n"
        f'model: "{system_id}"            # provenance: what you ran and when\n'
        "latency_ms: 0\n"
        "cost_usd: null\n"
        "artifacts: []                    # downloaded files, e.g. [model_plot.png]\n"
        "---\n\n"
        f"<!--\n{header}-->\n\n"
        f"{turn_stubs}\n"
    )


def manual_tasks(tasks: list[Task]) -> list[Task]:
    """Tasks served by the manual adapter (human-captured into submissions/)."""
    return [t for t in tasks if "manual" in t.required_capabilities]


def scaffold_submissions(
    tasks: list[Task],
    *,
    system_id: str,
    runs: int | None = None,
    seed: int | None = None,
    submissions_dir: str | Path = "submissions",
    force: bool = False,
    progress=None,
) -> list[Path]:
    """Write skeleton submission files for one system; return the paths written.

    Only tasks requiring the ``manual`` capability are scaffolded. Each (task × seed)
    yields ``<submissions_dir>/<slug(system_id)>/seed<n>/<task_id>.md``. Existing
    files are skipped (reported via ``progress``) unless ``force`` is set.
    """
    base = Path(submissions_dir)
    if not base.is_absolute():
        base = settings.repo_root / base
    sys_dir = base / slug(system_id)

    def emit(msg: str) -> None:
        if progress:
            progress(msg)

    written: list[Path] = []
    for task in manual_tasks(tasks):
        n_runs = task.runs or runs or settings.default_runs
        seeds = [seed] if seed is not None else range(n_runs)
        for s in seeds:
            path = sys_dir / f"seed{s}" / f"{task.id}.md"
            if path.exists() and not force:
                emit(f"skip (exists)  {path.relative_to(base.parent)}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                render_template(task, system_id=system_id, seed=s, runs=n_runs),
                encoding="utf-8",
            )
            written.append(path)
            emit(f"wrote          {path.relative_to(base.parent)}")
    return written
