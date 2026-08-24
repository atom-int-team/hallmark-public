"""Render a scoring summary as a leaderboard (markdown / HTML).

Two tables (Reasoning, Execution): rows = systems, columns = domains + overall
(mean with bootstrap 95% CI), plus cost_usd, latency_ms.
"""
from __future__ import annotations

import html

_TRACKS = ("reasoning", "execution")

_CAPTION = "Cells: mean [95% CI]. Scores are normalized regret / R² in [0, 1]; higher is better."


def _fmt_cell(cell: dict | None) -> str:
    if cell is None:
        return "—"
    if cell["n"] < 2 or (cell["lo"] == cell["hi"]):
        return f"{cell['mean']:.3f}"
    return f"{cell['mean']:.3f} [{cell['lo']:.2f}–{cell['hi']:.2f}]"


def _fmt_float(v: float | None, fmt: str) -> str:
    return format(v, fmt) if v is not None else "—"


def _columns(domains: list[str]) -> list[str]:
    return ["system", *domains, "overall", "cost_usd", "latency_ms"]


def _row_values(sysid: str, row: dict, domains: list[str]) -> list[str]:
    cells = [sysid]
    cells += [_fmt_cell(row["domains"].get(d)) for d in domains]
    cells.append(_fmt_cell(row["overall"]))
    cells.append(_fmt_float(row["cost_usd"], ".4f"))
    cells.append(_fmt_float(row["latency_ms"], ".0f"))
    return cells


def _sorted_rows(block: dict):
    return sorted(
        block["systems"].items(),
        key=lambda kv: (kv[1]["overall"] is None, -(kv[1]["overall"] or {}).get("mean", 0.0)),
    )


_DIAG_COLS = ["task", "judge R²", "f*", "floor", "naive boundary"]
_DIAG_CAPTION = ("judge R² = fraction of real-data variance the surrogate explains (fidelity — "
                 "score gaps below it are noise); naive boundary = best score the trivial "
                 "'push every knob to a domain edge' strategy reaches (a system at/below it "
                 "learned nothing the task tests).")


def _diag_values(d: dict) -> list[str]:
    return [d["task"], _fmt_float(d.get("r2"), ".2f"), _fmt_float(d.get("f_star"), ".4g"),
            _fmt_float(d.get("floor"), ".4g"), _fmt_float(d.get("naive"), ".3f")]


# --------------------------------------------------------------------------- #
# Markdown                                                                     #
# --------------------------------------------------------------------------- #


def _md_table(track: str, block: dict, domains: list[str]) -> str:
    cols = _columns(domains)
    lines = [
        f"### {track.capitalize()} track",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for sysid, row in _sorted_rows(block):
        lines.append("| " + " | ".join(_row_values(sysid, row, domains)) + " |")
    return "\n".join(lines)


def _md_diagnostics(diagnostics: list[dict]) -> str:
    lines = ["### Olympus judge diagnostics", "",
             "| " + " | ".join(_DIAG_COLS) + " |",
             "|" + "|".join(["---"] * len(_DIAG_COLS)) + "|"]
    for d in diagnostics:
        lines.append("| " + " | ".join(_diag_values(d)) + " |")
    lines.append(f"\n_{_DIAG_CAPTION}_")
    return "\n".join(lines)


def to_markdown(summary: dict, *, run_name: str | None = None, notes: list[str] | None = None,
                diagnostics: list[dict] | None = None) -> str:
    out = ["# Leaderboard"]
    if run_name:
        out.append(f"\n_run: `{run_name}`_")
    if not summary["tracks"]:
        out.append("\n_(no scored results)_")
        return "\n".join(out)
    for track in _TRACKS:
        block = summary["tracks"].get(track)
        if block:
            out.append("\n" + _md_table(track, block, summary["domains"]))
    if diagnostics:
        out.append("\n" + _md_diagnostics(diagnostics))
    if notes:
        out.append("\n## Anti-contamination notes\n")
        out += [f"- {n}" for n in notes]
    out.append("\n_" + _CAPTION + "_")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# HTML                                                                         #
# --------------------------------------------------------------------------- #


def _html_table(track: str, block: dict, domains: list[str]) -> str:
    cols = _columns(domains)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = []
    for sysid, row in _sorted_rows(block):
        tds = "".join(f"<td>{html.escape(v)}</td>" for v in _row_values(sysid, row, domains))
        rows.append(f"<tr>{tds}</tr>")
    return (
        f"<h3>{html.escape(track.capitalize())} track</h3>\n"
        f"<table border='1' cellpadding='4' cellspacing='0'>\n"
        f"<thead><tr>{head}</tr></thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody></table>"
    )


def _html_diagnostics(diagnostics: list[dict]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in _DIAG_COLS)
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in _diag_values(d)) + "</tr>"
        for d in diagnostics
    )
    return (
        "<h3>Olympus judge diagnostics</h3>\n"
        "<table border='1' cellpadding='4' cellspacing='0'>\n"
        f"<thead><tr>{head}</tr></thead>\n<tbody>\n{rows}\n</tbody></table>\n"
        f"<p><em>{html.escape(_DIAG_CAPTION)}</em></p>"
    )


def to_html(summary: dict, *, run_name: str | None = None, notes: list[str] | None = None,
            diagnostics: list[dict] | None = None) -> str:
    title = f"Leaderboard — {html.escape(run_name)}" if run_name else "Leaderboard"
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{title}</title>",
        f"<h1>{title}</h1>",
    ]
    if not summary["tracks"]:
        parts.append("<p><em>no scored results</em></p>")
    for track in _TRACKS:
        block = summary["tracks"].get(track)
        if block:
            parts.append(_html_table(track, block, summary["domains"]))
    if diagnostics:
        parts.append(_html_diagnostics(diagnostics))
    if notes:
        items = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
        parts.append(f"<h2>Anti-contamination notes</h2>\n<ul>{items}</ul>")
    parts.append(f"<p><em>{html.escape(_CAPTION)}</em></p>")
    return "\n".join(parts)
