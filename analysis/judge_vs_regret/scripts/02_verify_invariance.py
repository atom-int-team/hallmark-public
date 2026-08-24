"""Step 2 — the content-invariance gate (imported by 01, runnable as a report).

A style rewrite may only change voice, length and structure. The gate is a cheap
pre-filter run in the generation loop; the *authoritative* check is step 03, which
re-runs the real grader and demands a delta of exactly 0.

Rules, and why each exists:

* ``Final:`` lines identical — ``parse_ids`` scans them last-match-first, so an extra
  one silently rewrites the selection.
* no invented numbers — a rewrite that adds a numeric claim is no longer style-only.
  Dropping numbers is allowed (that is what "terse" means), hence subset, not equality.
* no invented candidate ids — same argument, and it catches an id smuggled into prose.
* length ratio — proof the style actually changed; without it a no-op rewrite would
  pass every other rule and quietly kill the manipulation.

Run:  python analysis/judge_vs_regret/scripts/02_verify_invariance.py
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from _common import DATA, latest_variants  # noqa: E402

# same shape as hallmark/graders/parse_ids.py::_FINAL_RE — the contract line the grader reads
_FINAL_RE = re.compile(r"final\b[^\n:]*:?\s*(.+)$", re.I | re.M)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ID_RE = re.compile(r"\b[a-z]{1,3}\d{2,4}\b", re.I)

# The manipulation is a CONTRAST between two poles, so the binding requirement is
# pair-level: the verbose text must be at least MIN_CONTRAST times the terse one.
# Judging each pole against the original was the wrong target — original lengths span
# 710 to 47k characters, so "twice the original" is trivial for one system and
# impossible for another, and H3 compares verbose with terse, never with the original.
TERSE_MAX_RATIO = 0.70  # the terse instruction is "shorten", measured against the original
MIN_CONTRAST = 2.0  # len(verbose) / len(terse)


def effective_reading(task_id: str, text: str):
    """What the deterministic grader actually extracts from ``text``.

    Reuses the repo's own regexes so the gate cannot drift from the grader. This
    replaces naive Final-line equality, which rejected legitimate rewrites: an
    original that says "### Final Answer:" above its contract line produces two
    regex matches, and a terse rewrite (told to drop headings) produces one — the
    contract line itself being identical. ``parse_ids`` reads the last match that
    yields ids, so only that reading has to be invariant.
    """
    from _common import gold, grading_spec

    grader = grading_spec(task_id)["grader"]
    spec = gold(task_id)

    if grader in ("b3_select", "select_hypervolume"):
        from hallmark.graders.parse_ids import _FINAL_RE as PID_FINAL
        from hallmark.graders.parse_ids import _SPLIT_RE, _filter

        valid, k = set(spec["candidates"]), int(spec["k"])
        for tail in reversed(PID_FINAL.findall(text or "")):
            picks = _filter(_SPLIT_RE.split(tail.strip()), valid)
            if picks:
                return ("ids", tuple(picks[:k]))
        return ("ids", ())

    from hallmark import oracle as ora
    from hallmark.graders.extract import _find_labeled_choice, _regex_extract_point

    variables = ora.variable_names(spec)
    point, complete = _regex_extract_point(text or "", variables)
    cats = {
        c: _find_labeled_choice(text or "", c, list(levels))
        for c, levels in (spec.get("categorical") or {}).items()
    }
    return ("point", complete, tuple(sorted(point.items())), tuple(sorted(cats.items())))


def _norm_tail(tail: str) -> str:
    """Normalize a Final tail the way the graders read it.

    ``strip_markdown_emphasis`` + edge trimming mirror ``extract._regex_extract_point``
    and ``parse_ids._EDGE_RE``: a rewrite that drops a trailing ``**`` is byte-different
    but grader-identical, and rejecting it would cost retries without protecting anything.
    """
    from hallmark.textutils import strip_markdown_emphasis

    return re.sub(r"\s+", " ", strip_markdown_emphasis(tail)).strip(" .,;:`*_-")


def final_tails(text: str) -> list[str]:
    return [_norm_tail(t) for t in _FINAL_RE.findall(text or "")]


def gate(orig: str, var: str, style: str, task_id: str | None = None) -> list[str]:
    """Return a list of violations; empty means the variant is admissible."""
    bad: list[str] = []
    if not (var or "").strip():
        return ["empty variant"]

    if task_id:  # exact: the grader must read the same proposal out of both texts
        ro, rv = effective_reading(task_id, orig), effective_reading(task_id, var)
        if ro != rv:
            bad.append(f"grader reading changed: {ro} -> {rv}")
    else:  # fallback when the task is unknown: normalized Final tails
        fo, fv = final_tails(orig), final_tails(var)
        if not fo:
            bad.append("original has no Final line (cell should not be in the sample)")
        if fo != fv:
            bad.append(f"Final line(s) changed: {fo!r} -> {fv!r}")

    new_nums = sorted(set(_NUM_RE.findall(var)) - set(_NUM_RE.findall(orig)))
    if new_nums:
        bad.append(f"invented numbers: {new_nums[:6]}")

    new_ids = sorted({i.lower() for i in _ID_RE.findall(var)} - {i.lower() for i in _ID_RE.findall(orig)})
    if new_ids:
        bad.append(f"invented ids: {new_ids[:6]}")

    ratio = len(var) / max(len(orig), 1)
    if style == "terse" and ratio > TERSE_MAX_RATIO:
        bad.append(f"terse is not terse (ratio {ratio:.2f} > {TERSE_MAX_RATIO})")
    return bad


def pair_contrast(terse: str, verbose: str) -> float:
    return len(verbose) / max(len(terse), 1)


def main() -> None:
    path = DATA / "variants.jsonl"
    if not path.exists():
        print(f"no {path} yet — run 01_make_variants.py first")
        return
    by_uid = {(r["uid"], r["style"]): r for r in latest_variants(accepted_only=False)}
    fails = 0
    for (uid, style), r in sorted(by_uid.items()):
        if style == "orig":
            continue
        orig = by_uid[(uid, "orig")]["text"]
        bad = gate(orig, r["text"], style, r.get("task_id"))
        if bad:
            fails += 1
            print(f"FAIL {uid} [{style}]: {'; '.join(bad)}")

    weak = 0
    for uid in sorted({u for (u, s) in by_uid if s == "orig"}):
        t, v = by_uid.get((uid, "terse")), by_uid.get((uid, "verbose"))
        if not (t and v):
            continue
        c = pair_contrast(t["text"], v["text"])
        if c < MIN_CONTRAST:
            weak += 1
            print(f"WEAK {uid}: verbose/terse contrast {c:.2f} < {MIN_CONTRAST}")

    print(f"\n{len(by_uid)} variants, {fails} failing the gate, "
          f"{weak} pairs below the contrast floor")
    sys.exit(1 if (fails or weak) else 0)


if __name__ == "__main__":
    main()
