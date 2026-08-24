"""LLM->JSON extraction: pull a structured proposal out of a free-form answer.

Graders that need a machine-readable design point (``oracle_eval``,
``olympus_design``) share this. Both response forms — a scalar design point and a
full recipe — pin a ``Final: name=value, ...`` contract line in the prompt, so the
deterministic parse (``_regex_extract_point`` / ``_regex_extract_design_final``)
runs FIRST and a compliant answer never reaches the network. The LLM is the
*fallback* for an answer that states its proposal some other way (prose, an odd
table); the grader does the *judgment* deterministically either way — keeping the
judgment out of the LLM. Seam ``_extract_llm`` is the single network call (tests
monkeypatch it). ``extract_backend="regex"`` forces the key-less path outright
(labeled ``"<var> <number>"`` pairs scanned over the whole answer) for offline
dry-runs, mirroring the judge's ``overlap`` backend.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable

from ..textutils import strip_markdown_emphasis
from .judge import _JSON_BLOCK_RE, _llm_judge_one, resolve_judge_args

_SYSTEM = (
    "You extract structured data from an AI system's answer. Return EXACTLY one "
    "JSON object with the requested fields and nothing else. Use numbers (not "
    "strings) for numeric fields. If the answer does not state a field, omit it. "
    "Do not infer or invent values the answer does not give."
)

_SYSTEM_GENERIC = (
    "You extract structured data from an AI system's answer. Return EXACTLY one "
    "JSON value (object or array) matching the instruction and nothing else. Use "
    "numbers (not strings) for numeric fields. Do not invent values the answer "
    "does not state."
)

_JSON_ANY_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(answer: str) -> str | None:
    """Pull the model's Python source out of an answer (deterministic, no LLM).

    Returns the longest fenced ```python``` block, or the raw text if it has no
    fence but defines a function. None when neither is present. Used by the
    vibe-coding grader ``code_exec_eval``.
    """
    blocks = _CODE_BLOCK_RE.findall(answer or "")
    if blocks:
        return max(blocks, key=len).strip()
    text = (answer or "").strip()
    return text if "def " in text else None

_USER_TEMPLATE = """Extract the FINAL proposed design point from the answer.
Return a JSON object with one numeric field per variable (omit any not stated):
{fields}

## Answer
{answer}
"""


async def _extract_llm(messages: list[dict], params: dict) -> str:
    """Single network call (the monkeypatch seam for tests)."""
    model, api_key, base_url = resolve_judge_args(params, "extract")
    return await _llm_judge_one(model, api_key, base_url, messages)


def _parse_json(text: str) -> dict:
    m = _JSON_BLOCK_RE.search(text or "")
    for cand in ([text, m.group(0)] if m else [text or ""]):
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# A 4-digit number carrying a space/NBSP thousands separator ("1 026" -> 1026). The
# grouped alternative is tried FIRST (regex alternation is ordered) and is deliberately
# NARROW: a SINGLE leading digit, one 3-digit group, no trailing digit. That covers every
# realistic spaced value in these fields' domains (temps 1 000-1 299 K/°C) while refusing
# to merge two adjacent values — "500 550", "560 1200", "980 980" all have a >=3-digit or
# trailing-digit first run, so they fall through to the plain alternative (-> 500/560/980),
# matching the old behaviour. Decimals ("7.2") and plain 4-digit runs ("1058") also take
# the plain branch. Precision over recall: a wrong number tanks a benchmark score.
_LABELED_NUM = r"-?\d[  ]\d{3}(?!\d)(?:\.\d+)?|-?\d+(?:\.\d+)?"


def _find_labeled_number(answer: str, field: str, siblings: Iterable[str] | None = None) -> float | None:
    """Last ``<field> ... <number>`` occurrence in ``answer`` (deterministic).

    Tolerant of the ways an answer restates a machine field name:
    - an underscore matches a literal underscore *or* a space
      (``solutionizing_temp`` vs "Solutionizing temp"), case-insensitively;
    - the abbreviated tail may be spelled out — a trailing ``[a-z]*`` lets
      ``aging_temp`` match "Aging temperature" / "Solutionizing temperature";
    - a ``z`` in the field name matches ``z`` *or* ``s`` so a British spelling
      ("solutionising temperature") matches the American field ("solutionizing_temp");
    - the number may carry a space/NBSP thousands separator ("1 026" -> 1026);
    - up to 30 separator chars (``:``, ``=``, ``|``, ``≈``, units, **and the
      wide whitespace padding of a markdown table cell**) before the number.

    Suffix fallback: a multi-word field (``solutionizing_temp``) whose answer names
    only its FIRST word ("solutionizing 1058 °C" — the model dropped "temp") is still
    recovered — but only when no ``siblings`` field shares that first word, so
    ``aging_temp`` can never steal ``aging_time``'s number. A fully-qualified match
    (the whole label) always wins over a first-word-only one. The fallback fires only when
    ``siblings`` is passed (the collision set); left unset it is disabled, so an uninformed
    caller can never trigger a cross-field steal.
    """
    answer = strip_markdown_emphasis(answer)

    def _label(text: str) -> str:
        # underscore -> space-or-underscore; z -> [sz] (US/UK). ``str.replace`` does a
        # single non-overlapping pass, so the ``z`` it introduces in "[sz]" is not rescanned.
        return re.escape(text).replace("_", "[ _]").replace("z", "[sz]")

    def _scan(label: str) -> float | None:
        pat = re.compile(rf"{label}[a-z]*\b([^0-9\-]{{0,30}})({_LABELED_NUM})", re.IGNORECASE)
        # Prefer the LAST mention whose gap to the number is "clean" — only separators
        # / units, no intervening WORD. A wordy gap means the number usually belongs to
        # a *different* field named in prose, not to this one. Concretely this rejects
        # "...drive aging_temp low, set Cr ~16" (which would read aging_temp=16, then
        # clamp and tank the score) in favour of the recipe line "aging_temp 493".
        # Falls back to the last raw hit when no clean one exists, so a single wordy-gap
        # mention is still recovered (no regression vs the old last-hit behaviour).
        clean: float | None = None
        last: float | None = None
        for m in pat.finditer(answer):
            last = float(m.group(2).replace(" ", " ").replace(" ", ""))
            if not re.search(r"[A-Za-z]{2,}", m.group(1)):
                clean = last
        return clean if clean is not None else last

    # 1) fully-qualified: the whole field label.
    v = _scan(_label(field))
    if v is not None:
        return v
    # 2) first-word-only fallback, only when unambiguous among siblings.
    first, _sep, rest = field.partition("_")
    if rest and siblings is not None:  # unknown siblings -> stay conservative, no fallback
        fl = first.lower()
        if not any(s != field and s.split("_", 1)[0].lower() == fl for s in siblings):
            return _scan(_label(first))
    return None


def _find_labeled_choice(answer: str, field: str, levels: list[str]) -> str | None:
    """Last categorical level stated for ``field`` (deterministic backfill).

    Looks for the field label followed (within a short window) by one of the
    allowed ``levels``, tolerating space/hyphen for underscores in both the
    label and the level (``Heat Treatment: solution-aged`` -> ``solution_aged``).
    Returns the canonical level string (matching ``levels`` exactly) or None.
    """
    def _norm(s: str) -> str:
        return re.sub(r"[ _-]+", "_", s.strip().lower())

    answer = strip_markdown_emphasis(answer)
    canon = {_norm(lv): lv for lv in levels}
    label = re.escape(field).replace("_", "[ _]")
    alt = "|".join(
        re.escape(lv).replace(r"\_", "[ _-]") for lv in sorted(levels, key=len, reverse=True)
    )
    hits = re.findall(rf"{label}[a-z]*\b[^A-Za-z0-9]{{0,30}}({alt})", answer, re.IGNORECASE)
    return canon.get(_norm(hits[-1])) if hits else None


def _regex_extract(answer: str, variables: list[str]) -> dict:
    """Key-less fallback: last ``var ... <number>`` occurrence per variable."""
    out: dict[str, float] = {}
    for var in variables:
        val = _find_labeled_number(answer, var, variables)
        if val is not None:
            out[var] = val
    return out


# A line that *begins* with "Final" (tolerating markdown fences ** ## > - etc.) — the
# contract line every olympus/b3 design prompt asks the SUT to end with, e.g.
# ``Final: O=0.34, Cu=0.22, ...``. The prompt's own echoed "e.g. Final: ..." example
# sits mid-sentence (line starts with "Reason ..."), so it never matches this anchor.
_FINAL_ANCHOR_RE = re.compile(r"(?im)^[ \t#*>_.\-]*final\b.*$")
# ``name=number`` / ``name: number`` pairs, name a standalone token (not inside a word
# or a formula like ``Cr2O3``). Underscores allowed so multi-word knobs survive.
# number tolerates a decimal AND scientific notation ("6.03e-05", "1.9E+3") — models
# routinely emit tiny fractions in exponent form; dropping the exponent (reading
# "6.03e-05" as "6.03") would put the value wildly out of domain and tank the score.
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_PAIR_RE = re.compile(
    rf"(?<![A-Za-z0-9])([A-Za-z][A-Za-z0-9_]*)\s*[:=]\s*({_NUM})"
)
# Bookkeeping words a model tacks onto its proposal ("...; all other elements=0",
# "(Sum = 1.0000)") — never a real variable/element name, so an unknown ``name=number``
# token that is one of these must NOT force the LLM fallback.
_NOISE_TOKENS = frozenset(
    {"sum", "total", "subtotal", "remainder", "elements", "element", "others",
     "other", "rest", "all", "balance", "bal", "note", "count", "n",
     # "…, Ca=0.25 (all other element fractions = 0)" — the trailing bookkeeping clause
     # parses as ``fractions=0``. Without these two the whole Final line was declared
     # incomplete and the answer went to the LLM extractor, which read back the same four
     # numbers the anchor line already states. Observed on b3_supercon_maxelem.
     "fraction", "fractions"}
)


# Bookkeeping a model tacks onto the *recipe* contract line ("predicted_strength=1483",
# "sigma_y = 1502") on top of the shared set above. None of these can ever be a design field,
# so an unknown ``name=number`` token that is one of them must NOT force the LLM fallback.
_RECIPE_NOISE_TOKENS = _NOISE_TOKENS | frozenset(
    {"predicted", "prediction", "predicted_strength", "predicted_yield_strength",
     "yield_strength", "yield", "strength", "sigma_y", "ys", "mpa", "gpa"}
)


def _regex_extract_point(answer: str, variables: list[str]) -> tuple[dict, bool]:
    """Deterministic ``<var>=<number>`` recovery, preferring the last ``Final:`` line.

    Returns ``(point, complete)``. ``complete`` is True when a well-formed proposal was
    parsed AND every ``name=number`` token in it maps to a known variable — i.e. the
    answer is fully machine-readable and the LLM extractor would add nothing, so the
    caller can skip the (slow) network call. When the proposal isn't cleanly parseable
    (unusual formatting, prose-stated values, stray ``name=number`` tokens) ``complete``
    is False and the caller falls back to the LLM extractor — so this never *regresses*
    a result, it only removes the LLM round-trip for the common contract-compliant case.

    Anchoring on the last ``Final`` line (and stopping at the next blank line) keeps the
    prompt's echoed example and any trailing prose out of the token scan. Absent a
    ``Final`` line the whole answer is scanned with last-occurrence-wins, which still
    recovers the real proposal when a model states it once inline.
    """
    var_set = set(variables)
    answer = strip_markdown_emphasis(answer)  # ``**Final:** Cr=18`` must parse like plain text
    finals = list(_FINAL_ANCHOR_RE.finditer(answer))
    if finals:
        seg = answer[finals[-1].start():]
        seg = re.split(r"\n\s*\n", seg, maxsplit=1)[0]  # just the Final paragraph
    else:
        seg = answer
    pairs = _PAIR_RE.findall(seg)
    point: dict[str, float] = {}
    for name, num in pairs:
        if name in var_set:
            point[name] = float(num)  # last occurrence wins
    unknown = [name for name, _ in pairs if name not in var_set]
    complete = bool(point) and all(u.lower() in _NOISE_TOKENS for u in unknown)
    return point, complete


def _find_comp_value(answer: str, elem: str, dom: tuple[float, float]) -> float | None:
    """Deterministic alloying-wt% recovery for ONE element symbol, domain-bounded.

    Matches the element symbol as a standalone token (not inside a word or a
    formula like ``Cr2O3``) optionally followed by a separator / ``wt%`` and a
    number, e.g. ``Cr 14.8`` · ``Cr=14.8`` · ``| Cr | 14.8 |`` · ``Ni ~ 18``.
    Only **in-domain** hits are kept, so a stray prose number (a year, a
    temperature) can't masquerade as a wt%. Returns the last in-domain hit (the
    stated recipe value). Caller restricts this to *multi-letter* symbols — bare
    ``C``/``V``/``N`` are too collision-prone in prose (Celsius, figure numbers).

    Emphasis is stripped first: a bolded table cell (``| **Cr** | 18.0 |``) puts
    ``**`` between the symbol and its separator, which the pattern's narrow
    separator class rejects — silently dropping the whole composition.
    """
    answer = strip_markdown_emphasis(answer)
    lo, hi = dom
    pat = (
        rf"(?<![A-Za-z]){re.escape(elem)}(?![A-Za-z0-9])"
        r"\s*[:=|~≈]?\s*(?:wt\.?\s*%?\s*)?(-?\d+(?:\.\d+)?)"
    )
    inb = [v for v in (float(h) for h in re.findall(pat, answer)) if lo <= v <= hi]
    return inb[-1] if inb else None


def _regex_extract_design(answer: str, spec) -> dict:
    """Deterministic single-recipe extraction (composition + processing + categorical).

    The key-less safety net for the recipe schema, mirroring ``_regex_extract``
    for the scalar schema. Composition is recovered per multi-letter element via
    ``_find_comp_value`` (domain-bounded); processing via ``_find_labeled_number``;
    categorical via ``_find_labeled_choice``.
    """
    from .. import oracle as ora

    comp, proc, _ = ora.recipe_fields(spec)
    out: dict = {}
    for e in comp:
        if len(e) >= 2 and (v := _find_comp_value(answer, e, ora.domain(spec, e))) is not None:
            out[e] = v
    for p in proc:
        if (v := _find_labeled_number(answer, p, proc)) is not None:
            out[p] = v
    for c, levels in spec.get("categorical", {}).items():
        if (ch := _find_labeled_choice(answer, c, list(levels))) is not None:
            out[c] = ch
    return out


def _regex_extract_design_final(answer: str, spec) -> tuple[dict, bool]:
    """Deterministic recipe recovery from the ``Final:`` contract line.

    The recipe counterpart of :func:`_regex_extract_point`, and it exists for the same
    reason: the recipe prompts pin a ``Final: name=value, ...`` block, so a compliant
    answer is fully machine-readable and the LLM extractor would add nothing. Returns
    ``(design, complete)``; ``complete=False`` sends the caller to the LLM, so this can
    only remove a round-trip, never regress a result.

    ``complete`` requires: at least one alloying element (``assemble_recipe``'s own floor
    for "a proposal was made at all"), every *active* processing knob and categorical
    (an inactive one is noise — ``assemble_recipe`` back-fills it, so demanding it here
    would send a sound answer to the LLM for nothing), and no stray ``name=number`` token
    outside the bookkeeping set. A stated balance element (``Fe=76.5``) is accepted and
    ignored: ``assemble_recipe`` recomputes it as ``comp_total - Σ`` and never reads the
    stated value, so it is neither a design field nor a sign of an unparseable answer.
    """
    from .. import oracle as ora

    comp, proc, _ = ora.recipe_fields(spec)
    cats = spec.get("categorical", {})
    balance = spec.get("balance_element")
    active = ora.active_variables(spec)

    answer = strip_markdown_emphasis(answer)  # ``**Final:** Cr=18`` must parse like plain text
    finals = list(_FINAL_ANCHOR_RE.finditer(answer))
    if finals:
        seg = answer[finals[-1].start():]
        seg = re.split(r"\n\s*\n", seg, maxsplit=1)[0]  # just the Final block
    else:
        seg = answer

    known = set(comp) | set(proc)
    design: dict = {}
    unknown: list[str] = []
    for name, num in _PAIR_RE.findall(seg):
        if name in known:
            design[name] = float(num)  # last occurrence wins
        elif name != balance:
            unknown.append(name)
    for c, levels in cats.items():
        if (ch := _find_labeled_choice(seg, c, list(levels))) is not None:
            design[c] = ch

    complete = (
        any(e in design for e in comp)
        and all(p in design for p in proc if p in active)
        and all(c in design for c in cats if c in active)
        and all(u.lower() in _RECIPE_NOISE_TOKENS for u in unknown)
    )
    return design, complete


_BLOCK_HEADER_RE = re.compile(
    r"(?im)^[ \t#*>\-.]*\b(?:recipe|candidate|design|experiment|sample|alloy|option|point|R)"
    r"\s*#?\s*\d+\b"
)


def _split_recipe_blocks(answer: str) -> list[str]:
    """Slice a multi-recipe answer into per-recipe text blocks on ``Recipe N`` /
    ``Candidate N`` / ``R1`` headers. Returns [] when fewer than two markers are
    found (caller then treats the whole answer as a single block)."""
    marks = list(_BLOCK_HEADER_RE.finditer(answer or ""))
    if len(marks) < 2:
        return []
    spans = [m.start() for m in marks] + [len(answer)]
    return [answer[spans[i]:spans[i + 1]] for i in range(len(marks))]


def _regex_extract_designs(answer: str, spec) -> list[dict]:
    """Deterministic multi-recipe extraction: split on recipe headers, then run the
    single-recipe regex per block. Falls back to one whole-answer extraction when no
    headers are present (one recipe stated without a numbered header)."""
    blocks = _split_recipe_blocks(answer)
    if not blocks:
        d = _regex_extract_design(answer, spec)
        return [d] if d else []
    return [d for d in (_regex_extract_design(b, spec) for b in blocks) if d]


async def extract_point(answer: str, variables: list[str], params: dict) -> dict:
    """Return ``{var: float}`` for every variable the answer actually states."""
    if not (answer or "").strip():
        return {}      # nothing to extract: skip the network call (and its nondeterminism)
    if params.get("extract_backend", "llm") == "regex":
        return _regex_extract(answer, variables)
    # Deterministic first: the design prompts pin a ``Final: var=value, ...`` contract
    # line, so a compliant answer is fully parseable offline — skip the LLM entirely.
    point, complete = _regex_extract_point(answer, variables)
    if complete:
        return point
    fields = "\n".join(f"- {v}" for v in variables)
    user = _USER_TEMPLATE.format(fields=fields, answer=answer)
    raw = await _extract_llm(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}], params
    )
    obj = _parse_json(raw)
    out: dict[str, float] = {}
    for v in variables:
        if v in obj:
            try:
                out[v] = float(obj[v])
            except (TypeError, ValueError):
                pass
    return out


def _parse_any(text: str):
    """Parse the first JSON object OR array in ``text`` (whole string first)."""
    text = (text or "").strip()
    cands = [text]
    m = _JSON_ANY_RE.search(text)
    if m:
        cands.append(m.group(0))
    for cand in cands:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


_LIST_KEYS = (
    "recipes", "designs", "points", "batch", "experiments", "candidates",
    "items", "selection", "intervals", "predictions", "results", "data",
)


def coerce_list(data) -> list:
    """Normalize a parsed JSON value into a list of records.

    The extraction model runs in JSON-*object* mode (``response_format`` in
    ``_llm_judge_one``), so it cannot emit a bare top-level array — a requested
    array comes back wrapped, e.g. ``{"recipes": [...]}`` or ``{"points": [...]}``.
    Without this, the multi-point graders (batch_doe, pareto, calibration,
    combinatorial_select) iterate the dict's keys and silently get nothing.

    Rules: a list passes through; a dict is unwrapped to its single list value
    (preferring common batch keys); a dict whose values are all records is taken
    as a map of records; otherwise a lone record becomes a one-element batch.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals  # a map keyed by id -> the records themselves
        return [data]  # a single record == a one-element batch
    return []


async def extract_json(answer: str, instruction: str, params: dict):
    """Generic LLM extraction returning a parsed JSON value (dict/list) or None.

    For the multi-point / record-shaped graders (batch_doe, pareto_hypervolume,
    calibration, combinatorial_select, …). There is no regex fallback for these
    richer shapes — offline tests monkeypatch ``_extract_llm`` or pass an inline
    answer the judge would return. Returns None if nothing parseable was found.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_GENERIC},
        {"role": "user", "content": f"{instruction}\n\n## Answer\n{answer}"},
    ]
    return _parse_any(await _extract_llm(messages, params))


def recipe_instruction(spec, *, multi: bool = False) -> str:
    """Field listing for extracting a full alloy recipe (composition + processing +
    categorical) from a free-form answer, derived from the oracle's recipe schema."""
    from .. import oracle as ora

    comp, proc, _ = ora.recipe_fields(spec)
    head = (
        "Extract the FINAL proposed recipes as a JSON ARRAY of objects, each with these fields:"
        if multi
        else "Extract the FINAL proposed recipe. Return ONE JSON object with these fields:"
    )
    lines = [
        head,
        f"- composition wt% (numbers; omit elements that are absent): {', '.join(comp)}",
        f"- processing parameters (numbers): {', '.join(proc)}",
    ]
    for c, levels in spec.get("categorical", {}).items():
        lines.append(f"- {c} (a string, exactly one of: {', '.join(levels)})")
    return "\n".join(lines)


async def extract_design(answer: str, spec, params: dict) -> dict:
    """Extract a single mixed recipe (numeric fields + categorical strings).

    Deterministic first: a ``Final:`` block that names every active field parses offline
    (``_regex_extract_design_final``) and returns without a network call. Everything below
    applies to the LLM fallback, which handles an answer that states its recipe some other
    way.

    The LLM extractor occasionally drops a field that is literally present in the
    answer (most often a field the model flagged as noise, e.g. "set to median").
    As a deterministic safety net, any processing/categorical field the extractor
    missed is back-filled by regex over the raw answer before returning — so a
    complete, valid recipe isn't zeroed by a single flaky omission.
    """
    from .. import oracle as ora

    if not (answer or "").strip():
        return {}      # nothing to extract: skip the network call (and its nondeterminism)
    if params.get("extract_backend", "llm") == "regex":
        return _regex_extract_design(answer, spec)

    # Deterministic first: the recipe prompts pin a ``Final: name=value, ...`` contract
    # block, so a compliant answer is fully parseable offline — skip the LLM entirely.
    design, complete = _regex_extract_design_final(answer, spec)
    if complete:
        return design

    data = await extract_json(answer, recipe_instruction(spec), params)
    data = data if isinstance(data, dict) else {}
    comp, proc, _ = ora.recipe_fields(spec)
    # Back-fill alloying elements the LLM extractor dropped (it occasionally loses a
    # whole composition when it sits in an unusual table layout). Restricted to
    # *multi-letter* symbols in an explicit, domain-bounded wt% context — bare
    # C/V/N are too collision-prone in prose. Without this a flaky extraction
    # collapses the recipe to the balance element (pure Fe) and scores 0.
    for e in comp:
        if e not in data and len(e) >= 2:
            if (v := _find_comp_value(answer, e, ora.domain(spec, e))) is not None:
                data[e] = v
    for field in proc:
        if field not in data and (v := _find_labeled_number(answer, field, proc)) is not None:
            data[field] = v
    # Categoricals (heat_treatment, route) have no numeric fallback; an active one
    # dropped by the extractor would zero an otherwise-complete recipe. Back-fill
    # by matching the labelled choice against the allowed levels.
    for cat, levels in spec.get("categorical", {}).items():
        if cat not in data and (c := _find_labeled_choice(answer, cat, list(levels))) is not None:
            data[cat] = c
    return data


async def extract_designs(answer: str, spec, params: dict) -> list[dict]:
    """Extract a list of mixed recipes (for the multi-recipe graders)."""
    if params.get("extract_backend", "llm") == "regex":
        return _regex_extract_designs(answer, spec)
    data = await extract_json(answer, recipe_instruction(spec, multi=True), params)
    return [d for d in coerce_list(data) if isinstance(d, dict)]
