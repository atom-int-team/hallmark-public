"""Small text helpers shared by adapters and graders.

Kept dependency-free so graders can run without any provider SDKs.
"""
from __future__ import annotations

import re

# <think>…</think> / <thinking>…</thinking> reasoning blocks some models emit.
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)

# DOI: 10.<registrant>/<suffix>; suffix is greedy up to whitespace/closing punctuation.
# ``*`` is excluded so markdown bold/italic (``**10.1234/foo**``) doesn't bleed into
# the captured DOI; ``?``/``#`` end the suffix so a URL query/fragment tacked onto a
# DOI link (``…/foo?utm_source=chatgpt.com``) is dropped. A real DOI contains none of
# these, and the trailing junk would otherwise 404 at CrossRef and be miscounted as a
# hallucination.
# ``(`` and ``)`` stay INSIDE the character class: legacy Elsevier/Wiley DOIs embed a
# parenthesised year (``10.1016/0025-5408(80)90012-4``). Excluding ``)`` truncated those
# to ``10.1016/0025-5408(80``, which 404s at CrossRef and is miscounted as a fabricated
# citation. A DOI written inside prose parentheses — ``(see 10.1234/abc)`` — is handled
# instead by balancing the parens in ``_clean``.
# Old Wiley SICI identifiers contain an angle-bracketed article identifier, e.g.
# ``10.1002/(SICI)...<1071::AID-ADMA1071>3.0.CO;2-H``. Match that family first;
# the general branch keeps excluding angle brackets so an adjacent HTML tag is
# never swallowed into an ordinary DOI.
_DOI_RE = re.compile(
    r"\b(?:10\.1002/\(sici\)[^\s\"']+|10\.\d{4,9}/[^\s\"'<>\]}*?#]+)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]}]+", re.IGNORECASE)

# Trailing punctuation that often clings to a DOI/URL inside prose (``*`` and `````
# for markdown emphasis / code spans). ``)`` is NOT here — it is stripped only when
# unbalanced, see ``_clean``.
_TRAILING = ".,;:]}>\"'*`"

# Unicode dashes copy-paste sometimes substitutes for an ASCII hyphen inside a DOI
# (e.g. U+2011 non-breaking hyphen) — normalize so the DOI resolves at CrossRef.
_UNICODE_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})

# Unicode sub/superscript digits models emit when typesetting chemical formulas
# (``KTaO₃``) or oxidation states (``Fe³⁺``). These map to ``str.isalnum() == True``
# but NOT to ``\d`` in Python regex, so a formula token built on ``\d`` (``[A-Z][a-z]?\d*``)
# matches nothing and the count is lost. Fold them to ASCII before formula extraction.
_UNICODE_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def normalize_dashes(text: str) -> str:
    """Map Unicode dashes/minuses (U+2212, en/em dash, …) to ASCII ``-``.

    Models routinely emit the typographic minus ``−`` (U+2212) in front of
    negative numbers; a numeric regex built on ASCII ``-`` would otherwise read
    ``−0.45`` as ``0.45`` and flip the sign. Apply before numeric extraction.
    """
    return text.translate(_UNICODE_DASHES)


def normalize_subscripts(text: str) -> str:
    """Map Unicode sub/superscript digits (``KTaO₃`` → ``KTaO3``) to ASCII.

    Apply before chemical-formula extraction: the subscript ``₃`` (U+2083) is a
    ``\\w`` word character but not a ``\\d`` digit, so a formula regex anchored on
    ``\\d`` and ``\\b`` fails to match ``KTaO₃`` at all and falls back to spurious
    earlier tokens (e.g. the Roman-numeral oxidation state ``IV``).
    """
    return (text or "").translate(_UNICODE_DIGITS)


def strip_think(text: str) -> str:
    """Remove <think>…</think> reasoning blocks; return the visible answer."""
    return _THINK_RE.sub("", text).strip()


# Markdown emphasis / code-span markers. Models typeset recipe tables as
# ``| **Cr** | 18.0 wt% |`` and the bold markers fuse to the label, so a regex
# expecting ``<symbol><separator><number>`` never matches. Only ``__`` (double)
# is stripped, never a single ``_`` — machine field names carry one
# (``aging_temp``, ``cooling_rate``) and stripping it would break their labels.
_MD_EMPHASIS_RE = re.compile(r"\*\*|\*|__|`")


def strip_markdown_emphasis(text: str) -> str:
    """Remove ``**``/``*``/``__``/`` ` `` markers so a label abuts its value.

    Apply before any *regex* extraction that keys off a label ("Cr", "aging_temp")
    followed by a number: the deterministic extractors allow only separator
    characters between the two, and a bold marker is not one. Never apply before
    LLM extraction — the markdown structure helps the model read the table.
    """
    return _MD_EMPHASIS_RE.sub("", text or "")


def _clean(token: str) -> str:
    """Strip prose punctuation clinging to a DOI/URL. A trailing ``)`` is removed only
    when it closes a paren the token never opened (``(see 10.1234/abc)``) — never the
    ``)`` that belongs *inside* a legacy DOI (``10.1016/0025-5408(80)90012-4``)."""
    token = token.strip(_TRAILING)
    # A closing prose parenthesis can be followed immediately by the next sentence
    # (``10.1234/abc).All ...``). Stop at the first close that the DOI itself never
    # opened; checking only ``endswith(")")`` would swallow ``).All`` as DOI text.
    balance = 0
    for i, char in enumerate(token):
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
            if balance < 0:
                token = token[:i].strip(_TRAILING)
                break
    while token.endswith(")") and token.count(")") > token.count("("):
        token = token[:-1].strip(_TRAILING)
    return token


def extract_dois(text: str) -> list[str]:
    """All DOIs in ``text``, normalized to lowercase, de-duplicated, order-preserving."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _DOI_RE.finditer(text):
        doi = _clean(m.group(0)).translate(_UNICODE_DASHES).lower()
        if doi not in seen:
            seen.add(doi)
            out.append(doi)
    return out


def mask_dois(text: str, fill: str = " ") -> str:
    """Blank out every DOI token, preserving character offsets.

    A DOI suffix routinely encodes digits that read as a year — ``10.1016/j.jcp.2014.12.018``
    is a 2015 paper. A year (or title) a citation *claims* must therefore be read from the
    prose around the identifier, never from the identifier itself. Offsets are preserved so
    spans found in the masked text index back into the original.
    """
    return _DOI_RE.sub(lambda m: fill * len(m.group(0)), text)


def extract_urls(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _URL_RE.finditer(text):
        url = _clean(m.group(0))
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def extract_citations(text: str) -> list[str]:
    """DOIs first (as bare ``10.…`` strings), then any other URLs."""
    dois = extract_dois(text)
    doi_set = set(dois)
    urls = [u for u in extract_urls(text) if not any(d in u.lower() for d in doi_set)]
    return dois + urls


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(system_id: str) -> str:
    """Filesystem-safe slug for a system_id (shared by runner cache + manual subs)."""
    return _SLUG_RE.sub("_", system_id).strip("_")


_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-words F1 between two strings (offline text-overlap proxy)."""
    p, g = set(tokenize(pred)), set(tokenize(gold))
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def title_similarity(a: str, b: str) -> float:
    """Similarity of two titles in [0, 1]: max of sequence ratio and token-set F1."""
    from difflib import SequenceMatcher

    na, nb = " ".join(tokenize(a)), " ".join(tokenize(b))
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    return max(ratio, token_f1(a, b))
