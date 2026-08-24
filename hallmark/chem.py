"""Chemistry helpers: element symbols, validity, composition parsing.

Prefers ``pymatgen`` when installed (``pip install -e ".[chem]"``) for element/
composition validity, and falls back to a built-in symbol table so the text
graders run without the heavy dependency.
"""
from __future__ import annotations

import re
from fractions import Fraction
from functools import reduce
from math import gcd

ELEMENTS: tuple[str, ...] = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I "
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu"
).split()
_ELEMENT_SET = set(ELEMENTS)

# Longest-first so "Co" wins over "C", "Mn" over an accidental "M", etc.
_ALT = "|".join(sorted(ELEMENTS, key=len, reverse=True))
ELEMENT_RE = re.compile(r"\b(" + _ALT + r")\b")

# "Cr 18", "Cr: 18.0", "Cr = 18 wt%", "Cr 18%", and markdown-table cells like
# "| Cr | 18.0 |" (pipe separator) or an approximate "Fe ~86.6". Case-sensitive
# on purpose: a case-insensitive match would treat English words (in, as, be,
# he) as elements.
_COMPOSITION_RE = re.compile(
    r"\b(" + _ALT + r")\b\s*[:=|]?\s*~?\s*(\d+(?:\.\d+)?)\s*(?:wt\.?\s*%|at\.?\s*%|%)?"
)


def valid_element(symbol: str) -> bool:
    """Whether ``symbol`` is a real chemical element (pymatgen if available)."""
    try:
        from pymatgen.core import Element

        return Element.is_valid_symbol(symbol)
    except Exception:
        return symbol in _ELEMENT_SET


def parse_composition(text: str, *, max_value: float = 100.0) -> dict[str, float]:
    """Extract an ``{element: amount}`` mapping from free text.

    Keeps the first occurrence of each element and drops implausible amounts
    (> ``max_value``). Element symbols are normalized to canonical case.
    """
    out: dict[str, float] = {}
    for m in _COMPOSITION_RE.finditer(text):
        sym, num = m.group(1), m.group(2)
        value = float(num)
        if value > max_value or sym in out:
            continue
        out[sym] = value
    return out


# A single, gap-free chemical formula token: "SrTiO3", "La0.5Sr0.5MnO3", "Sr1Ti1O3".
# (Distinct from _COMPOSITION_RE, which reads spaced "Cr 18 wt%" recipe text.)
_FORMULA_TOKEN_RE = re.compile(r"[A-Z][a-z]?\d*(?:\.\d+)?")
_FORMULA_FULL_RE = re.compile(r"(?:[A-Z][a-z]?\d*(?:\.\d+)?)+")
_FORMULA_PART_RE = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")


def parse_formula(text: str) -> dict[str, float] | None:
    """Parse ONE compact chemical formula into ``{element: amount}``; ``None`` on junk.

    ``"SrTiO3" -> {"Sr": 1.0, "Ti": 1.0, "O": 3.0}``. Prefers ``pymatgen.Composition``
    (handles parentheses/hydrates) and falls back to a built-in tokenizer that accepts
    only a gap-free run of ``Element[count]`` groups — so prose ("as", "in") and spaced
    recipe text are rejected (use ``parse_composition`` for those). Unlike
    ``parse_composition`` this is a strict *token* parser for the design-task answer.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        from pymatgen.core import Composition

        d = Composition(text).get_el_amt_dict()
        return {str(k): float(v) for k, v in d.items()} or None
    except Exception:
        pass
    if not _FORMULA_FULL_RE.fullmatch(text):
        return None
    out: dict[str, float] = {}
    for m in _FORMULA_PART_RE.finditer(text):
        sym, num = m.group(1), m.group(2)
        if sym not in _ELEMENT_SET:
            return None
        out[sym] = out.get(sym, 0.0) + (float(num) if num else 1.0)
    return out or None


def canonical_formula(text_or_comp: str | dict) -> str | None:
    """Canonical join key for a formula: integer coefficients reduced by gcd, elements
    alphabetical. ``"SrTiO3"`` / ``"Sr2Ti2O6"`` / ``"TiSrO3"`` all -> ``"O3SrTi"``.

    The single key used by the MLIP snapshot (file name + record ``key``) and by
    ``set_f1`` formula matching, so any spelling of the same composition collides.
    Returns ``None`` for unparseable input or non-positive amounts.
    """
    comp = text_or_comp if isinstance(text_or_comp, dict) else parse_formula(str(text_or_comp))
    if not comp:
        return None
    fr = {k: Fraction(float(v)).limit_denominator(10000) for k, v in comp.items()}
    if any(f <= 0 for f in fr.values()):
        return None
    denom = reduce(lambda a, b: a * b // gcd(a, b), [f.denominator for f in fr.values()], 1)
    ints = {k: int(f * denom) for k, f in fr.items()}
    g = reduce(gcd, list(ints.values())) or 1
    ints = {k: v // g for k, v in ints.items()}
    return "".join(f"{el}{ints[el]}" if ints[el] != 1 else el for el in sorted(ints))
