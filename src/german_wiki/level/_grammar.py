"""The grammar anchor: SPEC §5's syllabus table, as a lookup (signal 2).

SPEC §5 is blunt about why this is hardcoded rather than inferred: "This is a known,
stable syllabus; there's no reason to infer it." So the table below is the table from
§5, transcribed, and the only judgment in this module is *matching a node to a row*.

**That match is the fallible step, and it is graded accordingly.** A worked example from
the live corpus, which the naive version gets wrong: ``verben-mit-präpositionen`` is a B1
concept whose body mentions *Akkusativ* and *Dativ* in passing. SPEC puts Akkusativ at A2,
so a flat "scan the whole node for keywords" would confidently propose **B1 → A2** and
quietly relabel a B1 grammar rule as beginner material. Since ``cefr`` drives SPEC §5.1's
priority score, that reorders what you study.

So hits carry *where* they were found:

- a hit in ``title_de`` is what the node is **about** — it decides;
- a hit in ``body_md`` is something the node **mentions** — it is a weak signal, and
  ``_cefr`` sends it to the tiebreak rather than letting it override an existing level.

Nothing here writes, calls a model, or decides a final level; it reports what matched.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from ..models import CEFR

# Ascending, so "the highest structure a node teaches" is a max() over this.
CEFR_ORDER: dict[str, int] = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

Where = Literal["title", "body"]


@dataclass(frozen=True)
class GrammarRule:
    """One row of SPEC §5's table: a structure, its level, and how to spot it."""

    structure: str
    level: CEFR
    pattern: re.Pattern[str]


def _rule(structure: str, level: CEFR, pattern: str) -> GrammarRule:
    return GrammarRule(structure, level, re.compile(pattern, re.IGNORECASE))


# German compounds are why these mostly anchor at the START of a word but not the end:
# `\bakkusativ` matches "Akkusativ", "Akkusativobjekt" and "Akkusativergänzung", which are
# all the same signal. A trailing `\b` would miss every compound. Conversely there is no
# leading-substring matching, so `\bpräposition` would NOT fire inside
# "Wechselpräpositionen" -- which is correct, because that is its own row at a different
# level and has its own rule.
GRAMMAR_MAP: list[GrammarRule] = [
    # --- A1: Präsens, Nominativ ---
    _rule("präsens", "A1", r"\bpräsens"),
    _rule("nominativ", "A1", r"\bnominativ"),
    # --- A2: Perfekt, Akkusativ, Wechselpräpositionen ---
    # `\bperfekt` deliberately does not fire inside "Plusquamperfekt" (no word boundary),
    # and Plusquamperfekt is not in §5's table, so it contributes nothing rather than
    # being silently treated as A2.
    _rule("perfekt", "A2", r"\bperfekt"),
    _rule("akkusativ", "A2", r"\bakkusativ"),
    _rule("wechselpräposition", "A2", r"\bwechselpräposition"),
    # --- B1: Passiv, Konjunktiv II (basic), Relativsätze ---
    _rule("passiv", "B1", r"\bpassiv"),
    # §5 splits Konjunktiv II across B1 ("basic") and B2 ("full"). No keyword can tell
    # those apart, so this maps to the LOWER level: under-leveling sends the node to the
    # tiebreak or to your review, while over-leveling hides a B1 rule above your target
    # and it never surfaces in the study queue. Erring downward fails visibly.
    _rule("konjunktiv-ii", "B1", r"\bkonjunktiv[\s-]*(?:ii|2)\b"),
    _rule("relativsatz", "B1", r"\brelativsa|\brelativsä"),
    # --- B2: Konjunktiv II (full), Genitiv, erweiterte Infinitive ---
    _rule("genitiv", "B2", r"\bgenitiv"),
    _rule("erweiterter-infinitiv", "B2", r"\berweiterte[rn]?\s+infinitiv"),
    # --- C1: Partizipialattribute, Nominalstil, Konjunktiv I ---
    _rule("partizipialattribut", "C1", r"\bpartizipialattribut"),
    _rule("nominalstil", "C1", r"\bnominalstil"),
    # The negative lookahead -- NOT list ordering -- is what stops this swallowing
    # "Konjunktiv II" and marking a B1 node C1. Ordering would work only until someone
    # re-sorted the table; the lookahead holds regardless of position.
    _rule("konjunktiv-i", "C1", r"\bkonjunktiv[\s-]*(?:i(?!i)|1)\b"),
]


@dataclass(frozen=True)
class GrammarHit:
    structure: str
    level: CEFR
    where: Where


def _normalize(text: str) -> str:
    """NFC + casefold, so ``Wechselpräpositionen`` and a decomposed ``ä`` both match.

    NFC for the same reason node ids normalize (ADR-012): ``ä`` has two encodings that
    look identical, and a pattern written with one would silently miss the other.
    """
    return unicodedata.normalize("NFC", text).casefold()


def match(text: str, where: Where) -> list[GrammarHit]:
    """Every SPEC §5 structure appearing in ``text``, tagged with where it was found."""
    haystack = _normalize(text)
    return [
        GrammarHit(rule.structure, rule.level, where)
        for rule in GRAMMAR_MAP
        if rule.pattern.search(haystack)
    ]


def grammar_anchor(title_de: str, body_md: str) -> list[GrammarHit]:
    """All hits, title first. Order is presentational; ``strongest`` does the grading."""
    return [*match(title_de, "title"), *match(body_md, "body")]


def strongest(hits: list[GrammarHit]) -> tuple[CEFR, list[GrammarHit]] | None:
    """The level this node teaches, preferring what the title says it is about.

    Returns ``(level, hits_that_set_it)``, or ``None`` when nothing matched.

    Title hits shadow body hits **entirely** rather than being merely weighted: a node
    titled "Perfekt" is about the Perfekt even if its body mentions Genitiv in an aside,
    and averaging the two would produce a level neither signal supports. Within a scope,
    the highest level wins -- you cannot study a node until you can handle its hardest
    structure.
    """
    if not hits:
        return None
    scoped = [h for h in hits if h.where == "title"] or hits
    top = max(CEFR_ORDER[h.level] for h in scoped)
    winners = [h for h in scoped if CEFR_ORDER[h.level] == top]
    return winners[0].level, winners


def is_title_anchored(hits: list[GrammarHit]) -> bool:
    """True when the title itself named a §5 structure -- the confident case."""
    return any(h.where == "title" for h in hits)
