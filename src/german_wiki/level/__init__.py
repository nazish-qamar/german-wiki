"""CEFR leveling: rules first, model last (SPEC §5, §11 slice 6).

This package is the **only** import surface for leveling; internals are ``_``-prefixed,
matching the ``llm``, ``ingest``, ``embed`` and ``merge`` packages.

SPEC §5 opens with the problem: "Zero-shot LLM CEFR judgment is inconsistent across
sessions." So the level comes from three signals in strict precedence, and the model is
the last of them, not the first:

1. **Lexical** (``_lexical``) — a wordlist lookup. Ships as a working seam with **no
   data**: real Goethe/DWDS lists cannot be redistributed here, so ``vocab/cefr/*.txt`` is
   gitignored and absence is a supported state. A hand-written substitute was rejected —
   it would be the unreliable per-item judgment this whole approach exists to replace.
2. **Grammar** (``_grammar``) — SPEC §5's syllabus table as a hardcoded map. Pure lookup.
   Hits are graded by *where* they matched: the title says what a node is about, the body
   may only mention something in passing.
3. **Tiebreak** (``_tiebreak``) — one free ``glm-4.5-flash`` call, fired **only** when 1
   and 2 conflict or are both absent, and handed their results rather than asked cold.

``_cefr.derive_level`` enforces that precedence so the tiebreak cannot become the default
path, and ``_relevel`` re-derives levels on nodes that already exist — proposing through
``/proposals`` and ``gw review``, never writing directly (ADR-003).

Every model-derived level is marked ``llm:tiebreak`` in ``cefr_basis``, kept greppable as
the successor to slice 3's ``llm:extraction`` marker::

    grep -l 'cefr_basis: llm:tiebreak' nodes/    # the least-grounded levels you have
"""

from __future__ import annotations

from ._cefr import (
    HUMAN_SEED_MARKER,
    TIEBREAK_MARKER,
    LevelResult,
    derive_level,
    is_absent,
    is_placeholder,
)
from ._grammar import (
    CEFR_ORDER,
    GRAMMAR_MAP,
    GrammarHit,
    GrammarRule,
    grammar_anchor,
    is_title_anchored,
    strongest,
)
from ._lexical import LexicalHit, available, clear_cache, lexical_anchor, lookup
from ._relevel import FLAG_BASIS_ONLY, FLAG_TIEBREAK, RelevelResult, relevel, targets
from ._tiebreak import Tiebreak, TiebreakError, tiebreak

__all__ = [
    "CEFR_ORDER",
    "FLAG_BASIS_ONLY",
    "FLAG_TIEBREAK",
    "GRAMMAR_MAP",
    "HUMAN_SEED_MARKER",
    "TIEBREAK_MARKER",
    "GrammarHit",
    "GrammarRule",
    "LevelResult",
    "LexicalHit",
    "RelevelResult",
    "Tiebreak",
    "TiebreakError",
    "available",
    "clear_cache",
    "derive_level",
    "grammar_anchor",
    "is_absent",
    "is_placeholder",
    "is_title_anchored",
    "lexical_anchor",
    "lookup",
    "relevel",
    "strongest",
    "targets",
    "tiebreak",
]
