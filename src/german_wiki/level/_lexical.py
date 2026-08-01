"""The lexical anchor: a wordlist lookup (SPEC §5, signal 1).

SPEC §5 puts this first — Goethe publishes official A1/A2/B1 lists, and frequency rank is
a strong CEFR proxy above that. It is also the signal this repo **ships without data**.

The lists are not in git. Real ones (Goethe's Wortlisten, DWDS frequency classes) carry
licensing that cannot be redistributed here, so ``vocab/cefr/*.txt`` is gitignored and only
``vocab/cefr/README.md`` is tracked — which is also what keeps the directory present on a
fresh clone. See that README for the format and where to obtain a list.

**A hand-written starter list was rejected rather than deferred.** A CEFR list written from
recollection is exactly the unreliable per-item judgment SPEC §5 introduces the rules
approach to *replace*, and it would put unearned confidence into ``cefr_basis`` where
nobody could audit it. The project already applies this rule to model prices (ADR-008 §4:
a wrong rate is worse than a visible gap); the same holds for levels.

So **absence is a supported state, not an error**: a missing directory, a missing file, an
empty file and a file of blank lines all mean "no signal". Every caller must cope with
``None``, and until a list exists, pure-vocabulary nodes have no lexical anchor and no
grammar structure, so they reach the LLM tiebreak. Those levels stay findable with
``grep -l 'cefr_basis: llm:tiebreak' nodes/``.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .. import config
from ..models import CEFR

# Ascending. A lemma in several lists resolves to the LOWEST — you meet a word at the
# earliest level that teaches it, and later lists repeating it say nothing new.
LEVEL_FILES: list[tuple[CEFR, str]] = [
    ("A1", "a1.txt"),
    ("A2", "a2.txt"),
    ("B1", "b1.txt"),
    ("B2", "b2.txt"),
    ("C1", "c1.txt"),
    ("C2", "c2.txt"),
]

COMMENT_PREFIX = "#"


class LexicalHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lemma: str
    level: CEFR
    #  Which list it came from, for ``cefr_basis`` -- e.g. "goethe:a1".
    source: str


def _fold(text: str) -> str:
    """NFC + casefold, matching how the wordlist itself is read.

    Both sides go through this, so a precomposed ``ü`` in the list still matches a
    decomposed one in a title (the ADR-012 hazard, in a second place)."""
    return unicodedata.normalize("NFC", text).strip().casefold()


def wordlist_dir(cefr_dir: Path | str | None = None) -> Path:
    return Path(cefr_dir) if cefr_dir is not None else config.CEFR_DIR


def _read_list(path: Path) -> set[str]:
    """Lemmas in one file. A missing or unreadable file is an empty set, never a raise."""
    if not path.is_file():
        return set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {
        folded
        for line in lines
        if not line.strip().startswith(COMMENT_PREFIX) and (folded := _fold(line))
    }


@lru_cache(maxsize=8)
def _load(root: str) -> list[tuple[CEFR, str, frozenset[str]]]:
    """(level, source, lemmas) per level file, ascending. Cached per directory."""
    base = Path(root)
    loaded = []
    for level, filename in LEVEL_FILES:
        lemmas = _read_list(base / filename)
        if lemmas:
            loaded.append((level, f"goethe:{filename.removesuffix('.txt')}", frozenset(lemmas)))
    return loaded


def clear_cache() -> None:
    """Forget loaded wordlists — for tests, and after dropping a new list in."""
    _load.cache_clear()


def available(cefr_dir: Path | str | None = None) -> bool:
    """Whether any wordlist holds data. False is normal on a fresh clone."""
    return bool(_load(str(wordlist_dir(cefr_dir))))


def lookup(lemma: str, *, cefr_dir: Path | str | None = None) -> LexicalHit | None:
    """The earliest level listing ``lemma``, or ``None`` when nothing lists it.

    ``None`` is ambiguous by design and callers must treat it as such: it means either
    "no wordlist is installed" or "installed lists do not contain this word". Neither is
    evidence about the level, which is why this never returns a default.
    """
    needle = _fold(lemma)
    if not needle:
        return None
    for level, source, lemmas in _load(str(wordlist_dir(cefr_dir))):
        if needle in lemmas:
            return LexicalHit(lemma=needle, level=level, source=source)
    return None


def lexical_anchor(
    title_de: str, *, lemmas: list[str] | None = None, cefr_dir: Path | str | None = None
) -> LexicalHit | None:
    """The lexical signal for a node, or ``None``.

    Checks the node's declared ``lemmas`` first (``type: vocab`` word families carry
    them) and falls back to the German title. Returns the **highest** level among the
    family's members, because a family is only as easy as its hardest word — the opposite
    of ``lookup``'s lowest-wins rule, which resolves one word listed twice.
    """
    from ._grammar import CEFR_ORDER

    hits = [hit for word in (lemmas or []) if (hit := lookup(word, cefr_dir=cefr_dir))]
    if hits:
        return max(hits, key=lambda h: CEFR_ORDER[h.level])
    return lookup(title_de, cefr_dir=cefr_dir)
