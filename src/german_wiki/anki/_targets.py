"""What gets blanked out of an example sentence -- and when nothing does.

A cloze card is only as good as its deletion. Blanking the wrong word produces a card that
is *silently* useless: it looks like a real card, it comes up in reviews, and it teaches
either nothing or something false. Unlike most defects in this project that failure never
announces itself, so this module is built to refuse rather than to guess.

Three rungs, strict precedence, and the last one is a refusal:

1. **``**bold**`` in the sentence.** The author marked what is being taught. This is not a
   heuristic at all -- it is an assertion, and it always wins.
2. **The node's own structured content.** Only where the node has said something specific
   enough to act on; see the two rules below.
3. **Nothing.** No cloze card is emitted and the caller reports the example as unblanked.

Rung 2 follows ADR-014's rule for grid columns: *read the human's assertion, do not infer
the fact*. A node that writes ``teilnehmen an (+ Dativ)`` has explicitly claimed that `an`
governs a case, and blanking `an` teaches exactly that claim. A node that merely happens to
contain the word `an` has claimed nothing, and gets rung 3.

**No inflection matching, deliberately.** ``wasche`` is not ``waschen`` here, and
``wäschst`` is further still -- resolving it needs ablaut folding (a->ä), which would
introduce false stem matches, which is precisely the silent-bad-card failure this module
exists to avoid. An example whose lemma appears only in an inflected form falls to rung 3;
the fix is for the author to bold the word, which is one keystroke and unambiguous.

Nothing here writes, calls a model, or looks outside the node it is given.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..models import Node
from ..morph import fold, lemma_key

# The closed inventory. A closed set is what keeps rung 2 from being a guess: no matter
# what a body says, only these words can ever be selected. Grouped as the grammars do.
PREPOSITIONS: frozenset[str] = frozenset(
    {
        # Wechselpräpositionen (SPEC's A2 anchor)
        "an", "auf", "hinter", "in", "neben", "über", "unter", "vor", "zwischen",
        # Akkusativ
        "bis", "durch", "für", "gegen", "ohne", "um", "entlang",
        # Dativ
        "aus", "außer", "bei", "gegenüber", "mit", "nach", "seit", "von", "zu",
        # Genitiv
        "während", "wegen", "trotz", "statt", "innerhalb", "außerhalb", "aufgrund",
    }
)  # fmt: skip

# The case labels a node uses when it is asserting government.
_CASE = r"(?:Akkusativ|Dativ|Genitiv|Akk\.?|Dat\.?|Gen\.?)"

# Two spellings of the same assertion: `an (+ Dativ)`, `an (Dativ)`, `an + Dativ`.
# The captured word still has to be in PREPOSITIONS, which is what makes this safe.
_GOVERNED = [
    re.compile(rf"\b(\w+)\s*\(\s*\+?\s*{_CASE}\s*\)", re.IGNORECASE),
    re.compile(rf"\b(\w+)\s*\+\s*{_CASE}\b", re.IGNORECASE),
]

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

# Word-ish runs, keeping internal hyphens and apostrophes so `du-Ebene` stays one token.
_TOKEN = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)

TargetSource = Literal["bold", "structure"]


class ClozeTargets(BaseModel):
    """The substrings of a *plain* sentence to blank, and which rung supplied them."""

    model_config = ConfigDict(extra="forbid")

    terms: list[str]
    source: TargetSource


def strip_emphasis(text: str) -> str:
    """The sentence as it should appear on the card, markers removed."""
    return _ITALIC.sub(r"\1", _BOLD.sub(r"\1", text))


def bold_terms(german: str) -> list[str]:
    """The text inside each ``**...**`` span, in order."""
    return [match.group(1) for match in _BOLD.finditer(german)]


def governed_prepositions(body_md: str) -> set[str]:
    """Prepositions the body explicitly annotates with a case.

    The `(+ Dativ)` is the assertion. Without it a bare preposition in prose is just a
    word, and acting on it would blank `in` on a Konjunktiv II card.
    """
    found: set[str] = set()
    for pattern in _GOVERNED:
        for match in pattern.finditer(body_md):
            word = fold(match.group(1))
            if word in PREPOSITIONS:
                found.add(word)
    return found


def _lemma_keys(node: Node) -> set[str]:
    keys = {lemma_key(lemma) for lemma in node.lemmas or []}
    if node.root:
        keys.add(lemma_key(node.root))
    return {key for key in keys if key}


def _match_keys(plain: str, keys: set[str]) -> list[str]:
    """Tokens of ``plain`` whose folded form equals one of ``keys``, longest n-gram first.

    Compares token-by-token rather than searching the folded *sentence*, because
    ``str.casefold`` is not length-preserving (``groß`` -> ``gross``) and any index
    mapping between the two would silently slip on exactly the characters German cares
    about most.
    """
    tokens = list(_TOKEN.finditer(plain))
    widths = sorted({len(key.split()) for key in keys}, reverse=True)

    found: list[str] = []
    taken: set[int] = set()
    for width in widths:
        for start in range(len(tokens) - width + 1):
            if any(index in taken for index in range(start, start + width)):
                continue
            span = plain[tokens[start].start() : tokens[start + width - 1].end()]
            if fold(" ".join(span.split())) in keys:
                found.append(span)
                taken.update(range(start, start + width))
    return found


def structural_terms(node: Node, plain: str) -> list[str]:
    """Rung 2. Empty when the node has asserted nothing specific enough to act on."""
    if node.type == "vocab":
        return _match_keys(plain, _lemma_keys(node))
    if node.type == "grammar":
        return _match_keys(plain, governed_prepositions(node.body_md))
    # phrase / pattern / culture: no structural claim a rule can read. Rung 3.
    return []


def cloze_targets(node: Node, german: str) -> ClozeTargets | None:
    """The deletion for one example, or ``None`` when there is no honest one."""
    if bold := bold_terms(german):
        return ClozeTargets(terms=bold, source="bold")
    if structural := structural_terms(node, strip_emphasis(german)):
        return ClozeTargets(terms=structural, source="structure")
    return None
