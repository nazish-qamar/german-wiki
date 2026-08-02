"""Segmentation: split a verb only when the corpus vouches for the stem (SPEC §7.4).

``_prefixes.candidates`` will happily report that ``verstehen`` is ``ver-`` + ``stehen``,
because mechanically it is. SPEC §7.4 is a warning about exactly that::

    `verstehen` has nothing to do with `stehen` + directional `ver-`. `bekommen` ≠
    `be-` + `kommen` in any useful sense. Meanings have drifted centuries past
    transparency.

So this module adds the constraint that turns a *possible* split into a *proposable* one:
**the residual stem must already exist in your corpus**, as a family node's ``root:`` or
as one of its ``lemmas``. Prefix-shaped starts are never blindly stripped.

That constraint does real work in both directions:

- ``verstehen`` does not split, because you have no ``stehen`` node. If you later write
  one, it becomes splittable -- and at that point a human is looking at it, and
  ``_transparency`` will judge it ``opaque``, so the grid marks the cell irregular rather
  than teaching a false family. Two safeguards, in series.
- The grid grows *as you study* rather than speculating about vocabulary you have never
  met, which is also what keeps ``gw families`` from proposing hundreds of edges into
  empty space.

The second refusal is stress: a ``VARIABLE`` prefix yields no segmentation at all, because
``úmfahren`` and ``umfáhren`` are spelled identically (see ``_prefixes``). The one thing
that overrides it is a human writing ``separable:`` on the node -- §7.4's "the node holds
the truth", applied to segmentation as well as to meaning.

Nothing here writes, proposes, or calls a model. It answers one question about one word.
"""

from __future__ import annotations

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..models import Node
from ._prefixes import Separability, candidates

# Stripped before a lemma is indexed, so "die Wäsche" and "sich waschen" are findable by
# their bare forms. Articles and the reflexive pronoun are inflection, not the word.
_LEMMA_NOISE = ("der ", "die ", "das ", "sich ")

WithheldReason = Literal["no-prefix", "variable-stress", "no-corpus-evidence"]


def fold(text: str) -> str:
    """NFC + casefold. The same normalization the corpus index and the lookups use.

    NFC for the ADR-012 reason: ``ä`` has two encodings that render identically, and a
    stem written in one form would silently fail to match a root written in the other.
    """
    return unicodedata.normalize("NFC", text).strip().casefold()


def _lemma_key(lemma: str) -> str:
    folded = fold(lemma)
    for noise in _LEMMA_NOISE:
        folded = folded.removeprefix(noise)
    return folded.strip()


class Segmentation(BaseModel):
    """A split the corpus supports. Still a *proposal input*, not a written edge."""

    model_config = ConfigDict(extra="forbid")

    word: str
    prefix: str
    stem: str
    separability: Separability
    # Which node vouched for the stem -- the evidence, carried so a proposal can cite it.
    stem_node_id: str
    # True when a human's explicit ``separable:`` resolved a stress-ambiguous prefix.
    resolved_by_node: bool = False


class Withheld(BaseModel):
    """No split, and *why* -- the reason drives whether the grid says anything at all."""

    model_config = ConfigDict(extra="forbid")

    word: str
    reason: WithheldReason
    prefix: str | None = None
    stem: str | None = None

    @property
    def needs_human(self) -> bool:
        """Whether a person could resolve this, as opposed to more study resolving it.

        ``variable-stress`` is the only one: no amount of ingesting fixes a distinction
        the spelling does not carry, so these surface in ``gw gaps --ambiguous``.
        ``no-corpus-evidence`` resolves by itself the day the stem gets a node.
        """
        return self.reason == "variable-stress"


class CorpusIndex(BaseModel):
    """Stems the corpus can vouch for, mapped to the node that provides them."""

    model_config = ConfigDict(extra="forbid")

    stems: dict[str, str] = {}

    @classmethod
    def build(cls, nodes: list[Node]) -> CorpusIndex:
        """Index every ``root:`` and every lemma. Roots win a collision -- they are the
        node's own claim about what it is a family of."""
        stems: dict[str, str] = {}
        for node in nodes:
            for lemma in node.lemmas or []:
                if key := _lemma_key(lemma):
                    stems.setdefault(key, node.id)
        for node in nodes:  # second pass so roots overwrite lemma-derived entries
            if node.root and (key := fold(node.root)):
                stems[key] = node.id
        return cls(stems=stems)

    def vouches_for(self, stem: str) -> str | None:
        return self.stems.get(fold(stem))


def segment(
    word: str,
    *,
    corpus: CorpusIndex,
    declared_separable: bool | None = None,
) -> Segmentation | Withheld:
    """Split ``word``, or explain why not.

    ``declared_separable`` is the node's own ``separable:`` field when it has one. It is
    the **only** thing that unlocks a stress-ambiguous prefix, because it means a person
    who knows which word this is has already answered the question the spelling cannot.
    """
    options = candidates(word)
    if not options:
        return Withheld(word=word, reason="no-prefix")

    # Longest prefix first, so `vorbeikommen` is tried as vorbei+kommen before vor+beikommen.
    withheld: Withheld | None = None
    for prefix, stem, separability in options:
        stem_node = corpus.vouches_for(stem)
        if stem_node is None:
            withheld = withheld or Withheld(
                word=word, reason="no-corpus-evidence", prefix=prefix, stem=stem
            )
            continue

        if separability == "variable" and declared_separable is None:
            # The corpus vouches for the stem, so this WOULD grid -- and that is exactly
            # when withholding matters. Reported ahead of any weaker refusal below.
            return Withheld(word=word, reason="variable-stress", prefix=prefix, stem=stem)

        resolved = separability == "variable"
        return Segmentation(
            word=word,
            prefix=prefix,
            stem=stem,
            separability="separable"
            if resolved and declared_separable
            else "inseparable"
            if resolved
            else separability,
            stem_node_id=stem_node,
            resolved_by_node=resolved,
        )

    return withheld or Withheld(word=word, reason="no-corpus-evidence")
