"""Nodes to cards. Pure -- no genanki import, no filesystem, no model calls.

Kept separate from ``_deck`` so the interesting logic (what gets blanked, what a note's
identity is, what is skipped and why) is testable without the export dependency, and so the
dependency itself touches exactly one module.

**Note identity is the load-bearing decision here**, because it is what makes re-export a
sync rather than a duplication. See ``guid_for`` below; the short version is that a note is
identified by *what it teaches*, not by where it sits in a list, because
``merge.new_examples`` appends to the Examples section on every SAME merge and a positional
identity would discard your review history each time that happened.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..bodytext import Example, parse_examples
from ..models import Node
from ._format import generators_for
from ._html import to_html
from ._targets import cloze_targets, strip_emphasis

CardKind = Literal["cloze", "basic"]

# Why an example produced no card. Surfaced by `gw anki` so the export doubles as a report
# on the notes themselves -- both reasons are things only you can fix, in the node.
SkipReason = Literal["no-examples", "no-cloze-target", "no-prompt-for-full-cloze"]

_WHITESPACE = re.compile(r"\s+")


class Card(BaseModel):
    """One note, ready to hand to genanki. ``fields`` is ordered per its note type."""

    model_config = ConfigDict(extra="forbid")

    guid: str
    kind: CardKind
    fields: list[str]
    tags: list[str]
    node_id: str


class Skip(BaseModel):
    """Something that could have been a card and deliberately was not."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    reason: SkipReason
    detail: str = ""


def guid_fold(text: str) -> str:
    """Normalize text for note identity.

    **This function must never change.** Its output feeds every GUID in the deck, so any
    edit re-identifies every note: Anki would import them as new cards and silently discard
    the scheduling history the .apkg format was chosen to preserve. If a different fold is
    ever wanted, it belongs to a new deck, not to this one.

    Deliberately *not* shared with `merge._fold` or `morph.fold`, which look similar and
    answer different questions (the same separation ADR-010 drew for `embed`'s two
    normalizations). Those are free to be tuned as their callers learn; this one is not,
    and coupling them would mean a merge-sensitivity tweak quietly orphaning your deck.

    NFC per ADR-012 (`ä` has two encodings), casefold, and alphanumerics only -- so
    re-emphasising a sentence or fixing its punctuation keeps the card and its history.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in folded).split())


def guid_for(node_id: str, discriminator: str) -> str:
    """A stable note id from what the note teaches.

    The discriminator is the folded sentence for a cloze note and the literal ``"rule"``
    for a node's single front/back card -- never an index. Position would break on insert
    and reorder, and `merge.new_examples()` appends to the Examples section on every SAME
    merge, so insertion is a routine pipeline event rather than a rare hand edit.
    """
    digest = hashlib.sha1(f"{node_id}\x00{discriminator}".encode()).hexdigest()
    return digest[:32]


def _tag(value: str) -> str:
    """Anki tags are space-delimited, so a tag may not contain whitespace."""
    return _WHITESPACE.sub("_", value.strip())


def card_tags(node: Node, example: Example | None = None) -> list[str]:
    """Hierarchical tags, so one deck stays filterable by level, register and theme.

    ``node::<id>`` is SPEC §8.1's provenance requirement on the card itself: when a card
    looks wrong mid-review you need to reach the node that produced it -- and it is also
    how an orphaned card is found after its example is deleted, since an .apkg import can
    add and update but never delete.
    """
    tags = [
        _tag(f"cefr::{node.cefr}"),
        _tag(f"type::{node.type}"),
        _tag(f"node::{node.id}"),
    ]
    tags += [_tag(f"register::{value}") for value in node.register or []]
    tags += [_tag(f"theme::{value}") for value in node.themes or []]
    if example:
        tags += [_tag(f"ctx::{value}") for value in example.tags]
    # Sorted and de-duplicated so a re-export produces byte-identical notes.
    return sorted(set(tags))


def _source_field(node: Node) -> str:
    return f"{node.title_de} ({node.id})"


def _cloze_text(plain: str, terms: list[str]) -> str:
    """Wrap every target with ``{{c1::…}}`` in ONE pass.

    All terms share ``c1`` on purpose: bold marks *the construction being taught*, and
    `**Ich wäre Ihnen dankbar, wenn Sie** … **zusenden könnten**` is a single Konjunktiv II
    frame split around the middle field. Numbering them c1/c2 would generate a card that
    tests half a construction. Two cards means writing two examples.

    One pass, longest term first, so overlapping targets can never be double-wrapped.
    """
    ordered = sorted(set(terms), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(term) for term in ordered))
    return pattern.sub(lambda match: "{{c1::" + match.group(0) + "}}", plain)


def _covers_everything(plain: str, terms: list[str]) -> bool:
    return guid_fold(plain) == guid_fold(" ".join(terms))


def cloze_cards(node: Node) -> tuple[list[Card], list[Skip]]:
    """One cloze note per example that has an honest deletion."""
    examples = parse_examples(node.body_md)
    if not examples:
        return [], [Skip(node_id=node.id, reason="no-examples")]

    cards: list[Card] = []
    skips: list[Skip] = []
    for example in examples:
        plain = strip_emphasis(example.german)
        targets = cloze_targets(node, example.german)
        if targets is None:
            skips.append(Skip(node_id=node.id, reason="no-cloze-target", detail=plain))
            continue

        # A deletion covering the whole sentence leaves a front of bare `[...]`, so the
        # gloss becomes the prompt and the card is an EN->DE production card instead.
        # Without a gloss there is no front at all, and a card with nothing on its front
        # is not a card.
        full = _covers_everything(plain, targets.terms)
        if full and not example.note:
            skips.append(Skip(node_id=node.id, reason="no-prompt-for-full-cloze", detail=plain))
            continue

        cards.append(
            Card(
                guid=guid_for(node.id, guid_fold(plain)),
                kind="cloze",
                # Text, Prompt, Note, Source -- order matches the note type in `_deck`.
                fields=[
                    _cloze_text(plain, targets.terms),
                    # Prompt: front-side, and ONLY when the deletion hides everything.
                    # For a partial cloze it must stay empty or the gloss sits next to
                    # `Ich {{c1::wasche}} mir die Hände` and gives the answer away.
                    example.note or "" if full else "",
                    example.note or "",
                    _source_field(node),
                ],
                tags=card_tags(node, example),
                node_id=node.id,
            )
        )
    return cards, skips


def rule_card(node: Node) -> Card:
    """The node's own front/back card: what is this, and what does it say."""
    back = to_html(node.body_md)
    if node.title_en:
        back = f"<i>{node.title_en}</i><br>{back}"
    return Card(
        # "rule" rather than a hash of the body, so editing a rule UPDATES its card and
        # keeps the review history instead of retiring it and starting the interval over.
        guid=guid_for(node.id, "rule"),
        kind="basic",
        fields=[node.title_de, back, _source_field(node)],  # Front, Back, Source
        tags=card_tags(node),
        node_id=node.id,
    )


def cards_for(node: Node) -> tuple[list[Card], list[Skip]]:
    """Every card a node produces, per `CARD_FORMATS`."""
    cards: list[Card] = []
    skips: list[Skip] = []
    for generator in generators_for(node.type):
        if generator == "rule_basic":
            cards.append(rule_card(node))
        elif generator == "examples_cloze":
            produced, skipped = cloze_cards(node)
            cards += produced
            skips += skipped
    return cards, skips
