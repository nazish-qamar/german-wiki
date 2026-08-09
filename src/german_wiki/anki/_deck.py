"""The genanki binding: note types, templates, and writing the ``.apkg``.

The only module in the package that imports genanki, so the dependency has exactly one
edge and everything interesting about the export stays testable without it.

**The ids below are constants and must stay constants.** genanki identifies a note type and
a deck by integer id, and the documented failure -- the one every genanki tutorial warns
about after the fact -- is generating them at runtime: each export then declares a *new*
note type, so Anki imports the cards as unrelated notes and the previous ones become
orphans carrying all of your review history. Randomising these would silently undo the
reason `.apkg` was chosen over TSV in the first place.
"""

from __future__ import annotations

from pathlib import Path

import genanki

from ._cards import Card

DECK_NAME = "German Wiki"

# Fixed forever. See the module docstring. Values are in genanki's recommended range
# [2**30, 2**31); they are arbitrary but they are not allowed to change.
DECK_ID = 1928374650
CLOZE_MODEL_ID = 1928374651
BASIC_MODEL_ID = 1928374652

_CSS = """\
.card {
  font-family: -apple-system, Segoe UI, Roboto, sans-serif;
  font-size: 20px;
  text-align: left;
  color: #1a1a1a;
  background-color: #fdfdfd;
  padding: 16px;
}
.cloze { font-weight: bold; color: #0b6bcb; }
.prompt { font-size: 17px; color: #444; margin-top: 10px; }
.note { font-size: 17px; color: #444; margin-top: 10px; }
.source { font-size: 13px; color: #999; margin-top: 14px; }
table { border-collapse: collapse; margin-top: 8px; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: left; }
th { background: #f2f2f2; }
"""

# `Prompt` is wrapped in a conditional section so it appears on the FRONT only when it is
# non-empty -- which `_cards` arranges to be exactly when the deletion covers the whole
# sentence and the front would otherwise be a bare `[...]`. For a partial cloze it stays
# empty, keeping the gloss off the front where it would give the answer away.
CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    "German Wiki — Cloze",
    fields=[{"name": "Text"}, {"name": "Prompt"}, {"name": "Note"}, {"name": "Source"}],
    templates=[
        {
            "name": "Cloze",
            "qfmt": "{{cloze:Text}}{{#Prompt}}<div class='prompt'>{{Prompt}}</div>{{/Prompt}}",
            "afmt": (
                "{{cloze:Text}}"
                "{{#Note}}<div class='note'>{{Note}}</div>{{/Note}}"
                "<div class='source'>{{Source}}</div>"
            ),
        }
    ],
    css=_CSS,
    model_type=genanki.Model.CLOZE,
)

BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    "German Wiki — Rule",
    fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Source"}],
    templates=[
        {
            "name": "Rule",
            "qfmt": "{{Front}}",
            "afmt": ("{{FrontSide}}<hr id=answer>{{Back}}<div class='source'>{{Source}}</div>"),
        }
    ],
    css=_CSS,
)

_MODELS = {"cloze": CLOZE_MODEL, "basic": BASIC_MODEL}


def _note(card: Card) -> genanki.Note:
    return genanki.Note(
        model=_MODELS[card.kind],
        fields=card.fields,
        tags=card.tags,
        # The explicit guid is the whole point: it is what lets Anki recognise a note it
        # already has and update it, rather than inserting a duplicate.
        guid=card.guid,
    )


def write_deck(cards: list[Card], path: Path) -> Path:
    """Write ``cards`` to an ``.apkg`` at ``path``, creating parent directories."""
    deck = genanki.Deck(DECK_ID, DECK_NAME)
    # Sorted by guid so two runs over the same corpus emit notes in the same order; the
    # zip's own timestamps still differ, so the file is not byte-identical, but its
    # contents are.
    for card in sorted(cards, key=lambda card: card.guid):
        deck.add_note(_note(card))

    path.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(path))
    return path
