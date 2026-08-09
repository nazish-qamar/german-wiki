"""Anki export (SPEC §8.1): nodes become cards, and an existing scheduler does the rest.

*"Nodes -> cards; let an existing scheduler do spaced repetition. Do not build your own
SRS. Cloze-delete the example sentences."*

This package is **read-only with respect to `/nodes`** -- the first feature since slice 3
for which that is true, which is why ADR-003's review gate appears nowhere in it. Its output
is a derived artifact in `exports/`, rebuildable at any time and gitignored for the same
reason `data/index.db` is.

Three things are worth knowing before changing anything here:

- **`CARD_FORMATS` and `EXPORT_STATUSES` are the dials**, published here by the convention
  `embed.GRAY_LOW` and `merge.MAX_REGENERATIONS` set. Both were chosen before the deck had
  been studied once and are expected to move.
- **`_cards.guid_fold` is the opposite of a dial.** Note identity must never change, because
  changing it re-identifies every note and discards the review history the `.apkg` format
  exists to preserve.
- **The export can add and update, never delete.** Removing an example leaves an orphan
  card; the `node::<id>` tag is how you find it.
"""

from __future__ import annotations

from ._cards import Card, Skip, card_tags, cards_for, guid_for, rule_card
from ._deck import BASIC_MODEL_ID, CLOZE_MODEL_ID, DECK_ID, DECK_NAME, write_deck
from ._export import DEFAULT_EXPORT_PATH, ExcludedNode, ExportResult, build, export
from ._format import CARD_FORMATS, CEFR_ORDER, EXPORT_STATUSES, Generator, generators_for
from ._html import to_html
from ._targets import PREPOSITIONS, ClozeTargets, cloze_targets, strip_emphasis

__all__ = [
    "BASIC_MODEL_ID",
    # The dials. Expected to move once the deck has been studied for a week.
    "CARD_FORMATS",
    "CEFR_ORDER",
    "CLOZE_MODEL_ID",
    "DECK_ID",
    "DECK_NAME",
    "DEFAULT_EXPORT_PATH",
    "EXPORT_STATUSES",
    "PREPOSITIONS",
    "Card",
    "ClozeTargets",
    "ExcludedNode",
    "ExportResult",
    "Generator",
    "Skip",
    "build",
    "card_tags",
    "cards_for",
    "cloze_targets",
    "export",
    "generators_for",
    "guid_for",
    "rule_card",
    "strip_emphasis",
    "to_html",
    "write_deck",
]
