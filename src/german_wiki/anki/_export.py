"""Nodes in, one ``.apkg`` out, plus an honest account of what did not make it.

The report is not decoration. Every skip here names something only you can fix, in the node
itself -- an examples section that does not exist, or a sentence with nothing marked as the
thing being taught. So the export doubles as a review of the notes, in the same spirit as
SPEC §7.3's grid, where the empty cells are the point.

The counterpart to that honesty is the one thing this cannot do: an ``.apkg`` import adds
and updates but never *deletes*. Removing an example from a node leaves its card behind in
your collection, so removals are reported too and every note carries a ``node::<id>`` tag
to make the orphan findable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..models import Node, Status
from ._cards import Card, Skip, cards_for
from ._deck import write_deck
from ._format import EXPORT_STATUSES

DEFAULT_EXPORT_PATH = Path("exports") / "german-wiki.apkg"


class ExcludedNode(BaseModel):
    """A node held back by the status filter, not by anything wrong with it."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    status: Status


class ExportResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    cards: list[Card] = []
    skips: list[Skip] = []
    excluded: list[ExcludedNode] = []

    @property
    def node_ids(self) -> set[str]:
        return {card.node_id for card in self.cards}

    @property
    def guids(self) -> set[str]:
        return {card.guid for card in self.cards}


def build(nodes: list[Node], *, statuses: tuple[Status, ...] = EXPORT_STATUSES) -> ExportResult:
    """Turn nodes into cards. Writes nothing -- the dry-run half of an export."""
    result = ExportResult()
    for node in sorted(nodes, key=lambda node: node.id):
        if node.status not in statuses:
            result.excluded.append(ExcludedNode(node_id=node.id, status=node.status))
            continue
        cards, skips = cards_for(node)
        result.cards += cards
        result.skips += skips
    return result


def export(
    nodes: list[Node],
    path: Path = DEFAULT_EXPORT_PATH,
    *,
    statuses: tuple[Status, ...] = EXPORT_STATUSES,
) -> ExportResult:
    """Build and write. Returns the same report ``build`` would, plus the written path."""
    result = build(nodes, statuses=statuses)
    write_deck(result.cards, path)
    result.path = path
    return result
