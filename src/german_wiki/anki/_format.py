"""The two dials: which cards a node type produces, and which nodes are exported at all.

Both values were chosen before the deck had been studied even once, which is the entire
reason they are named constants in a module you can find rather than conditionals buried in
the builder. The expectation is that they move -- after a week of reviews you will know
whether a `pattern` node is better as a rule card or as nothing, and changing that should be
one line, not an archaeology exercise.

That is the same reasoning behind `embed.GRAY_LOW`, `merge.MAX_REGENERATIONS` and
`ingest.MIN_PAGE_CHARS`, and the same reasoning slice 8 recorded for its two PDF thresholds:
a guess that is easy to find is a guess that gets corrected.
"""

from __future__ import annotations

from typing import Literal, get_args

from ..models import CEFR, NodeType, Status

# What a generator makes. `rule_basic` is one front/back card per node; `examples_cloze` is
# one cloze card per example sentence that has an honest deletion (see `_targets`).
Generator = Literal["rule_basic", "examples_cloze"]

# Format follows node type. A node type may run several generators -- `grammar` gets both,
# because the rule and its applications are genuinely different recall acts: knowing that
# Wechselpräpositionen take Akkusativ for movement is not the same as producing `auf den`
# in a sentence under time pressure.
#
# Every NodeType literal must appear here. `test_anki_cards.py` asserts that against
# `models.NodeType`, so adding a sixth type fails there rather than raising KeyError
# halfway through an export.
CARD_FORMATS: dict[NodeType, tuple[Generator, ...]] = {
    "vocab": ("examples_cloze",),
    "phrase": ("examples_cloze",),
    "grammar": ("rule_basic", "examples_cloze"),
    "pattern": ("rule_basic",),
    "culture": ("rule_basic",),
}

# Which nodes reach the deck. `draft` is excluded because it has not been through review,
# and an unlearned card is cheaper to add later than a wrong one is to unlearn -- `gw anki
# --include-draft` overrides it without a code edit.
EXPORT_STATUSES: tuple[Status, ...] = ("reviewed", "stable")

# Kept as a tuple in CEFR order so a future `--max-level` can slice it. Not used as a filter
# yet: with six nodes there is nothing to filter, and SPEC §5.1's priority scoring is the
# feature that will actually want it, once its inputs exist.
CEFR_ORDER: tuple[CEFR, ...] = get_args(CEFR)


def generators_for(node_type: NodeType) -> tuple[Generator, ...]:
    """The generators to run for a node type. Raises on an unmapped type, deliberately.

    A missing mapping is a bug in `CARD_FORMATS`, and defaulting it to "no cards" would
    make a whole node type silently vanish from the deck.
    """
    return CARD_FORMATS[node_type]
