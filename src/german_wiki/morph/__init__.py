"""Morphological grids: root × prefix, and the gaps between (SPEC §7, §11 slice 7).

This package is the **only** import surface for morphology; internals are ``_``-prefixed,
matching the ``llm``, ``ingest``, ``embed``, ``merge`` and ``level`` packages.

SPEC §7 calls the root × prefix grid "the single highest-leverage structure in German
vocabulary": learn that ``an-`` means toward/on once, and every root's family becomes
semi-predictable. The grid also doubles as gap detection (§7.3) -- **empty cells are words
you have not learned yet.**

Three ideas carry the slice.

**Rules segment; the model judges only transparency.** Splitting ``abwaschen`` into
``ab-`` + ``waschen`` is mechanical, so ``_segment`` does it with a closed prefix
inventory and one hard constraint: the residual stem must have *corpus evidence*. Whether
a family is a real learning scaffold or a centuries-drifted coincidence
(``verstehen`` ≠ ``ver-`` + ``stehen``, §7.4) is semantic and not rule-decidable, so that
alone costs one free ``glm-4.5-flash`` call in ``_transparency``.

**A human may dangle; the pipeline may not.** Every link in the corpus currently points at
a node that does not exist, and that is the feature: a hand-written ``target: ankommen`` is
an *intention*, and §7.3 makes it the gap signal. The same edge written by the pipeline
would be a bug, so slice 5's ``apply_link`` keeps its refusal. Same artifact, opposite
rules, keyed on who wrote it -- the same shape as ADR-007's ``learn=True``.

**Dangling-ness is computed, never stored.** ``_grid`` asks "does this target exist right
now?" at read time. There is no flag to clear, so the day you write the ``ankommen`` node
its cell simply reads as learned, with no migration and nothing to maintain.

Writes go through ``/proposals`` and ``gw review`` like everything else (ADR-003).
SPEC §7.2's ingest-time auto-creation is deliberately **not** here: it is a write path
whose correctness depends on segmentation and transparency being trustworthy, and that has
to be watched on real material first (ADR-014).
"""

from __future__ import annotations

from ._analyse import MORPH_SOURCE_ID, Analysis, analyse
from ._grid import (
    Cell,
    CellState,
    Grid,
    PrefixAxis,
    RootAxis,
    build_grid,
    dangling_targets,
    is_family_node,
    is_prefix_node,
    morpheme_of,
)
from ._prefixes import INSEPARABLE, SEPARABLE, VARIABLE, Separability, candidates, classify
from ._segment import CorpusIndex, Segmentation, Withheld, segment
from ._transparency import TRUSTED, Transparency, TransparencyError, judge

__all__ = [
    "INSEPARABLE",
    "MORPH_SOURCE_ID",
    "SEPARABLE",
    "TRUSTED",
    "VARIABLE",
    "Analysis",
    "Cell",
    "CellState",
    "CorpusIndex",
    "Grid",
    "PrefixAxis",
    "RootAxis",
    "Segmentation",
    "Separability",
    "Transparency",
    "TransparencyError",
    "Withheld",
    "analyse",
    "build_grid",
    "candidates",
    "classify",
    "dangling_targets",
    "is_family_node",
    "is_prefix_node",
    "judge",
    "morpheme_of",
    "segment",
]
