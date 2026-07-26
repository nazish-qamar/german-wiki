"""Ingestion: raw text in, reviewable candidate nodes out (SPEC §2, §11 slice 3).

This package is the **only** import surface for ingestion; internals are
``_``-prefixed, matching the ``llm`` package's discipline.

The shape of this slice is set by ADR-003: nothing auto-writes to ``/nodes``.
``ingest_file`` writes complete, loadable node files to ``/queue``; ``promote_source``
is the only path that writes into ``/nodes``, and the only caller anywhere that
passes ``learn=True`` to grow the tag vocabulary (ADR-007). Manual review sits
between them -- rejecting a candidate is deleting its queue file. Slice 5 wraps that
same seam with LangGraph adjudication.

No merging and no dedup here: every candidate becomes a node (SPEC §11).
"""

from __future__ import annotations

from ._extract import Candidate, ExtractionError
from ._ingest import IngestResult, ingest_file
from ._nodes import list_queue
from ._promote import PromoteResult, Refusal, promote_source

__all__ = [
    "Candidate",
    "ExtractionError",
    "IngestResult",
    "PromoteResult",
    "Refusal",
    "ingest_file",
    "list_queue",
    "promote_source",
]
