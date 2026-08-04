"""Ingestion: raw text in, reviewable candidate nodes out (SPEC §2, §11 slice 3).

This package is the **only** import surface for ingestion; internals are
``_``-prefixed, matching the ``llm`` package's discipline.

The shape of this slice is set by ADR-003: nothing auto-writes to ``/nodes``.
``ingest_file`` writes complete, loadable node files to ``/queue``; ``write_approved``
is the only function that writes into ``/nodes``, and the only caller anywhere that
passes ``learn=True`` to grow the tag vocabulary (ADR-007). Manual review sits
between them -- rejecting a candidate is deleting its queue file.

Slice 5 wraps that same seam rather than replacing it: ``german_wiki.merge`` routes
every approved merge, link and create through ``write_approved`` too, so there is
still exactly one door into ``/nodes`` no matter which command opened it.

No merging and no dedup here: every candidate becomes a node (SPEC §11).
"""

from __future__ import annotations

from ._extract import MAX_CANDIDATES, Candidate, ExtractionError
from ._ingest import (
    Confirm,
    IngestResult,
    OcrRejected,
    PdfIngestResult,
    ingest_file,
    ingest_pdf,
)
from ._nodes import list_queue
from ._pdf import (
    MIN_PAGE_CHARS,
    MIN_TEXT_PAGE_RATIO,
    PdfError,
    PdfExtraction,
    extract_pages,
    is_pdf,
)
from ._promote import PromoteResult, Refusal, promote_source, write_approved
from ._raw import read_raw_text
from ._vision import MAX_IMAGE_BYTES, VisionError, is_image, transcribe

# The dials, on the public surface where `embed.GRAY_LOW` and `merge.MAX_REGENERATIONS`
# live. Slice 8's are guesses made before seeing many real German textbook PDFs, so they
# are the ones most likely to need moving -- which is exactly why they belong somewhere
# you would think to look rather than inside a private module.
__all__ = [
    "MAX_CANDIDATES",
    "MAX_IMAGE_BYTES",
    "MIN_PAGE_CHARS",
    "MIN_TEXT_PAGE_RATIO",
    "Candidate",
    "Confirm",
    "ExtractionError",
    "IngestResult",
    "OcrRejected",
    "PdfError",
    "PdfExtraction",
    "PdfIngestResult",
    "PromoteResult",
    "Refusal",
    "VisionError",
    "extract_pages",
    "ingest_file",
    "ingest_pdf",
    "is_image",
    "is_pdf",
    "list_queue",
    "promote_source",
    "read_raw_text",
    "transcribe",
    "write_approved",
]
