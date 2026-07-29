"""Local embeddings and duplicate detection (SPEC §3, §11 slice 4).

This package is the **only** import surface for embedding and dedup; internals are
``_``-prefixed, matching the ``llm`` and ``ingest`` packages.

Embeddings are always computed **in-process** via sentence-transformers and never
through an API (ADR-004) -- ``llm.complete()`` refuses the ``embeddings`` step by
design, and nothing here calls it. ``sentence_transformers`` is imported lazily, so
importing this package costs nothing and ``gw list`` never pays for torch.

Slice 4 **detects and reports only**. Nothing here writes to ``/nodes`` or
``/queue``; merging and adjudication are slice 5. The one thing it does write is
vectors into the derived SQLite index, which ADR-001 makes freely rebuildable --
a different layer from the source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._cache import clear as _cache_clear
from ._cache import stats as _cache_stats
from ._detect import (
    DEFAULT_K,
    GRAY_HIGH,
    GRAY_LOW,
    NEAR_EXACT_JACCARD,
    DuplicateReport,
    Match,
    find_duplicates,
)
from ._embed import EmbedResult, embed_nodes
from ._model import Embedder, embedding_model_name, load_embedder

__all__ = [
    "DEFAULT_K",
    "GRAY_HIGH",
    "GRAY_LOW",
    "NEAR_EXACT_JACCARD",
    "DuplicateReport",
    "EmbedResult",
    "Embedder",
    "Match",
    "cache_clear",
    "cache_stats",
    "embed_nodes",
    "embedding_model_name",
    "find_duplicates",
    "load_embedder",
]


def cache_stats(*, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Entry count, total bytes and mtime bounds of the embedding cache."""
    return _cache_stats(cache_dir=cache_dir)


def cache_clear(*, cache_dir: Path | str | None = None, older_than_days: int | None = None) -> int:
    """Remove cached vectors, returning how many. Only ever costs CPU to regenerate."""
    return _cache_clear(cache_dir=cache_dir, older_than_days=older_than_days)
