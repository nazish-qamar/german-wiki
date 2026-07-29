"""Turning nodes into vectors, cache-first.

Every vector is looked up in the disk cache before the model is touched, and the
model is only loaded at all if something is actually missing. That is what lets
``gw reindex`` repopulate the vector table without importing torch, and what makes
a second ``gw embed`` report "0 new".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..db import EMBEDDING_DIM
from ..models import Node
from . import _cache, _store
from ._model import Embedder, check_dimension, embedding_model_name
from ._text import embed_text


class EmbedResult(BaseModel):
    """What one embedding pass did. ``computed`` is the only part that cost time."""

    model_config = ConfigDict(extra="forbid")

    model: str
    total: int = 0
    computed: int = 0
    from_cache: int = 0
    stored: int = 0


def _resolve_model(
    embedder: Embedder | None, model: str | None, settings_path: Path | str | None
) -> str:
    if model is not None:
        return model
    if embedder is not None:
        return embedder.model
    return embedding_model_name(settings_path=settings_path)


def vectors_for(
    nodes: list[Node],
    *,
    embedder: Embedder | None = None,
    model: str | None = None,
    cache_dir: Path | str | None = None,
    settings_path: Path | str | None = None,
    compute: bool = True,
) -> tuple[dict[str, list[float]], EmbedResult]:
    """Vectors for ``nodes``, served from cache and computed only where missing.

    ``compute=False`` returns whatever the cache already holds and never loads a
    model -- the mode ``gw reindex`` uses.
    """
    model_name = _resolve_model(embedder, model, settings_path)
    result = EmbedResult(model=model_name, total=len(nodes))

    vectors: dict[str, list[float]] = {}
    missing: list[tuple[str, str]] = []  # (node_id, text)

    for node in nodes:
        text = embed_text(node)
        key = _cache.cache_key(model_name, text)
        hit = _cache.read(model_name, key, cache_dir=cache_dir, expect_dim=EMBEDDING_DIM)
        if hit is not None:
            vectors[node.id] = hit
            result.from_cache += 1
        else:
            missing.append((node.id, text))

    if missing and compute:
        if embedder is None:
            from ._model import load_embedder  # lazy: importing torch is expensive

            embedder = load_embedder(settings_path=settings_path)
        check_dimension(embedder.dimension, embedder.model)

        computed = embedder.encode([text for _, text in missing])
        for (node_id, text), vector in zip(missing, computed, strict=True):
            vectors[node_id] = vector
            _cache.write(
                model_name, _cache.cache_key(model_name, text), vector, cache_dir=cache_dir
            )
            result.computed += 1

    return vectors, result


def embed_nodes(
    nodes: list[Node],
    *,
    conn: sqlite3.Connection | None = None,
    embedder: Embedder | None = None,
    model: str | None = None,
    cache_dir: Path | str | None = None,
    settings_path: Path | str | None = None,
    compute: bool = True,
) -> EmbedResult:
    """Embed ``nodes`` and store their vectors in the index."""
    vectors, result = vectors_for(
        nodes,
        embedder=embedder,
        model=model,
        cache_dir=cache_dir,
        settings_path=settings_path,
        compute=compute,
    )
    if conn is not None and vectors:
        result.stored = _store.store_vectors(conn, vectors)
    return result
