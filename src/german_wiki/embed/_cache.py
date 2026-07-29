"""Disk cache for embedding vectors -- ADR-005's principle, applied locally.

Local embeddings cost no money, but they cost *time*: a full corpus re-encode on
CPU is the difference between ``gw embed`` being instant and being a coffee break.
Same reasoning as the model-call cache, same failure posture: a corrupt or
mismatched entry logs a warning, is removed, and degrades to a miss. A broken
cache never breaks a run.

**The model name is part of the key**, so two embedding models never collide and
can be A/B'd against the same corpus without clearing anything.

This is *recompute-avoidance*. The sqlite-vec table is *query structure*. They are
deliberately separate: the cache survives ``gw reindex`` (which drops every table),
which is exactly what lets reindex reload vectors without loading a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import config
from ..logutil import get_logger

logger = get_logger(__name__)

# Bump to invalidate every cached vector.
KEY_VERSION = 1

SUBDIR = "embeddings"

REQUIRED_KEYS = frozenset({"key", "model", "dim", "vector"})


def _model_slug(model: str) -> str:
    """Filesystem-safe directory name for a model id like ``intfloat/e5-small``."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "model"


def cache_key(model: str, text: str) -> str:
    """sha256 over canonical JSON of (model, exact model input).

    ``text`` is what actually gets encoded, ``query: `` prefix included, so
    "identical input never re-computes" is literally true rather than approximately.
    """
    material = {"v": KEY_VERSION, "model": model, "text": text}
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _root(cache_dir: Path | str | None) -> Path:
    return (Path(cache_dir) if cache_dir is not None else config.CACHE_DIR) / SUBDIR


def entry_path(model: str, key: str, *, cache_dir: Path | str | None = None) -> Path:
    return _root(cache_dir) / _model_slug(model) / key[:2] / f"{key}.json"


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - platform-specific
        logger.warning("could not remove cache entry %s: %s", path, exc)


def read(
    model: str,
    key: str,
    *,
    cache_dir: Path | str | None = None,
    expect_dim: int | None = None,
) -> list[float] | None:
    """Return the cached vector, or ``None`` for any kind of miss."""
    path = entry_path(model, key, cache_dir=cache_dir)
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("discarding unreadable embedding cache entry %s: %s", path, exc)
        _discard(path)
        return None

    if not isinstance(payload, dict) or not REQUIRED_KEYS.issubset(payload):
        logger.warning("discarding malformed embedding cache entry %s", path)
        _discard(path)
        return None

    vector = payload["vector"]
    if expect_dim is not None and len(vector) != expect_dim:
        # A stale entry from a differently-shaped model. Storing this into a
        # fixed-width vec0 column would fail or, worse, be accepted quietly.
        logger.warning(
            "discarding embedding cache entry %s: %d dimensions, expected %d",
            path,
            len(vector),
            expect_dim,
        )
        _discard(path)
        return None

    return vector


def write(
    model: str,
    key: str,
    vector: list[float],
    *,
    cache_dir: Path | str | None = None,
) -> None:
    """Write one vector atomically. A failure warns and returns; it never raises."""
    path = entry_path(model, key, cache_dir=cache_dir)
    tmp = path.with_name(f"{key}.{uuid.uuid4().hex}.tmp")
    payload: dict[str, Any] = {
        "key": key,
        "model": model,
        "dim": len(vector),
        "created_at": datetime.now(UTC).isoformat(),
        "vector": vector,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
        # os.replace, not rename: on Windows rename raises when the target exists.
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("embedding cache write failed for %s: %s", key, exc)
    finally:
        tmp.unlink(missing_ok=True)


def _entries(cache_dir: Path | str | None = None):
    return sorted(_root(cache_dir).glob("*/*/*.json"))


def stats(*, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Entry count, total bytes, and mtime bounds of the embedding cache."""
    entries = _entries(cache_dir)
    mtimes = [entry.stat().st_mtime for entry in entries]
    return {
        "entries": len(entries),
        "bytes": sum(entry.stat().st_size for entry in entries),
        "oldest": min(mtimes) if mtimes else None,
        "newest": max(mtimes) if mtimes else None,
    }


def clear(*, cache_dir: Path | str | None = None, older_than_days: int | None = None) -> int:
    """Remove cached vectors, optionally only those older than N days.

    Only ever costs CPU time to regenerate -- never data.
    """
    cutoff = time.time() - older_than_days * 86400 if older_than_days is not None else None
    removed = 0
    for entry in _entries(cache_dir):
        if cutoff is not None and entry.stat().st_mtime >= cutoff:
            continue
        _discard(entry)
        removed += 1
    return removed
