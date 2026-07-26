"""Content-hash disk cache wrapping every model call (ADR-005).

The tuning phase re-runs the pipeline over the same sources dozens of times.
Uncached that multiplies token spend by roughly 50x; this cache is what keeps the
project inside its budget, and it is why slice 2 comes before slice 3 (SPEC §11).

**What the key covers** is the whole design. It is the request as the provider
would see it -- provider, model, messages, sampling parameters, response format,
seed -- plus a ``prompt_version`` escape hatch for invalidating entries when a
downstream parser changes but the prompt text does not.

**What it deliberately excludes** is the pipeline step. Two steps issuing a
byte-identical request should share an entry, and renaming a step must not cost
money. The step is recorded in the stored payload for debugging, just not hashed.
Credentials, base URL, timeouts and retry counts are excluded for the same
reason: none of them change the response content.

Failures here are never fatal. A write that fails, an entry that is corrupt, a
stored request that disagrees with the recomputed key -- each logs a warning and
degrades to a cache miss, because a broken cache must never break a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .. import config
from ..logutil import get_logger

logger = get_logger(__name__)

# Bump to invalidate every existing entry.
KEY_VERSION = 1

# Entries live under <cache_dir>/llm/<first two hex chars>/<key>.json
SUBDIR = "llm"

REQUIRED_KEYS = frozenset({"key", "request", "text", "usage"})


def key_material(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
    max_tokens: int | None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    prompt_version: str | None = None,
) -> dict[str, Any]:
    """Build the exact dict that gets hashed. Nothing else influences the key."""
    return {
        "v": KEY_VERSION,
        "provider": provider,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
        "seed": seed,
        "prompt_version": prompt_version,
    }


def cache_key(material: dict[str, Any]) -> str:
    """sha256 over canonical JSON, so key order never affects the digest.

    Untruncated: a collision would silently serve the wrong answer, and there is
    no reason to trade that risk for shorter filenames.
    """
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def entry_path(key: str, *, cache_dir: Path | str | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    return root / SUBDIR / key[:2] / f"{key}.json"


def read(
    key: str,
    *,
    cache_dir: Path | str | None = None,
    expect_request: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the stored payload, or ``None`` for any kind of miss.

    ``expect_request`` is compared against the stored request. A mismatch means
    a hash collision or a hand-edited entry; it turns a silently wrong answer
    into a log line and a miss.
    """
    path = entry_path(key, cache_dir=cache_dir)
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("discarding unreadable cache entry %s: %s", path, exc)
        _discard(path)
        return None

    if not isinstance(payload, dict) or not REQUIRED_KEYS.issubset(payload):
        logger.warning("discarding malformed cache entry %s", path)
        _discard(path)
        return None

    if expect_request is not None and payload["request"] != expect_request:
        logger.warning("discarding cache entry %s: stored request does not match its key", path)
        _discard(path)
        return None

    logger.debug("cache hit %s", key)
    return payload


def write(key: str, payload: dict[str, Any], *, cache_dir: Path | str | None = None) -> None:
    """Write an entry atomically. A failure warns and returns; it never raises."""
    path = entry_path(key, cache_dir=cache_dir)
    tmp = path.with_name(f"{key}.{uuid.uuid4().hex}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )
        # os.replace, not rename: on Windows rename raises when the target exists.
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("cache write failed for %s: %s", key, exc)
    finally:
        tmp.unlink(missing_ok=True)


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - platform-specific
        logger.warning("could not remove cache entry %s: %s", path, exc)


def _entries(cache_dir: Path | str | None = None):
    root = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    return sorted((root / SUBDIR).glob("*/*.json"))


def stats(*, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Entry count, total bytes, and the mtime bounds of the cache."""
    entries = _entries(cache_dir)
    mtimes = [entry.stat().st_mtime for entry in entries]
    return {
        "entries": len(entries),
        "bytes": sum(entry.stat().st_size for entry in entries),
        "oldest": min(mtimes) if mtimes else None,
        "newest": max(mtimes) if mtimes else None,
    }


def clear(*, cache_dir: Path | str | None = None, older_than_days: int | None = None) -> int:
    """Remove cache entries, optionally only those older than N days.

    Returns the number removed. Entries are regenerable by definition, so this
    only ever costs tokens, never data.
    """
    cutoff = time.time() - older_than_days * 86400 if older_than_days is not None else None
    removed = 0
    for entry in _entries(cache_dir):
        if cutoff is not None and entry.stat().st_mtime >= cutoff:
            continue
        _discard(entry)
        removed += 1
    return removed
