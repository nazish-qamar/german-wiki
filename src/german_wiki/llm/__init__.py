"""The model layer: one swappable call interface, cached and cost-logged.

This package is the **only** import surface for model calls. Everything else is
private (``_``-prefixed) and must not be imported from outside this package --
``cli.py`` included, which is why the cache and ledger operations the CLI needs
are re-exported here rather than reached into. Import from here::

    from german_wiki.llm import complete, Prompt

Slice 2 ships the plumbing only: routing config, prompt assembly, the disk cache
(ADR-005) and token/cost accounting. Extraction prompts and response parsing
belong to slice 3.

``resolve_step`` is public so a **local** runner -- slice 4's embedder -- can read
its configured model id without going through ``complete()``, which refuses any
``kind: local`` provider by design (ADR-004). Resolution reads config; it never
calls anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._cache import clear as _cache_clear
from ._cache import stats as _cache_stats
from ._client import JSON_OBJECT, ChatClient, ModelResponse, complete
from ._parse import strip_fences
from ._prompt import Prompt, ShotPair
from ._settings import ResolvedStep, resolve_step
from ._usage import Usage
from ._usage import totals as _totals

__all__ = [
    "JSON_OBJECT",
    "ChatClient",
    "ModelResponse",
    "Prompt",
    "ResolvedStep",
    "ShotPair",
    "Usage",
    "cache_clear",
    "cache_stats",
    "complete",
    "cost_totals",
    "resolve_step",
    "strip_fences",
]


def cost_totals(**kwargs: Any) -> dict[str, Any]:
    """Token and cost totals from the usage ledger (SPEC §10).

    Accepts ``usage_log``, ``since`` and ``group_by``; see ``_usage.totals``.
    """
    return _totals(**kwargs)


def cache_stats(*, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Entry count, total bytes and mtime bounds of the model-call cache."""
    return _cache_stats(cache_dir=cache_dir)


def cache_clear(*, cache_dir: Path | str | None = None, older_than_days: int | None = None) -> int:
    """Remove cache entries, returning how many. Entries are always regenerable."""
    return _cache_clear(cache_dir=cache_dir, older_than_days=older_than_days)
