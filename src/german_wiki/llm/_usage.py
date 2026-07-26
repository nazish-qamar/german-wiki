"""The token/cost ledger: one JSONL record per model call (SPEC §10).

Records go to ``logs/llm_usage.jsonl``, which is **tracked in git** -- spend
history is a durable record, not a scratch file. It is append-only and never
rewritten, so diffs are pure additions.

Written with a plain file handle rather than a ``logging.Handler``: ``gw.log``
has a fixed human-readable formatter and ``propagate=False``, so putting JSON
through it would make the human log unreadable and the machine log unparseable.

Every key is always present (``null`` where inapplicable) so readers never need
defaults and the shape is assertable as an exact key set.

Two conventions worth knowing:

- ``priced`` describes the *model*, not the call. When it is false the model has
  no configured rate, and an uncached call records ``cost_usd: null`` rather than
  a guessed figure.
- A **cache hit always records ``cost_usd: 0.0``**, priced or not, because a hit
  demonstrably spends nothing. ``saved_usd`` then carries what the hit avoided,
  which is what makes "the cache has saved you $X" computable.

The running total is derived by summing this file on demand rather than kept in a
counter. A counter would be a second source of truth that can drift, and it could
not live in ``data/index.db`` anyway -- ``rebuild_schema`` drops every table on
each ``gw reindex``, which would silently zero the total.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .. import config

GroupBy = Literal["step", "model", "provider", "day"]

RECORD_KEYS = frozenset(
    {
        "ts",
        "step",
        "provider",
        "model",
        "cache_key",
        "cached",
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "priced",
        "cost_usd",
        "saved_usd",
        "latency_ms",
        "finish_reason",
        "error",
    }
)

_TOKEN_FIELDS = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


class Usage(BaseModel):
    """Token counts as reported by the provider."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # From ``usage.prompt_tokens_details.cached_tokens``: the provider's own
    # prompt-cache hit. This is the SPEC §10 feedback loop -- the only way to see
    # whether fixed-first/variable-last ordering is actually paying off.
    cached_prompt_tokens: int = 0


def _path(usage_log: Path | str | None) -> Path:
    return Path(usage_log) if usage_log is not None else config.USAGE_LOG_PATH


def log_call(
    *,
    step: str,
    provider: str,
    model: str,
    cache_key: str,
    cached: bool,
    usage: Usage,
    cost_usd: float | None,
    priced: bool | None = None,
    saved_usd: float = 0.0,
    latency_ms: int = 0,
    finish_reason: str | None = None,
    error: str | None = None,
    usage_log: Path | str | None = None,
) -> dict[str, Any]:
    """Append one record and return it."""
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "step": step,
        "provider": provider,
        "model": model,
        "cache_key": cache_key,
        "cached": cached,
        "prompt_tokens": usage.prompt_tokens,
        "cached_prompt_tokens": usage.cached_prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        # `priced` describes the MODEL, so a cache hit (cost 0.0) on an unpriced
        # model stays labelled unpriced. Derived only when not stated.
        "priced": (cost_usd is not None) if priced is None else priced,
        "cost_usd": cost_usd,
        "saved_usd": saved_usd,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
        "error": error,
    }

    path = _path(usage_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_records(*, usage_log: Path | str | None = None) -> list[dict[str, Any]]:
    """Every record in the ledger. A missing file reads as empty, not an error."""
    path = _path(usage_log)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _blank() -> dict[str, Any]:
    return {
        "calls": 0,
        "hits": 0,
        "hit_rate": 0.0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "saved_usd": 0.0,
        "unpriced_calls": 0,
    }


def _accumulate(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    bucket["calls"] += 1
    bucket["hits"] += 1 if record["cached"] else 0
    for field in _TOKEN_FIELDS:
        bucket[field] += record[field] or 0
    if record["cost_usd"] is None:
        bucket["unpriced_calls"] += 1
    else:
        bucket["cost_usd"] += record["cost_usd"]
    bucket["saved_usd"] += record["saved_usd"] or 0.0


def _finish(bucket: dict[str, Any]) -> dict[str, Any]:
    bucket["hit_rate"] = bucket["hits"] / bucket["calls"] if bucket["calls"] else 0.0
    return bucket


def _group_key(record: dict[str, Any], group_by: GroupBy) -> str:
    return record["ts"][:10] if group_by == "day" else record[group_by]


def totals(
    *,
    usage_log: Path | str | None = None,
    since: date | None = None,
    group_by: GroupBy | None = None,
    records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sum the ledger. ``groups`` is present only when ``group_by`` is given."""
    rows = list(records) if records is not None else read_records(usage_log=usage_log)
    if since is not None:
        cutoff = since.isoformat()
        rows = [r for r in rows if r["ts"][:10] >= cutoff]

    overall = _blank()
    groups: dict[str, dict[str, Any]] = {}
    for record in rows:
        _accumulate(overall, record)
        if group_by is not None:
            bucket = groups.setdefault(_group_key(record, group_by), _blank())
            _accumulate(bucket, record)

    result = _finish(overall)
    if group_by is not None:
        result["groups"] = {name: _finish(b) for name, b in sorted(groups.items())}
    return result
