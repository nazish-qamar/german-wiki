"""Ingestion orchestration: file -> raw -> extraction -> queue.

The whole flow is deliberately linear and writes nothing to ``/nodes``. Ordering
carries meaning:

1. Read and resolve the source id from the **content**, so a re-ingest is detected.
2. Write the raw ``.txt`` -- before the model runs, so the provenance record never
   depends on the model succeeding.
3. Extract. A failure here leaves the raw file with no sidecar: an incomplete
   ingest, free to retry because the call is cached (ADR-005).
4. Stage complete node files in ``/queue``.
5. Write the sidecar last, recording what the extraction actually produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..llm import ChatClient
from ..models import Node
from . import _nodes, _raw
from ._extract import extract


class IngestResult(BaseModel):
    """What one ``gw ingest`` produced. Nothing here has touched ``/nodes``."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    already_ingested: bool
    raw_path: Path
    queue_paths: list[Path]
    nodes: list[Node]
    cached: bool = False
    cost_usd: float | None = None


def ingest_file(
    path: Path | str,
    *,
    force: bool = False,
    client: ChatClient | None = None,
    raw_dir: Path | str | None = None,
    queue_dir: Path | str | None = None,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    today: date | None = None,
) -> IngestResult:
    """Ingest one plain-text file into the review queue.

    Raises ``ExtractionError`` if the model returned nothing usable -- notably on
    ``finish_reason == "length"``, which looks like an empty result but is a
    truncation.
    """
    path = Path(path)
    text = _raw.read_source(path)
    source_id, is_new = _raw.resolve_source_id(text, path, raw_dir=raw_dir, today=today)
    raw_path, _ = _raw.raw_paths(source_id, raw_dir=raw_dir)

    if not is_new and not force:
        # Same content already in /raw. Report what is still queued for it, if
        # anything -- an empty list means it was already promoted or rejected.
        queued = _nodes.list_queue(queue_dir=queue_dir).get(source_id, [])
        return IngestResult(
            source_id=source_id,
            already_ingested=True,
            raw_path=raw_path,
            queue_paths=queued,
            nodes=[],
        )

    _raw.store_raw(text, source_id, raw_dir=raw_dir)

    candidates, response = extract(
        text,
        client=client,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=dict(env) if env is not None else None,
    )

    nodes = _nodes.to_nodes(
        candidates,
        source_id=source_id,
        nodes_dir=nodes_dir,
        queue_dir=queue_dir,
        now=now,
    )
    queue_paths = (
        _nodes.write_queue(nodes, source_id, queue_dir=queue_dir, vocab_dir=vocab_dir)
        if nodes
        else []
    )

    _raw.write_sidecar(
        source_id,
        text=text,
        original=path,
        model=response.model,
        provider=response.provider,
        candidate_count=len(candidates),
        raw_dir=raw_dir,
    )

    return IngestResult(
        source_id=source_id,
        already_ingested=False,
        raw_path=raw_path,
        queue_paths=queue_paths,
        nodes=nodes,
        cached=response.cached,
        cost_usd=response.cost_usd,
    )
