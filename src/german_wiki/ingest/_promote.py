"""Promotion: the one path that writes into ``/nodes`` (ADR-003, ADR-007).

Everything upstream stages into ``/queue`` or ``/proposals``. This module is the
approved gate, and it is deliberately dumb -- no adjudication, no dedup, no
merging. Slice 5 wraps *this seam* with LangGraph ``interrupt()``; it does not
replace it, and ``german_wiki.merge`` writes through ``write_approved`` here
rather than reaching for ``storage.write_node`` itself.

``write_approved`` is the single ``write_node(..., learn=True)`` call site in the
codebase. ADR-007 says the tag known-sets grow only through the same
human-approved gate as ``/nodes`` writes; that gate is this one function, and an
AST test asserts the call lives inside *this function* -- not merely in this file,
which would stay green if the call drifted back into a caller.

It carries two jobs at once, so both are stated explicitly:

- **the vocabulary gate** (``learn=True``), and
- **the create-vs-overwrite precondition** (``expect_exists``), which is what
  keeps "promote a new node" and "rewrite a node a merge just regenerated" from
  being the same unchecked write.

Promotion is **not** a file move, for two reasons:

- ``load_node`` re-validates each file, so a hand-edit made during review is caught
  before it reaches ``/nodes`` rather than breaking the next ``gw reindex``.
- ``shutil.move`` would skip both of the guarantees above.

Nothing is ever overwritten by accident: a node id that already exists in
``/nodes`` is refused and left queued, so the collision is yours to resolve.
Failures are per-file -- one bad candidate never blocks the rest.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from .. import config, index, storage
from ..logutil import get_logger
from ..models import Node
from ..vocab import KNOWN_SET_FILES
from ._nodes import queue_dir_for

logger = get_logger(__name__)


def write_approved(
    node: Node,
    *,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    expect_exists: bool = False,
) -> Path:
    """Write one approved node into ``/nodes``. The only writer, and the only learner.

    ``expect_exists`` states what the caller believes about the target and refuses
    if reality disagrees, so the two write shapes cannot be confused:

    - ``False`` -- creating a node (promotion, or a DISTINCT adjudication). An
      existing file is a collision: refuse and leave the source staged.
    - ``True`` -- rewriting a node an approved merge regenerated. A *missing* file
      means the winner was renamed or deleted between proposal and review, and
      writing would silently resurrect it under stale content.

    ``learn=True`` is passed here and nowhere else in the codebase (ADR-007): the
    tag known-sets grow at exactly the moment a human-approved write lands.
    """
    root = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    target = root / f"{node.id}.md"
    exists = target.exists()

    if expect_exists and not exists:
        raise ValueError(
            f"expected an existing node at {target} to rewrite, but it is gone; "
            "re-run adjudication rather than recreating it from a stale proposal"
        )
    if not expect_exists and exists:
        raise ValueError(f"a node with id {node.id!r} already exists at {target}")

    storage.write_node(node, target, vocab_dir=vocab_dir, learn=True)
    return target


class Refusal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    path: Path
    reason: str


class PromoteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    promoted: list[str] = []
    refused: list[Refusal] = []
    learned_tags: dict[str, list[str]] = {}
    reindexed: dict[str, int] | None = None


def _vocab_snapshot(vocab_dir: Path | str | None) -> dict[str, list[str]]:
    root = Path(vocab_dir) if vocab_dir is not None else config.VOCAB_DIR
    snapshot = {}
    for field, filename in KNOWN_SET_FILES.items():
        path = root / filename
        snapshot[field] = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return snapshot


def promote_source(
    source_id: str,
    *,
    queue_dir: Path | str | None = None,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    reindex: bool = True,
) -> PromoteResult:
    """Move a source's reviewed candidates from ``/queue`` into ``/nodes``."""
    source_queue = queue_dir_for(source_id, queue_dir=queue_dir)
    queued = sorted(source_queue.glob("*.md")) if source_queue.is_dir() else []
    if not queued:
        raise ValueError(f"nothing queued for source {source_id!r} (looked in {source_queue})")

    nodes_root = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    before = _vocab_snapshot(vocab_dir)

    promoted: list[str] = []
    refused: list[Refusal] = []

    for path in queued:
        try:
            node = storage.load_node(path)
        except (ValueError, OSError, yaml.YAMLError) as exc:
            # Everything load_node can throw on a hand-edited file: broken YAML,
            # a bad enum or missing field (pydantic ValidationError is a
            # ValueError), or an id that no longer matches the stem. Anything
            # else is a bug and should propagate.
            refused.append(Refusal(node_id=path.stem, path=path, reason=str(exc)))
            logger.warning("refusing to promote %s: %s", path.name, exc)
            continue

        try:
            # The approved gate. expect_exists=False: promotion creates nodes and
            # must never overwrite one.
            write_approved(
                node, nodes_dir=nodes_root, vocab_dir=vocab_dir, expect_exists=False
            )
        except ValueError as exc:
            refused.append(Refusal(node_id=node.id, path=path, reason=str(exc)))
            logger.warning("refusing to promote %s: %s", path.name, exc)
            continue

        path.unlink()  # only on success -- a refused candidate stays queued
        promoted.append(node.id)

    after = _vocab_snapshot(vocab_dir)
    learned = {
        field: [tag for tag in after[field] if tag not in set(before[field])] for field in after
    }
    learned = {field: tags for field, tags in learned.items() if tags}

    reindexed = None
    if promoted and reindex:
        reindexed = index.reindex(nodes_dir=nodes_root, db_path=db_path)

    # Tidy up once every candidate has been dealt with.
    if source_queue.is_dir() and not any(source_queue.iterdir()):
        source_queue.rmdir()

    return PromoteResult(
        source_id=source_id,
        promoted=promoted,
        refused=refused,
        learned_tags=learned,
        reindexed=reindexed,
    )
