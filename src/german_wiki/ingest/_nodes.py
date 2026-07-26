"""Candidates become complete node files, staged in ``/queue`` (ADR-003).

Queued files are **real nodes**, not an intermediate format: each one loads through
``storage.load_node`` unchanged. That is what makes review possible with an editor
and nothing else -- you read the file that will land in ``/nodes``, and rejecting a
candidate is deleting it.

Two invariants this module owns:

- ``id`` equals the filename stem. ``write_node`` does not enforce it (only
  ``load_node`` does), so a mismatch here would produce a file that cannot be read
  back. Ids are derived from the title and de-collided against both ``/nodes`` and
  ``/queue``.
- **Nothing is ever silently overwritten.** A colliding id gets a numeric suffix.
  This slice has no dedup at all (SPEC §11) -- two sources describing the same
  concept produce two nodes, and slice 4 is what detects that.

Queue writes pass ``learn=False``: the queue is not the approved gate, so a new
tag must not enter the vocabulary here. ``_promote`` passes ``learn=True`` (ADR-007).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .. import config, storage
from ..logutil import get_logger
from ..models import Node
from ._extract import Candidate
from ._raw import slugify

logger = get_logger(__name__)

# Every machine-assigned CEFR level carries this prefix. SPEC §5 says zero-shot LLM
# CEFR judgment is inconsistent, and slice 6 replaces it with wordlist + grammar
# anchors -- so `grep 'cefr_basis: llm:extraction' nodes/` finds exactly the levels
# that still need re-deriving. Node.cefr is required, so leaving it blank is not an
# option; marking it is.
PROVISIONAL_CEFR = "llm:extraction"

# Everything a machine proposes starts as draft (SPEC §1.1 status).
INITIAL_STATUS = "draft"


def queue_dir_for(source_id: str, *, queue_dir: Path | str | None = None) -> Path:
    root = Path(queue_dir) if queue_dir is not None else config.QUEUE_DIR
    return root / source_id


def taken_ids(
    *, nodes_dir: Path | str | None = None, queue_dir: Path | str | None = None
) -> set[str]:
    """Every node id already used, in ``/nodes`` or anywhere in ``/queue``."""
    nodes_root = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    queue_root = Path(queue_dir) if queue_dir is not None else config.QUEUE_DIR
    taken = {p.stem for p in nodes_root.glob("*.md")} if nodes_root.is_dir() else set()
    if queue_root.is_dir():
        taken |= {p.stem for p in queue_root.glob("*/*.md")}
    return taken


def node_id_for(title_de: str, *, taken: set[str]) -> str:
    """A unique, filename-safe id for a candidate title.

    Collisions get ``-2``, ``-3``, … rather than overwriting. ``taken`` is mutated
    so a batch de-collides against itself as well as against what already exists.
    """
    base = slugify(title_de, max_length=60, fallback="konzept")
    candidate = base
    suffix = 1
    while candidate in taken:
        suffix += 1
        candidate = f"{base}-{suffix}"
    if candidate != base:
        logger.warning("node id %r is taken; using %r instead", base, candidate)
    taken.add(candidate)
    return candidate


def to_node(
    candidate: Candidate,
    *,
    source_id: str,
    node_id: str,
    now: datetime | None = None,
) -> Node:
    """Map one candidate onto a Node. Adds nothing the model did not return."""
    basis = (
        f"{PROVISIONAL_CEFR}; {candidate.cefr_basis}" if candidate.cefr_basis else PROVISIONAL_CEFR
    )
    return Node(
        id=node_id,
        title_de=candidate.title_de,
        title_en=candidate.title_en,
        type=candidate.type,
        cefr=candidate.cefr,
        cefr_basis=basis,
        register=list(candidate.register),
        themes=list(candidate.themes),
        source_ids=[source_id],  # provenance back to /raw (SPEC §8.1)
        confidence=candidate.confidence,
        status=INITIAL_STATUS,
        version=1,
        updated_at=now or datetime.now(UTC),
        body_md=candidate.body_md,
    )


def to_nodes(
    candidates: list[Candidate],
    *,
    source_id: str,
    nodes_dir: Path | str | None = None,
    queue_dir: Path | str | None = None,
    now: datetime | None = None,
) -> list[Node]:
    """Map a whole extraction, de-colliding ids across the batch and the repo."""
    taken = taken_ids(nodes_dir=nodes_dir, queue_dir=queue_dir)
    stamp = now or datetime.now(UTC)
    return [
        to_node(
            candidate,
            source_id=source_id,
            node_id=node_id_for(candidate.title_de, taken=taken),
            now=stamp,
        )
        for candidate in candidates
    ]


def write_queue(
    nodes: list[Node],
    source_id: str,
    *,
    queue_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
) -> list[Path]:
    """Stage nodes for review. Never touches ``/nodes``."""
    target = queue_dir_for(source_id, queue_dir=queue_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for node in nodes:
        path = target / f"{node.id}.md"  # stem == id, so load_node accepts it
        # learn=False: the queue is not the approved gate. Promote grows the
        # vocabulary, at the same moment it writes to /nodes (ADR-007).
        storage.write_node(node, path, vocab_dir=vocab_dir, learn=False)
        paths.append(path)
    return paths


def list_queue(*, queue_dir: Path | str | None = None) -> dict[str, list[Path]]:
    """Pending sources -> their queued node files, sorted."""
    root = Path(queue_dir) if queue_dir is not None else config.QUEUE_DIR
    if not root.is_dir():
        return {}
    pending = {}
    for source in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(source.glob("*.md"))
        if files:
            pending[source.name] = files
    return pending
