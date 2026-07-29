"""Three-tier duplicate detection, cheap → expensive (SPEC §3.1).

1. **Exact** — sha256 of normalized text. Free, catches copy-paste.
2. **Near-exact** — true Jaccard over character shingles, length-ratio prefiltered.
   Free. (Exact Jaccard rather than MinHash: see ``_text`` and ADR-010.)
3. **Semantic** — cosine over local embeddings. Free, but needs vectors.

Tiers 1–2 handle the majority for nothing, which is why SPEC §3.1 expects the LLM
(slice 5, tier 3) to fire on only ~10–20% of candidates. Detection is *not* an LLM
problem.

**This module reports; it never acts.** No merge, no dedup write, nothing to
``/nodes`` or ``/queue``. The return value is the entire output. What it does write
-- vectors, into the derived index -- is a different layer: ADR-001 makes that
table freely rebuildable, and slice 5 is what turns a gray-zone pair into a
decision.
"""

from __future__ import annotations

import hashlib
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..logutil import get_logger
from ..models import Node
from . import _store
from ._embed import EmbedResult, vectors_for
from ._model import Embedder
from ._text import could_reach, jaccard, normalize_for_hash, shingles

logger = get_logger(__name__)

# Tier 1. SPEC gives no number for near-exact; 0.85 means "copy-paste with edits".
NEAR_EXACT_JACCARD = 0.85

# Tier 2. SPEC §3.1 proposes 0.75–0.92 for the LLM-adjudication window, but those
# numbers do not survive contact with multilingual-e5, whose cosine scores compress
# into a narrow high band. Measured on the seed corpus plus a real ingested pair
# (see embed_text for the full table):
#
#   strongest UNRELATED pair    0.8635
#   weakest true DUPLICATE      0.9194
#   exact copy                  1.0000
#
# At SPEC's 0.75 floor every pair is gray, which would hand slice 5 a meaningless
# adjudication queue -- the exact outcome §3.1's cheap tiers exist to prevent.
# GRAY_LOW sits above the unrelated ceiling with headroom; GRAY_HIGH stays high
# enough that only near-identical text auto-classifies as a duplicate, leaving
# genuine paraphrases for the LLM to judge (SPEC §3.1: OVERLAP routes to merge).
#
# Per SPEC §3.3 these are *the* node-count dial, and §3.3 says bias aggressive --
# a flagged pair only costs one cheap adjudication call, a missed one costs a
# fragmented wiki. Nine unrelated pairs is thin evidence: revisit once real
# material accumulates.
GRAY_LOW = 0.87
GRAY_HIGH = 0.95

# Neighbours fetched per node. More than ten plausible duplicates for one node
# means the threshold needs tuning, not the limit.
DEFAULT_K = 10

Tier = Literal["exact", "near-exact", "semantic"]
Band = Literal["duplicate", "gray"]


class Match(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_id: str
    right_id: str
    tier: Tier
    similarity: float
    band: Band


class DuplicateReport(BaseModel):
    """Everything one detection run found. Nothing was written to /nodes."""

    model_config = ConfigDict(extra="forbid")

    matches: list[Match] = []
    compared: int = 0
    embedding: EmbedResult | None = None

    @property
    def duplicates(self) -> list[Match]:
        return [m for m in self.matches if m.band == "duplicate"]

    @property
    def gray(self) -> list[Match]:
        return [m for m in self.matches if m.band == "gray"]


def _band(similarity: float) -> Band:
    return "duplicate" if similarity >= GRAY_HIGH else "gray"


def _pair(a: str, b: str) -> tuple[str, str]:
    """Ordered, so a symmetric match is only ever reported once."""
    return (a, b) if a <= b else (b, a)


def _node_text(node: Node) -> str:
    return f"{node.title_de}\n{node.title_en}\n{node.body_md}"


def _exact(texts: dict[str, str], focus: set[str]) -> dict[tuple[str, str], float]:
    """Group by content hash; every pair inside a group is an exact duplicate."""
    by_hash: dict[str, list[str]] = {}
    for node_id, text in texts.items():
        digest = hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()
        by_hash.setdefault(digest, []).append(node_id)

    found = {}
    for group in by_hash.values():
        for a, b in combinations(sorted(group), 2):
            if a in focus or b in focus:
                found[_pair(a, b)] = 1.0
    return found


def _near_exact(
    texts: dict[str, str], focus: set[str], seen: set[tuple[str, str]]
) -> tuple[dict[tuple[str, str], float], int]:
    """Exact Jaccard over shingles, skipping pairs the length bound rules out."""
    sets = {node_id: shingles(text) for node_id, text in texts.items()}
    found: dict[tuple[str, str], float] = {}
    compared = 0

    for a, b in combinations(sorted(texts), 2):
        if a not in focus and b not in focus:
            continue
        if _pair(a, b) in seen:
            continue
        if not could_reach(sets[a], sets[b], NEAR_EXACT_JACCARD):
            continue  # cannot possibly reach the threshold; skip the set maths
        compared += 1
        score = jaccard(sets[a], sets[b])
        if score >= NEAR_EXACT_JACCARD:
            found[_pair(a, b)] = score
    return found, compared


def _cosine(a: list[float], b: list[float]) -> float:
    """Vectors are stored normalized, so the dot product IS the cosine."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _semantic(
    conn: sqlite3.Connection | None,
    vectors: dict[str, list[float]],
    focus: set[str],
    indexed: set[str],
    *,
    k: int,
    seen: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], float], int]:
    """Cosine similarity, via the vector index where possible.

    Ids already in ``vec_nodes`` use sqlite-vec KNN, which stays sublinear as the
    corpus grows. Ids that are not indexed -- queued candidates, which are not
    nodes yet and must not pollute the index -- are compared directly; there are
    at most a handful per source.
    """
    found: dict[tuple[str, str], float] = {}
    compared = 0
    unindexed = [node_id for node_id in sorted(vectors) if node_id not in indexed]

    for node_id in sorted(focus):
        vector = vectors.get(node_id)
        if vector is None:
            continue

        neighbours: list[tuple[str, float]] = []
        if conn is not None and indexed:
            neighbours.extend(_store.knn(conn, vector, k=k, exclude=node_id))
        # Compare against anything the index does not hold (candidates).
        neighbours.extend(
            (other, _cosine(vector, vectors[other])) for other in unindexed if other != node_id
        )

        for other, similarity in neighbours:
            pair = _pair(node_id, other)
            if pair in seen or pair in found:
                continue
            compared += 1
            if similarity >= GRAY_LOW:
                found[pair] = similarity
    return found, compared


def find_duplicates(
    nodes: list[Node],
    *,
    against: list[Node] | None = None,
    conn: sqlite3.Connection | None = None,
    embedder: Embedder | None = None,
    model: str | None = None,
    cache_dir: Path | str | None = None,
    settings_path: Path | str | None = None,
    k: int = DEFAULT_K,
    store: bool = True,
) -> DuplicateReport:
    """Report likely duplicates. Writes nothing to /nodes or /queue.

    ``nodes`` is the focus set -- every reported pair has at least one side in it.
    ``against`` is the existing corpus to compare with; omit it to scan ``nodes``
    against themselves.
    """
    focus = {node.id for node in nodes}
    shadowed = focus & {node.id for node in (against or [])}
    if shadowed:
        # Cannot happen via `gw ingest` -- slice 3 de-collides candidate ids against
        # /nodes -- but a hand-placed queue file could do it. Say so rather than
        # silently dropping one side when the id-keyed corpus merges.
        logger.warning(
            "%d id(s) appear in both the focus set and the corpus and were merged: %s",
            len(shadowed),
            ", ".join(sorted(shadowed)),
        )

    corpus: dict[str, Node] = {node.id: node for node in (against or [])}
    corpus.update({node.id: node for node in nodes})

    texts = {node_id: _node_text(node) for node_id, node in corpus.items()}

    matches: dict[tuple[str, str], tuple[Tier, float]] = {}

    exact = _exact(texts, focus)
    for pair, score in exact.items():
        matches[pair] = ("exact", score)

    near, compared = _near_exact(texts, focus, set(matches))
    for pair, score in near.items():
        matches[pair] = ("near-exact", score)

    # Tier 3 (semantic). Existing nodes get their vectors stored; queued candidates
    # are embedded for the comparison but never written to the index.
    indexed_nodes = [node for node in (against or []) if node.id not in focus] or list(
        corpus.values()
    )
    vectors, embedding = vectors_for(
        list(corpus.values()),
        embedder=embedder,
        model=model,
        cache_dir=cache_dir,
        settings_path=settings_path,
    )
    indexed: set[str] = set()
    if conn is not None and store:
        storable = {n.id: vectors[n.id] for n in indexed_nodes if n.id in vectors}
        if storable:
            _store.store_vectors(conn, storable)
        indexed = _store.stored_ids(conn) & set(vectors)

    semantic, sem_compared = _semantic(conn, vectors, focus, indexed, k=k, seen=set(matches))
    for pair, score in semantic.items():
        matches[pair] = ("semantic", score)

    report = DuplicateReport(
        compared=compared + sem_compared,
        embedding=embedding,
        matches=[
            Match(left_id=a, right_id=b, tier=tier, similarity=score, band=_band(score))
            for (a, b), (tier, score) in sorted(matches.items(), key=lambda kv: (-kv[1][1], kv[0]))
        ],
    )
    return report
