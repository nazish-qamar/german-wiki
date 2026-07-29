"""Vector storage and nearest-neighbour search over ``vec_nodes``.

The table is part of the derived index (ADR-001): dropped and rebuilt freely, never
authoritative. Vectors are written normalized, so the cosine distance sqlite-vec
returns converts to a similarity with a plain ``1 - distance`` -- nothing outside
this module ever handles a raw distance.
"""

from __future__ import annotations

import sqlite3

import sqlite_vec

from ..db import EMBEDDING_DIM


def store_vectors(conn: sqlite3.Connection, vectors: dict[str, list[float]]) -> int:
    """Upsert vectors keyed by node id. Returns how many were written."""
    written = 0
    for node_id, vector in vectors.items():
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"vector for node {node_id!r} has {len(vector)} dimensions, "
                f"but vec_nodes is float[{EMBEDDING_DIM}]"
            )
        # vec0 has no UPSERT; delete-then-insert keeps re-embedding idempotent.
        conn.execute("DELETE FROM vec_nodes WHERE node_id = ?", (node_id,))
        conn.execute(
            "INSERT INTO vec_nodes (node_id, embedding) VALUES (?, ?)",
            (node_id, sqlite_vec.serialize_float32(vector)),
        )
        written += 1
    conn.commit()
    return written


def stored_ids(conn: sqlite3.Connection) -> set[str]:
    """Node ids that currently have a vector."""
    return {r["node_id"] for r in conn.execute("SELECT node_id FROM vec_nodes")}


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM vec_nodes").fetchone()[0]


def knn(
    conn: sqlite3.Connection,
    vector: list[float],
    *,
    k: int = 10,
    exclude: str | None = None,
) -> list[tuple[str, float]]:
    """Nearest neighbours as ``(node_id, similarity)``, most similar first.

    ``similarity = 1 - distance`` because the column declares
    ``distance_metric=cosine`` and vectors are stored normalized.
    """
    if count(conn) == 0:
        return []
    # Ask for one extra so excluding self still leaves k results.
    limit = k + 1 if exclude is not None else k
    rows = conn.execute(
        "SELECT node_id, distance FROM vec_nodes "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(vector), limit),
    ).fetchall()

    out = [(r["node_id"], 1.0 - r["distance"]) for r in rows if r["node_id"] != exclude]
    return out[:k]
