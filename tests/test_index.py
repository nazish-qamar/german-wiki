"""reindex builds the derived index; query filters and staleness behave."""

from __future__ import annotations

import os
import time

import pytest

from german_wiki import index
from german_wiki.db import EMBEDDING_DIM, connect


def _ids(rows):
    return sorted(r["id"] for r in rows)


def test_reindex_counts(tmp_nodes, tmp_db):
    counts = index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    assert counts == {"nodes": 4, "links": 9, "themes": 6}


def test_schema_tables_include_the_vector_table(tmp_nodes, tmp_db):
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    conn = connect(tmp_db)
    try:
        names = {
            r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert {"nodes", "links", "node_themes", "meta"} <= names
    assert "vec_nodes" in names  # arrived in slice 4


def test_vec_table_is_declared_at_the_pinned_dimension(tmp_nodes, tmp_db):
    """The DDL width and db.EMBEDDING_DIM must never drift apart."""
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    conn = connect(tmp_db)
    try:
        ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'vec_nodes'").fetchone()[
            "sql"
        ]
    finally:
        conn.close()
    assert f"float[{EMBEDDING_DIM}]" in ddl
    assert "distance_metric=cosine" in ddl


def test_reindex_leaves_the_vector_table_empty(tmp_nodes, tmp_db):
    """reindex never computes embeddings -- `gw list` must not pull in torch."""
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    conn = connect(tmp_db)
    try:
        assert conn.execute("SELECT count(*) FROM vec_nodes").fetchone()[0] == 0
    finally:
        conn.close()


def test_rebuild_is_idempotent_with_the_virtual_table(tmp_nodes, tmp_db):
    """vec0 creates shadow tables; a second reindex must not trip over them."""
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    counts = index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    assert counts["nodes"] == 4


def test_set_meta_upserts(tmp_nodes, tmp_db):
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    conn = connect(tmp_db)
    try:
        index.set_meta(conn, "probe", "one")
        index.set_meta(conn, "probe", "two")
        conn.commit()
        assert index.get_meta(conn, "probe") == "two"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "filt,expected",
    [
        ({"type": "grammar"}, ["wechselpraepositionen"]),
        ({"type": "vocab"}, ["familie-waschen"]),
        ({"type": "pattern"}, ["prefix-an"]),
        ({"type": "phrase"}, ["um-hilfe-bitten"]),
        (
            {"cefr": "A2"},
            ["familie-waschen", "prefix-an", "um-hilfe-bitten", "wechselpraepositionen"],
        ),
        ({"cefr": "B1"}, []),
        ({"theme": "küche"}, ["familie-waschen"]),
        ({"theme": "büro"}, ["um-hilfe-bitten"]),
        ({"type": "vocab", "cefr": "A2"}, ["familie-waschen"]),
        ({"type": "grammar", "theme": "küche"}, []),  # AND-combined
    ],
)
def test_query_filters(tmp_nodes, tmp_db, filt, expected):
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    conn = connect(tmp_db)
    try:
        rows = index.query_nodes(conn, **filt)
    finally:
        conn.close()
    assert _ids(rows) == expected


def test_staleness(tmp_nodes, tmp_db):
    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)
    conn = connect(tmp_db)
    try:
        assert index.is_stale(conn, nodes_dir=tmp_nodes) is False
        # make one node file newer than the last reindex (absolute future time —
        # the copied seeds keep their original, older mtime)
        target = next(tmp_nodes.glob("*.md"))
        future = time.time() + 1000
        os.utime(target, (future, future))
        assert index.is_stale(conn, nodes_dir=tmp_nodes) is True
    finally:
        conn.close()


def test_stale_when_never_indexed(tmp_nodes, tmp_db):
    """A fresh DB (no reindex meta) counts as stale."""
    from german_wiki.db import rebuild_schema

    conn = connect(tmp_db)
    try:
        rebuild_schema(conn)
        assert index.is_stale(conn, nodes_dir=tmp_nodes) is True
    finally:
        conn.close()
