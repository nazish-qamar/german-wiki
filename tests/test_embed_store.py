"""vec_nodes round-trip and KNN, in similarity terms rather than raw distance."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from german_wiki import index
from german_wiki.db import EMBEDDING_DIM, connect, rebuild_schema
from german_wiki.embed import _store


def _unit(*leading: float) -> list[float]:
    """A normalized EMBEDDING_DIM vector whose first components are given."""
    vector = list(leading) + [0.0] * (EMBEDDING_DIM - len(leading))
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture
def conn(tmp_db: Path):
    connection = connect(tmp_db)
    rebuild_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


# --- round-trip ---


def test_store_and_count(conn) -> None:
    assert _store.count(conn) == 0
    written = _store.store_vectors(conn, {"a": _unit(1.0), "b": _unit(0.0, 1.0)})
    assert written == 2
    assert _store.count(conn) == 2
    assert _store.stored_ids(conn) == {"a", "b"}


def test_storing_the_same_id_twice_replaces_rather_than_duplicates(conn) -> None:
    """Re-embedding a changed node must not leave two vectors behind."""
    _store.store_vectors(conn, {"a": _unit(1.0)})
    _store.store_vectors(conn, {"a": _unit(0.0, 1.0)})

    assert _store.count(conn) == 1
    (top,) = _store.knn(conn, _unit(0.0, 1.0), k=1)
    assert top[0] == "a"
    assert top[1] == pytest.approx(1.0, abs=1e-5)


def test_wrong_width_vector_is_refused(conn) -> None:
    """The last line of defence before sqlite-vec sees a malformed value."""
    with pytest.raises(ValueError, match=rf"3 dimensions.*float\[{EMBEDDING_DIM}\]"):
        _store.store_vectors(conn, {"a": [0.1, 0.2, 0.3]})
    assert _store.stored_ids(conn) == set()


# --- knn returns similarity, not distance ---


def test_knn_returns_similarity_ordered_most_similar_first(conn) -> None:
    _store.store_vectors(
        conn,
        {
            "same": _unit(1.0),
            "near": _unit(0.99, 0.1),
            "far": _unit(0.0, 1.0),
        },
    )
    results = _store.knn(conn, _unit(1.0), k=3)

    assert [node_id for node_id, _ in results] == ["same", "near", "far"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[-1][1] == pytest.approx(0.0, abs=1e-5)
    assert all(-1.01 <= score <= 1.01 for _, score in results)


def test_knn_honours_k(conn) -> None:
    _store.store_vectors(conn, {f"n{i}": _unit(1.0, i / 10) for i in range(6)})
    assert len(_store.knn(conn, _unit(1.0), k=2)) == 2


def test_knn_can_exclude_self(conn) -> None:
    """Comparing a node against the corpus must not rank it against itself."""
    _store.store_vectors(conn, {"a": _unit(1.0), "b": _unit(0.9, 0.1)})
    results = _store.knn(conn, _unit(1.0), k=5, exclude="a")
    assert [node_id for node_id, _ in results] == ["b"]


def test_knn_still_returns_k_results_when_excluding(conn) -> None:
    _store.store_vectors(conn, {f"n{i}": _unit(1.0, i / 20) for i in range(5)})
    assert len(_store.knn(conn, _unit(1.0), k=3, exclude="n0")) == 3


def test_knn_on_an_empty_table(conn) -> None:
    assert _store.knn(conn, _unit(1.0), k=5) == []


# --- interaction with the rebuildable index (ADR-001) ---


def test_reindex_drops_stored_vectors(tmp_nodes: Path, tmp_db: Path) -> None:
    """The vec table is derived: reindex wipes it, and the disk cache is what
    makes repopulating cheap. This is why cache and table stay separate."""
    conn = connect(tmp_db)
    try:
        rebuild_schema(conn)
        _store.store_vectors(conn, {"a": _unit(1.0)})
        assert _store.count(conn) == 1
    finally:
        conn.close()

    index.reindex(nodes_dir=tmp_nodes, db_path=tmp_db)

    conn = connect(tmp_db)
    try:
        assert _store.count(conn) == 0
    finally:
        conn.close()


def test_vectors_survive_a_reconnect(tmp_db: Path) -> None:
    conn = connect(tmp_db)
    try:
        rebuild_schema(conn)
        _store.store_vectors(conn, {"a": _unit(1.0)})
    finally:
        conn.close()

    conn = connect(tmp_db)
    try:
        assert _store.stored_ids(conn) == {"a"}
    finally:
        conn.close()
