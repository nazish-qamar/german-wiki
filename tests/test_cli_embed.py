"""gw embed / gw dupes / gw cache: the cache proof and the report-only guarantee."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeEmbedder
from typer.testing import CliRunner

from german_wiki import storage
from german_wiki.cli import app
from german_wiki.embed import _cache as embed_cache
from german_wiki.llm import _cache as llm_cache

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


@pytest.fixture(autouse=True)
def _no_real_model(monkeypatch):
    """The offline suite must never load sentence-transformers."""
    monkeypatch.setattr("german_wiki.embed._model.load_embedder", lambda **kwargs: FakeEmbedder())


@pytest.fixture(autouse=True)
def indexed(tmp_nodes: Path, tmp_db: Path):
    """embed and dupes read the derived index, so it has to exist first."""
    runner.invoke(app, ["reindex", "--nodes-dir", str(tmp_nodes), "--db", str(tmp_db)], env=WIDE)
    return tmp_db


def _embed(tmp_nodes, tmp_db, tmp_cache):
    return runner.invoke(
        app,
        [
            "embed",
            "--nodes-dir",
            str(tmp_nodes),
            "--db",
            str(tmp_db),
            "--cache-dir",
            str(tmp_cache),
        ],
        env=WIDE,
    )


def _dupes(tmp_nodes, tmp_db, tmp_cache, *extra):
    return runner.invoke(
        app,
        [
            "dupes",
            "--nodes-dir",
            str(tmp_nodes),
            "--db",
            str(tmp_db),
            "--cache-dir",
            str(tmp_cache),
            *extra,
        ],
        env=WIDE,
    )


# --- gw embed: the counts ARE the cache proof ---


def test_embed_reports_new_and_cached(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    result = _embed(tmp_nodes, tmp_db, tmp_cache)
    assert result.exit_code == 0
    assert "Embedded 4 new, 0 from cache" in result.stdout
    assert "4 vector(s)" in result.stdout


def test_a_second_embed_computes_nothing(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    """The embedding-layer equivalent of slice 2's call_count == 1 assertion."""
    _embed(tmp_nodes, tmp_db, tmp_cache)
    result = _embed(tmp_nodes, tmp_db, tmp_cache)
    assert "Embedded 0 new, 4 from cache" in result.stdout


def test_embed_names_the_model(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    assert "model " in _embed(tmp_nodes, tmp_db, tmp_cache).stdout


def test_embed_on_an_empty_node_dir(tmp_path: Path, tmp_db: Path, tmp_cache: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _embed(empty, tmp_db, tmp_cache)
    assert result.exit_code == 0
    assert "No nodes to embed" in result.stdout


def test_embed_without_an_index_names_the_fix(
    tmp_nodes: Path, tmp_path: Path, tmp_cache: Path
) -> None:
    result = _embed(tmp_nodes, tmp_path / "absent.db", tmp_cache)
    assert result.exit_code == 1
    assert "gw reindex" in _combined(result)


def test_dupes_on_a_pre_slice_4_index_names_the_fix(
    tmp_nodes: Path, tmp_path: Path, tmp_cache: Path
) -> None:
    """An index built before the vector table existed must say so, not crash."""
    import sqlite3

    old = tmp_path / "old.db"
    sqlite3.connect(old).execute("CREATE TABLE nodes (id TEXT)").connection.close()

    result = _dupes(tmp_nodes, old, tmp_cache)
    assert result.exit_code == 1
    assert "no vector table" in _combined(result)
    assert "gw reindex" in _combined(result)


# --- gw reindex reloads cached vectors without loading a model ---


def test_reindex_restores_cached_vectors(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    _embed(tmp_nodes, tmp_db, tmp_cache)
    result = runner.invoke(
        app,
        [
            "reindex",
            "--nodes-dir",
            str(tmp_nodes),
            "--db",
            str(tmp_db),
            "--cache-dir",
            str(tmp_cache),
        ],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "restored 4 cached vector(s)" in result.stdout


def test_reindex_without_a_cache_says_so(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    result = runner.invoke(
        app,
        [
            "reindex",
            "--nodes-dir",
            str(tmp_nodes),
            "--db",
            str(tmp_db),
            "--cache-dir",
            str(tmp_cache),
        ],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "run gw embed" in result.stdout.replace("\n", " ")


# --- gw dupes ---


def test_dupes_on_the_seed_corpus_finds_none(
    tmp_nodes: Path, tmp_db: Path, tmp_cache: Path
) -> None:
    """Four hand-authored seeds about different things: no duplicates expected."""
    result = _dupes(tmp_nodes, tmp_db, tmp_cache)
    assert result.exit_code == 0
    assert "No duplicates found among 4 node(s)" in result.stdout


def test_dupes_finds_a_planted_copy(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    original = tmp_nodes / "prefix-an.md"
    copy = tmp_nodes / "prefix-an-kopie.md"
    copy.write_text(
        original.read_text(encoding="utf-8").replace("id: prefix-an", "id: prefix-an-kopie"),
        encoding="utf-8",
    )

    result = _dupes(tmp_nodes, tmp_db, tmp_cache)
    assert result.exit_code == 0
    assert "prefix-an" in result.stdout
    assert "duplicate" in result.stdout
    assert "nothing was written to /nodes" in result.stdout.replace("\n", " ")


def test_dupes_writes_nothing_to_nodes(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    """Report only. The DB is excluded on purpose -- it legitimately gains vectors."""
    before = {p.name: p.read_bytes() for p in tmp_nodes.glob("*.md")}
    _dupes(tmp_nodes, tmp_db, tmp_cache)
    assert {p.name: p.read_bytes() for p in tmp_nodes.glob("*.md")} == before


def test_dupes_mentions_slice_5_for_gray_pairs(
    tmp_nodes: Path, tmp_db: Path, tmp_cache: Path
) -> None:
    original = tmp_nodes / "prefix-an.md"
    node = storage.load_node(original)
    near = node.model_copy(update={"id": "prefix-an-fast", "body_md": node.body_md + " Fast."})
    storage.write_node(near, tmp_nodes / "prefix-an-fast.md")

    result = _dupes(tmp_nodes, tmp_db, tmp_cache)
    assert "prefix-an-fast" in result.stdout


def test_dupes_against_a_queued_source(
    tmp_nodes: Path, tmp_db: Path, tmp_cache: Path, tmp_queue: Path
) -> None:
    source = tmp_queue / "20260726-test-abc12345"
    source.mkdir(parents=True)
    # A realistic candidate: same content, de-collided id -- exactly what slice 3's
    # node_id_for produces when a title already exists in /nodes.
    node = storage.load_node(tmp_nodes / "prefix-an.md")
    storage.write_node(node.model_copy(update={"id": "prefix-an-2"}), source / "prefix-an-2.md")

    result = _dupes(
        tmp_nodes,
        tmp_db,
        tmp_cache,
        "--queue",
        "20260726-test-abc12345",
        "--queue-dir",
        str(tmp_queue),
    )
    assert result.exit_code == 0
    assert "prefix-an" in result.stdout
    assert "duplicate" in result.stdout


def test_dupes_unknown_queue_source_errors(
    tmp_nodes: Path, tmp_db: Path, tmp_cache: Path, tmp_queue: Path
) -> None:
    result = _dupes(tmp_nodes, tmp_db, tmp_cache, "--queue", "nope", "--queue-dir", str(tmp_queue))
    assert result.exit_code == 1
    assert "Nothing queued for source" in _combined(result)


# --- gw cache covers BOTH caches ---


def test_cache_stats_lists_both(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    _embed(tmp_nodes, tmp_db, tmp_cache)
    result = runner.invoke(app, ["cache", "stats", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert result.exit_code == 0
    assert "llm" in result.stdout
    assert "embeddings" in result.stdout


def test_cache_stats_when_both_are_empty(tmp_cache: Path) -> None:
    result = runner.invoke(app, ["cache", "stats", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert result.exit_code == 0
    assert "Caches are empty" in result.stdout


def test_cache_clear_embeddings_leaves_the_llm_cache_intact(
    tmp_nodes: Path, tmp_db: Path, tmp_cache: Path
) -> None:
    _embed(tmp_nodes, tmp_db, tmp_cache)
    llm_cache.write(
        "deadbeef",
        {"key": "deadbeef", "request": {}, "text": "x", "usage": {}},
        cache_dir=tmp_cache,
    )

    result = runner.invoke(
        app,
        ["cache", "clear", "--yes", "--kind", "embeddings", "--cache-dir", str(tmp_cache)],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert embed_cache.stats(cache_dir=tmp_cache)["entries"] == 0
    assert llm_cache.stats(cache_dir=tmp_cache)["entries"] == 1


def test_cache_clear_all_clears_both(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    _embed(tmp_nodes, tmp_db, tmp_cache)
    llm_cache.write(
        "deadbeef",
        {"key": "deadbeef", "request": {}, "text": "x", "usage": {}},
        cache_dir=tmp_cache,
    )

    runner.invoke(app, ["cache", "clear", "--yes", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert embed_cache.stats(cache_dir=tmp_cache)["entries"] == 0
    assert llm_cache.stats(cache_dir=tmp_cache)["entries"] == 0


def test_cache_clear_rejects_an_unknown_kind(tmp_cache: Path) -> None:
    result = runner.invoke(
        app, ["cache", "clear", "--yes", "--kind", "nope", "--cache-dir", str(tmp_cache)], env=WIDE
    )
    assert result.exit_code == 1
    assert "Unknown --kind value" in _combined(result)


def test_cache_clear_still_requires_yes(tmp_nodes: Path, tmp_db: Path, tmp_cache: Path) -> None:
    _embed(tmp_nodes, tmp_db, tmp_cache)
    result = runner.invoke(app, ["cache", "clear", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert result.exit_code == 1
    assert embed_cache.stats(cache_dir=tmp_cache)["entries"] == 4
