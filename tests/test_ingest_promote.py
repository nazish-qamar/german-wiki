"""Promotion: the ADR-003 gate into /nodes and the only ADR-007 learn=True caller."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from german_wiki import storage
from german_wiki.ingest import _nodes, _promote
from german_wiki.ingest._extract import Candidate

NOW = datetime(2026, 7, 26, 9, 14, 3, tzinfo=UTC)
SOURCE_ID = "20260726-notes-a1b2c3d4"


def _candidate(**overrides) -> Candidate:
    data = {
        "title_de": "Wechselpräpositionen",
        "title_en": "Two-way prepositions",
        "type": "grammar",
        "cefr": "A2",
        "cefr_basis": "grammar:test",
        "register": ["alltag"],
        "themes": ["haushalt"],
        "body_md": "Akkusativ bei Bewegung, Dativ bei Ort.",
        "confidence": 0.9,
    }
    data.update(overrides)
    return Candidate(**data)


@pytest.fixture
def empty_nodes(tmp_path: Path) -> Path:
    path = tmp_path / "nodes"
    path.mkdir()
    return path


def _queue(tmp_queue: Path, empty_nodes: Path, tmp_vocab: Path, candidates, source=SOURCE_ID):
    nodes = _nodes.to_nodes(
        candidates, source_id=source, nodes_dir=empty_nodes, queue_dir=tmp_queue, now=NOW
    )
    return _nodes.write_queue(nodes, source, queue_dir=tmp_queue, vocab_dir=tmp_vocab)


def _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db, source=SOURCE_ID, **overrides):
    return _promote.promote_source(
        source,
        queue_dir=tmp_queue,
        nodes_dir=empty_nodes,
        vocab_dir=tmp_vocab,
        db_path=tmp_db,
        **overrides,
    )


# --- the happy path ---


def test_promotes_every_queued_node(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(
        tmp_queue,
        empty_nodes,
        tmp_vocab,
        [_candidate(), _candidate(title_de="Dativ", title_en="Dative")],
    )
    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert sorted(result.promoted) == ["dativ", "wechselpraepositionen"]
    assert sorted(p.stem for p in empty_nodes.glob("*.md")) == ["dativ", "wechselpraepositionen"]
    assert result.refused == []


def test_promoted_nodes_survive_the_round_trip(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate()])
    _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    node = storage.load_node(empty_nodes / "wechselpraepositionen.md")
    assert node.status == "draft"
    assert node.source_ids == [SOURCE_ID]
    assert node.title_de == "Wechselpräpositionen"


def test_queue_entries_are_removed_on_success(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate()])
    _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert list(tmp_queue.glob("*/*.md")) == []
    assert not (tmp_queue / SOURCE_ID).exists()  # empty source dir cleaned up


def test_reindex_runs_and_the_node_is_queryable(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate(), _candidate(title_de="Dativ")])
    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert result.reindexed == {"nodes": 2, "links": 0, "themes": 2}
    assert tmp_db.exists()


def test_reindex_can_be_skipped(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate()])
    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db, reindex=False)
    assert result.reindexed is None
    assert not tmp_db.exists()


# --- ADR-007: this is the only place the vocabulary grows ---


def test_promote_learns_new_tags(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    """The load-bearing ADR-007 assertion: the known-set grows at the approved gate."""
    themes_file = tmp_vocab / "themes.txt"
    before = themes_file.read_text(encoding="utf-8")
    assert "arzt" not in before

    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate(themes=["arzt"])])
    # Staging alone must not have learned it.
    assert themes_file.read_text(encoding="utf-8") == before

    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert "arzt" in themes_file.read_text(encoding="utf-8").splitlines()
    assert result.learned_tags == {"themes": ["arzt"]}


def test_known_tags_are_not_relearned(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate(themes=["haushalt"])])
    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)
    assert result.learned_tags == {}


def test_tags_are_normalized_into_nodes(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    """`kitchen` is an alias for `küche` in vocab/aliases.yaml."""
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate(themes=["  KITCHEN "])])
    _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert storage.load_node(empty_nodes / "wechselpraepositionen.md").themes == ["küche"]


# --- refusals: never overwrite, never block the batch ---


def test_existing_node_is_refused_and_stays_queued(
    tmp_queue, empty_nodes, tmp_vocab, tmp_db
) -> None:
    paths = _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate()])
    existing = empty_nodes / "wechselpraepositionen.md"
    existing.write_text("---\nhandmade\n---\n", encoding="utf-8")

    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert result.promoted == []
    assert len(result.refused) == 1
    assert "already exists" in result.refused[0].reason
    assert existing.read_text(encoding="utf-8") == "---\nhandmade\n---\n"  # untouched
    assert paths[0].exists()  # still queued


def test_an_invalid_hand_edit_is_refused_but_others_promote(
    tmp_queue, empty_nodes, tmp_vocab, tmp_db
) -> None:
    """Validation on promote is why this is not a plain file move."""
    paths = _queue(
        tmp_queue,
        empty_nodes,
        tmp_vocab,
        [_candidate(), _candidate(title_de="Dativ", title_en="Dative")],
    )
    broken = next(p for p in paths if p.stem == "dativ")
    broken.write_text("---\ncefr: NOT-A-LEVEL\n---\nkaputt\n", encoding="utf-8")

    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert result.promoted == ["wechselpraepositionen"]
    assert [r.node_id for r in result.refused] == ["dativ"]
    assert broken.exists()  # left for you to fix
    assert (empty_nodes / "wechselpraepositionen.md").exists()


def test_refused_queue_dir_is_not_removed(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    _queue(tmp_queue, empty_nodes, tmp_vocab, [_candidate()])
    (empty_nodes / "wechselpraepositionen.md").write_text("x", encoding="utf-8")
    _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)
    assert (tmp_queue / SOURCE_ID).is_dir()


def test_unknown_source_raises(tmp_queue, empty_nodes, tmp_vocab, tmp_db) -> None:
    with pytest.raises(ValueError, match="nothing queued for source"):
        _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db, source="does-not-exist")


def test_rejecting_a_candidate_means_deleting_its_file(
    tmp_queue, empty_nodes, tmp_vocab, tmp_db
) -> None:
    """The review mechanism this slice ships: delete, then promote."""
    paths = _queue(
        tmp_queue,
        empty_nodes,
        tmp_vocab,
        [_candidate(), _candidate(title_de="Unsinn", title_en="Nonsense")],
    )
    next(p for p in paths if p.stem == "unsinn").unlink()

    result = _run(tmp_queue, empty_nodes, tmp_vocab, tmp_db)

    assert result.promoted == ["wechselpraepositionen"]
    assert not (empty_nodes / "unsinn.md").exists()
