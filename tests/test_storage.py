"""Round-trip fidelity and id/stem enforcement."""

from __future__ import annotations

import pytest

from german_wiki import storage

from conftest import SEED_IDS


@pytest.mark.parametrize("sid", SEED_IDS)
def test_round_trip_equal(seed_nodes, tmp_path, tmp_vocab, sid):
    """load -> write -> load yields an equal Node (semantic round-trip)."""
    node = seed_nodes[sid]
    out = tmp_path / f"{sid}.md"
    storage.write_node(node, out, vocab_dir=tmp_vocab)
    reloaded = storage.load_node(out)
    assert reloaded == node


def test_absent_vs_empty_preserved(seed_nodes, tmp_path, tmp_vocab):
    we = seed_nodes["wechselpräpositionen"]
    pa = seed_nodes["prefix-an"]

    we_out = tmp_path / "wechselpräpositionen.md"
    pa_out = tmp_path / "prefix-an.md"
    storage.write_node(we, we_out, vocab_dir=tmp_vocab)
    storage.write_node(pa, pa_out, vocab_dir=tmp_vocab)

    # present-empty stays [], absent stays absent (no invented field)
    assert storage.load_node(we_out).themes == []
    assert storage.load_node(pa_out).themes is None
    assert "themes:" not in pa_out.read_text(encoding="utf-8")
    assert "cefr_basis:" not in pa_out.read_text(encoding="utf-8")


def test_id_must_match_filename_stem(seed_nodes, tmp_path, tmp_vocab):
    node = seed_nodes["prefix-an"]
    wrong = tmp_path / "not-the-id.md"
    storage.write_node(node, wrong, vocab_dir=tmp_vocab)
    with pytest.raises(ValueError, match="does not match filename stem"):
        storage.load_node(wrong)


def test_load_all_nodes_sorted(nodes_dir):
    ids = [n.id for n in storage.load_all_nodes(nodes_dir)]
    assert ids == sorted(SEED_IDS)


def test_umlauts_written_literally(seed_nodes, tmp_path, tmp_vocab):
    """German chars must be literal UTF-8 in the file, not \\uXXXX escapes."""
    node = seed_nodes["familie-waschen"]  # has Wäsche, Waschmaschine, etc.
    out = tmp_path / "familie-waschen.md"
    storage.write_node(node, out, vocab_dir=tmp_vocab)
    text = out.read_text(encoding="utf-8")
    assert "ä" in text          # literal, not escaped
    assert "\\u00e4" not in text


def test_datetime_round_trips(seed_nodes, tmp_path, tmp_vocab):
    from datetime import datetime, timezone
    node = seed_nodes["prefix-an"].model_copy(
        update={"updated_at": datetime(2026, 1, 15, tzinfo=timezone.utc)}
    )
    out = tmp_path / "prefix-an.md"
    storage.write_node(node, out, vocab_dir=tmp_vocab)
    assert storage.load_node(out).updated_at == node.updated_at