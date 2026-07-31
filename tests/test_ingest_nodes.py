"""Candidates become complete, loadable node files staged in /queue."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from german_wiki import storage
from german_wiki.ingest import _nodes
from german_wiki.ingest._extract import Candidate

NOW = datetime(2026, 7, 26, 9, 14, 3, tzinfo=UTC)
SOURCE_ID = "20260726-notes-a1b2c3d4"


def _candidate(**overrides) -> Candidate:
    data = {
        "title_de": "Wechselpräpositionen",
        "title_en": "Two-way prepositions",
        "type": "grammar",
        "cefr": "A2",
        "cefr_basis": "grammar:wechselpräpositionen",
        "register": ["alltag"],
        "themes": ["haushalt"],
        "body_md": "Akkusativ bei Bewegung, Dativ bei Ort.\n\n## Examples\n- Ich gehe in die Küche.",
        "confidence": 0.9,
    }
    data.update(overrides)
    return Candidate(**data)


def _records() -> list[logging.LogRecord]:
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger("german_wiki.ingest._nodes").addHandler(_Collector())
    return collected


# --- ids ---


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Wechselpräpositionen", "wechselpräpositionen"),
        ("waschen (Wortfamilie)", "waschen-wortfamilie"),
        ("Straße & Verkehr", "straße-verkehr"),
        ("um Hilfe bitten", "um-hilfe-bitten"),
        ("Präfix an-", "präfix-an"),
        ("!!!", "konzept"),
    ],
)
def test_node_id_keeps_real_german(title, expected) -> None:
    """ADR-012: a node id is human-facing, so it carries the actual word.

    Renamed from ``..._reproduces_the_seed_convention``: the hand-authored seeds used
    ``ae``/``ss`` digraphs, and this deliberately no longer matches them. Source ids
    still do -- see ``test_ingest_raw.py``, which asserts the opposite behaviour on
    purpose.
    """
    assert _nodes.node_id_for(title, taken=set()) == expected


def test_node_id_normalizes_precomposed_and_decomposed_umlauts() -> None:
    """The hazard that made ASCII slugs defensible, now handled rather than dodged.

    ``ä`` is U+00E4 precomposed and U+0061 U+0308 decomposed -- identical on screen,
    different bytes, and macOS stores filenames in the second form while Linux and
    Windows use the first. Without NFC these would be two node ids for one word, which
    is precisely the fragmentation ADR-006 exists to prevent.
    """
    precomposed = "Prüfung"
    decomposed = "Prüfung"
    assert precomposed != decomposed  # genuinely different strings
    assert _nodes.node_id_for(precomposed, taken=set()) == "prüfung"
    assert _nodes.node_id_for(decomposed, taken=set()) == "prüfung"


def test_id_collision_within_a_batch_gets_a_suffix() -> None:
    taken: set[str] = set()
    ids = [_nodes.node_id_for("Dativ", taken=taken) for _ in range(3)]
    assert ids == ["dativ", "dativ-2", "dativ-3"]


def test_id_collision_with_an_existing_node_gets_a_suffix(tmp_nodes: Path, tmp_queue: Path) -> None:
    """Never silently overwrite a node the user already has."""
    taken = _nodes.taken_ids(nodes_dir=tmp_nodes, queue_dir=tmp_queue)
    assert "prefix-an" in taken
    assert _nodes.node_id_for("prefix an", taken=taken) == "prefix-an-2"


def test_id_collision_with_a_queued_node_gets_a_suffix(tmp_nodes: Path, tmp_queue: Path) -> None:
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="dativ", now=NOW)
    _nodes.write_queue([node], SOURCE_ID, queue_dir=tmp_queue)

    taken = _nodes.taken_ids(nodes_dir=tmp_nodes, queue_dir=tmp_queue)
    assert _nodes.node_id_for("Dativ", taken=taken) == "dativ-2"


def test_collision_warns() -> None:
    records = _records()
    _nodes.node_id_for("Dativ", taken={"dativ"})
    assert any("is taken" in r.getMessage() for r in records)


def test_taken_ids_on_missing_dirs_is_empty(tmp_path: Path) -> None:
    assert _nodes.taken_ids(nodes_dir=tmp_path / "no", queue_dir=tmp_path / "nope") == set()


# --- mapping ---


def test_mapping_carries_provenance_and_draft_status() -> None:
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="x", now=NOW)
    assert node.source_ids == [SOURCE_ID]
    assert node.status == "draft"
    assert node.version == 1
    assert node.updated_at == NOW


def test_mapping_preserves_candidate_content() -> None:
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="x", now=NOW)
    assert node.title_de == "Wechselpräpositionen"
    assert node.type == "grammar"
    assert node.cefr == "A2"
    assert node.register == ["alltag"]
    assert node.themes == ["haushalt"]
    assert node.confidence == 0.9
    assert "Akkusativ bei Bewegung" in node.body_md


def test_cefr_basis_is_marked_provisional_with_the_model_reason() -> None:
    """SPEC §5: LLM CEFR is unreliable; slice 6 must be able to find every one."""
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="x", now=NOW)
    assert node.cefr_basis == "llm:extraction; grammar:wechselpräpositionen"
    assert node.cefr_basis.startswith(_nodes.PROVISIONAL_CEFR)


def test_cefr_basis_is_marked_provisional_without_one() -> None:
    node = _nodes.to_node(_candidate(cefr_basis=None), source_id=SOURCE_ID, node_id="x", now=NOW)
    assert node.cefr_basis == "llm:extraction"


def test_mapping_invents_no_links_or_examples() -> None:
    """Slices 4-7 own relations; extraction must not fabricate them."""
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="x", now=NOW)
    assert node.links == []
    assert node.examples is None
    assert node.lemmas is None
    assert node.root is None


def test_to_nodes_decollides_across_the_batch(tmp_nodes: Path, tmp_queue: Path) -> None:
    candidates = [_candidate(), _candidate(), _candidate()]
    nodes = _nodes.to_nodes(
        candidates, source_id=SOURCE_ID, nodes_dir=tmp_nodes, queue_dir=tmp_queue, now=NOW
    )
    assert [n.id for n in nodes] == [
        "wechselpräpositionen-2",  # the seed already owns the bare id
        "wechselpräpositionen-3",
        "wechselpräpositionen-4",
    ]


# --- the queue ---


def test_queued_files_are_named_for_their_id(tmp_queue: Path) -> None:
    nodes = [_nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="dativ", now=NOW)]
    paths = _nodes.write_queue(nodes, SOURCE_ID, queue_dir=tmp_queue)
    assert paths[0] == tmp_queue / SOURCE_ID / "dativ.md"


def test_every_queued_file_loads_as_a_node(tmp_queue: Path, tmp_vocab: Path) -> None:
    """The point of the queue: you review real nodes, not an intermediate format."""
    nodes = _nodes.to_nodes(
        [_candidate(), _candidate(title_de="Dativ", title_en="Dative")],
        source_id=SOURCE_ID,
        nodes_dir=tmp_queue,  # empty, so no seed collisions
        queue_dir=tmp_queue,
        now=NOW,
    )
    paths = _nodes.write_queue(nodes, SOURCE_ID, queue_dir=tmp_queue, vocab_dir=tmp_vocab)

    for path, original in zip(paths, nodes, strict=True):
        reloaded = storage.load_node(path)
        assert reloaded == original


def test_queue_write_does_not_grow_the_vocabulary(tmp_queue: Path, tmp_vocab: Path) -> None:
    """ADR-007: only the approved gate (promote) learns. The queue must not."""
    before = (tmp_vocab / "themes.txt").read_text(encoding="utf-8")
    node = _nodes.to_node(
        _candidate(themes=["völlig-neues-thema"]), source_id=SOURCE_ID, node_id="x", now=NOW
    )
    _nodes.write_queue([node], SOURCE_ID, queue_dir=tmp_queue, vocab_dir=tmp_vocab)

    assert (tmp_vocab / "themes.txt").read_text(encoding="utf-8") == before


def test_write_queue_never_touches_nodes_dir(tmp_queue: Path, tmp_nodes: Path) -> None:
    before = sorted(p.name for p in tmp_nodes.glob("*.md"))
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="neu", now=NOW)
    _nodes.write_queue([node], SOURCE_ID, queue_dir=tmp_queue)
    assert sorted(p.name for p in tmp_nodes.glob("*.md")) == before


def test_umlauts_are_literal_in_queued_files(tmp_queue: Path, tmp_vocab: Path) -> None:
    node = _nodes.to_node(_candidate(), source_id=SOURCE_ID, node_id="x", now=NOW)
    path = _nodes.write_queue([node], SOURCE_ID, queue_dir=tmp_queue, vocab_dir=tmp_vocab)[0]
    text = path.read_text(encoding="utf-8")
    assert "Wechselpräpositionen" in text
    assert "\\u00e4" not in text


# --- listing ---


def test_list_queue_groups_by_source(tmp_queue: Path) -> None:
    for source in ("src-a", "src-b"):
        nodes = [_nodes.to_node(_candidate(), source_id=source, node_id=f"{source}-n", now=NOW)]
        _nodes.write_queue(nodes, source, queue_dir=tmp_queue)

    pending = _nodes.list_queue(queue_dir=tmp_queue)
    assert sorted(pending) == ["src-a", "src-b"]
    assert len(pending["src-a"]) == 1


def test_list_queue_on_a_missing_dir_is_empty(tmp_path: Path) -> None:
    assert _nodes.list_queue(queue_dir=tmp_path / "absent") == {}


def test_list_queue_skips_empty_source_dirs(tmp_queue: Path) -> None:
    (tmp_queue / "leer").mkdir(parents=True)
    assert _nodes.list_queue(queue_dir=tmp_queue) == {}
