"""Three-tier detection: what it finds, how it bands, and that it writes nothing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from conftest import FakeEmbedder

from german_wiki.db import connect, rebuild_schema
from german_wiki.embed import _detect, _store
from german_wiki.models import Node

BODY = (
    "Wechselpräpositionen stehen mit Akkusativ, wenn eine Bewegung gemeint ist, "
    "und mit Dativ, wenn ein Ort gemeint ist."
)


def _node(node_id: str, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": "Wechselpräpositionen",
        "title_en": "Two-way prepositions",
        "type": "grammar",
        "cefr": "A2",
        "status": "draft",
        "body_md": BODY,
    }
    data.update(overrides)
    return Node(**data)


@pytest.fixture
def conn(tmp_db: Path):
    connection = connect(tmp_db)
    rebuild_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def _find(nodes, conn, tmp_cache, **kwargs):
    kwargs.setdefault("embedder", FakeEmbedder())
    return _detect.find_duplicates(nodes, conn=conn, cache_dir=tmp_cache, **kwargs)


# --- tier 1: exact ---


def test_identical_text_is_an_exact_duplicate(conn, tmp_cache) -> None:
    report = _find([_node("a"), _node("b")], conn, tmp_cache)

    exact = [m for m in report.matches if m.tier == "exact"]
    assert len(exact) == 1
    assert (exact[0].left_id, exact[0].right_id) == ("a", "b")
    assert exact[0].similarity == 1.0
    assert exact[0].band == "duplicate"


def test_case_and_whitespace_differences_are_still_exact(conn, tmp_cache) -> None:
    """Tier-1 normalization is aggressive on purpose."""
    report = _find([_node("a"), _node("b", body_md=BODY.upper() + "   ")], conn, tmp_cache)
    assert any(m.tier == "exact" for m in report.matches)


def test_a_symmetric_pair_is_reported_once(conn, tmp_cache) -> None:
    report = _find([_node("a"), _node("b")], conn, tmp_cache)
    pairs = {(m.left_id, m.right_id) for m in report.matches}
    assert ("b", "a") not in pairs
    assert len(report.matches) == 1


# --- tier 1: near-exact ---


def test_a_small_edit_is_near_exact(conn, tmp_cache) -> None:
    edited = BODY.replace("ein Ort gemeint ist", "ein Ort gemeint sein sollte")
    report = _find([_node("a"), _node("b", body_md=edited)], conn, tmp_cache)

    near = [m for m in report.matches if m.tier == "near-exact"]
    assert len(near) == 1
    assert _detect.NEAR_EXACT_JACCARD <= near[0].similarity < 1.0


def test_unrelated_nodes_are_not_reported(conn, tmp_cache) -> None:
    other = _node(
        "b",
        title_de="Die Waschmaschine",
        title_en="The washing machine",
        body_md="Die Waschmaschine steht in der Küche und wäscht die Wäsche.",
    )
    report = _find([_node("a"), other], conn, tmp_cache)
    assert report.matches == []


def test_exact_wins_over_near_exact(conn, tmp_cache) -> None:
    """A pair is classified by the cheapest tier that catches it."""
    report = _find([_node("a"), _node("b")], conn, tmp_cache)
    assert [m.tier for m in report.matches] == ["exact"]


# --- tier 2: semantic banding ---


@pytest.mark.parametrize(
    "similarity,expected",
    [
        (0.99, "duplicate"),
        (_detect.GRAY_HIGH, "duplicate"),
        (0.90, "gray"),
        (_detect.GRAY_LOW, "gray"),
    ],
)
def test_banding_at_the_spec_boundaries(similarity, expected) -> None:
    """SPEC §3.1: >= GRAY_HIGH is a duplicate, GRAY_LOW..GRAY_HIGH earns adjudication."""
    assert _detect._band(similarity) == expected


def test_semantically_close_nodes_land_in_the_gray_zone(conn, tmp_cache) -> None:
    """Different words, related meaning -- the case only embeddings catch."""
    a = _node("a")
    b = _node(
        "b",
        title_de="Lokale Präpositionen",
        title_en="Local prepositions",
        body_md="Ortsangaben verlangen den Dativ; Richtungsangaben verlangen den Akkusativ.",
    )
    from german_wiki.embed._text import embed_text

    # Place b's vector deliberately near a's, which is what a real model would do
    # for a paraphrase and what a hash or shingle comparison cannot see.
    embedder = FakeEmbedder(similar_to={embed_text(b): embed_text(a)})
    report = _find([a, b], conn, tmp_cache, embedder=embedder)

    semantic = [m for m in report.matches if m.tier == "semantic"]
    assert len(semantic) == 1
    assert semantic[0].similarity >= _detect.GRAY_LOW
    assert report.gray or report.duplicates


def test_below_the_gray_floor_is_silent(conn, tmp_cache) -> None:
    a = _node("a")
    b = _node(
        "b", title_de="Ganz anderes", title_en="Entirely other", body_md="Nichts damit zu tun."
    )
    assert _find([a, b], conn, tmp_cache).matches == []


# --- comparing candidates against an existing corpus ---


def test_candidates_are_compared_against_existing_nodes(conn, tmp_cache) -> None:
    existing = _node("existing")
    candidate = _node("candidate")

    report = _find([candidate], conn, tmp_cache, against=[existing])

    assert len(report.matches) == 1
    assert {report.matches[0].left_id, report.matches[0].right_id} == {"candidate", "existing"}


def test_pairs_entirely_outside_the_focus_are_not_reported(conn, tmp_cache) -> None:
    """Two existing nodes duplicating each other is not this run's question."""
    report = _find(
        [_node("candidate", body_md="Etwas völlig anderes über Kochen.")],
        conn,
        tmp_cache,
        against=[_node("old-a"), _node("old-b")],
    )
    assert report.matches == []


def test_candidates_are_compared_to_each_other(conn, tmp_cache) -> None:
    """Slice 3 has no dedup, so one source can emit two takes on one concept."""
    report = _find([_node("cand-a"), _node("cand-b")], conn, tmp_cache, against=[_node("old")])
    pairs = {(m.left_id, m.right_id) for m in report.matches}
    assert ("cand-a", "cand-b") in pairs


# --- the report-only invariant, asserted in BOTH directions ---


def test_detection_writes_nothing_to_the_source_of_truth(
    conn, tmp_cache, tmp_nodes: Path, tmp_queue: Path, tmp_vocab: Path
) -> None:
    """Source of truth untouched; the derived index legitimately gains vectors.

    Those are different layers. /nodes, /queue and vocab are authoritative and must
    be byte-identical afterwards. data/index.db is derived (ADR-001) and SHOULD
    change -- detection embeds lazily, and that is not a violation.
    """
    (tmp_queue / "src").mkdir(parents=True)
    shutil.copy2(next(tmp_nodes.glob("*.md")), tmp_queue / "src" / "copy.md")

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
        }

    before = {
        "nodes": snapshot(tmp_nodes),
        "queue": snapshot(tmp_queue),
        "vocab": snapshot(tmp_vocab),
    }
    assert _store.stored_ids(conn) == set()

    report = _find([_node("a"), _node("b")], conn, tmp_cache)

    # Direction 1: authoritative files are untouched.
    assert snapshot(tmp_nodes) == before["nodes"]
    assert snapshot(tmp_queue) == before["queue"]
    assert snapshot(tmp_vocab) == before["vocab"]

    # Direction 2: the derived index DID gain vectors. Asserting it is unchanged
    # would be wrong -- embedding is a separate layer from detection.
    assert _store.stored_ids(conn) == {"a", "b"}
    assert report.embedding is not None
    assert report.embedding.computed == 2


def test_a_second_run_recomputes_nothing(conn, tmp_cache) -> None:
    """The embedding cache makes re-running detection free."""
    embedder = FakeEmbedder()
    nodes = [_node("a"), _node("b")]

    first = _find(nodes, conn, tmp_cache, embedder=embedder)
    second = _find(nodes, conn, tmp_cache, embedder=embedder)

    assert first.embedding.computed == 2
    assert second.embedding.computed == 0
    assert second.embedding.from_cache == 2
    assert embedder.encode_count == 1  # only the first run did any work


def test_empty_corpus_reports_nothing(conn, tmp_cache) -> None:
    report = _find([], conn, tmp_cache)
    assert report.matches == []
    assert report.compared == 0
