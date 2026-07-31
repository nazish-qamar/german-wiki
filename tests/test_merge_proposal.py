"""Proposals are the durable state (ADR-011), so the file has to round-trip exactly."""

from __future__ import annotations

from pathlib import Path

import pytest

from german_wiki.merge import Proposal, _proposal, delete_proposal, list_proposals, load_proposal


def _proposal_obj(**overrides) -> Proposal:
    data = {
        "id": "merge-b-1234abcd",
        "kind": "merge",
        "outcome": "OVERLAP",
        "candidate": "b",
        "counterpart": "a",
        "winner": "a",
        "loser": "b",
        "similarity": 0.91,
        "tier": "semantic",
        "band": "gray",
        "confidence": 0.8,
        "reason": "same tense, B adds the auxiliary",
        "body_md": "Merged body.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n",
    }
    data.update(overrides)
    return Proposal(**data)


def test_a_proposal_round_trips_through_its_file(tmp_proposals: Path) -> None:
    """Semantic round-trip, not byte-identical -- the convention ``storage.py`` sets.

    Trailing newlines on the body are normalized away, which is harmless because
    ``write_node`` re-normalizes them on the way into ``/nodes`` anyway.
    """
    original = _proposal_obj()
    path = _proposal.write_proposal(original, proposals_dir=tmp_proposals)
    loaded = load_proposal(path)

    assert loaded.body_md.rstrip("\n") == original.body_md.rstrip("\n")
    assert loaded.model_dump(exclude={"body_md"}) == original.model_dump(exclude={"body_md"})


def test_umlauts_survive_the_round_trip(tmp_proposals: Path) -> None:
    original = _proposal_obj(reason="Wechselpräpositionen — Küche, Büro", body_md="Größe.\n")
    path = _proposal.write_proposal(original, proposals_dir=tmp_proposals)
    assert load_proposal(path).reason == "Wechselpräpositionen — Küche, Büro"


def test_the_body_is_the_proposed_content(tmp_proposals: Path) -> None:
    """A merge proposal's body IS the Markdown that will land in /nodes."""
    path = _proposal.write_proposal(_proposal_obj(), proposals_dir=tmp_proposals)
    assert path.read_text(encoding="utf-8").rstrip().endswith("- Ich habe gearbeitet. (I worked.)")


def test_a_hand_edit_during_review_is_what_loads_back(tmp_proposals: Path) -> None:
    """ADR-009's idiom: review needs nothing but an editor, and the edit is the approval."""
    path = _proposal.write_proposal(_proposal_obj(), proposals_dir=tmp_proposals)
    text = path.read_text(encoding="utf-8").replace("Merged body.", "Body I rewrote by hand.")
    path.write_text(text, encoding="utf-8")

    assert load_proposal(path).body_md.startswith("Body I rewrote by hand.")


def test_a_hand_edit_that_breaks_the_schema_is_caught_on_load(tmp_proposals: Path) -> None:
    path = _proposal.write_proposal(_proposal_obj(), proposals_dir=tmp_proposals)
    path.write_text(
        path.read_text(encoding="utf-8").replace("kind: merge", "kind: obliterate"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_proposal(path)


def test_the_id_must_match_the_filename_stem(tmp_proposals: Path) -> None:
    path = _proposal.write_proposal(_proposal_obj(), proposals_dir=tmp_proposals)
    renamed = path.with_name("something-else.md")
    path.rename(renamed)
    with pytest.raises(ValueError, match="does not match filename stem"):
        load_proposal(renamed)


# --- ids ---


def test_ids_are_stable_across_runs() -> None:
    """Re-running `gw adjudicate` must overwrite a pending proposal, not duplicate it."""
    first = _proposal.proposal_id("merge", "b", "a")
    second = _proposal.proposal_id("merge", "b", "a")
    assert first == second


def test_ids_separate_the_kinds_and_the_relations() -> None:
    """One candidate can earn several proposals; none of them may collide."""
    ids = {
        _proposal.proposal_id("merge", "b", "a"),
        _proposal.proposal_id("create", "b"),
        _proposal.proposal_id("link", "b", "a", "governs"),
        _proposal.proposal_id("link", "b", "a", "prerequisite_for"),
    }
    assert len(ids) == 4


def test_a_long_candidate_id_stays_a_usable_filename() -> None:
    pid = _proposal.proposal_id("merge", "x" * 200, "a")
    assert len(pid) < 60
    assert "/" not in pid and "\\" not in pid


# --- the directory ---


def test_listing_is_empty_before_anything_is_proposed(tmp_proposals: Path) -> None:
    assert list_proposals(proposals_dir=tmp_proposals) == []


def test_listing_puts_the_strongest_pair_first(tmp_proposals: Path) -> None:
    for pid, sim in (("merge-a-1", 0.88), ("merge-b-2", 0.97), ("merge-c-3", 0.91)):
        _proposal.write_proposal(
            _proposal_obj(id=pid, similarity=sim), proposals_dir=tmp_proposals
        )
    assert [p.id for p in list_proposals(proposals_dir=tmp_proposals)] == [
        "merge-b-2",
        "merge-c-3",
        "merge-a-1",
    ]


def test_deleting_resolves_a_proposal_and_tidies_up(tmp_proposals: Path) -> None:
    """Approval and rejection both end here, so /proposals only holds pending work."""
    _proposal.write_proposal(_proposal_obj(), proposals_dir=tmp_proposals)
    assert delete_proposal("merge-b-1234abcd", proposals_dir=tmp_proposals) is True
    assert list_proposals(proposals_dir=tmp_proposals) == []
    assert not tmp_proposals.exists()  # empty directory removed


def test_deleting_something_absent_is_not_an_error(tmp_proposals: Path) -> None:
    assert delete_proposal("nope", proposals_dir=tmp_proposals) is False


# --- what approving would write ---


@pytest.mark.parametrize(
    ("kind", "writes"), [("merge", True), ("create", True), ("link", False), ("discard", False)]
)
def test_only_merge_and_create_write_a_body(kind, writes) -> None:
    """ADR-010: an approved edge changes frontmatter, never a body."""
    assert _proposal_obj(kind=kind).writes_body is writes


def test_the_pair_is_order_independent() -> None:
    assert _proposal_obj(candidate="b", counterpart="a").pair == ("a", "b")
    assert _proposal_obj(candidate="a", counterpart="b").pair == ("a", "b")
    assert _proposal_obj(counterpart=None).pair is None
