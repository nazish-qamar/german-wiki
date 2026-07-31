"""Applying an approved decision: what lands, what is archived, and what stays untouched."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from german_wiki import merge, storage
from german_wiki.merge import ApplyError, Proposal, _apply, _ledger
from german_wiki.models import Link, Node

WINNER_BODY = "Perfekt mit haben.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n"
LOSER_BODY = "Perfekt mit sein.\n\n## Examples\n- Ich bin gefahren. (I drove.)\n"
MERGED_BODY = "Perfekt mit haben und sein.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n"


def _node(node_id: str, body: str, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": "Perfekt",
        "title_en": "Perfect",
        "type": "grammar",
        "cefr": "A2",
        "status": "stable",
        "body_md": body,
        "source_ids": [f"src-{node_id}"],
        "version": 1,
    }
    data.update(overrides)
    return Node(**data)


@pytest.fixture
def world(tmp_path: Path, tmp_vocab: Path):
    """A winner in /nodes, a loser staged in /queue, and an immutable /raw beside them."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    queue = tmp_path / "queue" / "src-1"
    queue.mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "src-perfekt-haben.txt").write_bytes(b"Quelle A.\n")
    (raw / "src-perfekt-sein.txt").write_bytes(b"Quelle B.\n")

    storage.write_node(_node("perfekt-haben", WINNER_BODY), nodes / "perfekt-haben.md", vocab_dir=tmp_vocab)
    storage.write_node(_node("perfekt-sein", LOSER_BODY), queue / "perfekt-sein.md", vocab_dir=tmp_vocab)

    return {
        "nodes": nodes,
        "queue": queue,
        "raw": raw,
        "vocab": tmp_vocab,
        "merged": tmp_path / "_merged",
        "ledger": tmp_path / "decisions.jsonl",
    }


def _merge_proposal(world, **overrides) -> Proposal:
    data = {
        "id": "merge-perfekt-sein-abc",
        "kind": "merge",
        "outcome": "OVERLAP",
        "candidate": "perfekt-sein",
        "counterpart": "perfekt-haben",
        "winner": "perfekt-haben",
        "loser": "perfekt-sein",
        "similarity": 0.91,
        "tier": "semantic",
        "band": "gray",
        "confidence": 0.85,
        "changelog": "added the sein auxiliary",
        "candidate_path": str(world["queue"] / "perfekt-sein.md"),
        "provider": "zai",
        "model": "glm-4.5-flash",
        "body_md": MERGED_BODY,
        "source_id": "src-1",
    }
    data.update(overrides)
    return Proposal(**data)


def _apply_kwargs(world) -> dict:
    return {
        "nodes_dir": world["nodes"],
        "vocab_dir": world["vocab"],
        "decisions_log": world["ledger"],
    }


def _raw_snapshot(raw: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(raw.iterdir())}


# --- merge ---


def test_an_approved_merge_rewrites_the_winner(world) -> None:
    result = _apply.apply_merge(
        _merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world)
    )
    winner = storage.load_node(world["nodes"] / "perfekt-haben.md")

    assert result.written == ["perfekt-haben"]
    assert winner.body_md.rstrip("\n") == MERGED_BODY.rstrip("\n")
    assert winner.version == 2  # bumped for human legibility; the ledger is the count


def test_a_merge_unions_provenance_mechanically(world) -> None:
    """SPEC §8.1 calls the pointer back to /raw non-negotiable, so no model touches it."""
    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    winner = storage.load_node(world["nodes"] / "perfekt-haben.md")
    assert winner.source_ids == ["src-perfekt-haben", "src-perfekt-sein"]


def test_a_hand_edited_proposal_body_is_what_gets_written(world) -> None:
    edited = "Body I rewrote during review.\n"
    _apply.apply_merge(
        _merge_proposal(world, body_md=edited), merged_dir=world["merged"], **_apply_kwargs(world)
    )
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").body_md.rstrip() == edited.rstrip()


def test_the_loser_is_archived_never_silently_deleted(world) -> None:
    """SPEC §3.2, and the archive is git-tracked because it is the audit trail."""
    before = (world["queue"] / "perfekt-sein.md").read_bytes()
    result = _apply.apply_merge(
        _merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world)
    )

    archived = world["merged"] / "perfekt-sein.md"
    assert result.archived == ["perfekt-sein"]
    assert archived.read_bytes() == before  # byte-for-byte, so git diff means something
    assert not (world["queue"] / "perfekt-sein.md").exists()

    pointer = json.loads((world["merged"] / "perfekt-sein.json").read_text(encoding="utf-8"))
    assert pointer["merged_into"] == "perfekt-haben"
    assert pointer["outcome"] == "OVERLAP"
    assert pointer["decision_id"] == result.decision_id


def test_archiving_twice_does_not_overwrite_the_first_tombstone(world) -> None:
    world["merged"].mkdir(parents=True)
    (world["merged"] / "perfekt-sein.md").write_text("an older casualty\n", encoding="utf-8")

    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    assert (world["merged"] / "perfekt-sein.md").read_text(encoding="utf-8") == "an older casualty\n"
    assert (world["merged"] / "perfekt-sein-2.md").is_file()


def test_raw_is_byte_identical_after_a_merge(world) -> None:
    """§12.1's anchor only works if nothing in this slice can touch it."""
    before = _raw_snapshot(world["raw"])
    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    assert _raw_snapshot(world["raw"]) == before


# --- status as a live trust signal (ADR-011 §7) ---


def test_an_overlap_demotes_a_stable_node_to_reviewed(world) -> None:
    """The body was rewritten and confirmed only as a diff, so `stable` would overclaim."""
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").status == "stable"

    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").status == "reviewed"


def test_a_same_leaves_a_stable_node_stable(world) -> None:
    """SAME appends mechanically -- nothing was re-encoded, so nothing was un-vetted.

    This is the same asymmetry the regeneration cap draws, and deliberately so: the
    operation that counts toward MAX_REGENERATIONS is the operation that demotes.
    """
    _apply.apply_merge(
        _merge_proposal(world, outcome="SAME"),
        merged_dir=world["merged"],
        **_apply_kwargs(world),
    )
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").status == "stable"


@pytest.mark.parametrize("status", ["draft", "reviewed"])
def test_demotion_caps_at_reviewed_and_never_forces_draft(world, status) -> None:
    """A regeneration stops a node overclaiming; it does not undo the review it had."""
    storage.write_node(
        _node("perfekt-haben", WINNER_BODY, status=status),
        world["nodes"] / "perfekt-haben.md",
        vocab_dir=world["vocab"],
    )
    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").status == status


def test_the_cap_and_the_demotion_share_one_definition_of_drift(world) -> None:
    """The invariant, pinned: "OVERLAP is the drift event", applied to both guards.

    Keying `_status_after` off `_ledger.REGENERATING_OUTCOMES` is what stops the two
    guards from silently disagreeing after a future edit to either one.
    """
    for outcome in ("SAME", "OVERLAP"):
        demotes = _apply._status_after(_node("x", WINNER_BODY, status="stable"), outcome) != "stable"
        counts = outcome in _ledger.REGENERATING_OUTCOMES
        assert demotes is counts, f"{outcome}: demotes={demotes} but counts={counts}"


def test_an_approved_link_does_not_touch_status(world) -> None:
    """ADR-010: an edge changes frontmatter, but trust is about the body."""
    storage.write_node(
        _node("perfekt-sein", LOSER_BODY),
        world["nodes"] / "perfekt-sein.md",
        vocab_dir=world["vocab"],
    )
    _apply.apply_link(_link_proposal(world), **_apply_kwargs(world))
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").status == "stable"


def test_a_merge_onto_a_missing_winner_is_refused(world) -> None:
    """write_approved's precondition: a stale proposal must not resurrect a deleted node."""
    (world["nodes"] / "perfekt-haben.md").unlink()
    with pytest.raises(ApplyError):
        _apply.apply_merge(
            _merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world)
        )


# --- link ---


def _link_proposal(world, **overrides) -> Proposal:
    data = {
        "id": "link-perfekt-sein-gov",
        "kind": "link",
        "outcome": "DISTINCT_RELATED",
        "candidate": "perfekt-sein",
        "counterpart": "perfekt-haben",
        "relation": "governs",
        "direction": "a_to_b",
        "confidence": 0.85,
        "similarity": 0.88,
        "body_md": "Proposed edge summary that must never be written.\n",
    }
    data.update(overrides)
    return Proposal(**data)


def test_an_approved_link_touches_frontmatter_only(world) -> None:
    """ADR-010: a proposed edge writes nothing to the bodies."""
    storage.write_node(
        _node("perfekt-sein", LOSER_BODY), world["nodes"] / "perfekt-sein.md", vocab_dir=world["vocab"]
    )
    before = {
        node_id: storage.load_node(world["nodes"] / f"{node_id}.md").body_md
        for node_id in ("perfekt-haben", "perfekt-sein")
    }

    result = _apply.apply_link(_link_proposal(world), **_apply_kwargs(world))
    after = {
        node_id: storage.load_node(world["nodes"] / f"{node_id}.md")
        for node_id in ("perfekt-haben", "perfekt-sein")
    }

    assert result.written == ["perfekt-haben"]  # direction a_to_b: counterpart -> candidate
    assert after["perfekt-haben"].links == [
        Link(target="perfekt-sein", relation="governs", confidence=0.85)
    ]
    assert after["perfekt-sein"].links == []
    for node_id, body in before.items():
        assert after[node_id].body_md == body


def test_direction_decides_which_node_carries_the_edge(world) -> None:
    storage.write_node(
        _node("perfekt-sein", LOSER_BODY), world["nodes"] / "perfekt-sein.md", vocab_dir=world["vocab"]
    )
    _apply.apply_link(_link_proposal(world, direction="b_to_a"), **_apply_kwargs(world))

    assert storage.load_node(world["nodes"] / "perfekt-sein.md").links[0].target == "perfekt-haben"
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").links == []


def test_a_dangling_edge_is_refused_with_the_fix_named(world) -> None:
    """The candidate is still staged, so its `create` proposal has not been approved yet."""
    with pytest.raises(ApplyError, match="Approve its `create` proposal first"):
        _apply.apply_link(_link_proposal(world), **_apply_kwargs(world))
    assert storage.load_node(world["nodes"] / "perfekt-haben.md").links == []


def test_re_applying_a_link_is_idempotent(world) -> None:
    storage.write_node(
        _node("perfekt-sein", LOSER_BODY), world["nodes"] / "perfekt-sein.md", vocab_dir=world["vocab"]
    )
    _apply.apply_link(_link_proposal(world), **_apply_kwargs(world))
    _apply.apply_link(_link_proposal(world), **_apply_kwargs(world))
    assert len(storage.load_node(world["nodes"] / "perfekt-haben.md").links) == 1


# --- create and discard ---


def test_an_approved_create_promotes_the_candidate(world) -> None:
    proposal = Proposal(
        id="create-perfekt-sein-abc",
        kind="create",
        outcome="DISTINCT",
        candidate="perfekt-sein",
        candidate_path=str(world["queue"] / "perfekt-sein.md"),
        body_md=LOSER_BODY,
    )
    result = _apply.apply_create(proposal, **_apply_kwargs(world))

    assert result.written == ["perfekt-sein"]
    assert (world["nodes"] / "perfekt-sein.md").is_file()
    assert not (world["queue"] / "perfekt-sein.md").exists()


def test_a_create_that_would_overwrite_is_refused(world) -> None:
    proposal = Proposal(
        id="create-perfekt-haben-abc",
        kind="create",
        outcome="DISTINCT",
        candidate="perfekt-haben",
        body_md=LOSER_BODY,
    )
    with pytest.raises(ValueError, match="already exists"):
        _apply.apply_create(proposal, **_apply_kwargs(world))


def test_a_rejection_writes_nothing_but_is_recorded(world) -> None:
    before = sorted(p.name for p in world["nodes"].iterdir())
    result = _apply.apply_discard(
        _merge_proposal(world), approved=False, decisions_log=world["ledger"]
    )

    assert result.written == [] and result.archived == []
    assert sorted(p.name for p in world["nodes"].iterdir()) == before
    assert (world["queue"] / "perfekt-sein.md").is_file()  # candidate stays staged

    [record] = _ledger.read_all(decisions_log=world["ledger"])
    assert record.approved is False
    assert record.winner == "perfekt-haben" and record.loser == "perfekt-sein"


# --- the ledger ---


def test_every_decision_records_which_model_decided_it(world) -> None:
    """ADR-011: flash verdicts are pipeline development, so they must stay a grep."""
    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    [record] = _ledger.read_all(decisions_log=world["ledger"])

    assert (record.provider, record.model) == ("zai", "glm-4.5-flash")
    assert record.basis == "llm"
    assert record.changelog == "added the sein auxiliary"
    assert record.similarity == 0.91


def test_an_approved_merge_counts_toward_the_cap(world) -> None:
    _apply.apply_merge(_merge_proposal(world), merged_dir=world["merged"], **_apply_kwargs(world))
    assert _ledger.merge_count("perfekt-haben", decisions_log=world["ledger"]) == 1


def test_the_ledger_is_append_only(world) -> None:
    _apply.apply_discard(_merge_proposal(world), approved=False, decisions_log=world["ledger"])
    first = world["ledger"].read_text(encoding="utf-8")
    _apply.apply_discard(
        _merge_proposal(world, id="other"), approved=False, decisions_log=world["ledger"]
    )
    assert world["ledger"].read_text(encoding="utf-8").startswith(first)


# --- session ordering ---


def test_creates_are_applied_before_links(world) -> None:
    """A link must never reference a node whose own proposal is still pending."""
    ordered = merge.review_order(
        [_link_proposal(world), _merge_proposal(world), Proposal(
            id="create-x", kind="create", outcome="DISTINCT", candidate="x"
        )]
    )
    assert [p.kind for p in ordered] == ["create", "merge", "link"]
