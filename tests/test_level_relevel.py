"""Re-levelling existing nodes: who gets picked, and that nothing is written."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki import storage
from german_wiki.level import FLAG_BASIS_ONLY, FLAG_TIEBREAK, _lexical, _relevel
from german_wiki.models import Node

EMPTY = Path(__file__).parent / "fixtures" / "cefr-empty"


@pytest.fixture(autouse=True)
def _clear_cache():
    _lexical.clear_cache()
    yield
    _lexical.clear_cache()


def _node(node_id: str, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": "Die Wochentage",
        "title_en": "Weekdays",
        "type": "vocab",
        "cefr": "A2",
        "status": "draft",
        "body_md": "Montag, Dienstag.",
    }
    data.update(overrides)
    return Node(**data)


@pytest.fixture
def corpus(tmp_path: Path, tmp_vocab: Path) -> Path:
    """One node of each basis shape: placeholder, missing, and hand-authored."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    for node in (
        _node("placeholder", cefr_basis="llm:extraction; a guess"),
        _node("no-basis", cefr_basis=None),
        _node("hand-authored", cefr_basis="freq:high; goethe:A1(waschen)"),
        _node("already-derived", cefr_basis="llm:tiebreak(A2); grammar:none; lexical:none"),
    ):
        storage.write_node(node, nodes / f"{node.id}.md", vocab_dir=tmp_vocab)
    return nodes


def _client(level: str = "B1") -> FakeChatClient:
    return FakeChatClient(text=json.dumps({"cefr": level, "reason": "because"}))


# --- target selection ---


def test_targets_are_placeholders_and_missing_bases(corpus: Path) -> None:
    """A missing basis counts as much as a placeholder: SPEC §5 says always store one."""
    ids = [n.id for n in _relevel.targets(storage.load_all_nodes(corpus))]
    assert sorted(ids) == ["no-basis", "placeholder"]


def test_a_hand_authored_basis_is_left_alone(corpus: Path) -> None:
    """It records human judgment, which is the entire reason the field exists."""
    assert "hand-authored" not in [n.id for n in _relevel.targets(storage.load_all_nodes(corpus))]


def test_an_already_derived_basis_is_not_re_targeted(corpus: Path) -> None:
    """`llm:tiebreak` is a derived level with its evidence attached, not a placeholder.

    Treating it as one would re-derive it on every run and never converge.
    """
    assert "already-derived" not in [
        n.id for n in _relevel.targets(storage.load_all_nodes(corpus))
    ]


def test_all_widens_to_every_node(corpus: Path) -> None:
    ids = [n.id for n in _relevel.targets(storage.load_all_nodes(corpus), everything=True)]
    assert len(ids) == 4


# --- proposal construction ---


def test_a_no_op_produces_no_proposal() -> None:
    """Approving no-ops trains you to approve without reading, which breaks the gate."""
    from german_wiki.level._cefr import LevelResult

    node = _node("n", cefr="A2", cefr_basis="grammar:perfekt(A2)")
    same = LevelResult(cefr="A2", basis="grammar:perfekt(A2)")
    assert _relevel.build_proposal(node, same) is None


def test_an_undecidable_result_produces_no_proposal() -> None:
    from german_wiki.level._cefr import LevelResult

    node = _node("n")
    assert _relevel.build_proposal(node, LevelResult(cefr=None, basis="unresolved")) is None


def test_a_basis_only_change_is_flagged() -> None:
    """The level is right but unexplained -- worth proposing, worth marking as minor."""
    from german_wiki.level._cefr import LevelResult

    node = _node("n", cefr="A2", cefr_basis=None)
    proposal = _relevel.build_proposal(node, LevelResult(cefr="A2", basis="grammar:perfekt(A2)"))
    assert FLAG_BASIS_ONLY in proposal.flags
    assert proposal.cefr == "A2"


def test_a_model_derived_level_is_flagged() -> None:
    from german_wiki.level._cefr import LevelResult

    node = _node("n", cefr="A2")
    proposal = _relevel.build_proposal(
        node, LevelResult(cefr="B1", basis="llm:tiebreak(B1)", used_tiebreak=True)
    )
    assert FLAG_TIEBREAK in proposal.flags
    assert proposal.basis == "llm"


def test_a_rules_derived_level_records_its_basis_as_rules() -> None:
    from german_wiki.level._cefr import LevelResult

    proposal = _relevel.build_proposal(
        _node("n"), LevelResult(cefr="B1", basis="grammar:passiv(B1)")
    )
    assert proposal.basis == "rules"
    assert FLAG_TIEBREAK not in proposal.flags


def test_the_proposal_body_shows_both_fields_changing() -> None:
    from german_wiki.level._cefr import LevelResult

    node = _node("n", cefr="A2", cefr_basis="llm:extraction; guess")
    proposal = _relevel.build_proposal(node, LevelResult(cefr="B1", basis="grammar:passiv(B1)"))
    assert "`A2` → `B1`" in proposal.body_md
    assert "llm:extraction; guess" in proposal.body_md
    assert proposal.writes_body is False  # a relevel never rewrites the node body


# --- the run ---


def test_relevel_writes_proposals_and_never_nodes(
    corpus: Path, tmp_path: Path, models_config: Path
) -> None:
    before = {p.name: p.read_bytes() for p in corpus.glob("*.md")}
    proposals_dir = tmp_path / "proposals"

    result = _relevel.relevel(
        nodes_dir=corpus,
        proposals_dir=proposals_dir,
        cefr_dir=EMPTY,
        client=_client("B1"),
        settings_path=models_config,
        cache_dir=tmp_path / "cache",
        usage_log=tmp_path / "usage.jsonl",
    )

    assert result.considered == 2
    assert len(result.proposals) == 2
    assert {p.name: p.read_bytes() for p in corpus.glob("*.md")} == before
    assert all(p.is_file() for p in result.paths)
    assert all(p.kind == "relevel" for p in result.proposals)


def test_relevel_never_changes_a_hand_authored_level_it_cannot_ground(
    corpus: Path, tmp_path: Path, models_config: Path
) -> None:
    """End-to-end pin for the `prefix-an` regression.

    ``no-basis`` is a hand-authored node with no anchors. Re-levelling must record
    ``human:seed`` and leave ``cefr`` exactly where the human put it -- never let an
    ungrounded tiebreak move it because the explanation field was empty.
    """
    client = _client("C1")  # what an ungrounded tiebreak would have proposed
    before = storage.load_node(corpus / "no-basis.md")

    result = _relevel.relevel(
        nodes_dir=corpus,
        proposals_dir=tmp_path / "proposals",
        cefr_dir=EMPTY,
        client=client,
        settings_path=models_config,
        cache_dir=tmp_path / "cache",
        usage_log=tmp_path / "usage.jsonl",
    )

    [proposal] = [p for p in result.proposals if p.candidate == "no-basis"]
    assert proposal.cefr == before.cefr == "A2", "the human's level must survive"
    assert proposal.cefr_basis == "human:seed"
    assert FLAG_BASIS_ONLY in proposal.flags
    assert FLAG_TIEBREAK not in proposal.flags
    assert proposal.basis == "rules"


def test_a_human_seed_basis_converges(corpus: Path, tmp_path: Path, models_config: Path) -> None:
    """`human:seed` is an explanation, so a second run must not re-target it."""
    from german_wiki.level import _cefr

    assert _cefr.is_placeholder("human:seed") is False


def test_rules_only_mode_reports_instead_of_calling_out(corpus: Path, tmp_path: Path) -> None:
    """`--no-llm` must produce an honest gap, not a guessed level.

    The two targets split, and the split is the point: ``placeholder`` has a machine level
    the rules cannot confirm, so it is reported unresolved; ``no-basis`` has a *human*
    level the rules cannot confirm, which is a different situation with a real answer.
    """
    client = _client()
    result = _relevel.relevel(
        nodes_dir=corpus,
        proposals_dir=tmp_path / "proposals",
        cefr_dir=EMPTY,
        allow_llm=False,
        client=client,
    )
    assert client.call_count == 0
    assert result.unresolved == ["placeholder"]
    assert [p.candidate for p in result.proposals] == ["no-basis"]
    assert result.proposals[0].cefr_basis == "human:seed"


def test_proposal_ids_are_stable_across_runs(
    corpus: Path, tmp_path: Path, models_config: Path
) -> None:
    """Re-running overwrites the pending proposal rather than duplicating it."""
    kwargs = {
        "nodes_dir": corpus,
        "proposals_dir": tmp_path / "proposals",
        "cefr_dir": EMPTY,
        "settings_path": models_config,
        "cache_dir": tmp_path / "cache",
        "usage_log": tmp_path / "usage.jsonl",
    }
    first = _relevel.relevel(client=_client("B1"), **kwargs)
    second = _relevel.relevel(client=_client("B1"), **kwargs)

    assert [p.id for p in first.proposals] == [p.id for p in second.proposals]
    assert len(list((tmp_path / "proposals").glob("*.md"))) == 2
