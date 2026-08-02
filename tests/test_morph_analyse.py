"""On-demand analysis: what it proposes, what it leaves alone, and that it writes nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.models import Link, Node
from german_wiki.morph._analyse import analyse


def _family(node_id: str, root: str, lemmas: list[str], **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": root,
        "title_en": root,
        "type": "vocab",
        "cefr": "A2",
        "status": "stable",
        "root": root,
        "lemmas": lemmas,
        "body_md": "",
    }
    data.update(overrides)
    return Node(**data)


def _prefix(morpheme: str) -> Node:
    return Node(
        id=f"prefix-{morpheme}",
        title_de=f"{morpheme}- (Präfix)",
        title_en=f"prefix {morpheme}-",
        type="pattern",
        cefr="A2",
        status="stable",
        separable=True,
        body_md="",
    )


def _verdict(value: str = "high") -> str:
    return json.dumps({"transparency": value, "reason": "because", "confidence": 0.9})


# --- what it proposes ---


def test_a_missing_prefix_node_is_proposed_as_a_create() -> None:
    result = analyse(
        [_family("f", "waschen", ["waschen", "abwaschen"], family_transparency="high")],
        judge_transparency=False,
    )
    creates = result.by_kind("create")
    assert [p.candidate for p in creates] == ["prefix-ab"]


def test_a_proposed_prefix_node_does_not_invent_its_meaning() -> None:
    """SPEC §4.1 forbids unsourced claims, and a prefix inventory knows separability,
    not meaning. The body is a template with a TODO, hand-editable at review.

    The template names **no prefix glosses at all**, not even as illustrations: an example
    like "e.g. an- = toward/on" would read as this node's own content whenever the node
    being created happens to be `prefix-an`.
    """
    result = analyse(
        [_family("f", "waschen", ["waschen", "abwaschen"], family_transparency="high")],
        judge_transparency=False,
    )
    body = result.by_kind("create")[0].body_md
    assert "TODO" in body
    assert "abwaschen" in body  # it does state the evidence that motivated it
    # No glosses: the pipeline never writes "<prefix>- = <meaning>" for any prefix.
    assert " = " not in body
    for gloss in ("toward", "out/off", "up/open", "away"):
        assert gloss not in body.lower()


def test_a_family_edge_is_proposed_when_not_already_stated() -> None:
    result = analyse(
        [
            _prefix("auf"),
            _family("f", "waschen", ["waschen", "aufwaschen"], family_transparency="high"),
        ],
        judge_transparency=False,
    )
    links = result.by_kind("link")
    assert [(p.candidate, p.counterpart, p.relation) for p in links] == [
        ("f", "prefix-auf", "same_family")
    ]


# --- what it leaves alone ---


def test_an_existing_edge_is_not_re_proposed_even_when_it_dangles() -> None:
    """A dangling target is your intention (SPEC §7.3), not an outstanding task.

    This is the case the real corpus is in: familie-waschen already points at prefix-ab,
    which does not exist. Re-proposing it would turn the roadmap into a chore list.
    """
    family = _family(
        "f",
        "waschen",
        ["waschen", "abwaschen"],
        family_transparency="high",
        links=[Link(target="prefix-ab", relation="same_family")],
    )
    result = analyse([family], judge_transparency=False)
    assert result.by_kind("link") == []


def test_an_existing_transparency_is_never_re_judged(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    """A model verdict must not overwrite yours -- so it is not even asked for."""
    client = FakeChatClient(text=_verdict("opaque"))
    result = analyse(
        [_family("f", "waschen", ["waschen", "abwaschen"], family_transparency="high")],
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert client.call_count == 0
    assert result.judged == 0
    assert result.by_kind("morphology") == []


def test_a_family_without_transparency_is_judged_once(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text=_verdict("drifted"))
    result = analyse(
        [_family("f", "waschen", ["waschen", "abwaschen", "aufwaschen"])],
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert client.call_count == 1  # per family, not per lemma
    proposal = result.by_kind("morphology")[0]
    assert proposal.family_transparency == "drifted"
    assert proposal.kind == "morphology" and proposal.outcome == "MORPHOLOGY"


def test_rules_only_mode_makes_no_model_call(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text=_verdict())
    analyse(
        [_family("f", "waschen", ["waschen", "abwaschen"])],
        judge_transparency=False,
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert client.call_count == 0


# --- the withheld claim, end to end ---


def test_a_stress_ambiguous_lemma_is_reported_not_proposed() -> None:
    """No link, no create, no cell -- it surfaces as needing a human instead."""
    result = analyse(
        [_family("f", "fahren", ["fahren", "umfahren"], family_transparency="high")],
        judge_transparency=False,
    )
    assert [w.word for w in result.ambiguous] == ["umfahren"]
    assert result.proposals == []


def test_a_lemma_with_no_corpus_evidence_is_neither_proposed_nor_flagged() -> None:
    """It resolves itself as you study, so it is not asked about."""
    result = analyse(
        [_family("f", "waschen", ["waschen", "verstehen"], family_transparency="high")],
        judge_transparency=False,
    )
    assert [w.word for w in result.unresolved] == ["verstehen"]
    assert result.ambiguous == []
    assert result.by_kind("create") == []


# --- it writes nothing ---


def test_analysis_touches_no_files(tmp_path: Path, tmp_vocab: Path) -> None:
    from german_wiki import storage

    nodes_dir = tmp_path / "nodes"
    nodes_dir.mkdir()
    family = _family("f", "waschen", ["waschen", "abwaschen"], family_transparency="high")
    storage.write_node(family, nodes_dir / "f.md", vocab_dir=tmp_vocab)
    before = {p.name: p.read_bytes() for p in nodes_dir.iterdir()}

    analyse(storage.load_all_nodes(nodes_dir), judge_transparency=False)

    assert {p.name: p.read_bytes() for p in nodes_dir.iterdir()} == before


def test_proposal_ids_are_stable_across_runs() -> None:
    corpus = [_family("f", "waschen", ["waschen", "abwaschen"], family_transparency="high")]
    first = {p.id for p in analyse(corpus, judge_transparency=False).proposals}
    second = {p.id for p in analyse(corpus, judge_transparency=False).proposals}
    assert first == second


@pytest.mark.parametrize("kind", ["link", "create"])
def test_rule_derived_proposals_record_that_they_are_rules(kind) -> None:
    """`basis` records what produced a proposal, which is how you know what to trust."""
    result = analyse(
        [
            _prefix("auf"),
            _family("f", "waschen", ["waschen", "aufwaschen", "abwaschen"],
                    family_transparency="high"),
        ],
        judge_transparency=False,
    )
    for proposal in result.by_kind(kind):
        assert proposal.basis == "rules"
