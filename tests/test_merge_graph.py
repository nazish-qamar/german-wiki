"""The state machine: where it stops, what it costs, and where it routes.

The load-bearing assertion in this file is that the propose pass **cannot write**. It is
checked two ways -- behaviourally (nothing on disk changed) and structurally (the graph
has no edge from adjudicate to any apply node), because a behavioural test only proves
the path was not taken today.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import FakeChatClient, FakeEmbedder

from german_wiki import merge, storage
from german_wiki.db import connect, rebuild_schema
from german_wiki.merge import _graph
from german_wiki.models import Node

BODY = "Das Perfekt bildet man mit haben + Partizip II.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n"
OTHER_BODY = "Wechselpräpositionen stehen mit Akkusativ oder Dativ.\n"


def _node(node_id: str, title: str, body: str = BODY, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": title,
        "title_en": title,
        "type": "grammar",
        "cefr": "A2",
        "status": "draft",
        "body_md": body,
        "source_ids": ["seed"],
    }
    data.update(overrides)
    return Node(**data)


def _verdict(outcome: str, **extra) -> str:
    body = {"outcome": outcome, "confidence": 0.9, "reason": "because"}
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


def _merged(body: str = "Zusammengeführt.\n", changelog: str = "merged") -> str:
    return json.dumps({"body_md": body, "changelog": changelog}, ensure_ascii=False)


@pytest.fixture
def world(tmp_path: Path, tmp_vocab: Path, models_config: Path):
    """One existing node, one staged candidate, and everything redirected to tmp."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    queue = tmp_path / "queue" / "src-1"
    queue.mkdir(parents=True)

    existing = _node("perfekt-haben", "Perfekt mit haben")
    storage.write_node(existing, nodes / "perfekt-haben.md", vocab_dir=tmp_vocab)
    candidate = _node("perfekt-sein", "Perfekt mit sein", source_ids=["src-1"])
    storage.write_node(candidate, queue / "perfekt-sein.md", vocab_dir=tmp_vocab)

    conn = connect(tmp_path / "index.db")
    rebuild_schema(conn)

    # Place the candidate's embedding right next to the existing node's, so the pair
    # lands in the gray band and the LLM tier actually fires.
    from german_wiki.embed._text import embed_text

    embedder = FakeEmbedder(similar_to={embed_text(candidate): embed_text(existing)})

    ctx = merge.Context(
        nodes_dir=nodes,
        queue_dir=tmp_path / "queue",
        proposals_dir=tmp_path / "proposals",
        merged_dir=tmp_path / "_merged",
        vocab_dir=tmp_vocab,
        raw_dir=tmp_path / "raw",
        decisions_log=tmp_path / "decisions.jsonl",
        cache_dir=tmp_path / "cache",
        settings_path=models_config,
        usage_log=tmp_path / "usage.jsonl",
        conn=conn,
        embedder=embedder,
    )
    try:
        yield ctx
    finally:
        conn.close()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --- structure: the gate is topological ---


def test_no_edge_leads_from_adjudicate_to_an_apply_node() -> None:
    """ADR-003 enforced by the graph's shape, not by a convention someone could forget.

    Even a resume of the propose pass cannot reach a write: there is no path. The only
    way into `route` is to enter the graph already holding a human decision.
    """
    graph = merge.build_graph().get_graph()
    reachable = {e.target for e in graph.edges if e.source == "adjudicate"}
    assert reachable == {"__end__"}

    into_route = {e.source for e in graph.edges if e.target == "route"}
    assert into_route == {"__start__"}


def test_entry_selects_the_apply_half_only_when_a_decision_is_present() -> None:
    assert _graph._entry({"candidate_id": "x"}) == "extract"
    assert _graph._entry({"candidate_id": "x", "decision": {"approved": True}}) == "route"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("merge", "merge"), ("link", "link"), ("create", "create"), ("discard", "discard")],
)
def test_the_router_dispatches_each_kind(kind, expected) -> None:
    state = {"decision": {"approved": True, "proposal": {"kind": kind}}}
    assert _graph._route(state) == expected


def test_a_rejection_always_routes_to_discard() -> None:
    """Whatever was proposed, "no" writes nothing."""
    state = {"decision": {"approved": False, "proposal": {"kind": "merge"}}}
    assert _graph._route(state) == "discard"


# --- behaviour: the propose pass writes nothing ---


def test_the_propose_pass_stops_at_the_interrupt_and_writes_nothing(world) -> None:
    ctx = world
    nodes_before = _snapshot(ctx.nodes_dir)
    queue_before = _snapshot(ctx.queue_dir)

    ctx.client = FakeChatClient(text=[_verdict("OVERLAP", b_adds="sein"), _merged()])
    result = merge.propose_for_source("src-1", ctx=ctx)

    assert result.candidates == 1
    assert len(result.proposals) == 1
    # ADR-003: neither the source of truth nor the staging area moved.
    assert _snapshot(ctx.nodes_dir) == nodes_before
    assert _snapshot(ctx.queue_dir) == queue_before
    # ...and the proposal is on disk, awaiting a human.
    assert result.paths[0].is_file()


def test_an_overlap_proposes_a_merge_with_a_regenerated_body(world) -> None:
    ctx = world
    ctx.client = FakeChatClient(
        text=[_verdict("OVERLAP", b_adds="sein"), _merged("Neu zusammengeführt.\n", "added sein")]
    )
    proposal = merge.propose_for_source("src-1", ctx=ctx).proposals[0]

    assert proposal.kind == "merge"
    assert proposal.outcome == "OVERLAP"
    assert proposal.winner == "perfekt-haben"  # the established node keeps its id
    assert proposal.loser == "perfekt-sein"
    assert proposal.body_md == "Neu zusammengeführt.\n"
    assert proposal.changelog == "added sein"
    assert proposal.basis == "llm"


def test_a_distinct_related_verdict_yields_a_create_and_a_link(world) -> None:
    """ADR-010's finding: a candidate can be its own node *and* earn a typed edge."""
    ctx = world
    ctx.client = FakeChatClient(
        text=[_verdict("DISTINCT_RELATED", relation="governs", direction="a_to_b")]
    )
    proposals = merge.propose_for_source("src-1", ctx=ctx).proposals

    kinds = {p.kind: p for p in proposals}
    assert set(kinds) == {"create", "link"}
    assert kinds["link"].relation == "governs"
    assert kinds["link"].direction == "a_to_b"
    assert kinds["create"].candidate == "perfekt-sein"
    # The edge proposal writes no body -- ADR-010 says relations touch frontmatter only.
    assert kinds["link"].writes_body is False


def test_a_distinct_verdict_yields_only_a_create(world) -> None:
    ctx = world
    ctx.client = FakeChatClient(text=_verdict("DISTINCT"))
    proposals = merge.propose_for_source("src-1", ctx=ctx).proposals
    assert [p.kind for p in proposals] == ["create"]


# --- cost shape (SPEC §3.1: the LLM fires on ~10-20%) ---


def test_the_duplicate_band_decides_without_a_model_call(world) -> None:
    """>= GRAY_HIGH is SAME by threshold -- but still a proposal.

    Confidence saves the model call, not the human gate: ADR-003 means a threshold never
    auto-writes, however sure it is.
    """
    ctx = world
    # An exact copy: tier 1 catches it at similarity 1.0, comfortably in the duplicate band.
    shutil.copy2(
        ctx.nodes_dir / "perfekt-haben.md", ctx.queue_dir / "src-1" / "perfekt-sein.md"
    )
    text = (ctx.queue_dir / "src-1" / "perfekt-sein.md").read_text(encoding="utf-8")
    (ctx.queue_dir / "src-1" / "perfekt-sein.md").write_text(
        text.replace("id: perfekt-haben", "id: perfekt-sein"), encoding="utf-8"
    )

    ctx.client = FakeChatClient(text=_verdict("DISTINCT"))
    proposals = merge.propose_for_source("src-1", ctx=ctx).proposals

    assert ctx.client.call_count == 0  # the model was never asked
    assert proposals[0].outcome == "SAME"
    assert proposals[0].basis == "threshold"
    assert proposals[0].kind == "merge"  # still needs approval


def test_a_candidate_with_no_neighbour_costs_nothing(world, tmp_path: Path) -> None:
    ctx = world
    ctx.embedder = FakeEmbedder()  # nothing placed near anything
    (ctx.queue_dir / "src-1" / "perfekt-sein.md").write_text(
        (ctx.queue_dir / "src-1" / "perfekt-sein.md")
        .read_text(encoding="utf-8")
        .replace(BODY.strip(), "Etwas völlig anderes über Kochen und Küchengeräte."),
        encoding="utf-8",
    )
    ctx.client = FakeChatClient(text=_verdict("DISTINCT"))

    proposals = merge.propose_for_source("src-1", ctx=ctx).proposals
    assert ctx.client.call_count == 0
    assert [p.kind for p in proposals] == ["create"]


def test_a_same_verdict_needs_no_regeneration_call(world) -> None:
    """SPEC §3.2: keep A's body, gain B's examples. Mechanical, so nothing is re-encoded."""
    ctx = world
    ctx.client = FakeChatClient(text=_verdict("SAME"))
    proposal = merge.propose_for_source("src-1", ctx=ctx).proposals[0]

    assert ctx.client.call_count == 1  # classification only; no merge call
    assert proposal.outcome == "SAME"
    assert proposal.kind == "merge"


# --- the cap, from inside the graph ---


def test_a_capped_node_proposes_manual_instead_of_merging(world) -> None:
    """SPEC §12.1's hard refuse, surfaced rather than silently dropped."""
    ctx = world
    for i in range(merge.MAX_REGENERATIONS):
        merge._ledger.append(
            merge.Decision(
                decision_id=f"d{i}",
                proposal_id="p",
                decided_at="2026-07-31T00:00:00+00:00",
                approved=True,
                kind="merge",
                outcome="OVERLAP",
                winner="perfekt-haben",
                loser="x",
            ),
            decisions_log=ctx.decisions_log,
        )

    ctx.client = FakeChatClient(text=[_verdict("OVERLAP", b_adds="sein"), _merged()])
    proposal = merge.propose_for_source("src-1", ctx=ctx).proposals[0]

    assert proposal.kind == "discard"
    assert proposal.outcome == "MANUAL"
    assert merge.FLAG_CAP in proposal.flags
    assert ctx.client.call_count == 1  # classified, but never paid for a regeneration


def test_an_already_decided_pair_is_not_re_proposed(world) -> None:
    ctx = world
    merge._ledger.append(
        merge.Decision(
            decision_id="d1",
            proposal_id="p",
            decided_at="2026-07-31T00:00:00+00:00",
            approved=False,
            kind="merge",
            outcome="OVERLAP",
            winner="perfekt-haben",
            loser="perfekt-sein",
        ),
        decisions_log=ctx.decisions_log,
    )
    ctx.client = FakeChatClient(text=_verdict("OVERLAP"))
    proposals = merge.propose_for_source("src-1", ctx=ctx).proposals

    assert ctx.client.call_count == 0
    assert [p.kind for p in proposals] == ["create"]


def test_an_unreadable_ledger_only_forgets_rejections(world) -> None:
    """Asymmetry check: permissive for re-proposing, strict for the cap."""
    ctx = world
    ctx.decisions_log.write_text("{broken\n", encoding="utf-8")
    ctx.client = FakeChatClient(text=[_verdict("DISTINCT")])

    result = merge.propose_for_source("src-1", ctx=ctx)
    assert result.ledger_readable is False
    assert [p.kind for p in result.proposals] == ["create"]  # ran anyway
