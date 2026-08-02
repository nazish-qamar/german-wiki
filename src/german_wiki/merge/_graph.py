"""The LangGraph state machine: extract -> embed -> retrieve -> adjudicate -> route.

```
                     START
                       │
              ┌────────┴────────┐   conditional: is a human decision
              │                 │   already in state?
        (no decision)     (decision present)
              │                 │
          extract               │     load the staged candidate
              │                 │
           embed                │     vectors_for(), cache-first
              │                 │
          retrieve              │     tiers 1-3 from slice 4, top-k
              │                 │
        adjudicate              │     four outcomes, then interrupt()
              │                 │       ══ THE GATE ══
             END                │
                                │
                             route          conditional on kind
                               │
        ┌────────┬────────┬────┴────┬────────────┬─────────┐
      merge     link    create   relevel    morphology  discard
        └────────┴────────┴────┬────┴────────────┴─────────┘
                              END
```

Two kinds change frontmatter and nothing else, and both join here rather than getting
their own command-with-a-write, because ADR-003 gates writes to ``/nodes`` and one review
queue is the whole point:

- ``relevel`` (slice 6) — ``cefr`` / ``cefr_basis`` (SPEC §5).
- ``morphology`` (slice 7) — ``root`` / ``lemmas`` / ``separable`` /
  ``family_transparency`` (SPEC §7). Its transparency field is an outright model judgment
  about meaning, so it could never have been anything but reviewed.

**The propose pass has no edge to any apply node.** ``adjudicate`` ends at ``END``, and the
only way into ``route`` is to enter the graph with a human decision already in the state --
which only ``gw review`` does, after showing you the diff. So ADR-003 is enforced by the
topology, not merely by a convention someone could forget: there is no path from "the model
said OVERLAP" to a write.

``interrupt()`` is what stops ``adjudicate`` from returning: its payload is the proposal
set, which the driver catches and writes to ``/proposals``. Nothing resumes it, and nothing
needs to -- the durable state is the file, not a paused graph (ADR-011), which is why
``InMemorySaver`` suffices.

The apply nodes are **thin wrappers** over ``_apply``. The routing is implemented exactly
once; ``gw review`` drives these nodes rather than a parallel copy, because two
implementations are how one of them quietly stops honouring the gate.

Two scoping decisions worth stating:

- **The graph's unit is one candidate.** A candidate has one *fate* (merge into something,
  or become its own node) and may additionally earn typed edges. So one run emits one fate
  proposal plus zero or more link proposals -- which is exactly ADR-010's finding, where a
  candidate was simultaneously distinct from a node and ``governs``-related to it.
- **Candidates are compared against ``/nodes``, not against each other.** Two candidates
  from one source both proposing to merge into each other has no stable resolution, and
  ADR-006's 5-8 cap is the guard against a source atomizing in the first place.
  ``gw dupes --queue`` still reports intra-source pairs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from .. import config, embed, storage
from ..embed import DEFAULT_K, Embedder
from ..llm import ChatClient
from ..logutil import get_logger
from ..models import Node
from . import _adjudicate, _apply, _ledger, _proposal, _regenerate
from ._adjudicate import Adjudication
from ._proposal import (
    FLAG_CAP,
    FLAG_LEDGER_MISSING,
    FLAG_LOW_CONFIDENCE,
    FLAG_UNSOURCED,
    LOW_CONFIDENCE,
    Proposal,
)

logger = get_logger(__name__)

# Outcomes whose fate is "fold this candidate into an existing node".
MERGING_OUTCOMES = frozenset({"SAME", "OVERLAP"})


@dataclass
class Context:
    """Everything injectable, carried as LangGraph's typed runtime context.

    Kept out of the graph state on purpose: state is checkpointed and should stay plain
    data, while this holds live handles (a DB connection, a fake client in tests) and
    path overrides.
    """

    nodes_dir: Path | None = None
    queue_dir: Path | None = None
    proposals_dir: Path | None = None
    merged_dir: Path | None = None
    vocab_dir: Path | None = None
    raw_dir: Path | None = None
    decisions_log: Path | None = None
    cache_dir: Path | None = None
    settings_path: Path | None = None
    usage_log: Path | None = None
    conn: sqlite3.Connection | None = None
    client: ChatClient | None = None
    embedder: Embedder | None = None
    env: Mapping[str, str] | None = None
    k: int = DEFAULT_K
    now: datetime | None = None
    # Pairs already decided are not re-proposed. Resolved once per session by the driver.
    decided: set[tuple[str, str]] = field(default_factory=set)

    def stamp(self) -> datetime:
        return self.now or datetime.now(UTC)

    @property
    def llm_kwargs(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "settings_path": self.settings_path,
            "cache_dir": self.cache_dir,
            "usage_log": self.usage_log,
            "env": dict(self.env) if self.env is not None else None,
        }


class MergeState(TypedDict, total=False):
    """Plain data only -- live handles live on ``Context``."""

    candidate_id: str
    candidate_path: str | None
    source_id: str | None
    candidate: dict
    neighbours: list[dict]
    embedding: dict | None
    proposals: list[dict]
    # Present only on the apply pass. Its presence is what selects the entry point.
    decision: dict | None
    result: dict | None


def _nodes_root(ctx: Context) -> Path:
    return ctx.nodes_dir if ctx.nodes_dir is not None else config.NODES_DIR


# --- propose-pass nodes ---


def extract_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    """Resolve the candidate concept this run decides.

    Named for SPEC §2's pipeline stage, but it does not re-run the extractor: slice 3
    already extracted this source into ``/queue``, and re-extracting would duplicate
    ``gw ingest`` for no gain. What it does is load that staged candidate, which is a
    complete, loadable node file by ADR-009's design.
    """
    ctx = runtime.context
    path = state.get("candidate_path")
    if path:
        node = storage.load_node(path)
    else:
        node = storage.load_node(_nodes_root(ctx) / f"{state['candidate_id']}.md")
    return {"candidate": node.model_dump(), "candidate_id": node.id}


def embed_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    """Ensure the corpus has vectors, cache-first (ADR-010).

    Costs nothing on a warm cache, and loads the local model at most once per run when
    it is cold. Embeddings are always in-process, never an API (ADR-004).
    """
    ctx = runtime.context
    corpus = storage.load_all_nodes(_nodes_root(ctx))
    if not corpus:
        return {"embedding": None}
    result = embed.embed_nodes(
        corpus,
        conn=ctx.conn,
        embedder=ctx.embedder,
        cache_dir=ctx.cache_dir,
        settings_path=ctx.settings_path,
    )
    return {"embedding": result.model_dump()}


def retrieve_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    """Top-k neighbours above ``GRAY_LOW``, via slice 4's three tiers.

    Detection is not an LLM problem (SPEC §3.1): tiers 1 and 2 are free, and tier 3 is a
    local embedding. Only what survives this step can cost a model call.
    """
    ctx = runtime.context
    candidate = Node.model_validate(state["candidate"])
    corpus = storage.load_all_nodes(_nodes_root(ctx))
    if not corpus:
        return {"neighbours": []}

    report = embed.find_duplicates(
        [candidate],
        against=corpus,
        conn=ctx.conn,
        embedder=ctx.embedder,
        cache_dir=ctx.cache_dir,
        settings_path=ctx.settings_path,
        k=ctx.k,
    )

    neighbours = []
    for match in report.matches:
        other = match.right_id if match.left_id == candidate.id else match.left_id
        if other == candidate.id:
            continue
        pair = (candidate.id, other) if candidate.id <= other else (other, candidate.id)
        if pair in ctx.decided:
            logger.info("skipping %s ↔ %s: already decided", *pair)
            continue
        neighbours.append(
            {
                "id": other,
                "similarity": match.similarity,
                "tier": match.tier,
                "band": match.band,
            }
        )
    return {"neighbours": neighbours}


def _pick_winner(existing: Node, candidate: Node, *, candidate_is_staged: bool) -> tuple[Node, Node]:
    """Which node survives a merge. Stated in the proposal so it stays overridable.

    A node already in ``/nodes`` beats a staged candidate: it keeps its id, so every
    inbound link stays valid. Between two established nodes, the one with more sources
    has more provenance behind it, and a lexicographic tiebreak keeps the choice
    deterministic across runs.
    """
    if candidate_is_staged:
        return existing, candidate
    if len(candidate.source_ids) > len(existing.source_ids):
        return candidate, existing
    if len(candidate.source_ids) < len(existing.source_ids):
        return existing, candidate
    return (existing, candidate) if existing.id <= candidate.id else (candidate, existing)


def _raw_sources(ctx: Context, *nodes: Node) -> list[str]:
    """The immutable ``/raw`` texts behind these nodes, for the drift check only."""
    from ..ingest import read_raw_text

    texts = []
    for node in nodes:
        texts.append(node.body_md)
        for source_id in node.source_ids:
            raw = read_raw_text(source_id, raw_dir=ctx.raw_dir)
            if raw:
                texts.append(raw)
    return texts


def _same_body(winner: Node, loser: Node) -> str:
    """SPEC §3.2: keep A's body, gain B's new examples. No model call, nothing invented."""
    extra = _regenerate.new_examples(winner, loser)
    if not extra:
        return winner.body_md
    body = winner.body_md.rstrip()
    if not _regenerate.example_lines(winner.body_md):
        body = f"{body}\n\n## Examples"
    return body + "\n" + "\n".join(f"- {line}" for line in extra) + "\n"


def _flags(verdict: Adjudication, *extra: str) -> list[str]:
    flags = [f for f in extra if f]
    if verdict.confidence is not None and verdict.confidence < LOW_CONFIDENCE:
        flags.append(FLAG_LOW_CONFIDENCE)
    return flags


def _merge_proposal(
    ctx: Context,
    *,
    candidate: Node,
    other: Node,
    neighbour: dict,
    verdict: Adjudication,
    basis: str,
    provider: str | None,
    model: str | None,
    source_id: str | None,
    candidate_path: str | None,
) -> Proposal:
    """Build the fate proposal for a SAME/OVERLAP verdict, cap-checked."""
    winner, loser = _pick_winner(other, candidate, candidate_is_staged=candidate_path is not None)
    common = {
        "outcome": verdict.outcome,
        "basis": basis,
        "source_id": source_id,
        "candidate": candidate.id,
        "counterpart": other.id,
        "winner": winner.id,
        "loser": loser.id,
        "similarity": neighbour["similarity"],
        "tier": neighbour["tier"],
        "band": neighbour["band"],
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "b_adds": verdict.b_adds,
        "candidate_path": candidate_path,
        "provider": provider,
        "model": model,
        "created_at": _proposal.now_iso(ctx.stamp()),
    }

    if verdict.outcome == "SAME":
        # No regeneration, so no cap check and no model call: the body is not re-encoded.
        return Proposal(
            id=_proposal.proposal_id("merge", candidate.id, other.id),
            kind="merge",
            body_md=_same_body(winner, loser),
            changelog=f"SAME as {winner.id}: kept its body, added provenance and new examples.",
            flags=_flags(verdict),
            **common,
        )

    cap = _regenerate.check_cap(winner, decisions_log=ctx.decisions_log)
    if not cap.allowed:
        # SPEC §12.1's hard guard. Emitted as a proposal rather than dropped, so the
        # refusal is visible and actionable instead of a pair that silently stops
        # appearing.
        return Proposal(
            id=_proposal.proposal_id("discard", candidate.id, other.id),
            kind="discard",
            outcome="MANUAL",
            body_md=f"Merge refused.\n\n{cap.reason}\n",
            changelog=cap.reason,
            flags=_flags(
                verdict, FLAG_CAP, "" if cap.ledger_readable else FLAG_LEDGER_MISSING
            ),
            **{k: v for k, v in common.items() if k != "outcome" and k != "basis"},
            basis=basis,
        )

    merged, response = _regenerate.regenerate(
        winner,
        loser,
        b_adds=verdict.b_adds,
        sources=_raw_sources(ctx, winner, loser),
        **ctx.llm_kwargs,
    )
    return Proposal(
        id=_proposal.proposal_id("merge", candidate.id, other.id),
        kind="merge",
        body_md=merged.body_md,
        changelog=merged.changelog,
        flags=_flags(
            verdict,
            FLAG_UNSOURCED if merged.unsourced else "",
            "" if cap.ledger_readable else FLAG_LEDGER_MISSING,
        ),
        **{**common, "model": response.model, "provider": response.provider},
    )


def adjudicate_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    """Classify every neighbour, assemble the proposals, and stop at the gate.

    Two ways to reach a verdict, and only one of them costs money:

    - band ``duplicate`` (>= ``GRAY_HIGH``) is SAME **by threshold**, with no model call.
      It is still a proposal: a confident threshold is not permission to write (ADR-003).
      Confidence saves the call, not the human gate.
    - band ``gray`` goes to the model, which is the ~10-20% SPEC §3.1 budgets for.

    The candidate's *fate* is the strongest merging verdict, or "create" if there is
    none. ``DISTINCT_RELATED`` verdicts additionally become link proposals -- but only
    when the fate is create, since edges on a node about to be archived would dangle.
    """
    ctx = runtime.context
    candidate = Node.model_validate(state["candidate"])
    source_id = state.get("source_id")
    candidate_path = state.get("candidate_path")
    nodes_root = _nodes_root(ctx)

    fate: Proposal | None = None
    links: list[Proposal] = []

    for neighbour in state.get("neighbours", []):
        other = storage.load_node(nodes_root / f"{neighbour['id']}.md")

        if neighbour["band"] == "duplicate":
            verdict = Adjudication(
                outcome="SAME",
                confidence=neighbour["similarity"],
                reason=(
                    f"cosine {neighbour['similarity']:.3f} at or above the duplicate "
                    f"threshold {embed.GRAY_HIGH}; no model call was needed"
                ),
            )
            basis, provider, model = "threshold", None, None
        else:
            # A is the established node, B the candidate -- the orientation SPEC §3.1's
            # prompt and §3.2's "append B's source_id to A" both assume.
            verdict, response = _adjudicate.adjudicate(other, candidate, **ctx.llm_kwargs)
            basis, provider, model = "llm", response.provider, response.model

        if verdict.outcome in MERGING_OUTCOMES and fate is None:
            fate = _merge_proposal(
                ctx,
                candidate=candidate,
                other=other,
                neighbour=neighbour,
                verdict=verdict,
                basis=basis,
                provider=provider,
                model=model,
                source_id=source_id,
                candidate_path=candidate_path,
            )
            break  # a candidate has one fate; further neighbours are moot once it merges

        if verdict.outcome == "DISTINCT_RELATED":
            links.append(
                Proposal(
                    id=_proposal.proposal_id(
                        "link", candidate.id, other.id, verdict.relation
                    ),
                    kind="link",
                    outcome="DISTINCT_RELATED",
                    basis=basis,
                    source_id=source_id,
                    candidate=candidate.id,
                    counterpart=other.id,
                    relation=verdict.relation,
                    direction=verdict.direction,
                    similarity=neighbour["similarity"],
                    tier=neighbour["tier"],
                    band=neighbour["band"],
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                    candidate_path=candidate_path,
                    provider=provider,
                    model=model,
                    created_at=_proposal.now_iso(ctx.stamp()),
                    body_md=(
                        f"Proposed edge: **{other.id} -{verdict.relation}-> {candidate.id}**\n"
                        if verdict.direction == "a_to_b"
                        else f"Proposed edge: **{candidate.id} -{verdict.relation}-> {other.id}**\n"
                    )
                    + f"\n{verdict.reason}\n\n"
                    "_Approving this changes frontmatter only; no body is rewritten._\n",
                    flags=_flags(verdict),
                )
            )

    proposals: list[Proposal] = []
    if fate is not None:
        # A merged-away candidate cannot hold edges, so its link proposals are dropped.
        proposals.append(fate)
    else:
        proposals.append(_create_proposal(ctx, candidate, source_id, candidate_path))
        proposals.extend(links)

    payload = [p.model_dump() for p in proposals]

    # ══ THE GATE (ADR-003) ══
    # Everything above is pure computation over cached inputs. This is where the run
    # stops: the payload is the proposal set, the driver persists it to /proposals, and
    # nothing resumes. No apply node is reachable from here -- see the module docstring.
    interrupt(payload)
    return {"proposals": payload}  # unreachable in practice; kept honest for the type


def _create_proposal(
    ctx: Context, candidate: Node, source_id: str | None, candidate_path: str | None
) -> Proposal:
    return Proposal(
        id=_proposal.proposal_id("create", candidate.id),
        kind="create",
        outcome="DISTINCT",
        basis="llm",
        source_id=source_id,
        candidate=candidate.id,
        candidate_path=candidate_path,
        # No body: the staged queue file at `candidate_path` is the authoritative,
        # hand-editable content (ADR-011, amended). Carrying a copy here is what let a
        # reviewer's edit be silently discarded -- two editable artifacts holding the same
        # text, with the proposal quietly winning.
        body_md="" if candidate_path else candidate.body_md,
        reason="No existing node was judged the same concept.",
        created_at=_proposal.now_iso(ctx.stamp()),
    )


# --- apply-pass nodes: thin wrappers, one implementation (see _apply) ---


def route_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    return {}


def _route(state: MergeState) -> str:
    decision = state.get("decision") or {}
    if not decision.get("approved"):
        return "discard"
    kind = (decision.get("proposal") or {}).get("kind", "discard")
    return kind if kind in ("merge", "link", "create", "relevel", "morphology") else "discard"


def _decided(state: MergeState) -> tuple[Proposal, bool]:
    decision = state["decision"]
    return Proposal.model_validate(decision["proposal"]), bool(decision.get("approved"))


def merge_apply_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    proposal, _ = _decided(state)
    return {
        "result": _apply.apply_merge(
            proposal,
            nodes_dir=ctx.nodes_dir,
            vocab_dir=ctx.vocab_dir,
            merged_dir=ctx.merged_dir,
            decisions_log=ctx.decisions_log,
            now=ctx.stamp(),
        ).model_dump()
    }


def link_apply_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    proposal, _ = _decided(state)
    return {
        "result": _apply.apply_link(
            proposal,
            nodes_dir=ctx.nodes_dir,
            vocab_dir=ctx.vocab_dir,
            decisions_log=ctx.decisions_log,
            now=ctx.stamp(),
        ).model_dump()
    }


def create_apply_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    proposal, _ = _decided(state)
    return {
        "result": _apply.apply_create(
            proposal,
            nodes_dir=ctx.nodes_dir,
            vocab_dir=ctx.vocab_dir,
            decisions_log=ctx.decisions_log,
            now=ctx.stamp(),
        ).model_dump()
    }


def relevel_apply_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    proposal, _ = _decided(state)
    return {
        "result": _apply.apply_relevel(
            proposal,
            nodes_dir=ctx.nodes_dir,
            vocab_dir=ctx.vocab_dir,
            decisions_log=ctx.decisions_log,
            now=ctx.stamp(),
        ).model_dump()
    }


def morphology_apply_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    proposal, _ = _decided(state)
    return {
        "result": _apply.apply_morphology(
            proposal,
            nodes_dir=ctx.nodes_dir,
            vocab_dir=ctx.vocab_dir,
            decisions_log=ctx.decisions_log,
            now=ctx.stamp(),
        ).model_dump()
    }


def discard_apply_node(state: MergeState, runtime: Runtime[Context]) -> dict:
    ctx = runtime.context
    proposal, approved = _decided(state)
    return {
        "result": _apply.apply_discard(
            proposal,
            approved=approved,
            decisions_log=ctx.decisions_log,
            now=ctx.stamp(),
        ).model_dump()
    }


# --- assembly ---


def _entry(state: MergeState) -> str:
    """The only way into the apply half is to arrive holding a human decision."""
    return "route" if state.get("decision") else "extract"


def build_graph():
    """Compile the state machine with a fresh in-memory checkpointer.

    Fresh per call on purpose: ``interrupt()`` requires a checkpointer, but nothing here
    resumes a thread, and a shared saver would make two runs of the same candidate in one
    process collide on their thread id.
    """
    builder = StateGraph(MergeState, context_schema=Context)
    builder.add_node("extract", extract_node)
    builder.add_node("embed", embed_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("adjudicate", adjudicate_node)
    builder.add_node("route", route_node)
    builder.add_node("merge", merge_apply_node)
    builder.add_node("link", link_apply_node)
    builder.add_node("create", create_apply_node)
    builder.add_node("relevel", relevel_apply_node)
    builder.add_node("morphology", morphology_apply_node)
    builder.add_node("discard", discard_apply_node)

    builder.add_conditional_edges(START, _entry, {"extract": "extract", "route": "route"})
    builder.add_edge("extract", "embed")
    builder.add_edge("embed", "retrieve")
    builder.add_edge("retrieve", "adjudicate")
    # No edge from adjudicate to route. The gate is structural, not conventional.
    builder.add_edge("adjudicate", END)

    builder.add_conditional_edges(
        "route",
        _route,
        {
            "merge": "merge",
            "link": "link",
            "create": "create",
            "relevel": "relevel",
            "morphology": "morphology",
            "discard": "discard",
        },
    )
    for node in ("merge", "link", "create", "relevel", "morphology", "discard"):
        builder.add_edge(node, END)

    return builder.compile(checkpointer=InMemorySaver())


# --- drivers: the two entry points into the one graph ---


class ProposeResult(BaseModel):
    """What one ``gw adjudicate`` produced. Nothing here touched ``/nodes``."""

    model_config = ConfigDict(extra="forbid")

    sources: list[str] = Field(default_factory=list)
    candidates: int = 0
    proposals: list[Proposal] = Field(default_factory=list)
    paths: list[Path] = Field(default_factory=list)
    skipped_decided: int = 0
    ledger_readable: bool = True

    def by_kind(self, kind: str) -> list[Proposal]:
        return [p for p in self.proposals if p.kind == kind]


def _decided_pairs(ctx: Context) -> tuple[set[tuple[str, str]], bool]:
    """Pairs already decided, so a rejected pair is not re-proposed every run.

    Unlike the regeneration cap, an unreadable ledger is handled *permissively* here,
    and the asymmetry is deliberate: forgetting a past rejection costs one redundant
    question, while forgetting a past merge would let the cap fail open. Cheap to be
    wrong in this direction, expensive in that one.
    """
    try:
        return _ledger.decided_pairs(decisions_log=ctx.decisions_log), True
    except _ledger.LedgerUnreadable as exc:
        logger.warning("decision ledger unreadable (%s); previously decided pairs may re-appear", exc)
        return set(), False


def propose_for_source(
    source_id: str | None = None,
    *,
    ctx: Context | None = None,
) -> ProposeResult:
    """Run the propose pass over a source's staged candidates.

    One graph run per candidate. Each stops at ``interrupt()``; the payload is that
    candidate's proposal set, which is written to ``/proposals`` and waits for review.
    **Nothing is written to ``/nodes``.**
    """
    from ..ingest import list_queue

    ctx = ctx or Context()
    pending = list_queue(queue_dir=ctx.queue_dir)
    if source_id is not None:
        if source_id not in pending:
            raise ValueError(f"nothing queued for source {source_id!r}")
        pending = {source_id: pending[source_id]}
    if not pending:
        return ProposeResult()

    ctx.decided, ledger_readable = _decided_pairs(ctx)
    graph = build_graph()
    result = ProposeResult(sources=sorted(pending), ledger_readable=ledger_readable)

    for sid, paths in pending.items():
        for path in paths:
            result.candidates += 1
            state: MergeState = {
                "candidate_id": path.stem,
                "candidate_path": str(path),
                "source_id": sid,
            }
            run = graph.invoke(
                state,
                {"configurable": {"thread_id": f"propose-{sid}-{path.stem}"}},
                context=ctx,
            )

            interrupts = run.get("__interrupt__") or ()
            if not interrupts:
                # adjudicate() always interrupts, so this means the node never ran.
                logger.warning("no proposals produced for candidate %s", path.stem)
                continue

            for payload in interrupts[0].value:
                proposal = Proposal.model_validate(payload)
                result.proposals.append(proposal)
                result.paths.append(
                    _proposal.write_proposal(proposal, proposals_dir=ctx.proposals_dir)
                )

    return result


def apply_decision(
    proposal: Proposal,
    *,
    approved: bool,
    ctx: Context | None = None,
) -> _apply.ApplyResult:
    """Drive the apply half with a human decision. The only path that writes to /nodes.

    Enters the graph at ``route`` -- reachable only because ``decision`` is in the input
    state, which only a reviewed approval puts there.
    """
    ctx = ctx or Context()
    graph = build_graph()
    state: MergeState = {
        "candidate_id": proposal.candidate,
        "candidate_path": proposal.candidate_path,
        "source_id": proposal.source_id,
        "decision": {"proposal": proposal.model_dump(), "approved": approved},
    }
    run = graph.invoke(
        state, {"configurable": {"thread_id": f"apply-{proposal.id}"}}, context=ctx
    )
    result = run.get("result")
    if result is None:
        raise _apply.ApplyError(f"proposal {proposal.id!r} produced no result")
    return _apply.ApplyResult.model_validate(result)
