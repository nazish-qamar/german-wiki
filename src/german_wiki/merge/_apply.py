"""Applying an approved decision: the four routes, implemented exactly once.

Every write in this module goes through ``ingest.write_approved`` -- the slice-3 promote
seam -- so there is still exactly one door into ``/nodes`` no matter which command opened
it (ADR-009, ADR-011). Nothing here calls ``storage.write_node`` directly.

These four functions are the **single implementation** of the routing. ``_graph``'s
``merge`` / ``link`` / ``create`` / ``discard`` nodes are thin wrappers over them, and
``gw review`` drives the graph rather than a parallel copy. Implementing routing twice --
once as a graph node and once as a review helper -- is how the two drift apart and one
quietly stops honouring the gate.

Nothing here decides anything. Reaching any of these functions means a human already
approved (or rejected) the proposal; the outcome and the content are read off the
proposal file, hand-edits included.

Two invariants worth stating because they are easy to erode:

- **A link write never touches a body** (ADR-010). ``apply_link`` mutates ``links`` and
  nothing else, and refuses a dangling edge rather than writing one.
- **The loser is archived, never silently deleted** (SPEC §3.2). ``/_merged`` holds the
  losing file byte-for-byte plus a pointer sidecar, mirroring ``/raw``'s two-file idiom,
  and it is git-tracked because it is the audit trail the merge diff refers back to.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .. import config, storage
from ..ingest import write_approved
from ..logutil import get_logger
from ..models import Link, Node
from . import _ledger, _proposal
from ._proposal import Proposal

logger = get_logger(__name__)

# Applied in this order within one review session. Creates first, so a link never
# references a node that has not landed yet; discards last because they write nothing.
KIND_ORDER = {"create": 0, "merge": 1, "link": 2, "discard": 3}

# The trust ladder, and the one rung a machine regeneration knocks a node down.
#
# ``status`` is a trust signal, not a label that rides along: ``stable`` claims the node
# has been vetted as a whole. An OVERLAP merge rewrites the body and is confirmed only as
# a *diff* -- the reviewer approved what changed, not the resulting node read end to end --
# so keeping ``stable`` would overclaim. Capped at ``reviewed``: a regeneration does not
# undo the review that already happened, it just stops the node claiming more than it
# re-earned. ``draft`` and ``reviewed`` are already at or below that ceiling, so they are
# untouched -- demotion never forces a node down to ``draft``.
REGENERATION_DEMOTES = {"stable": "reviewed"}


def _status_after(winner: Node, outcome: str) -> str:
    """The winner's status after a merge (ADR-011 §7).

    Keyed on ``_ledger.REGENERATING_OUTCOMES`` deliberately, so this and the SPEC §12.1
    regeneration cap are driven by **one** definition of "which outcome is the drift
    event". SAME appends provenance and examples mechanically -- no model call, no
    re-encoding -- so it neither counts toward the cap nor demotes. If that set ever
    changes, both guards move together instead of silently disagreeing.
    """
    if outcome not in _ledger.REGENERATING_OUTCOMES:
        return winner.status
    return REGENERATION_DEMOTES.get(winner.status, winner.status)


class ApplyError(RuntimeError):
    """An approved decision could not be applied; nothing was written."""


class ApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    kind: str
    approved: bool
    written: list[str] = Field(default_factory=list)
    archived: list[str] = Field(default_factory=list)
    decision_id: str
    note: str = ""

    @property
    def needs_reindex(self) -> bool:
        return bool(self.written or self.archived)


# --- shared plumbing ---


def _record(
    proposal: Proposal,
    *,
    approved: bool,
    decisions_log: Path | str | None,
    now: datetime | None = None,
    note: str = "",
) -> str:
    """Append the durable decision record and return its id."""
    stamp = now or datetime.now(UTC)
    decision_id = _ledger.new_decision_id(proposal.id, now=stamp)
    _ledger.append(
        _ledger.Decision(
            decision_id=decision_id,
            proposal_id=proposal.id,
            decided_at=stamp.isoformat(),
            approved=approved,
            kind=proposal.kind,
            outcome=proposal.outcome,
            winner=proposal.winner,
            loser=proposal.loser,
            source_id=proposal.source_id,
            similarity=proposal.similarity,
            tier=proposal.tier,
            confidence=proposal.confidence,
            relation=proposal.relation,
            direction=proposal.direction,
            basis=proposal.basis,
            provider=proposal.provider,
            model=proposal.model,
            changelog=proposal.changelog or (note or None),
            flags=list(proposal.flags),
        ),
        decisions_log=decisions_log,
    )
    return decision_id


def _node_path(node_id: str, nodes_dir: Path | str | None) -> Path:
    root = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    return root / f"{node_id}.md"


def _load_side(proposal: Proposal, node_id: str, nodes_dir: Path | str | None) -> Node:
    """Load one side of a pair from ``/nodes``, or from the queue if it is a candidate."""
    path = _node_path(node_id, nodes_dir)
    if path.is_file():
        return storage.load_node(path)
    if proposal.candidate == node_id and proposal.candidate_path:
        staged = Path(proposal.candidate_path)
        if staged.is_file():
            return storage.load_node(staged)
    raise ApplyError(
        f"cannot find node {node_id!r} for proposal {proposal.id!r}; it was neither in "
        f"{path.parent} nor staged at {proposal.candidate_path!r}. Re-run `gw adjudicate` "
        "-- this proposal was written against a state that no longer exists."
    )


def _origin_path(proposal: Proposal, node_id: str, nodes_dir: Path | str | None) -> Path | None:
    """Where ``node_id`` currently lives on disk, if anywhere."""
    path = _node_path(node_id, nodes_dir)
    if path.is_file():
        return path
    if proposal.candidate == node_id and proposal.candidate_path:
        staged = Path(proposal.candidate_path)
        if staged.is_file():
            return staged
    return None


def archive_loser(
    proposal: Proposal,
    origin: Path,
    *,
    merged_dir: Path | str | None = None,
    decision_id: str,
    now: datetime | None = None,
) -> Path:
    """Move the losing node into ``/_merged`` with a pointer sidecar (SPEC §3.2).

    The ``.md`` is a **byte-for-byte** copy: the archived file must stay exactly what was
    merged away, which is why the pointer metadata lives in a sidecar rather than in its
    frontmatter. That also keeps ``Node`` free of ``merged_into``-style fields it would
    otherwise have to accept everywhere, since it is ``extra="forbid"``.
    """
    root = Path(merged_dir) if merged_dir is not None else config.MERGED_DIR
    root.mkdir(parents=True, exist_ok=True)

    stem = origin.stem
    target = root / f"{stem}.md"
    suffix = 1
    while target.exists():  # two different nodes can share an id across time
        suffix += 1
        target = root / f"{stem}-{suffix}.md"

    target.write_bytes(origin.read_bytes())
    target.with_suffix(".json").write_text(
        json.dumps(
            {
                "node_id": stem,
                "merged_into": proposal.winner,
                "merged_at": (now or datetime.now(UTC)).isoformat(),
                "outcome": proposal.outcome,
                "decision_id": decision_id,
                "proposal_id": proposal.id,
                "origin_path": str(origin),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    origin.unlink()
    return target


# --- the four routes ---


def apply_merge(
    proposal: Proposal,
    *,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    merged_dir: Path | str | None = None,
    decisions_log: Path | str | None = None,
    now: datetime | None = None,
) -> ApplyResult:
    """Fold the loser into the winner: rewrite one node, archive the other.

    Handles both SAME and OVERLAP. They differ in how ``proposal.body_md`` was produced --
    mechanically for SAME (existing body plus the loser's new examples), by regeneration
    for OVERLAP -- and that difference outlives the proposal in exactly one place:
    ``_status_after``, because OVERLAP is the lossy re-encoding and SAME is not.
    """
    if not (proposal.winner and proposal.loser):
        raise ApplyError(f"merge proposal {proposal.id!r} names no winner/loser pair")

    stamp = now or datetime.now(UTC)
    winner = _load_side(proposal, proposal.winner, nodes_dir)
    loser = _load_side(proposal, proposal.loser, nodes_dir)
    loser_path = _origin_path(proposal, proposal.loser, nodes_dir)
    if loser_path is None:
        raise ApplyError(f"losing node {proposal.loser!r} has no file to archive")

    # Provenance is unioned mechanically, never by the model: a merge must not be able to
    # lose the pointer back to /raw that SPEC §8.1 calls non-negotiable.
    source_ids = list(dict.fromkeys([*winner.source_ids, *loser.source_ids]))

    merged = winner.model_copy(
        update={
            "body_md": proposal.body_md,  # hand-edits during review included, verbatim
            "source_ids": source_ids,
            "version": (winner.version or 1) + 1,
            "status": _status_after(winner, proposal.outcome),
            "updated_at": stamp,
        }
    )

    decision_id = _record(proposal, approved=True, decisions_log=decisions_log, now=stamp)
    # expect_exists=True: the winner must already be there. A missing file means it was
    # renamed or removed since the proposal was written, and recreating it from a stale
    # body would resurrect content nobody approved.
    write_approved(merged, nodes_dir=nodes_dir, vocab_dir=vocab_dir, expect_exists=True)
    archived = archive_loser(
        proposal, loser_path, merged_dir=merged_dir, decision_id=decision_id, now=stamp
    )

    logger.info(
        "merged %s into %s (%s); loser archived to %s",
        proposal.loser,
        proposal.winner,
        proposal.outcome,
        archived,
    )
    return ApplyResult(
        proposal_id=proposal.id,
        kind="merge",
        approved=True,
        written=[merged.id],
        archived=[proposal.loser],
        decision_id=decision_id,
        note=proposal.changelog or "",
    )


def apply_link(
    proposal: Proposal,
    *,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    decisions_log: Path | str | None = None,
    now: datetime | None = None,
) -> ApplyResult:
    """Add one typed edge (SPEC §4.2). Touches ``links`` and nothing else.

    Refuses a dangling edge. If the candidate has not been created yet, its ``create``
    proposal is still pending -- approve that first; ``KIND_ORDER`` makes a single review
    session do so automatically.
    """
    if not (proposal.relation and proposal.direction and proposal.counterpart):
        raise ApplyError(f"link proposal {proposal.id!r} is missing relation/direction/target")

    stamp = now or datetime.now(UTC)
    if proposal.direction == "a_to_b":
        source_id, target_id = proposal.counterpart, proposal.candidate
    else:
        source_id, target_id = proposal.candidate, proposal.counterpart

    for node_id in (source_id, target_id):
        if not _node_path(node_id, nodes_dir).is_file():
            raise ApplyError(
                f"cannot link {source_id!r} -> {target_id!r}: {node_id!r} is not in /nodes "
                "yet. Approve its `create` proposal first, then re-run `gw review`."
            )

    source = storage.load_node(_node_path(source_id, nodes_dir))
    if any(link.target == target_id and link.relation == proposal.relation for link in source.links):
        decision_id = _record(
            proposal, approved=True, decisions_log=decisions_log, now=stamp, note="already linked"
        )
        return ApplyResult(
            proposal_id=proposal.id,
            kind="link",
            approved=True,
            decision_id=decision_id,
            note=f"{source_id} already has a {proposal.relation} edge to {target_id}",
        )

    linked = source.model_copy(
        update={
            "links": [
                *source.links,
                Link(target=target_id, relation=proposal.relation, confidence=proposal.confidence),
            ],
            "updated_at": stamp,
        }
    )
    # body_md is carried through model_copy untouched -- ADR-010: a proposed edge writes
    # nothing to the bodies. The proposal's own body_md is a summary and is NOT applied.

    decision_id = _record(proposal, approved=True, decisions_log=decisions_log, now=stamp)
    write_approved(linked, nodes_dir=nodes_dir, vocab_dir=vocab_dir, expect_exists=True)

    logger.info("linked %s -%s-> %s", source_id, proposal.relation, target_id)
    return ApplyResult(
        proposal_id=proposal.id,
        kind="link",
        approved=True,
        written=[linked.id],
        decision_id=decision_id,
        note=f"{source_id} -{proposal.relation}-> {target_id}",
    )


def apply_create(
    proposal: Proposal,
    *,
    nodes_dir: Path | str | None = None,
    vocab_dir: Path | str | None = None,
    decisions_log: Path | str | None = None,
    now: datetime | None = None,
) -> ApplyResult:
    """Promote the candidate as a new node -- the ordinary slice-3 path, gated."""
    stamp = now or datetime.now(UTC)
    node = _load_side(proposal, proposal.candidate, nodes_dir)
    staged = _origin_path(proposal, proposal.candidate, nodes_dir)

    if proposal.body_md.strip():
        node = node.model_copy(update={"body_md": proposal.body_md})

    decision_id = _record(proposal, approved=True, decisions_log=decisions_log, now=stamp)
    write_approved(node, nodes_dir=nodes_dir, vocab_dir=vocab_dir, expect_exists=False)
    if staged is not None and staged.is_file() and staged != _node_path(node.id, nodes_dir):
        staged.unlink()  # only after the write succeeded

    logger.info("created node %s from candidate", node.id)
    return ApplyResult(
        proposal_id=proposal.id,
        kind="create",
        approved=True,
        written=[node.id],
        decision_id=decision_id,
    )


def apply_discard(
    proposal: Proposal,
    *,
    approved: bool = False,
    decisions_log: Path | str | None = None,
    now: datetime | None = None,
    note: str = "",
) -> ApplyResult:
    """Write nothing, but record the decision.

    Covers every rejection and every ``MANUAL`` refusal. The record is what keeps a
    rejected pair from being re-proposed on the next run, and it is also what makes
    ADR-003's revisit condition ("acceptance rate consistently > 95%") measurable at all.
    """
    decision_id = _record(
        proposal,
        approved=approved,
        decisions_log=decisions_log,
        now=now,
        note=note or "rejected at review",
    )
    return ApplyResult(
        proposal_id=proposal.id,
        kind="discard",
        approved=approved,
        decision_id=decision_id,
        note=note or "nothing written",
    )


def review_order(proposals: list[Proposal]) -> list[Proposal]:
    """Order a review session so dependencies resolve: creates, merges, links, discards."""
    return sorted(
        proposals,
        key=lambda p: (KIND_ORDER.get(p.kind, 9), -(p.similarity or 0.0), p.id),
    )


def resolve(proposal: Proposal, *, proposals_dir: Path | str | None = None) -> bool:
    """Remove a decided proposal. Approval and rejection both end here."""
    return _proposal.delete_proposal(proposal.id, proposals_dir=proposals_dir)
