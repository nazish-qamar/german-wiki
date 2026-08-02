"""Proposals: the durable state between adjudication and review (ADR-011).

LangGraph's ``interrupt()`` needs a checkpointer to persist a *paused graph*, but nothing
here needs a paused graph to survive the process. The cross-session state is a **file**:
one Markdown-with-frontmatter proposal per proposed decision, written by ``gw adjudicate``
and read by ``gw review`` minutes or days later. ``InMemorySaver`` is therefore
sufficient, and re-entering the graph on the apply pass is free because adjudication is
content-hash cached (ADR-005) and embeddings are cached (ADR-010) -- **the cache is the
checkpointer**. That avoids a second persistence mechanism that would drift from the
files and the index.

The format is deliberately the same idiom as a queued node (ADR-009): a real file you can
open in an editor, with the **proposed content as the body**, so hand-editing a merge
before approving it needs nothing but a text editor -- and what you edited is exactly what
gets written.

``/proposals`` is gitignored like ``/queue``. A proposal is transient by construction: it
is resolved by approval-or-rejection and deleted either way, so the directory only ever
holds pending work. The durable record is the commit to ``/nodes`` that an approval
produces, plus ``logs/decisions.jsonl``.

**Merges and links share one queue and one format** (ADR-010). A proposed typed edge is a
reviewed write, not a side effect: §4.3's reading only holds because a proposed edge gets
the same human gate as a merge. A second, lighter-feeling queue for links is exactly the
loophole that reading has to avoid, so ``kind`` is a field here rather than a directory.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import frontmatter
from pydantic import BaseModel, ConfigDict, Field

from .. import config
from ._ledger import Kind

# MANUAL is not a model outcome -- it is what the regeneration cap emits when a merge is
# refused (SPEC §12.1). It reaches review so the refusal is visible and actionable rather
# than a pair that silently stops being proposed.
#
# RELEVEL (slice 6) is likewise not an adjudication outcome: it is a CEFR re-derivation
# (SPEC §5) riding the same queue, because ADR-003 gates every write to /nodes and a level
# change drives §5.1's study order.
#
# MORPHOLOGY (slice 7) rides it for the same reason: root/lemmas/separable are derived
# from rules, but `family_transparency` is a model judgment about meaning (SPEC §7.4), and
# a family wrongly marked `high` teaches a false pattern.
ProposalOutcome = Literal[
    "SAME", "OVERLAP", "DISTINCT_RELATED", "DISTINCT", "MANUAL", "RELEVEL", "MORPHOLOGY"
]

# Flags surfaced in the review diff. Advisory: they aim attention, they do not veto.
FLAG_UNSOURCED = "unsourced-examples"
FLAG_CAP = "regeneration-cap"
FLAG_LEDGER_MISSING = "ledger-missing"
FLAG_LOW_CONFIDENCE = "low-confidence"

# Below this the verdict is worth a second look regardless of outcome.
LOW_CONFIDENCE = 0.6

_ORDER = [
    "id",
    "kind",
    "outcome",
    "basis",
    "source_id",
    "candidate",
    "counterpart",
    "winner",
    "loser",
    "relation",
    "direction",
    "cefr",
    "cefr_basis",
    "root",
    "lemmas",
    "separable",
    "family_transparency",
    "similarity",
    "tier",
    "band",
    "confidence",
    "reason",
    "b_adds",
    "changelog",
    "flags",
    "candidate_path",
    "provider",
    "model",
    "created_at",
]


class Proposal(BaseModel):
    """One proposed change, awaiting approval. Writing this file changes nothing."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Kind
    outcome: ProposalOutcome
    # How the proposal was reached: "llm" when a model decided, "threshold" when the
    # >= GRAY_HIGH similarity band did, "rules" when a pure lookup did (SPEC §5's grammar
    # map). None of these is lighter than the others -- ADR-003 means a proposal cannot
    # write without review however it was produced. They only record what it cost and how
    # much to trust it.
    basis: Literal["llm", "threshold", "rules"] = "llm"

    source_id: str | None = None
    candidate: str  # the B side: the node under consideration
    counterpart: str | None = None  # the A side; absent for a bare create

    winner: str | None = None  # merge only
    loser: str | None = None  # merge only

    relation: str | None = None  # link only
    direction: str | None = None  # link only

    # relevel only (slice 6): the derived level and the evidence behind it. Named for the
    # Node fields they replace, so the review diff reads as the frontmatter change it is.
    cefr: str | None = None
    cefr_basis: str | None = None

    # morphology only (slice 7), same naming rule. `family_transparency` is the one that
    # carries a model's judgment; the rest are rule-derived (SPEC §7.4).
    root: str | None = None
    lemmas: list[str] | None = None
    separable: bool | None = None
    family_transparency: str | None = None

    similarity: float | None = None
    tier: str | None = None
    band: str | None = None
    confidence: float | None = None
    reason: str = ""
    b_adds: str | None = None
    changelog: str | None = None
    flags: list[str] = Field(default_factory=list)

    # Where the candidate currently lives (a /queue file), so apply can consume it.
    candidate_path: str | None = None

    provider: str | None = None
    model: str | None = None
    created_at: str | None = None

    # The proposed content. For `merge` and `create` this is what gets written, verbatim,
    # so an edit made here during review IS the approved content. For `link` it is a
    # human-readable summary only -- an approved edge changes frontmatter, never a body
    # (ADR-010) -- and for `discard`/MANUAL it is a note.
    body_md: str = ""

    @property
    def writes_body(self) -> bool:
        """Whether approving this proposal writes ``body_md`` into a node."""
        return self.kind in ("merge", "create")

    @property
    def pair(self) -> tuple[str, str] | None:
        """The ordered id pair this proposal decides, if it decides one."""
        if self.counterpart is None:
            return None
        a, b = self.candidate, self.counterpart
        return (a, b) if a <= b else (b, a)


def proposal_id(
    kind: str,
    candidate: str,
    counterpart: str | None = None,
    relation: str | None = None,
) -> str:
    """A stable, filename-safe id derived from *what is proposed*, not when.

    Stability is the point: re-running ``gw adjudicate`` over the same source produces
    the same id, so a pending proposal is overwritten in place rather than duplicated.
    Node ids are already slugs, so no further sanitizing is needed -- only truncation.
    """
    key = "|".join([kind, candidate, counterpart or "", relation or ""])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"{kind}-{candidate[:40].strip('-')}-{digest}"


def proposal_path(proposal_id: str, *, proposals_dir: Path | str | None = None) -> Path:
    root = Path(proposals_dir) if proposals_dir is not None else config.PROPOSALS_DIR
    return root / f"{proposal_id}.md"


def dumps_proposal(proposal: Proposal) -> str:
    """Serialize to Markdown with frontmatter, in a stable key order."""
    data = proposal.model_dump(exclude={"body_md"})
    meta = {k: data[k] for k in _ORDER if k in data and data[k] is not None}
    for k, v in data.items():  # any field added later, deterministically appended
        if k not in meta and v is not None:
            meta[k] = v
    post = frontmatter.Post(proposal.body_md, **meta)
    text = frontmatter.dumps(
        post, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000
    )
    return text.rstrip("\n") + "\n"


def write_proposal(
    proposal: Proposal, *, proposals_dir: Path | str | None = None
) -> Path:
    """Stage one proposal for review. Never touches ``/nodes``."""
    path = proposal_path(proposal.id, proposals_dir=proposals_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_proposal(proposal), encoding="utf-8", newline="\n")
    return path


def load_proposal(path: Path | str) -> Proposal:
    """Parse one proposal file, validating any hand-edit made during review."""
    path = Path(path)
    post = frontmatter.load(str(path))
    meta = dict(post.metadata)
    meta["body_md"] = post.content
    proposal = Proposal.model_validate(meta)
    if proposal.id != path.stem:
        raise ValueError(
            f"proposal id {proposal.id!r} does not match filename stem {path.stem!r} ({path})"
        )
    return proposal


def list_proposals(*, proposals_dir: Path | str | None = None) -> list[Proposal]:
    """Every pending proposal, most similar first, then by id for stability."""
    root = Path(proposals_dir) if proposals_dir is not None else config.PROPOSALS_DIR
    if not root.is_dir():
        return []
    proposals = [load_proposal(p) for p in sorted(root.glob("*.md"))]
    return sorted(proposals, key=lambda p: (-(p.similarity or 0.0), p.id))


def delete_proposal(proposal_id: str, *, proposals_dir: Path | str | None = None) -> bool:
    """Resolve a proposal by removing it. Approval and rejection both end here."""
    path = proposal_path(proposal_id, proposals_dir=proposals_dir)
    if not path.exists():
        return False
    path.unlink()
    root = path.parent
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()
    return True


def now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).isoformat()
