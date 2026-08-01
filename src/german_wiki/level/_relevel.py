"""Re-deriving levels on nodes that already exist (SPEC §5, slice 6).

Slice 3 stamped ``cefr_basis: llm:extraction`` on every extracted candidate precisely so
those guesses could be found again (ADR-009). This is the module that finds them and
proposes something better.

**It proposes; it never writes.** Output goes to ``/proposals`` as ``kind: relevel`` and
through the same ``gw review`` gate as merges and links. The lookup is deterministic, but
the step before it — deciding that a node *is about* a given SPEC §5 structure — is an
interpretation that can be wrong, and ``cefr`` drives §5.1's study order, so a bad relevel
reorders what you learn. That is the pedagogy-corruption risk review exists to catch.

**Hand-authored bases are left alone by default.** ``freq:high; goethe:A1(waschen)`` on a
seed node is human judgment, and the whole reason ``cefr_basis`` exists is to record it.
``--all`` overrides that, and is opt-in for exactly that reason.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .. import config, storage
from ..llm import ChatClient
from ..logutil import get_logger
from ..merge import Proposal, now_iso, proposal_id, write_proposal
from ..models import Node
from . import _cefr
from ._cefr import LevelResult

logger = get_logger(__name__)

# Marks a proposal whose level came from the model rather than the rules.
FLAG_TIEBREAK = "llm-tiebreak"
# Marks a proposal that only fills in a missing basis, leaving the level alone.
FLAG_BASIS_ONLY = "basis-only"


class RelevelResult(BaseModel):
    """What one ``gw relevel`` produced. Nothing here touched ``/nodes``."""

    model_config = ConfigDict(extra="forbid")

    considered: int = 0
    proposals: list[Proposal] = Field(default_factory=list)
    paths: list[Path] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)


def targets(nodes: list[Node], *, everything: bool = False) -> list[Node]:
    """Nodes whose level should be re-derived.

    Default: a machine placeholder (``llm:extraction…``) or **no basis at all**. The
    second case matters as much as the first — SPEC §5 says "always store ``cefr_basis``",
    so a node without one has an unexplained level, which is the same problem wearing a
    different hat.
    """
    if everything:
        return list(nodes)
    return [n for n in nodes if _cefr.is_placeholder(n.cefr_basis)]


def build_proposal(node: Node, result: LevelResult) -> Proposal | None:
    """A ``relevel`` proposal, or ``None`` when there is nothing to change.

    Returns ``None`` when the derived level *and* basis both already match — proposing a
    no-op would train you to approve without reading, which is how a review gate stops
    working.
    """
    if result.cefr is None:
        return None
    if result.cefr == node.cefr and result.basis == node.cefr_basis:
        return None

    flags = []
    if result.used_tiebreak:
        flags.append(FLAG_TIEBREAK)
    if result.cefr == node.cefr:
        flags.append(FLAG_BASIS_ONLY)

    return Proposal(
        id=proposal_id("relevel", node.id),
        kind="relevel",
        outcome="RELEVEL",
        basis="llm" if result.used_tiebreak else "rules",
        candidate=node.id,
        cefr=result.cefr,
        cefr_basis=result.basis,
        reason=(
            f"grammar: {result.grammar_summary}. lexical: {result.lexical_summary}."
        ),
        flags=flags,
        created_at=now_iso(),
        body_md=(
            f"**cefr:** `{node.cefr}` → `{result.cefr}`\n\n"
            f"**cefr_basis:**\n\n"
            f"- before: `{node.cefr_basis or '(none)'}`\n"
            f"- after:  `{result.basis}`\n\n"
            f"Signals given to the decision:\n\n"
            f"- grammar: {result.grammar_summary}\n"
            f"- lexical: {result.lexical_summary}\n\n"
            "_Approving this changes those two frontmatter fields only; the body is not "
            "rewritten._\n"
        ),
    )


def relevel(
    *,
    nodes_dir: Path | str | None = None,
    proposals_dir: Path | str | None = None,
    cefr_dir: Path | str | None = None,
    everything: bool = False,
    allow_llm: bool = True,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
) -> RelevelResult:
    """Derive levels for the target nodes and stage the changes for review."""
    root = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    selected = targets(storage.load_all_nodes(root), everything=everything)
    result = RelevelResult(considered=len(selected))

    for node in selected:
        derived = _cefr.derive_level(
            node,
            cefr_dir=cefr_dir,
            allow_llm=allow_llm,
            client=client,
            settings_path=settings_path,
            cache_dir=cache_dir,
            usage_log=usage_log,
            env=env,
        )
        if derived.cefr is None:
            # Rules were silent and the tiebreak was unavailable. Reported rather than
            # guessed at: an unexplained level is what this slice exists to remove.
            result.unresolved.append(node.id)
            logger.info("could not derive a level for %s: %s", node.id, derived.basis)
            continue

        proposal = build_proposal(node, derived)
        if proposal is None:
            result.unchanged.append(node.id)
            continue

        result.proposals.append(proposal)
        result.paths.append(write_proposal(proposal, proposals_dir=proposals_dir))

    return result
