"""On-demand analysis: what the corpus implies, as reviewable proposals (SPEC §7.2).

SPEC §7.2 wants this to happen automatically at ingest -- "the merge step links it to
both its root node and its prefix node, creating either if absent". It does not happen
automatically here, and that is deliberate (ADR-014): auto-creation is a write path whose
correctness rests on segmentation and transparency being trustworthy, and every earlier
slice earned its automation by being watched first (detection before merge, queue before
``/nodes``, propose before apply). So slice 7 proposes; a later slice may auto-accept.

Three outputs, all through the existing queue and none of them new machinery:

- a missing prefix node          -> ``create``
- a family/prefix edge           -> ``link`` (relation ``same_family``, SPEC §4.2)
- a family's transparency        -> ``morphology``

**A proposed prefix node's body is a template, not invented semantics.** It lists the
members that motivated it and leaves the directional meaning as an explicit TODO.
Generating "``an-`` means toward/on" from a prefix inventory would be exactly the
unsourced claim SPEC §4.1 forbids -- the inventory knows separability, not meaning. The
proposal body is hand-editable at review (ADR-011), so you write the semantics there and
what you wrote is what lands.

**Variable-stress prefixes get no ``create`` proposal at all.** A prefix node must commit
to ``separable:`` to be a grid row, and for ``um``/``durch``/``über`` that commitment is
precisely what the machine cannot make. They surface as needing a human instead.

**Nothing already stated is re-proposed.** An existing link is left alone even when it
dangles -- a dangling target is your intention (SPEC §7.3), not a task -- and a family
that already declares ``family_transparency`` is never re-judged, so a model verdict
cannot overwrite yours.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..llm import ChatClient
from ..logutil import get_logger
from ..merge import Proposal, now_iso, proposal_id
from ..models import Node
from . import _transparency
from ._grid import is_family_node, is_prefix_node, morpheme_of
from ._prefixes import classify
from ._segment import CorpusIndex, Segmentation, Withheld, fold, segment

logger = get_logger(__name__)

SAME_FAMILY = "same_family"

# Staging directory and provenance marker for nodes the corpus *implied* rather than a
# source producing. Greppable: `grep -l morph-analysis nodes/` finds every node that
# arrived this way rather than from ingested material.
MORPH_SOURCE_ID = "morph-analysis"

PREFIX_NODE_BODY = """\
**TODO — write what `{prefix}-` does to a verb's sense.**

That one sentence is the reason this node exists: SPEC §7.1 makes the prefix's meaning the
learnable thing, and learning it once pays off across every verb that uses it.

It is left blank on purpose. A prefix inventory knows *separability*, not *meaning*, so
anything written here by the pipeline would be a claim with no source behind it (SPEC
§4.1). Replace this line before approving — what you write is what lands.

Observed in this corpus:

{members}
"""


class Analysis(BaseModel):
    """What one ``gw families`` run found. Nothing here has touched ``/nodes``."""

    model_config = ConfigDict(extra="forbid")

    proposals: list[Proposal] = Field(default_factory=list)
    # Complete node files that must be STAGED before their `create` proposals can apply.
    #
    # `apply_create` promotes a staged node; it has always assumed one exists, because
    # every create until slice 7 came from `gw ingest`. A create with no file anywhere
    # cannot work: `Proposal` carries an id and a body, but `Node` also requires
    # `title_de`, `title_en`, `type` and `status`, so a node simply cannot be built from a
    # proposal. Rather than widen `Proposal` into a second copy of `Node`'s schema, the
    # node is staged in `/queue` exactly as ingestion stages one -- which also makes it
    # hand-editable before approval, like every other queued node (ADR-009).
    staged: list[Node] = Field(default_factory=list)
    # Stress-ambiguous verbs: a person must say which word it is (see ``_prefixes``).
    ambiguous: list[Withheld] = Field(default_factory=list)
    # Splittable in principle, but the stem has no node yet. Resolves itself as you study.
    unresolved: list[Withheld] = Field(default_factory=list)
    segmented: list[Segmentation] = Field(default_factory=list)
    judged: int = 0

    def by_kind(self, kind: str) -> list[Proposal]:
        return [p for p in self.proposals if p.kind == kind]


def _prefix_node_id(morpheme: str) -> str:
    return f"prefix-{morpheme}"


def _prefix_node(morpheme: str, members: list[tuple[str, Node]]) -> Node:
    """Build the complete prefix node a ``create`` proposal will promote.

    Every required ``Node`` field is filled from evidence or from an explicit convention;
    nothing is invented:

    - ``cefr`` is the **lowest** level among the families that motivated it — you meet a
      prefix at the earliest verb that uses it. ``cefr_basis`` names those families, so
      slice 6's ``gw relevel`` can re-derive it and you can see what drove it (SPEC §5).
    - ``separable`` comes from the inventory. Variable prefixes never reach here.
    - ``family_transparency`` is left unset: that is a judgment about meaning (SPEC §7.4),
      and this node has no meaning written yet.
    - ``body_md`` is the TODO template — the meaning is yours to write at review.
    """
    from ..level import CEFR_ORDER

    families = {node.id: node for _, node in members}
    cefr = min((n.cefr for n in families.values()), key=lambda c: CEFR_ORDER[c])
    return Node(
        id=_prefix_node_id(morpheme),
        title_de=f"{morpheme}- (Präfix)",
        title_en=f"prefix {morpheme}-",
        type="pattern",
        cefr=cefr,
        cefr_basis=f"morph:derived(min of {', '.join(sorted(families))})",
        status="draft",  # machine-proposed, like every ingested candidate (slice 3)
        separable=classify(morpheme) == "separable",
        # Provenance is the analysis itself: this node exists because N verbs in the
        # corpus use this prefix, and the body lists them. There is no /raw source
        # because nothing was ingested -- the corpus implied it.
        source_ids=[MORPH_SOURCE_ID],
        body_md=PREFIX_NODE_BODY.format(
            prefix=morpheme,
            members="\n".join(f"- {word}" for word, _ in sorted(members)),
        ),
    )


def analyse(
    nodes: list[Node],
    *,
    judge_transparency: bool = True,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    now: str | None = None,
) -> Analysis:
    """Segment every family's lemmas and propose what the corpus implies. Writes nothing."""
    corpus = CorpusIndex.build(nodes)
    ids = {n.id for n in nodes}
    stamp = now or now_iso()

    prefix_nodes: dict[str, Node] = {}
    for node in nodes:
        if is_prefix_node(node) and (morpheme := morpheme_of(node)):
            prefix_nodes[morpheme] = node

    result = Analysis()
    wanted_prefixes: dict[str, list[tuple[str, Node]]] = {}

    for family in (n for n in nodes if is_family_node(n)):
        declared = {fold(lk.target) for lk in family.links if lk.relation == SAME_FAMILY}

        for lemma in family.lemmas or []:
            outcome = segment(lemma, corpus=corpus)
            if isinstance(outcome, Withheld):
                if outcome.needs_human:
                    result.ambiguous.append(outcome)
                elif outcome.reason == "no-corpus-evidence":
                    result.unresolved.append(outcome)
                continue

            result.segmented.append(outcome)
            if outcome.separability == "variable":
                continue  # unreachable today, but the rule is the rule
            wanted_prefixes.setdefault(outcome.prefix, []).append((outcome.word, family))

            target = _prefix_node_id(outcome.prefix)
            if target in declared:
                continue  # already stated -- including when it dangles (SPEC §7.3)
            result.proposals.append(
                Proposal(
                    id=proposal_id("link", family.id, target, SAME_FAMILY),
                    kind="link",
                    outcome="DISTINCT_RELATED",
                    basis="rules",
                    candidate=family.id,
                    counterpart=target,
                    relation=SAME_FAMILY,
                    # family -> prefix, matching the direction the seed corpus already uses.
                    direction="b_to_a",
                    reason=(
                        f"{outcome.word} = {outcome.prefix}- + {outcome.stem}; "
                        f"{outcome.stem} is vouched for by {outcome.stem_node_id}"
                    ),
                    created_at=stamp,
                    body_md=f"Proposed edge: **{family.id} -{SAME_FAMILY}-> {target}**\n",
                )
            )
            declared.add(target)

        if judge_transparency and family.family_transparency is None:
            exemplar = next(
                (s for s in result.segmented if s.stem_node_id == family.id), None
            )
            if exemplar is not None:
                members = ", ".join(
                    s.word for s in result.segmented if s.stem_node_id == family.id
                )
                verdict, _ = _transparency.judge(
                    root=family.root or "",
                    prefix=exemplar.prefix,
                    word=exemplar.word,
                    gloss=f"family members: {members}",
                    client=client,
                    settings_path=settings_path,
                    cache_dir=cache_dir,
                    usage_log=usage_log,
                    env=env,
                )
                result.judged += 1
                result.proposals.append(
                    Proposal(
                        id=proposal_id("morphology", family.id),
                        kind="morphology",
                        outcome="MORPHOLOGY",
                        basis="llm",
                        candidate=family.id,
                        family_transparency=verdict.transparency,
                        confidence=verdict.confidence,
                        reason=verdict.reason,
                        created_at=stamp,
                        body_md=(
                            f"`family_transparency: {verdict.transparency}`\n\n"
                            f"{verdict.reason}\n\n"
                            "_Approving this changes frontmatter only; no body is "
                            "rewritten. SPEC §7.4: the node holds the truth, the grid "
                            "only predicts._\n"
                        ),
                    )
                )

    for morpheme, members in sorted(wanted_prefixes.items()):
        if morpheme in prefix_nodes or _prefix_node_id(morpheme) in ids:
            continue
        node = _prefix_node(morpheme, members)
        result.staged.append(node)
        result.proposals.append(
            Proposal(
                id=proposal_id("create", node.id),
                kind="create",
                outcome="DISTINCT",
                basis="rules",
                candidate=node.id,
                source_id=MORPH_SOURCE_ID,
                reason=f"{len(members)} verb(s) in the corpus use {morpheme}-",
                created_at=stamp,
                # The body this create will write. ``candidate_path`` is filled in by the
                # caller once the node is staged -- see ``Analysis.staged``.
                body_md=node.body_md,
            )
        )

    if result.ambiguous:
        logger.info(
            "%d stress-ambiguous verb(s) withheld from the grid: %s",
            len(result.ambiguous),
            ", ".join(w.word for w in result.ambiguous),
        )
    return result
