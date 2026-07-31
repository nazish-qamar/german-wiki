"""Merge regeneration and the SPEC §12.1 drift defence.

SPEC §4.1: on OVERLAP, **regenerate rather than concatenate**. One canonical explanation
covering both, every distinct example preserved, every exception preserved, and -- the
constraint that matters -- *do not add information not present in either source*.
Hallucinated grammar rules are the worst failure mode for a study tool, because you would
be memorizing fiction.

SPEC §12.1 names the risk this creates: every regeneration is a lossy re-encoding, so
after ten merges a node can quietly diverge from anything in its sources. Three
mitigations, and they are deliberately different in strength:

1. **Unsourced examples -> flag.** Pull the German example sentences out of the
   regenerated body and check each against A, B and their ``/raw`` texts. Misses are
   marked and shown in the review diff. This check is *fuzzy* -- a legitimate
   reformatting produces a non-verbatim example that is not drift -- so a hard refusal
   would false-positive and get routed around. The human approval gate (ADR-003) is the
   real guard; this only aims attention at the lines most worth reading.

2. **Regeneration cap -> hard refuse.** An exact integer with no false positives, and it
   guards the one drift a reviewer structurally *cannot* see: the reviewer judges one
   diff at a time and never sees cumulative divergence across many merges.

3. **``/raw`` stays immutable.** Nothing here writes to it. The cap prevents drift
   accumulating; raw-immutability is what lets a capped node be re-derived from its
   sources rather than re-merged from its already-drifted state. The two are halves of
   one defence.

Note what raw is and is not used for: it feeds the *check*, not the prompt. SPEC §4.1's
merge prompt is A and B, and §12.1 calls raw the re-verification anchor. Sending it to
the model would bloat every merge call and give the model more material to blend, which
is the opposite of the constraint.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ..llm import JSON_OBJECT, ChatClient, ModelResponse, Prompt, ShotPair, complete
from ..logutil import get_logger
from ..models import Node
from . import _ledger
from ._adjudicate import STEP, AdjudicationError, _strip_fences

logger = get_logger(__name__)

PROMPT_VERSION = "merge@1"

# SPEC §12.1: "cap regenerations per node". Five is the point at which a body has been
# re-encoded enough times that comparing it to /raw is more trustworthy than merging it
# again -- and a capped node is not stuck, it routes to manual merge, where re-deriving
# from the immutable sources is the intended move.
#
# This is the dial. Raising it trades drift risk for convenience; the honest test of a
# new value is whether a node at that count still reads like its sources.
MAX_REGENERATIONS = 5


class MergedBody(BaseModel):
    """A regenerated body, plus everything review needs to judge it."""

    model_config = ConfigDict(extra="forbid")

    body_md: str
    changelog: str
    # Example sentences that could not be traced to A, B or /raw. Advisory (see above).
    unsourced: list[str] = []


class CapCheck(BaseModel):
    """Whether this node may be regenerated again, and on what evidence."""

    model_config = ConfigDict(extra="forbid")

    allowed: bool
    count: int | None  # None == unknown, because the ledger could not be read
    reason: str = ""
    ledger_readable: bool = True


def check_cap(
    node: Node,
    *,
    decisions_log: Path | str | None = None,
    limit: int = MAX_REGENERATIONS,
) -> CapCheck:
    """Resolve the regeneration count for ``node``, failing safe when it is unknown.

    Three states, not two -- and the third is the whole point:

    ==========================  ======================  ==========================
    ledger                      node                    result
    ==========================  ======================  ==========================
    readable                    any                     authoritative count
    missing / corrupt           ``version`` unset or 1  proceed (fresh-clone case)
    missing / corrupt           ``version`` > 1         **refuse** (wipe case)
    ==========================  ======================  ==========================

    A missing ledger must never read as "count = 0, proceed": that silently disarms the
    guard. But refusing unconditionally would dead-end a fresh clone, where the ledger is
    legitimately absent because nothing has ever been merged. ``Node.version`` separates
    the two. It is used *only* as a tripwire and never as the count -- it is
    hand-editable, so it cannot be trusted to permit a merge, but it can be trusted to
    forbid one, since a wrong value in that direction costs only a refusal.
    """
    try:
        count = _ledger.merge_count(node.id, decisions_log=decisions_log)
    except _ledger.LedgerUnreadable as exc:
        if (node.version or 1) > 1:
            return CapCheck(
                allowed=False,
                count=None,
                ledger_readable=False,
                reason=(
                    f"{exc} Node {node.id!r} is at version {node.version}, so it has been "
                    "written by the pipeline before and its regeneration count cannot be "
                    "verified. Refusing to merge until the ledger is restored "
                    "(`git restore logs/decisions.jsonl`); merge this one by hand if you "
                    "cannot."
                ),
            )
        return CapCheck(
            allowed=True,
            count=None,
            ledger_readable=False,
            reason=(
                f"{exc} Node {node.id!r} has never been written by the pipeline "
                "(version unset or 1), so there is no count to have lost."
            ),
        )

    if count >= limit:
        return CapCheck(
            allowed=False,
            count=count,
            reason=(
                f"node {node.id!r} has already been regenerated {count} time(s), at the "
                f"cap of {limit} (SPEC §12.1). Merge this by hand, or re-derive the node "
                "from its /raw sources rather than merging its drifted body again."
            ),
        )
    return CapCheck(allowed=True, count=count)


# --- example sentences: extraction and sourcing ---

_EXAMPLES_HEADING = re.compile(r"^#{1,6}\s*(examples|beispiele)\b", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s")
_BULLET = re.compile(r"^\s*[-*+]\s+")
_TRAILING_TAGS = re.compile(r"\s*\[[^\[\]]*\]\s*$")
_TRAILING_GLOSS = re.compile(r"\s*\([^()]*\)\s*$")


def example_lines(body_md: str) -> list[str]:
    """The German example sentences in a body's ``## Examples`` section.

    Scoped to that section on purpose. The extraction prompt mandates the shape
    (``- German sentence (English gloss)``), and prose elsewhere in a body is
    explanation, which is legitimately paraphrased on merge. Example sentences are where
    invention actually hurts -- a fabricated sentence is a fact you would memorize.
    """
    out: list[str] = []
    in_section = False
    for line in body_md.splitlines():
        if _EXAMPLES_HEADING.match(line.strip()):
            in_section = True
            continue
        if in_section and _HEADING.match(line.strip()):
            in_section = False
            continue
        if not in_section or not _BULLET.match(line):
            continue

        text = _BULLET.sub("", line).strip()
        text = _TRAILING_TAGS.sub("", text)  # [alltag, du-Ebene]
        text = _TRAILING_GLOSS.sub("", text)  # (English gloss)
        if text.strip():
            out.append(text.strip())
    return out


def _fold(text: str) -> str:
    """Fold for comparison: NFKC, lowercase, letters and digits only.

    Aggressive because it answers "did this sentence come from somewhere?", where
    Markdown emphasis, punctuation and spacing are all noise. Deliberately *not* shared
    with ``embed``'s two normalizations, which answer different questions (ADR-010).
    """
    folded = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in folded).split())


def unsourced_examples(body_md: str, sources: list[str]) -> list[str]:
    """Example sentences in ``body_md`` that appear in none of ``sources``.

    Advisory. Substring containment over folded text, so reformatting and emphasis do
    not trip it, but a genuinely invented sentence does.
    """
    haystack = _fold("\n".join(s for s in sources if s))
    return [ex for ex in example_lines(body_md) if _fold(ex) and _fold(ex) not in haystack]


def new_examples(winner: Node, loser: Node) -> list[str]:
    """Example lines the loser has and the winner lacks (SPEC §3.2, the SAME path).

    Mechanical: no model call. SAME means B teaches nothing new, so its *body* is
    discarded -- but its examples and provenance are still worth keeping, and copying
    them verbatim cannot invent anything.

    **Deliberately conservative, and known to over-keep.** The comparison is textual, so
    it cannot see that "Kannst du mir helfen?" restates a register level the winner
    already demonstrates with "Kannst du mir mal kurz helfen?". Observed on the first
    real SAME merge (`um-hilfe-bitten`, 2026-07-31): three appended lines that were
    plainer phrasings of levels already in the node's table, trimmed by hand afterwards.

    That is the right default and should not be "fixed" by loosening the comparison. The
    two errors are not symmetric: over-keeping produces clutter a human deletes in
    seconds, while dropping an example silently removes material from a study note and
    is unrecoverable without re-reading /raw. A merge must never decide on its own that
    a sentence is redundant.

    *Maybe later, once slice 6 lands:* SPEC §6.2 tags register on each **example
    sentence**, not just the node. With those tags present, "this example demonstrates a
    du-Ebene request, which the winner already covers" becomes a structural query over
    tags rather than a semantic judgment -- which is the only version of this refinement
    that would be safe, because it can state *why* it skipped a line. Until then, keep
    everything and let the human trim.
    """
    have = {_fold(ex) for ex in example_lines(winner.body_md)}
    return [ex for ex in example_lines(loser.body_md) if _fold(ex) not in have]


# --- the prompt: fixed content first, variable content last (SPEC §10) ---

SYSTEM = """\
You merge two overlapping German learning concepts into one canonical explanation for a \
personal study wiki.

Write ONE explanation that covers both A and B.
Preserve every distinct example sentence from either.
Preserve any exception or edge case mentioned in either.
Do NOT add information that is not present in either source.

That last rule is absolute. This wiki is used to study from, so an invented grammar rule \
or an invented example sentence becomes something the reader memorizes as fact. If A and \
B disagree, say so in the body rather than resolving it. If something is unclear, leave \
it out or state the uncertainty - never fill the gap with your own knowledge of German, \
however confident you are.

Write the body in Markdown, in the style of the sources: a short statement of the rule or \
meaning, then any exception, then an "## Examples" section listing example sentences as \
- German sentence (English gloss)

Keep every example sentence from both sources, verbatim where possible. Drop an example \
only when it is a word-for-word duplicate of another.

Also return a one-line changelog saying what the merge changed - what B contributed, and \
anything you dropped as redundant.\
"""

OUTPUT_SCHEMA = """\
Respond with a single JSON object, no prose and no code fences:

{
  "body_md": "string (Markdown, the merged canonical explanation)",
  "changelog": "one line describing what this merge changed"
}\
"""

FEW_SHOT = [
    ShotPair(
        user=(
            "A: Perfekt mit haben (Perfect tense with haben)\n\n"
            "Das Perfekt bildet man mit *haben* + Partizip II.\n\n"
            "## Examples\n"
            "- Ich habe gearbeitet. (I worked.)\n\n"
            "---\n\n"
            "B: Perfekt: haben oder sein (Perfect: haben or sein)\n\n"
            "Verben der Bewegung nehmen *sein*.\n\n"
            "## Examples\n"
            "- Ich bin nach Berlin gefahren. (I drove to Berlin.)\n\n"
            "---\n\n"
            "B adds: the *sein* auxiliary for verbs of motion"
        ),
        assistant=json.dumps(
            {
                "body_md": (
                    "Das Perfekt bildet man mit einem Hilfsverb + Partizip II. Die "
                    "meisten Verben nehmen *haben*; Verben der Bewegung nehmen *sein*.\n\n"
                    "## Examples\n"
                    "- Ich habe gearbeitet. (I worked.)\n"
                    "- Ich bin nach Berlin gefahren. (I drove to Berlin.)"
                ),
                "changelog": "Added the sein auxiliary for verbs of motion from B.",
            },
            ensure_ascii=False,
        ),
    )
]


def _render(node: Node, label: str) -> str:
    return f"{label}: {node.title_de} ({node.title_en})\n\n{node.body_md.strip()}"


def build_prompt(winner: Node, loser: Node, *, b_adds: str | None = None) -> Prompt:
    """Assemble the merge prompt. ``/raw`` is deliberately not included -- see module doc."""
    variable = f"{_render(winner, 'A')}\n\n---\n\n{_render(loser, 'B')}"
    if b_adds and b_adds.strip():
        variable = f"{variable}\n\n---\n\nB adds: {b_adds.strip()}"
    return Prompt(
        system=SYSTEM,
        output_schema=OUTPUT_SCHEMA,
        few_shot=FEW_SHOT,
        variable=variable,
        version=PROMPT_VERSION,
    )


def parse(response: ModelResponse, *, sources: list[str] | None = None) -> MergedBody:
    """Validate a merge response, or raise ``AdjudicationError``.

    Truncation is checked before parsing, for the same reason as in ``_adjudicate``:
    a truncated response can still contain parseable-looking content, and a half-written
    body is far worse here -- it would silently drop the tail of a node.
    """
    if response.finish_reason == "length":
        raise AdjudicationError(
            f"merge regeneration truncated at {response.usage.completion_tokens} "
            f"completion tokens (finish_reason=length); raise max_tokens for step "
            f"{response.step!r} in config/models.yaml. Reasoning tokens count toward the "
            "cap, and a truncated merge would silently drop the end of the node.",
            response=response,
        )

    body = _strip_fences(response.text)
    if not body:
        raise AdjudicationError(
            f"merge regeneration returned no content (finish_reason={response.finish_reason})",
            response=response,
        )

    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AdjudicationError(
            f"merge regeneration did not return valid JSON: {exc}; got {body[:200]!r}",
            response=response,
        ) from exc

    try:
        merged = MergedBody.model_validate(data)
    except ValidationError as exc:
        raise AdjudicationError(
            f"merge regeneration did not match the schema: {exc}", response=response
        ) from exc

    if not merged.body_md.strip():
        raise AdjudicationError("merge regeneration returned an empty body", response=response)

    merged.unsourced = unsourced_examples(merged.body_md, sources or [])
    if merged.unsourced:
        logger.warning(
            "merge produced %d example sentence(s) not traceable to either source or "
            "/raw; flagged for review (SPEC §4.1 'do not add'): %s",
            len(merged.unsourced),
            "; ".join(merged.unsourced[:3]),
        )
    return merged


def regenerate(
    winner: Node,
    loser: Node,
    *,
    b_adds: str | None = None,
    sources: list[str] | None = None,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[MergedBody, ModelResponse]:
    """Regenerate one canonical body covering both nodes. Writes nothing.

    ``sources`` are the raw texts behind the pair, used only for the unsourced-example
    check. They are not sent to the model.
    """
    response = complete(
        STEP,
        build_prompt(winner, loser, b_adds=b_adds),
        client=client,
        response_format=JSON_OBJECT,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=env,
        use_cache=use_cache,
        refresh=refresh,
    )
    return parse(response, sources=sources), response
