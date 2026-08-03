"""Adjudication: what *kind* of connection is this pair? (SPEC §3.1 tier 3, ADR-010)

SPEC §3.1 frames adjudication as ``SAME | OVERLAP | DISTINCT``, which assumes every
flagged pair is a **redundancy** question. Real output from the first ingested source
says otherwise: of three gray-zone pairs the weakest was ``um-hilfe-bitten`` ↔
``verben-mit-praepositionen`` at 0.882 -- not a duplicate at all, but *bitten **um** +
Akkusativ*, which is a verb-preposition combination and therefore a ``governs`` relation
(SPEC §4.2).

So high similarity means "these are connected", and **how** is the thing this module has
to decide. Hence four outcomes, not three:

- ``SAME``             -> discard B's body, keep its provenance and new examples (§3.2)
- ``OVERLAP``          -> merge (§4.1)
- ``DISTINCT_RELATED`` -> propose a typed edge (§4.2), write nothing to the bodies
- ``DISTINCT``         -> leave alone; the candidate becomes its own node

Without the fourth branch every genuine relation gets mis-answered as a merge question
and either fragments the wiki or corrupts a node body.

**All four route through review** (ADR-010). Nothing here writes anything; this module
returns a verdict and the graph turns it into a proposal.

``complete()`` is parse-free by design, so this module owns the failure modes -- the same
division of labour as ``ingest/_extract.py``. The one that bites is
``finish_reason == "length"``: GLM spends completion tokens on internal reasoning before
emitting content, so a tight cap returns a well-formed *empty* response. That is an error
here, not an empty verdict, and the error carries the provider's reasoning trace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..llm import JSON_OBJECT, ChatClient, ModelResponse, Prompt, ShotPair, complete, strip_fences
from ..logutil import get_logger
from ..models import Node

logger = get_logger(__name__)

STEP = "adjudication"

# Enters the cache key, so bumping it re-adjudicates without touching prompt text --
# use it when THIS parser changes what it expects back (ADR-008).
PROMPT_VERSION = "adjudicate@1"

Outcome = Literal["SAME", "OVERLAP", "DISTINCT_RELATED", "DISTINCT"]

# SPEC §4.2's seven typed relations, closed HERE in the response schema so the model
# cannot invent `related_to` and hand us the generic backlink hairball §4.2 exists to
# prevent. ``models.Link.relation`` itself stays an open ``str``: constraining what a
# model may *propose* is a different question from constraining what a node may hold,
# and only the first one needs to fail loudly (cf. ADR-007's enum-vs-open-vocab split).
Relation = Literal[
    "contrasts_with",
    "prerequisite_for",
    "formal_variant_of",
    "same_family",
    "false_friend_of",
    "governs",
    "exception_to",
]

# Which way the edge points. A `governs` edge from the verb to the case it takes is not
# the same claim as the reverse, and §4.2's whole value is in the direction.
Direction = Literal["a_to_b", "b_to_a"]


class Adjudication(BaseModel):
    """One verdict on one pair. Carries no side effects and no writes."""

    model_config = ConfigDict(extra="forbid")

    outcome: Outcome
    confidence: float = 0.5
    reason: str = ""
    # SPEC §3.1's prompt: "If OVERLAP, state what B adds that A lacks."
    b_adds: str | None = None
    # DISTINCT_RELATED only.
    relation: Relation | None = None
    direction: Direction | None = None


class AdjudicationError(RuntimeError):
    """Adjudication produced no usable verdict.

    Carries the ``ModelResponse`` when there was one, so callers can surface the
    provider's reasoning trace -- the only way to see why a truncated call produced
    no content. Mirrors ``ingest.ExtractionError``.
    """

    def __init__(self, message: str, *, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.reasoning_content = response.reasoning_content if response else None


# --- the prompt: fixed content first, variable content last (SPEC §10) ---

SYSTEM = """\
You compare two German learning concepts from a personal study wiki and decide how they \
are related. Both were flagged as similar by an embedding search; your job is to say what \
that similarity actually means.

Answer with exactly one outcome:

SAME - the same learnable concept, expressed twice. B teaches nothing A does not already \
teach. Choose this only when a learner reading A would gain nothing from B.

OVERLAP - substantially the same concept, but B carries real content A lacks: an extra \
rule, an exception, a register, examples worth keeping. These should become one richer \
node.

DISTINCT_RELATED - two genuinely different concepts that are nonetheless connected, and \
the connection is worth recording as a typed edge. This is common and easy to miss: a \
fixed expression and the grammar rule it obeys are NOT duplicates, they are a relation. \
When you choose this, name the relation and its direction.

DISTINCT - two different concepts with no connection worth recording. High text \
similarity alone is not a connection; shared vocabulary or shared sentence shape is not \
a relation.

Available relations (DISTINCT_RELATED only), used strictly:
- governs: a verb or expression requires a particular case or preposition
- prerequisite_for: one must be learned before the other is usable
- contrasts_with: a pair learned against each other (Konjunktiv I vs II)
- formal_variant_of: the same intent at a different formality level
- same_family: a shared root or stem
- false_friend_of: looks like a word in the other language, means something else
- exception_to: an irregular member of a rule

Direction: "a_to_b" means the edge runs from A to B; "b_to_a" the reverse. For governs, \
the edge runs FROM the verb or expression TO the case or preposition it takes.

Bias toward merging over fragmenting: a study wiki fails by accumulating thin, \
disconnected notes. But never merge two concepts merely because they share words - that \
is what DISTINCT_RELATED is for.

Judge only what is written. Do not use knowledge of German that is not present in either \
text to decide they are the same. If you are unsure, lower the confidence and say why in \
"reason"; do not guess an outcome confidently.\
"""

OUTPUT_SCHEMA = """\
Respond with a single JSON object, no prose and no code fences:

{
  "outcome": "SAME|OVERLAP|DISTINCT_RELATED|DISTINCT",
  "confidence": 0.0,
  "reason": "one sentence explaining the decision",
  "b_adds": "what B adds that A lacks (OVERLAP only, otherwise null)",
  "relation": "governs|prerequisite_for|contrasts_with|formal_variant_of|same_family|false_friend_of|exception_to (DISTINCT_RELATED only, otherwise null)",
  "direction": "a_to_b|b_to_a (DISTINCT_RELATED only, otherwise null)"
}\
"""


def _render(node: Node, label: str) -> str:
    return (
        f"{label}: {node.title_de} ({node.title_en})\n"
        f"type: {node.type} | cefr: {node.cefr}\n\n"
        f"{node.body_md.strip()}"
    )


def _pair_text(a: Node, b: Node) -> str:
    return f"{_render(a, 'A')}\n\n---\n\n{_render(b, 'B')}"


def _shot(a: str, b: str, answer: dict[str, Any]) -> ShotPair:
    return ShotPair(
        user=f"{a}\n\n---\n\n{b}",
        assistant=json.dumps(answer, ensure_ascii=False),
    )


# Three exemplars, teaching the two boundaries that are actually hard: SAME vs OVERLAP,
# and redundancy vs relation (the ADR-010 finding). DISTINCT is the residual case and is
# well covered by the instructions above.
#
# NOTE, deliberately: none of these is `um-hilfe-bitten` ↔ `verben-mit-praepositionen`,
# even though that pair is the canonical worked example in ADR-010. It is the assertion
# in tests/test_merge_live.py, and few-shotting the model on the exact pair it is then
# tested against would prove nothing -- the test would be measuring recall of the prompt,
# not generalization. Shot 3 is structurally analogous (fixed expression vs the grammar
# rule it obeys, resolving to `governs`) with entirely different content.
FEW_SHOT = [
    _shot(
        "A: Die Wochentage (The days of the week)\ntype: vocab | cefr: A1\n\n"
        "Montag, Dienstag, Mittwoch, Donnerstag, Freitag, Samstag, Sonntag. "
        "Alle Wochentage sind maskulin: der Montag.",
        "B: Wochentage (Weekdays)\ntype: vocab | cefr: A1\n\n"
        "Die sieben Tage der Woche: Montag bis Sonntag. Sie haben alle den Artikel *der*.",
        {
            "outcome": "SAME",
            "confidence": 0.95,
            "reason": "Both list the seven weekdays and state that all are masculine.",
            "b_adds": None,
            "relation": None,
            "direction": None,
        },
    ),
    _shot(
        "A: Perfekt mit haben (Perfect tense with haben)\ntype: grammar | cefr: A2\n\n"
        "Das Perfekt bildet man mit *haben* + Partizip II. "
        "Ich habe gearbeitet. Sie hat gegessen.",
        "B: Perfekt: haben oder sein (Perfect: haben or sein)\ntype: grammar | cefr: A2\n\n"
        "Die meisten Verben bilden das Perfekt mit *haben*. Verben der Bewegung und der "
        "Zustandsänderung nehmen *sein*: Ich bin gefahren. Er ist eingeschlafen.",
        {
            "outcome": "OVERLAP",
            "confidence": 0.9,
            "reason": "Same tense, but B covers the auxiliary choice A only half-states.",
            "b_adds": "the *sein* auxiliary for verbs of motion and change of state",
            "relation": None,
            "direction": None,
        },
    ),
    _shot(
        "A: Angst haben vor + Dativ (To be afraid of)\ntype: phrase | cefr: B1\n\n"
        "Die feste Wendung *Angst haben vor* verlangt immer den Dativ. "
        "Ich habe Angst vor dem Hund.",
        "B: Präpositionen mit Dativ (Prepositions taking the dative)\n"
        "type: grammar | cefr: A2\n\n"
        "Die Präpositionen *aus, bei, mit, nach, seit, von, zu* stehen immer mit dem "
        "Dativ. Bei Wechselpräpositionen wie *vor* entscheidet die Bedeutung.",
        {
            "outcome": "DISTINCT_RELATED",
            "confidence": 0.85,
            "reason": "A fixed expression is not the rule it obeys; A selects vor + Dativ.",
            "b_adds": None,
            "relation": "governs",
            "direction": "a_to_b",
        },
    ),
]


def build_prompt(a: Node, b: Node) -> Prompt:
    """Assemble the adjudication prompt; only ``variable`` changes per pair."""
    return Prompt(
        system=SYSTEM,
        output_schema=OUTPUT_SCHEMA,
        few_shot=FEW_SHOT,
        variable=_pair_text(a, b),
        version=PROMPT_VERSION,
    )


# --- parsing ---


def _coherent(verdict: Adjudication) -> Adjudication:
    """Reconcile fields the schema cannot constrain against each other.

    Asymmetric on purpose. A ``DISTINCT_RELATED`` without a relation is *unusable* --
    there is no edge to propose -- so it fails. Relation fields on any other outcome
    are merely meaningless, so they are dropped with a warning: discarding a stray
    field is cheaper than throwing away an otherwise sound verdict and paying for a
    re-run.
    """
    if verdict.outcome == "DISTINCT_RELATED":
        if verdict.relation is None or verdict.direction is None:
            raise AdjudicationError(
                "outcome DISTINCT_RELATED needs both a relation and a direction; "
                f"got relation={verdict.relation!r} direction={verdict.direction!r}"
            )
        return verdict

    if verdict.relation is not None or verdict.direction is not None:
        logger.warning(
            "adjudication returned outcome %s with relation=%r direction=%r; dropping "
            "them (a typed edge is only proposed for DISTINCT_RELATED)",
            verdict.outcome,
            verdict.relation,
            verdict.direction,
        )
        verdict = verdict.model_copy(update={"relation": None, "direction": None})

    if verdict.outcome == "OVERLAP" and not (verdict.b_adds or "").strip():
        # SPEC §3.1 asks for it explicitly; its absence weakens the merge prompt but
        # does not invalidate the verdict.
        logger.warning("adjudication returned OVERLAP without stating what B adds")
    return verdict


def parse(response: ModelResponse) -> Adjudication:
    """Validate a model response into a verdict, or raise ``AdjudicationError``.

    Guard order matters: truncation is checked before parsing, because a truncated
    response can still contain parseable-looking content.
    """
    if response.finish_reason == "length":
        raise AdjudicationError(
            f"adjudication truncated at {response.usage.completion_tokens} completion "
            f"tokens (finish_reason=length); raise max_tokens for step {response.step!r} "
            "in config/models.yaml. Reasoning tokens count toward the cap.",
            response=response,
        )

    body = strip_fences(response.text)
    if not body:
        raise AdjudicationError(
            f"adjudication returned no content (finish_reason={response.finish_reason})",
            response=response,
        )

    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AdjudicationError(
            f"adjudication did not return valid JSON: {exc}; got {body[:200]!r}",
            response=response,
        ) from exc

    # Providers emit `null` for the fields the schema marks "otherwise null"; pydantic
    # accepts that for the optionals, but an explicit null outcome should read as the
    # schema violation it is rather than a confusing type error.
    if isinstance(data, dict) and data.get("outcome") is None:
        raise AdjudicationError(
            f"adjudication returned no outcome; got {body[:200]!r}", response=response
        )

    try:
        verdict = Adjudication.model_validate(data)
    except ValidationError as exc:
        raise AdjudicationError(
            f"adjudication did not match the verdict schema: {exc}", response=response
        ) from exc

    try:
        return _coherent(verdict)
    except AdjudicationError as exc:
        raise AdjudicationError(str(exc), response=response) from exc


def adjudicate(
    a: Node,
    b: Node,
    *,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[Adjudication, ModelResponse]:
    """Ask the model how ``a`` and ``b`` are related. Writes nothing anywhere."""
    response = complete(
        STEP,
        build_prompt(a, b),
        client=client,
        response_format=JSON_OBJECT,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=env,
        use_cache=use_cache,
        refresh=refresh,
    )
    return parse(response), response
