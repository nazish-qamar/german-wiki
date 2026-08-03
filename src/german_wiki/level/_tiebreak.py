"""The LLM tiebreak: SPEC §5 signal 3, and the last resort (slice 6).

SPEC §5 opens by stating the problem this signal has: "Zero-shot LLM CEFR judgment is
inconsistent across sessions." It is included anyway, under two conditions the spec is
specific about — it fires **only when signals 1 and 2 conflict or are absent**, and it is
called **with those signals already in context** rather than cold. Both conditions are
enforced by ``_cefr``; this module just runs the call.

It routes to ``cefr_tiebreak``, deliberately the free ``glm-4.5-flash`` and not the paid
``glm-4.6`` that adjudication uses. A tiebreak can fire on every node lacking a grammar
match, so it must stay free — and it is the least-grounded signal in the system, which
argues for spending less on it rather than more.

Every level it produces is marked ``llm:tiebreak`` in ``cefr_basis``, which stays
greppable on purpose: it is the successor to slice 3's ``llm:extraction`` marker
(ADR-009), so "which levels are still not rules-grounded?" remains one command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ..llm import JSON_OBJECT, ChatClient, ModelResponse, Prompt, ShotPair, complete, strip_fences
from ..models import CEFR

STEP = "cefr_tiebreak"

# Enters the cache key, so bumping it re-runs tiebreaks without touching prompt text.
PROMPT_VERSION = "cefr@1"


class Tiebreak(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cefr: CEFR
    reason: str = ""


class TiebreakError(RuntimeError):
    """The tiebreak produced no usable level. Carries the response for inspection."""

    def __init__(self, message: str, *, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.reasoning_content = response.reasoning_content if response else None


# --- the prompt: fixed content first, variable content last (SPEC §10) ---

SYSTEM = """\
You assign a CEFR level (A1-C2) to a single German learning concept for a personal study \
wiki.

You are the LAST resort, not the first. Two rule-based signals ran before you and are \
given below:

- GRAMMAR: whether the concept names a structure from a fixed syllabus map. A match found \
in the node's TITLE says what the node is about; a match found only in its BODY may be a \
passing mention and is weaker evidence.
- LEXICAL: whether the headword appears in a levelled vocabulary list.

Use them. Where a signal is present and unambiguous, follow it. You are being asked \
because they conflict or are both absent, so say which way you resolved it.

Judge the concept a learner must already handle to study this node, not the easiest thing \
it mentions. A node explaining a B1 rule with A1 example sentences is B1.

Guidance for the common case where both signals are absent: level by the vocabulary and \
structures the node actually requires. Everyday concrete words and present-tense \
statements are A1-A2; abstract, institutional or formal-register material is B2-C1.

Answer with one level and one sentence. If you are unsure, prefer the LOWER level: an \
under-levelled node surfaces too early and gets corrected, while an over-levelled one \
sits above the target level and is never seen.\
"""

OUTPUT_SCHEMA = """\
Respond with a single JSON object, no prose and no code fences:

{
  "cefr": "A1|A2|B1|B2|C1|C2",
  "reason": "one sentence naming which signal you followed, or why you overrode both"
}\
"""

FEW_SHOT = [
    ShotPair(
        user=(
            "TITLE: Die Wochentage (The days of the week)\n"
            "TYPE: vocab\n"
            "GRAMMAR: no structure from the syllabus map matched\n"
            "LEXICAL: no wordlist installed\n\n"
            "BODY:\nMontag, Dienstag, Mittwoch … Alle Wochentage sind maskulin."
        ),
        assistant=json.dumps(
            {
                "cefr": "A1",
                "reason": "Both signals absent; everyday concrete vocabulary with no "
                "structure beyond noun gender.",
            },
            ensure_ascii=False,
        ),
    ),
    ShotPair(
        user=(
            "TITLE: Antrag auf Wohngeld stellen (Applying for housing benefit)\n"
            "TYPE: phrase\n"
            "GRAMMAR: akkusativ (A2) matched in the BODY only\n"
            "LEXICAL: no wordlist installed\n\n"
            "BODY:\nHiermit beantrage ich … Nominalstil, unpersönliche Konstruktionen, "
            "Behördendeutsch."
        ),
        assistant=json.dumps(
            {
                "cefr": "B2",
                "reason": "Overrode the body-only A2 hit: Akkusativ is mentioned in "
                "passing, while the concept itself is institutional register.",
            },
            ensure_ascii=False,
        ),
    ),
]


def build_prompt(
    *, title_de: str, title_en: str, node_type: str, body_md: str, grammar: str, lexical: str
) -> Prompt:
    """Assemble the tiebreak prompt with signals 1 and 2 already in context (SPEC §5)."""
    variable = (
        f"TITLE: {title_de} ({title_en})\n"
        f"TYPE: {node_type}\n"
        f"GRAMMAR: {grammar}\n"
        f"LEXICAL: {lexical}\n\n"
        f"BODY:\n{body_md.strip()}"
    )
    return Prompt(
        system=SYSTEM,
        output_schema=OUTPUT_SCHEMA,
        few_shot=FEW_SHOT,
        variable=variable,
        version=PROMPT_VERSION,
    )


def parse(response: ModelResponse) -> Tiebreak:
    """Validate a tiebreak response, or raise ``TiebreakError``.

    Truncation is checked before parsing, matching ``_extract`` and ``_adjudicate``:
    GLM spends completion tokens on reasoning first, so a tight cap returns well-formed
    empty content that would otherwise read as "no level".
    """
    if response.finish_reason == "length":
        raise TiebreakError(
            f"cefr tiebreak truncated at {response.usage.completion_tokens} completion "
            f"tokens (finish_reason=length); raise max_tokens for step {response.step!r} "
            "in config/models.yaml. Reasoning tokens count toward the cap.",
            response=response,
        )

    body = strip_fences(response.text)
    if not body:
        raise TiebreakError(
            f"cefr tiebreak returned no content (finish_reason={response.finish_reason})",
            response=response,
        )

    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TiebreakError(
            f"cefr tiebreak did not return valid JSON: {exc}; got {body[:200]!r}",
            response=response,
        ) from exc

    try:
        return Tiebreak.model_validate(data)
    except ValidationError as exc:
        # Catches an invented level like "B3" or "beginner": CEFR is a strict enum
        # (ADR-007 keeps it that way precisely so a typo fails loudly).
        raise TiebreakError(
            f"cefr tiebreak did not match the schema: {exc}", response=response
        ) from exc


def tiebreak(
    *,
    title_de: str,
    title_en: str,
    node_type: str,
    body_md: str,
    grammar: str,
    lexical: str,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[Tiebreak, ModelResponse]:
    """Ask the model to resolve a level. Writes nothing."""
    response = complete(
        STEP,
        build_prompt(
            title_de=title_de,
            title_en=title_en,
            node_type=node_type,
            body_md=body_md,
            grammar=grammar,
            lexical=lexical,
        ),
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
