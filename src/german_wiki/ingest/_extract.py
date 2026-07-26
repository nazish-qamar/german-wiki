"""Candidate extraction: raw German text in, structured concepts out (SPEC §2).

The design choice that matters is in SPEC §2: **don't extract notes, extract
claims.** One textbook page yields several discrete concepts, each with its own
proposed title -- never one undifferentiated blob. The 5-8 cap (SPEC §2, ADR-006)
is the guardrail: a source trying to emit twenty is atomizing rather than
conceptualizing, which is what fragments the wiki.

The response schema reuses ``models.NodeType`` / ``models.CEFR``, so a model
answering ``"noun"`` or ``"B3"`` fails validation here rather than reaching a node
file. Nothing invented downstream: candidates carry only what the model returned.

``complete()`` is parse-free by design, so **this module owns the failure modes**.
The one that bites is ``finish_reason == "length"``: GLM-4.5 spends completion
tokens on internal reasoning before emitting any content, so a tight cap returns a
perfectly well-formed *empty* response. Treating that as "no candidates" would
silently drop a source. It is an error here, and the error carries the provider's
reasoning trace so the truncation is inspectable.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..llm import JSON_OBJECT, ChatClient, ModelResponse, Prompt, ShotPair, complete
from ..logutil import get_logger
from ..models import CEFR, NodeType

logger = get_logger(__name__)

# SPEC §2 / ADR-006. Exceeding it is a signal, not a fatal error -- see extract().
MAX_CANDIDATES = 8

# Enters the cache key, so bumping it re-runs extraction without touching a word
# of prompt text -- use it when THIS parser changes what it expects back.
PROMPT_VERSION = "extract@1"

STEP = "extraction"


with warnings.catch_warnings():
    # Same cosmetic clash models.py documents: `register` shadows the unused
    # BaseModel.register ABC hook. Scoped to the class definition rather than a
    # global filter, which pytest resets between tests.
    warnings.filterwarnings("ignore", message=r'Field name "register".*', category=UserWarning)

    class Candidate(BaseModel):
        """One extracted concept, before it becomes a Node."""

        model_config = ConfigDict(extra="forbid")

        title_de: str
        title_en: str
        type: NodeType
        cefr: CEFR
        cefr_basis: str | None = None
        register: list[str] = Field(default_factory=list)
        themes: list[str] = Field(default_factory=list)
        body_md: str
        confidence: float = 0.5


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(default_factory=list)


class ExtractionError(RuntimeError):
    """Extraction produced nothing usable.

    Carries the ``ModelResponse`` when there was one, so callers can surface the
    provider's reasoning trace -- the only way to see why a truncated call
    produced no content.
    """

    def __init__(self, message: str, *, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.reasoning_content = response.reasoning_content if response else None


# --- the prompt: fixed content first, variable content last (SPEC §10) ---

SYSTEM = """\
You extract discrete, learnable German concepts from a source text for a personal \
study wiki. You are given raw German material; you return structured candidates.

Extract CLAIMS, not notes. One page usually contains several distinct concepts. \
Never return one undifferentiated blob summarising the whole text.

What earns its own candidate:
- a word FAMILY sharing a stem (waschen / abwaschen / die Wäsche), not each inflected form
- a GRAMMAR RULE, not every example sentence of it
- a PATTERN, such as one intent expressed across registers (du-Ebene vs Sie-Ebene)
- a concept you would review as a unit, or link to from another note

What does NOT earn one:
- an individual inflected form
- a single word that merely fits an existing family
- a word that is just "a noun meaning X"
- an example sentence

Return between 1 and {max_candidates} candidates. Fewer, richer concepts are always \
better than many thin ones. If the text supports more than {max_candidates}, you are \
atomizing: group related items into families and patterns instead. If the text \
contains nothing learnable, return an empty list.

For each candidate:
- title_de / title_en: short, specific titles naming the concept
- type: one of grammar, vocab, phrase, pattern, culture
- cefr: a single level A1-C2, your best estimate
- cefr_basis: one short phrase saying what drove that level
- register: situational registers, lowercase German (alltag, büro, formell, \
umgangssprachlich, schriftlich); omit if unclear
- themes: situational themes, lowercase German (küche, büro, arzt, amt, café); \
omit if unclear
- body_md: a compact Markdown explanation. State the rule or meaning, then any \
exception. Add an "## Examples" section with example sentences drawn from the \
source, formatted as: - German sentence (English gloss)
- confidence: 0.0-1.0, your certainty this is a well-formed, distinct concept

Ground every claim in the source text. Do NOT add grammar rules, vocabulary or \
examples that are not present in or directly implied by it. If something is \
uncertain, lower the confidence and say so in body_md rather than inventing detail.\
"""

OUTPUT_SCHEMA = """\
Respond with a single JSON object, no prose and no code fences:

{
  "candidates": [
    {
      "title_de": "string",
      "title_en": "string",
      "type": "grammar|vocab|phrase|pattern|culture",
      "cefr": "A1|A2|B1|B2|C1|C2",
      "cefr_basis": "string",
      "register": ["string"],
      "themes": ["string"],
      "body_md": "string (Markdown)",
      "confidence": 0.0
    }
  ]
}\
"""

FEW_SHOT = [
    ShotPair(
        user=(
            "Wenn man höflich sein will, sagt man nicht 'Gib mir das Salz', sondern "
            "'Könntest du mir bitte das Salz geben?'. Im Büro benutzt man die Sie-Form: "
            "'Könnten Sie mir bitte helfen?'"
        ),
        assistant=json.dumps(
            {
                "candidates": [
                    {
                        "title_de": "Höfliche Bitte über Register",
                        "title_en": "Polite request across registers",
                        "type": "pattern",
                        "cefr": "A2",
                        "cefr_basis": "grammar:konjunktiv-ii-hoeflichkeit",
                        "register": ["alltag", "büro", "formell"],
                        "themes": ["büro"],
                        "body_md": (
                            "A request rises in politeness by moving from the imperative "
                            "to Konjunktiv II, and from **du** to **Sie**.\n\n"
                            "## Examples\n"
                            "- Gib mir das Salz. (Give me the salt.)\n"
                            "- Könntest du mir bitte das Salz geben? "
                            "(Could you please pass me the salt?)\n"
                            "- Könnten Sie mir bitte helfen? (Could you please help me?)"
                        ),
                        "confidence": 0.9,
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
]


def build_prompt(text: str) -> Prompt:
    """Assemble the extraction prompt; only ``variable`` changes per source."""
    return Prompt(
        system=SYSTEM.format(max_candidates=MAX_CANDIDATES),
        output_schema=OUTPUT_SCHEMA,
        few_shot=FEW_SHOT,
        variable=text,
        version=PROMPT_VERSION,
    )


# --- parsing ---


def _strip_fences(text: str) -> str:
    """Drop a ```json ... ``` wrapper. Providers emit them despite response_format."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse(response: ModelResponse) -> list[Candidate]:
    """Validate a model response into candidates, or raise ``ExtractionError``.

    Guard order matters: truncation is checked before parsing, because a truncated
    response can still contain parseable-looking content.
    """
    if response.finish_reason == "length":
        raise ExtractionError(
            f"extraction truncated at {response.usage.completion_tokens} completion tokens "
            f"(finish_reason=length); raise max_tokens for step {response.step!r} in "
            "config/models.yaml. Reasoning tokens count toward the cap.",
            response=response,
        )

    body = _strip_fences(response.text)
    if not body:
        raise ExtractionError(
            f"extraction returned no content (finish_reason={response.finish_reason})",
            response=response,
        )

    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"extraction did not return valid JSON: {exc}; got {body[:200]!r}",
            response=response,
        ) from exc

    try:
        extraction = Extraction.model_validate(data)
    except ValidationError as exc:
        raise ExtractionError(
            f"extraction did not match the candidate schema: {exc}", response=response
        ) from exc

    candidates = extraction.candidates
    if len(candidates) > MAX_CANDIDATES:
        # ADR-006: the overage is a signal the extractor is atomizing. Keep the
        # model's own first N rather than discarding a whole source's work.
        logger.warning(
            "extraction returned %d candidates, over the cap of %d; keeping the first %d "
            "(the extractor is atomizing rather than conceptualizing -- SPEC §3.4)",
            len(candidates),
            MAX_CANDIDATES,
            MAX_CANDIDATES,
        )
        candidates = candidates[:MAX_CANDIDATES]
    return candidates


def extract(
    text: str,
    *,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[list[Candidate], ModelResponse]:
    """Run the extraction step over ``text``. Returns candidates and the raw response."""
    response = complete(
        STEP,
        build_prompt(text),
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
