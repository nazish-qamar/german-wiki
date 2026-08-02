"""Transparency: is this family a learning scaffold or a coincidence? (SPEC §7.4)

The one judgment in this slice that rules cannot make. Segmentation is mechanical --
a closed prefix inventory plus corpus evidence -- but SPEC §7.4 is about semantics::

    `verstehen` has nothing to do with `stehen` + directional `ver-`. `bekommen` ≠
    `be-` + `kommen` in any useful sense. Meanings have drifted centuries past
    transparency.

    The family link is a **learning scaffold, not an etymological claim.**

So the model answers exactly one question -- ``high`` / ``drifted`` / ``opaque`` -- and
nothing else. It does not decide whether a split happened; ``_segment`` already did, from
evidence. Keeping the two apart is what stops a model's fluency being mistaken for
morphological fact, and it is the same division slice 6 drew between the CEFR grammar map
(rules) and the tiebreak (judgment).

It runs on the **free** ``glm-4.5-flash`` step, never the paid ``glm-4.6``: it is a
per-candidate call that fires as verbs arrive, and a wrong answer marks a grid cell rather
than corrupting a node body. Two things backstop it -- the review gate, because
``family_transparency`` is written to ``/nodes`` (ADR-003), and §7.4's own rule that the
node holds the truth while the grid only predicts.

``complete()`` is parse-free by design, so this module owns the failure modes, including
the ``finish_reason == "length"`` guard that has bitten in every slice since 3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from ..llm import JSON_OBJECT, ChatClient, ModelResponse, Prompt, ShotPair, complete
from ..logutil import get_logger
from ..models import FamilyTransparency

logger = get_logger(__name__)

STEP = "transparency"

# Enters the cache key, so bumping it re-judges without touching prompt text (ADR-008).
PROMPT_VERSION = "transparency@1"

# The value the grid trusts. Anything else marks predicted cells as irregular (see
# ``_grid._UNTRUSTWORTHY_FAMILIES``), which is §7.4's "useful watch-out signal".
TRUSTED: FamilyTransparency = "high"


class Transparency(BaseModel):
    """One verdict about one family. Carries no writes."""

    model_config = ConfigDict(extra="forbid")

    transparency: FamilyTransparency
    reason: str = ""
    confidence: float = 0.5


class TransparencyError(RuntimeError):
    """No usable verdict. Carries the response so a truncation is inspectable."""

    def __init__(self, message: str, *, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.reasoning_content = response.reasoning_content if response else None


SYSTEM = """\
You judge whether a German word family is a useful learning scaffold or a historical \
coincidence.

You are given a root, a prefix, and the resulting verb. The split itself is already \
established - do not question it. Answer only how far the meaning still follows from its \
parts:

high - the prefix's usual sense plus the root's sense predicts the verb's meaning. A \
learner who knows both can work it out. (waschen + ab- -> abwaschen, wash off/up.)

drifted - a connection is still visible, but the meaning has specialised enough that a \
learner would guess wrong at least some of the time. Worth teaching as related, with a \
warning.

opaque - the modern meanings have no useful relationship. The words merely look related. \
Teaching them as a family would actively mislead. (stehen vs verstehen; kommen vs \
bekommen.)

Judge the meanings words have TODAY, for a learner. Etymology is irrelevant: two words \
can share a genuine origin and still be opaque, which is exactly the case this rating \
exists to catch. When unsure between two ratings, choose the more cautious one - a family \
wrongly marked high teaches a false pattern, while one wrongly marked drifted only costs \
a warning the learner can ignore.\
"""

OUTPUT_SCHEMA = """\
Respond with a single JSON object, no prose and no code fences:

{
  "transparency": "high|drifted|opaque",
  "reason": "one sentence naming what the prefix does to the meaning, or why it no longer does",
  "confidence": 0.0
}\
"""

FEW_SHOT = [
    ShotPair(
        user="root: waschen (to wash)\nprefix: ab- (off, away)\nverb: abwaschen (to wash up)",
        assistant=json.dumps(
            {
                "transparency": "high",
                "reason": "ab- contributes its usual 'off/away' sense to washing.",
                "confidence": 0.9,
            },
            ensure_ascii=False,
        ),
    ),
    ShotPair(
        user="root: stehen (to stand)\nprefix: ver- (inseparable)\nverb: verstehen (to understand)",
        assistant=json.dumps(
            {
                "transparency": "opaque",
                "reason": "Understanding has no live relationship to standing; the pair only looks related.",
                "confidence": 0.95,
            },
            ensure_ascii=False,
        ),
    ),
    ShotPair(
        user="root: stellen (to place)\nprefix: an- (toward, on)\nverb: anstellen (to hire; to queue)",
        assistant=json.dumps(
            {
                "transparency": "drifted",
                "reason": "Placing is still faintly visible in queuing, but 'to hire' would not be guessed.",
                "confidence": 0.8,
            },
            ensure_ascii=False,
        ),
    ),
]


def build_prompt(*, root: str, prefix: str, word: str, gloss: str = "") -> Prompt:
    """Fixed content first, the family last (SPEC §10)."""
    variable = f"root: {root}\nprefix: {prefix}-\nverb: {word}"
    if gloss.strip():
        variable = f"{variable}\nknown meaning: {gloss.strip()}"
    return Prompt(
        system=SYSTEM,
        output_schema=OUTPUT_SCHEMA,
        few_shot=FEW_SHOT,
        variable=variable,
        version=PROMPT_VERSION,
    )


def _strip_fences(text: str) -> str:
    """Drop a ```json ... ``` wrapper. Third occurrence of this helper, and the point at
    which the rule of three says to extract it -- deferred deliberately until slice 7's
    own churn settles, and noted here so it is not forgotten."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse(response: ModelResponse) -> Transparency:
    """Validate a verdict, or raise. Truncation is checked before parsing, as ever."""
    if response.finish_reason == "length":
        raise TransparencyError(
            f"transparency judgment truncated at {response.usage.completion_tokens} "
            f"completion tokens (finish_reason=length); raise max_tokens for step "
            f"{response.step!r} in config/models.yaml.",
            response=response,
        )

    body = _strip_fences(response.text)
    if not body:
        raise TransparencyError(
            f"transparency returned no content (finish_reason={response.finish_reason})",
            response=response,
        )

    try:
        data: Any = json.loads(body)
    except json.JSONDecodeError as exc:
        raise TransparencyError(
            f"transparency did not return valid JSON: {exc}; got {body[:200]!r}",
            response=response,
        ) from exc

    try:
        return Transparency.model_validate(data)
    except ValidationError as exc:
        raise TransparencyError(
            f"transparency did not match the schema: {exc}", response=response
        ) from exc


def judge(
    *,
    root: str,
    prefix: str,
    word: str,
    gloss: str = "",
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[Transparency, ModelResponse]:
    """Rate one family. Writes nothing; the verdict reaches ``/nodes`` only via review."""
    response = complete(
        STEP,
        build_prompt(root=root, prefix=prefix, word=word, gloss=gloss),
        client=client,
        response_format=JSON_OBJECT,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=env,
        use_cache=use_cache,
        refresh=refresh,
    )
    verdict = parse(response)
    if verdict.transparency != TRUSTED:
        logger.info(
            "family %s rated %s: %s", word, verdict.transparency, verdict.reason
        )
    return verdict, response
