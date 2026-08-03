"""Undoing what the provider did to the response, before anyone interprets it.

This is a narrow, deliberate exception to the rule in ``_client``: **no *domain* parsing
lives in this package.** What a response *means* — candidates, a verdict, a level, a
transparency rating — belongs to the slice that asked the question, and each of those owns
its own schema and failure modes.

A markdown code fence is a different thing. It is not content; it is an artifact the
*provider* added around the content, despite being asked for ``response_format:
json_object``. Stripping it is the same category as the other provider quirks this layer
already absorbs — capturing ``reasoning_content``, surfacing ``finish_reason`` — and every
caller that asks for JSON hits it identically.

**Extracted at the rule of three** (ADR-010's policy, applied). ``ingest/_extract``,
``merge/_adjudicate`` and ``morph/_transparency`` each carried a byte-identical copy;
verified identical at the AST level before collapsing, so this is a pure move with no
behaviour reconciled. The docstrings differed, which is commentary, not behaviour.
"""

from __future__ import annotations

FENCE = "```"


def strip_fences(text: str) -> str:
    """Drop a ``` ```json … ``` ``` wrapper. Providers emit them despite response_format.

    Tolerant by design, because the failure it guards is cosmetic and the alternative is a
    JSON error that reads like a model failure:

    - no fence, or an unterminated one, returns the text unchanged apart from stripping;
    - the opening line goes whatever language tag it carries (``json``, ``JSON``, none);
    - an empty string, or a bare fence with nothing in it, yields ``""`` rather than
      raising — callers already treat empty content as their own error, with a message
      that names the step and the finish_reason.
    """
    stripped = text.strip()
    if not stripped.startswith(FENCE):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip().startswith(FENCE):
        lines = lines[:-1]
    return "\n".join(lines).strip()
