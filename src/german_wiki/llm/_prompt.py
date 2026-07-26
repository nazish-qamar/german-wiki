"""Prompt assembly: fixed content first, variable content last (SPEC §10).

Providers cache a request's leading tokens. Putting the invariant part of a
prompt -- system instructions, output schema, few-shot exemplars -- ahead of the
per-source text means every call in a run shares a cacheable prefix, which is
the secondary cost lever after the content cache itself (ADR-005).

That ordering is enforced by the *shape* of this API rather than by convention:
``Prompt`` has exactly one variable field and ``to_messages()`` always emits it
last, so a call site cannot get the order wrong by forgetting.

This module holds no prompt *content*. The extraction system prompt and its
few-shot exemplars belong to slice 3.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ShotPair(BaseModel):
    """One few-shot exemplar. FIXED content -- never varies per input."""

    model_config = ConfigDict(extra="forbid")

    user: str
    assistant: str


class Prompt(BaseModel):
    """A single model request, split into its cacheable prefix and its tail.

    Field order here *is* the wire order. ``system``, ``output_schema`` and
    ``few_shot`` form the invariant prefix a provider can cache; ``variable`` is
    the only part that changes per source, and it is always the final message.

    ``output_schema`` is named to avoid shadowing ``BaseModel.schema``. It is
    folded into the system message rather than sent as a second system message:
    it is exactly as invariant, it is one boundary fewer, and every
    OpenAI-compatible provider accepts a single leading system message where not
    all handle two.

    ``version`` does not reach the provider. It enters the cache key, so a
    downstream parser change can invalidate cached responses without editing a
    word of prompt text.
    """

    model_config = ConfigDict(extra="forbid")

    system: str
    output_schema: str | None = None
    few_shot: list[ShotPair] = Field(default_factory=list)
    variable: str
    version: str | None = None

    def to_messages(self) -> list[dict[str, str]]:
        """Assemble the OpenAI-compatible message list, variable content last."""
        system = self.system
        if self.output_schema:
            system = f"{system}\n\n{self.output_schema}"

        messages = [{"role": "system", "content": system}]
        for shot in self.few_shot:
            messages.append({"role": "user", "content": shot.user})
            messages.append({"role": "assistant", "content": shot.assistant})
        messages.append({"role": "user", "content": self.variable})
        return messages
