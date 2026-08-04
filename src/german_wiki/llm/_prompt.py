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

import base64
import hashlib

from pydantic import BaseModel, ConfigDict, Field


class ShotPair(BaseModel):
    """One few-shot exemplar. FIXED content -- never varies per input."""

    model_config = ConfigDict(extra="forbid")

    user: str
    assistant: str


class ImagePart(BaseModel):
    """One image attached to a prompt (slice 8, SPEC §2's image/PDF input).

    ``sha256`` is the image's **identity for caching**, taken over the raw bytes rather
    than the base64 rendering, so the same file always resolves to the same key however
    it was encoded. ``_cache`` substitutes this hash for the data URI in the key material:
    the cache needs a stable identifier, not the pixels (ADR-015).

    Deliberately *not* the source of truth for the image -- that is ``/raw`` (SPEC §1.2).
    This is a transport wrapper that exists for the duration of one call.
    """

    model_config = ConfigDict(extra="forbid")

    data: bytes
    media_type: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def to_data_uri(self) -> str:
        """The form the provider actually receives. Never enters the cache key."""
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"

    def to_content_part(self) -> dict[str, object]:
        return {"type": "image_url", "image_url": {"url": self.to_data_uri()}}


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
    # Slice 8. Attached to the FINAL user message, after ``variable`` -- the fixed
    # instruction stays the cacheable prefix and the image is the part that varies
    # (SPEC §10), which is the same ordering discipline text calls follow.
    images: list[ImagePart] = Field(default_factory=list)
    version: str | None = None

    def to_messages(self) -> list[dict]:
        """Assemble the OpenAI-compatible message list, variable content last.

        **With no images the output is byte-identical to before** -- plain string content
        throughout. That is deliberate: switching every text call to the multimodal
        content-array form would change ``messages``, hence every cache key, hence
        re-spend on material already paid for (ADR-005). Only a prompt that actually
        carries an image pays the different shape.
        """
        system = self.system
        if self.output_schema:
            system = f"{system}\n\n{self.output_schema}"

        messages: list[dict] = [{"role": "system", "content": system}]
        for shot in self.few_shot:
            messages.append({"role": "user", "content": shot.user})
            messages.append({"role": "assistant", "content": shot.assistant})

        if not self.images:
            messages.append({"role": "user", "content": self.variable})
            return messages

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.variable},
                    *(image.to_content_part() for image in self.images),
                ],
            }
        )
        return messages
