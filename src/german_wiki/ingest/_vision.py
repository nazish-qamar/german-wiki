"""Image → German text (SPEC §2's OCR step, §9's vision row, §11 slice 8).

This is the **only genuinely new part** of slice 8. Once there is text, it goes through
``extract → queue → adjudicate → review`` exactly as a ``.txt`` file does; nothing
downstream knows or cares that a camera was involved.

It is an ordinary ``complete()`` call, which is what makes that true. Routing, the
content-hash cache, the cost ledger and the ``finish_reason`` guard all come from slice 2
rather than being re-implemented here -- the image reaches the provider because ``Prompt``
grew an ``images`` field, not because vision got its own path to the API.

**Truncation is the dangerous failure, and it is why the guard matters more here than
anywhere.** A truncated JSON verdict is obviously broken; a truncated *transcription* is
not. It is a well-formed page of German that simply stops early, and it would flow
silently into ``/raw`` -- which SPEC §1.2 makes immutable and §12.1 makes the anchor you
re-verify drifted nodes against. Losing the second half of a page there is unrecoverable
without re-OCRing the image. So ``finish_reason == "length"`` is a failed call, per the
lesson slice 2 learned from GLM's reasoning tokens.

**No JSON.** Unlike every other step in this project, the vision response is plain text --
asking a model to wrap a page of German prose in JSON invites escaping bugs and wastes
output tokens on syntax. The transcription *is* the answer.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from ..llm import ChatClient, ImagePart, ModelResponse, Prompt, complete
from ..logutil import get_logger

logger = get_logger(__name__)

STEP = "vision"

# Enters the cache key, so bumping it re-OCRs without touching prompt text (ADR-008).
PROMPT_VERSION = "vision@1"

# What the provider will accept as an inline image.
SUFFIX_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

IMAGE_SUFFIXES = frozenset(SUFFIX_MEDIA_TYPES)

# Refused rather than silently downscaled. Resizing would need Pillow and would change the
# bytes -- and those bytes are what `/raw` records and what the cache key hashes, so a
# silent resize would make the stored provenance disagree with what was actually read.
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class VisionError(RuntimeError):
    """OCR produced nothing usable.

    Carries the ``ModelResponse`` when there was one so a truncation is inspectable,
    mirroring ``ExtractionError`` and ``AdjudicationError``.
    """

    def __init__(self, message: str, *, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.response = response
        self.reasoning_content = response.reasoning_content if response else None


SYSTEM = """\
You transcribe German text from an image for a personal study wiki.

Return the text exactly as it appears. This is transcription, not translation, \
summarisation or correction:

- Keep German spelling exactly: ä, ö, ü, ß are distinct letters, never ae/oe/ue/ss. \
Capitalisation carries meaning in German - keep it.
- Keep the reading order and the line and paragraph breaks of the page.
- Keep tables as Markdown tables when the layout is tabular, and keep headings as headings.
- Transcribe every language present. If part of the page is English (a gloss, a \
translation column, an instruction), transcribe that too, in place.
- Do NOT fix perceived spelling or grammar mistakes. If the page says something odd, the \
page is what matters - this text becomes the permanent record of what was on it.
- Do NOT add commentary, headers, or notes of your own. No "Here is the text:".

If part of the image is genuinely illegible, write [unleserlich] at that point rather \
than guessing at the word. A marked gap can be checked against the original; an invented \
word cannot be distinguished from a real one.

If the image contains no text at all, return nothing.\
"""

INSTRUCTION = "Transcribe the German text in this image."


def media_type_for(path: Path) -> str:
    """The image's media type, from its suffix and then the system's guess."""
    suffix = path.suffix.lower()
    if suffix in SUFFIX_MEDIA_TYPES:
        return SUFFIX_MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed and guessed.startswith("image/"):
        return guessed
    raise VisionError(
        f"{path.name} is not a supported image ({', '.join(sorted(IMAGE_SUFFIXES))})"
    )


def is_image(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def load_image(path: Path | str) -> ImagePart:
    """Read an image for transport. Refuses oversized files rather than resizing."""
    path = Path(path)
    data = path.read_bytes()
    if not data:
        raise VisionError(f"{path.name} is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise VisionError(
            f"{path.name} is {len(data) / 1e6:.1f} MB, over the {MAX_IMAGE_BYTES / 1e6:.0f} MB "
            "limit. Downscale it before ingesting -- this is not done automatically because "
            "the bytes ingested are the bytes /raw records and the cache keys on."
        )
    return ImagePart(data=data, media_type=media_type_for(path))


def build_prompt(image: ImagePart) -> Prompt:
    """Fixed instruction first as the cacheable prefix, image last (SPEC §10)."""
    return Prompt(
        system=SYSTEM,
        variable=INSTRUCTION,
        images=[image],
        version=PROMPT_VERSION,
    )


def parse(response: ModelResponse) -> str:
    """The transcription, or raise. Truncation is checked before anything else."""
    if response.finish_reason == "length":
        raise VisionError(
            f"transcription truncated at {response.usage.completion_tokens} completion "
            f"tokens (finish_reason=length); raise max_tokens for step {response.step!r} "
            "in config/models.yaml. A truncated page reads as a complete one, which is "
            "why this is an error rather than a short result.",
            response=response,
        )

    text = response.text.strip()
    if not text:
        raise VisionError(
            f"transcription returned no text (finish_reason={response.finish_reason}). "
            "If the image genuinely has no German on it, there is nothing to ingest.",
            response=response,
        )
    return text


def transcribe(
    path: Path | str,
    *,
    client: ChatClient | None = None,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: dict[str, str] | None = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> tuple[str, ModelResponse]:
    """OCR one image. Writes nothing -- the caller decides what to do with the text."""
    image = load_image(path)
    response = complete(
        STEP,
        build_prompt(image),
        client=client,
        settings_path=settings_path,
        cache_dir=cache_dir,
        usage_log=usage_log,
        env=env,
        use_cache=use_cache,
        refresh=refresh,
    )
    text = parse(response)
    logger.info(
        "transcribed %s: %d chars%s", Path(path).name, len(text), " (cached)" if response.cached else ""
    )
    return text, response
