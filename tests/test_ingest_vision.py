"""Vision OCR: the prompt, the truncation guard, and what counts as a failure."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.ingest import _vision
from german_wiki.ingest._vision import VisionError, transcribe
from german_wiki.llm import ModelResponse, Usage

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes" * 4


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "seite.png"
    path.write_bytes(PNG)
    return path


def _response(text: str, *, finish_reason: str = "stop", reasoning: str | None = None):
    return ModelResponse(
        text=text,
        step="vision",
        provider="zai",
        model="glm-4.5v",
        usage=Usage(prompt_tokens=800, completion_tokens=200, total_tokens=1000),
        cached=False,
        cost_usd=0.001,
        saved_usd=0.0,
        cache_key="k",
        finish_reason=finish_reason,
        reasoning_content=reasoning,
    )


# --- loading ---


def test_an_image_loads_with_its_media_type(image) -> None:
    part = _vision.load_image(image)
    assert part.media_type == "image/png"
    assert part.data == PNG


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.png", "image/png"), ("a.jpg", "image/jpeg"), ("a.JPEG", "image/jpeg"),
     ("a.webp", "image/webp")],
)
def test_supported_suffixes_map_to_media_types(name, expected) -> None:
    assert _vision.media_type_for(Path(name)) == expected


def test_a_non_image_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(VisionError, match="not a supported image"):
        _vision.media_type_for(tmp_path / "notes.txt")


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(VisionError, match="is empty"):
        _vision.load_image(empty)


def test_an_oversized_image_is_refused_not_resized(tmp_path: Path, monkeypatch) -> None:
    """Resizing would change the bytes -- and those bytes are what /raw records and what
    the cache keys on, so a silent resize would make provenance disagree with what was
    actually read."""
    monkeypatch.setattr(_vision, "MAX_IMAGE_BYTES", 32)
    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG" + b"x" * 200)

    with pytest.raises(VisionError, match="Downscale it"):
        _vision.load_image(big)


# --- the prompt ---


def test_the_instruction_comes_first_and_the_image_last(image) -> None:
    """SPEC §10: the fixed instruction is the cacheable prefix, the image is what varies."""
    messages = _vision.build_prompt(_vision.load_image(image)).to_messages()
    content = messages[-1]["content"]

    assert messages[0]["role"] == "system"
    assert content[0]["type"] == "text"
    assert content[-1]["type"] == "image_url"


def test_the_prompt_forbids_transliterating_umlauts() -> None:
    """The exact German OCR failure the checkpoint exists to catch (Bäckerei -> Backerei).

    ADR-012 made node *ids* ASCII deliberately; the transcription must not be, or the
    error is baked into /raw where §12.1 says the truth lives.
    """
    system = _vision.SYSTEM
    assert "ae/oe/ue/ss" in system
    assert "ß" in system
    assert "not translation" in system.lower() or "not translation," in system.lower()


def test_the_prompt_asks_for_a_marked_gap_rather_than_a_guess() -> None:
    """An invented word cannot be told apart from a real one after the fact."""
    assert "[unleserlich]" in _vision.SYSTEM


def test_no_json_is_requested(image) -> None:
    """The transcription IS the answer.

    Every other step in this project asks for JSON; wrapping a page of German prose in it
    would only invite escaping bugs and spend output tokens on syntax.
    """
    prompt = _vision.build_prompt(_vision.load_image(image))
    assert prompt.output_schema is None


# --- failure modes this module owns ---


def test_truncation_is_a_failure_not_a_short_page() -> None:
    """The dangerous one: a truncated transcription reads as a complete page.

    Unlike a truncated JSON verdict, nothing about the output looks wrong -- it is
    well-formed German that stops early, and it would flow into immutable /raw.
    """
    with pytest.raises(VisionError) as exc:
        _vision.parse(_response("Die Wechselpräp", finish_reason="length", reasoning="Looking…"))

    assert "max_tokens" in str(exc.value)
    assert "reads as a complete one" in str(exc.value)
    assert exc.value.reasoning_content == "Looking…"


def test_truncation_is_checked_before_the_text_is_used() -> None:
    """A truncated response still contains plausible-looking content."""
    with pytest.raises(VisionError, match="truncated"):
        _vision.parse(_response("Ein vollständiger Satz.", finish_reason="length"))


def test_an_empty_transcription_is_an_error_not_an_empty_source() -> None:
    """Ingesting an empty source looks like success while the content is lost."""
    with pytest.raises(VisionError, match="no text"):
        _vision.parse(_response("   "))


def test_a_normal_transcription_comes_back_stripped() -> None:
    assert _vision.parse(_response("\n  Guten Tag.\n ")) == "Guten Tag."


# --- the call ---


def test_transcribe_returns_text_and_the_response(
    image, models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text="Die Wechselpräpositionen stehen mit Akkusativ.")
    text, response = transcribe(
        image,
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert text.startswith("Die Wechselpräpositionen")
    assert response.step == "vision"
    assert client.call_count == 1


def test_re_transcribing_the_same_image_is_free(
    image, models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    """ADR-005 where it matters most: vision is PAID and runs per source, so re-processing
    a scan during tuning must not re-pay."""
    client = FakeChatClient(text=["Erste OCR", "Zweite OCR"])
    common = {
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
    }
    first, _ = transcribe(image, **common)
    second, response = transcribe(image, **common)

    assert client.call_count == 1
    assert second == first == "Erste OCR"
    assert response.cached is True


def test_a_different_image_is_not_served_from_the_first_ones_entry(
    tmp_path: Path, models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    """The collision that would silently corrupt every node derived from the second scan."""
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    one.write_bytes(b"\x89PNG-one")
    two.write_bytes(b"\x89PNG-two")

    client = FakeChatClient(text=["Seite eins", "Seite zwei"])
    common = {
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
    }
    assert transcribe(one, **common)[0] == "Seite eins"
    assert transcribe(two, **common)[0] == "Seite zwei"
    assert client.call_count == 2
