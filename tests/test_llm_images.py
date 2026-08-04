"""Images in prompts, and the cache identity that keeps two scans apart (ADR-015)."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.llm import ImagePart, Prompt, complete
from german_wiki.llm._cache import cache_key, key_material

PNG_A = b"\x89PNG\r\n\x1a\n-image-A"
PNG_B = b"\x89PNG\r\n\x1a\n-image-B"


def _image(data: bytes = PNG_A) -> ImagePart:
    return ImagePart(data=data, media_type="image/png")


def _prompt(images: list[ImagePart] | None = None) -> Prompt:
    return Prompt(
        system="Transcribe the German text in this image.",
        variable="",
        images=images or [],
        version="vision@1",
    )


def _key(prompt: Prompt) -> str:
    return cache_key(
        key_material(
            provider="zai",
            model="glm-4.5v",
            messages=prompt.to_messages(),
            temperature=0.0,
            max_tokens=8192,
        )
    )


# --- the message shape ---


def test_a_prompt_without_images_is_byte_identical_to_before() -> None:
    """No existing cache entry may be invalidated by this feature.

    Switching every text call to the multimodal content-array form would change
    `messages`, hence every key, hence re-spend on material already paid for (ADR-005).
    """
    messages = _prompt().to_messages()
    assert all(isinstance(m["content"], str) for m in messages)
    assert messages[-1] == {"role": "user", "content": ""}


def test_an_image_rides_on_the_final_user_message_after_the_text() -> None:
    """SPEC §10: fixed instruction first as the cacheable prefix, image last."""
    messages = _prompt([_image()]).to_messages()
    content = messages[-1]["content"]

    assert messages[0]["role"] == "system"
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[-1]["type"] == "image_url"


def test_the_provider_receives_a_real_data_uri() -> None:
    part = _image().to_content_part()
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == PNG_A


def test_several_images_keep_their_order() -> None:
    messages = _prompt([_image(PNG_A), _image(PNG_B)]).to_messages()
    urls = [p["image_url"]["url"] for p in messages[-1]["content"] if p["type"] == "image_url"]
    assert len(urls) == 2
    assert base64.b64decode(urls[0].split(",", 1)[1]) == PNG_A


# --- cache identity: the failure this exists to prevent ---


def test_two_different_images_never_share_a_cache_entry() -> None:
    """The catastrophic case, asserted directly.

    Same instruction, different scan. If the key covered only the text these would
    collide and the cache would serve image A's transcription for image B -- silently,
    and looking exactly like a successful OCR.
    """
    assert _key(_prompt([_image(PNG_A)])) != _key(_prompt([_image(PNG_B)]))


def test_the_same_image_resolves_to_the_same_key() -> None:
    """The other half: re-running a scan during tuning must be free, not re-paid."""
    assert _key(_prompt([_image(PNG_A)])) == _key(_prompt([_image(PNG_A)]))


def test_an_image_changes_the_key_relative_to_text_alone() -> None:
    assert _key(_prompt()) != _key(_prompt([_image()]))


def test_the_key_material_carries_the_hash_not_the_base64() -> None:
    """The whole point of redacting: entries stay kilobytes, and `/raw` stays the only
    copy of the image (SPEC §1.2)."""
    material = key_material(
        provider="zai",
        model="glm-4.5v",
        messages=_prompt([_image()]).to_messages(),
        temperature=0.0,
        max_tokens=8192,
    )
    blob = json.dumps(material)
    digest = hashlib.sha256(PNG_A).hexdigest()

    assert digest in blob
    assert base64.b64encode(PNG_A).decode() not in blob
    assert "data:image/png" not in blob


def test_the_hash_is_over_bytes_not_the_base64_rendering() -> None:
    """So the same file keys identically however it was encoded on the way in."""
    material = key_material(
        provider="zai",
        model="glm-4.5v",
        messages=_prompt([_image()]).to_messages(),
        temperature=0.0,
        max_tokens=8192,
    )
    part = material["messages"][-1]["content"][-1]
    assert part == {"type": "image_url", "sha256": hashlib.sha256(PNG_A).hexdigest()}


def test_text_only_key_material_is_untouched_by_redaction() -> None:
    material = key_material(
        provider="zai",
        model="free",
        messages=[{"role": "user", "content": "Guten Tag"}],
        temperature=0.0,
        max_tokens=100,
    )
    assert material["messages"] == [{"role": "user", "content": "Guten Tag"}]


# --- end to end through complete() ---


def test_a_repeated_image_call_hits_the_cache(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    """ADR-005 for vision. This matters more than anywhere else: vision is PAID and runs
    per source, so re-processing a textbook scan during tuning must cost nothing."""
    client = FakeChatClient(text=["Erste Antwort", "Zweite Antwort"])
    common = {
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
    }
    first = complete("vision", _prompt([_image()]), **common)
    second = complete("vision", _prompt([_image()]), **common)

    assert client.call_count == 1
    assert second.cached is True
    assert second.text == first.text == "Erste Antwort"


def test_a_different_image_misses_and_calls_again(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text=["Antwort A", "Antwort B"])
    common = {
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
    }
    complete("vision", _prompt([_image(PNG_A)]), **common)
    second = complete("vision", _prompt([_image(PNG_B)]), **common)

    assert client.call_count == 2
    assert second.cached is False
    assert second.text == "Antwort B"


def test_the_stored_entry_holds_no_image_data(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    complete(
        "vision",
        _prompt([_image()]),
        client=FakeChatClient(text="Antwort"),
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    entry = next((tmp_cache / "llm").rglob("*.json"))
    blob = entry.read_text(encoding="utf-8")

    assert hashlib.sha256(PNG_A).hexdigest() in blob
    assert base64.b64encode(PNG_A).decode() not in blob
    assert len(blob) < 4000, "an entry holding base64 would be far larger"


@pytest.mark.parametrize("media_type", ["image/png", "image/jpeg", "image/webp"])
def test_media_type_reaches_the_data_uri(media_type) -> None:
    part = ImagePart(data=PNG_A, media_type=media_type).to_content_part()
    assert part["image_url"]["url"].startswith(f"data:{media_type};base64,")
