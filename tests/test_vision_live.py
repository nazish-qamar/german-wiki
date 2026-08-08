"""Optional live OCR against a real vision model. Skipped unless GW_LIVE_TESTS=1.

The rest of the vision suite is offline and free by construction. This file is the only
one that opens a socket, and it exists to answer the question the offline tests cannot:
**does the configured model actually read German?**

    GW_LIVE_TESTS=1 pytest -m live -k vision

That question is not academic. ADR-015 routes vision to the free ``glm-4.6v-flash``
following the same tune-on-free pattern as extraction and adjudication, and ADR-011 §5
recorded what happened the last time a free model's quality was assumed rather than
measured: flash got both real gray-zone pairs wrong. The upgrade path (``glm-4.6v`` at
$0.30/$0.90) is priced and one line away *if* these assertions start failing.

**Umlauts and ß are the acceptance criterion**, not overall plausibility. German OCR
fails in ways that survive a glance -- ``Bäckerei`` -> ``Backerei``, ``groß`` -> ``gross``
-- and the transcription is frozen into ``/raw``, which SPEC §1.2 makes immutable and
§12.1 makes the anchor a drifted node is checked against. A transcription that reads
fluently but has flattened its diacritics is exactly the failure the OCR checkpoint
exists to catch, so it is what this test looks at.

Writes nothing: ``transcribe`` returns text and has no write path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from german_wiki.ingest import transcribe
from german_wiki.llm import resolve_step

LIVE = os.environ.get("GW_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE, reason="set GW_LIVE_TESTS=1 to run live model API tests"),
]

# A page rendered at test time rather than committed as a binary blob, so what is being
# asked of the model is readable in this file. Every umlaut and the ß appear on purpose.
GERMAN_LINES = [
    "Die Bäckerei öffnet um sechs Uhr.",
    "Könnten Sie mir bitte helfen?",
    "Die Straße ist sehr groß.",
    "Ich möchte gern über Österreich sprechen.",
]

# The characters that must survive. Each is a distinct failure mode:
#   ä/ö/ü  -> flattened to a/o/u, or transliterated to ae/oe/ue
#   ß      -> turned into B, or expanded to ss
DIACRITICS = ("ä", "ö", "ü", "ß")


def _render_png(lines: list[str]) -> bytes:
    """A minimal PNG of the text, drawn with Pillow if it is available."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow is not a project dependency
        pytest.skip(
            "rendering the fixture needs Pillow, which this project does not depend on. "
            "Install it in the venv to run this test, or point GW_VISION_FIXTURE at a "
            "photo of a German page."
        )

    image = Image.new("RGB", (900, 320), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((30, 30 + index * 60), line, fill="black")

    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def german_page(tmp_path: Path) -> Path:
    """The fixture image: a real photo if you point at one, else a rendered page."""
    supplied = os.environ.get("GW_VISION_FIXTURE")
    if supplied:
        path = Path(supplied)
        if not path.is_file():
            pytest.skip(f"GW_VISION_FIXTURE does not exist: {path}")
        return path

    path = tmp_path / "seite.png"
    path.write_bytes(_render_png(GERMAN_LINES))
    return path


def test_live_the_configured_model_reads_german(german_page, tmp_path: Path) -> None:
    text, response = transcribe(
        german_page, cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl"
    )

    assert text.strip(), "the model returned nothing"
    assert response.finish_reason != "length", "raise max_tokens for the vision step"
    # Not an exact-match assertion: OCR of a rendered page is not deterministic, and
    # demanding byte equality would make this a flaky test rather than a useful one.
    assert "Bäckerei" in text or "äckerei" in text, f"got: {text[:200]!r}"


def test_live_diacritics_survive(german_page, tmp_path: Path) -> None:
    """The acceptance criterion for the free model (ADR-015).

    If this fails, the answer is the recorded upgrade path -- switch the vision step to
    ``glm-4.6v`` -- not to loosen the assertion. A transcription that flattens umlauts is
    wrong in a way that reads as correct, and it is about to be frozen into /raw.
    """
    text, _ = transcribe(
        german_page, cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl"
    )

    missing = [ch for ch in DIACRITICS if ch not in text]
    assert not missing, (
        f"{resolve_step('vision').model} dropped {missing} from the transcription. "
        f"Got: {text[:300]!r}. ADR-015's upgrade path is glm-4.6v ($0.30/$0.90)."
    )


def test_live_it_transcribes_rather_than_translates(german_page, tmp_path: Path) -> None:
    """The prompt forbids translation; a model that helpfully renders English instead
    would destroy the source before it was ever stored."""
    text, _ = transcribe(
        german_page, cache_dir=tmp_path / "cache", usage_log=tmp_path / "usage.jsonl"
    )
    lowered = text.lower()
    assert "bakery" not in lowered
    assert "could you please help" not in lowered


def test_live_re_transcribing_is_free(german_page, tmp_path: Path) -> None:
    """ADR-005 against a real provider. Vision is the step where this matters most --
    it runs per source, and re-processing a scan during tuning must not re-pay."""
    cache = tmp_path / "cache"
    log = tmp_path / "usage.jsonl"
    first_text, first = transcribe(german_page, cache_dir=cache, usage_log=log)
    second_text, second = transcribe(german_page, cache_dir=cache, usage_log=log)

    assert first.cached is False
    assert second.cached is True
    assert second_text == first_text
