"""gw ingest for images and PDFs: the checkpoint, and what /raw holds when it fails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from german_wiki.cli import app
from german_wiki.llm import ModelResponse, Usage

runner = CliRunner()
WIDE = {"COLUMNS": "200"}

PNG = b"\x89PNG\r\n\x1a\n" + b"seite-eins" * 8

TRANSCRIPTION = (
    "Die Bäckerei öffnet um sechs Uhr.\n"
    "Man sagt höflich: Könnten Sie mir bitte helfen?\n"
)

CANDIDATES = {
    "candidates": [
        {
            "title_de": "Höfliche Bitte",
            "title_en": "Polite request",
            "type": "phrase",
            "cefr": "A2",
            "cefr_basis": "grammar:konjunktiv-ii",
            "register": ["formell"],
            "themes": ["alltag"],
            "body_md": "Könnten Sie mir bitte helfen?",
            "confidence": 0.9,
        }
    ]
}

CONFIG = {
    "version": 1,
    "providers": {
        "zai": {
            "kind": "api",
            "base_url": "https://example.invalid/v4",
            "api_key_env": "ZAI_API_KEY",
        }
    },
    "pricing": {"zai": {"free-model": {"input": 0.0, "cached_input": 0.0, "output": 0.0}}},
    "defaults": {"provider": "zai", "model": "free-model", "temperature": 0.0, "max_tokens": 4096},
    "steps": {"extraction": {"status": "active"}, "vision": {"status": "active"}},
}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


@pytest.fixture
def world(tmp_path: Path, tmp_vocab: Path, monkeypatch):
    settings = tmp_path / "models.yaml"
    settings.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("GW_MODELS_CONFIG", str(settings))
    monkeypatch.setenv("ZAI_API_KEY", "test-key")

    image = tmp_path / "seite.png"
    image.write_bytes(PNG)

    nodes = tmp_path / "nodes"
    nodes.mkdir()
    return {
        "root": tmp_path,
        "image": image,
        "nodes": nodes,
        "raw": tmp_path / "raw",
        "queue": tmp_path / "queue",
        "vocab": tmp_vocab,
        "cache": tmp_path / "cache",
    }


def _flags(w) -> list[str]:
    return [
        "--raw-dir", str(w["raw"]),
        "--queue-dir", str(w["queue"]),
        "--nodes-dir", str(w["nodes"]),
        "--vocab-dir", str(w["vocab"]),
        "--cache-dir", str(w["cache"]),
    ]


def _response(text: str, *, step: str, finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(
        text=text,
        step=step,
        provider="zai",
        model="free-model",
        usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        cached=False,
        cost_usd=0.0,
        saved_usd=0.0,
        cache_key="k",
        finish_reason=finish_reason,
    )


def _stub(monkeypatch, *, ocr: str = TRANSCRIPTION, ocr_finish: str = "stop") -> dict:
    """Canned vision + extraction responses, with a call counter per step."""
    seen: dict[str, int] = {"vision": 0, "extraction": 0}

    def _vision_complete(step, prompt, **kwargs):
        seen["vision"] += 1
        return _response(ocr, step=step, finish_reason=ocr_finish)

    def _extract_complete(step, prompt, **kwargs):
        seen["extraction"] += 1
        return _response(json.dumps(CANDIDATES, ensure_ascii=False), step=step)

    monkeypatch.setattr("german_wiki.ingest._vision.complete", _vision_complete)
    monkeypatch.setattr("german_wiki.ingest._extract.complete", _extract_complete)
    return seen


def _raw_files(w) -> list[str]:
    return sorted(p.name for p in w["raw"].iterdir()) if w["raw"].exists() else []


# --- the happy path ---


def test_an_image_reaches_the_queue_through_the_ordinary_pipeline(world, monkeypatch) -> None:
    """Vision is a new *input*, not a new pipeline — the tail is slice 3's, unchanged."""
    calls = _stub(monkeypatch)
    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE
    )

    assert result.exit_code == 0, _combined(result)
    assert calls == {"vision": 1, "extraction": 1}
    assert list(world["queue"].glob("*/*.md")), "a candidate should be staged"
    # ADR-003 holds for this input type too.
    assert list(world["nodes"].glob("*.md")) == []


def test_all_three_raw_artifacts_land(world, monkeypatch) -> None:
    _stub(monkeypatch)
    runner.invoke(app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE)

    names = _raw_files(world)
    assert any(n.endswith(".png") for n in names), "the image itself"
    assert any(n.endswith(".txt") for n in names), "the accepted transcription"
    assert any(n.endswith(".json") for n in names), "the sidecar"


def test_the_sidecar_records_where_the_text_came_from(world, monkeypatch) -> None:
    _stub(monkeypatch)
    runner.invoke(app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE)

    sidecar = json.loads(next(world["raw"].glob("*.json")).read_text(encoding="utf-8"))
    assert sidecar["artifact_suffix"] == ".png"
    assert sidecar["ocr_model"] == "free-model"
    assert sidecar["ocr_edited"] is False
    assert "ocr_sha256" in sidecar


def test_umlauts_survive_the_whole_round_trip(world, monkeypatch) -> None:
    """The thing the checkpoint exists to protect. ADR-012 made node *ids* ASCII; the
    transcription must not be, or `Bäckerei` is lost at the point of no return."""
    _stub(monkeypatch)
    runner.invoke(app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE)

    stored = next(world["raw"].glob("*.txt")).read_text(encoding="utf-8")
    assert "Bäckerei" in stored
    assert "öffnet" in stored


# --- the checkpoint (ADR-015) ---


def test_rejecting_leaves_no_transcription_but_keeps_the_image(world, monkeypatch) -> None:
    """The load-bearing one.

    /raw is immutable AND is §12.1's re-verification anchor, so a bad transcription
    frozen there corrupts the reference you would use to detect a bad node. Declining
    must therefore leave NO .txt — while keeping the image, so a retry is one command
    and costs nothing (the OCR call is cached).
    """
    _stub(monkeypatch)
    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="r\n", env=WIDE
    )

    assert result.exit_code == 1
    names = _raw_files(world)
    assert any(n.endswith(".png") for n in names), "the image must remain"
    assert not any(n.endswith(".txt") for n in names), "nothing may be frozen"
    assert not any(n.endswith(".json") for n in names), "no sidecar for a non-ingest"
    assert list(world["queue"].glob("*/*.md")) == []


def test_an_edit_is_what_gets_frozen_and_extracted(world, monkeypatch) -> None:
    """Correcting the OCR is the whole point of the checkpoint, so the correction — not
    the model's version — must be what lands and what extraction sees."""
    corrected = "Die Bäckerei öffnet um SECHS Uhr, korrigiert.\n"
    _stub(monkeypatch)
    monkeypatch.setattr("click.edit", lambda *a, **k: corrected)

    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="e\n", env=WIDE
    )
    assert result.exit_code == 0, _combined(result)

    stored = next(world["raw"].glob("*.txt")).read_text(encoding="utf-8")
    assert "korrigiert" in stored

    sidecar = json.loads(next(world["raw"].glob("*.json")).read_text(encoding="utf-8"))
    assert sidecar["ocr_edited"] is True
    # The digest of what the MODEL said, so the correction stays auditable.
    assert sidecar["ocr_sha256"] != sidecar["content_sha256"]


def test_closing_the_editor_without_saving_keeps_the_original(world, monkeypatch) -> None:
    _stub(monkeypatch)
    monkeypatch.setattr("click.edit", lambda *a, **k: None)

    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="e\n", env=WIDE
    )
    assert result.exit_code == 0, _combined(result)
    assert "Bäckerei" in next(world["raw"].glob("*.txt")).read_text(encoding="utf-8")


def test_yes_skips_the_checkpoint(world, monkeypatch) -> None:
    _stub(monkeypatch)
    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), "--yes", *_flags(world)], env=WIDE
    )
    assert result.exit_code == 0, _combined(result)
    assert any(n.endswith(".txt") for n in _raw_files(world))


def test_the_checkpoint_shows_the_text_and_says_what_to_look_at(world, monkeypatch) -> None:
    _stub(monkeypatch)
    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE
    )
    out = _combined(result)
    assert "Bäckerei" in out
    assert "umlauts" in out.lower()


# --- the image lands before any model call ---


def test_a_failed_ocr_still_leaves_the_image_in_raw(world, monkeypatch) -> None:
    """Slice 3's ordering rule, holding for a new input type: the raw record must never
    depend on the model succeeding. The image is the true source; the text is derived."""
    _stub(monkeypatch, ocr="Die Bäckerei öff", ocr_finish="length")

    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE
    )

    assert result.exit_code == 1
    assert "OCR failed" in _combined(result)
    assert any(n.endswith(".png") for n in _raw_files(world)), "image survives the failure"
    assert not any(n.endswith(".txt") for n in _raw_files(world))


def test_a_truncated_transcription_is_refused_not_stored(world, monkeypatch) -> None:
    """A truncated page reads as a complete one — the reason `length` is a failed call."""
    _stub(monkeypatch, ocr="Die Bäckerei öffnet um", ocr_finish="length")
    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), "--yes", *_flags(world)], env=WIDE
    )
    assert result.exit_code == 1
    assert "max_tokens" in _combined(result)


# --- --dry-run ---


def test_dry_run_shows_the_ocr_and_writes_nothing(world, monkeypatch) -> None:
    calls = _stub(monkeypatch)
    result = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), "--dry-run", *_flags(world)], env=WIDE
    )

    assert result.exit_code == 0, _combined(result)
    assert "Bäckerei" in result.stdout
    assert calls["vision"] == 1
    assert calls["extraction"] == 0
    assert _raw_files(world) == [], "not even the image"
    assert not world["queue"].exists()


def test_dry_run_refuses_a_non_image(world, monkeypatch) -> None:
    text = world["root"] / "notes.txt"
    text.write_text("Guten Tag.", encoding="utf-8")
    result = runner.invoke(
        app, ["ingest", "-f", str(text), "--dry-run", *_flags(world)], env=WIDE
    )
    assert result.exit_code == 1
    assert "applies to images" in _combined(result)


# --- re-ingest ---


def test_re_ingesting_the_same_image_is_detected(world, monkeypatch) -> None:
    _stub(monkeypatch)
    runner.invoke(app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE)
    second = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE
    )
    assert "Already ingested" in _combined(second)


def test_an_aborted_checkpoint_is_resumable_not_already_ingested(world, monkeypatch) -> None:
    """An image stored with no .txt beside it is an INCOMPLETE ingest, not a finished one.

    Treating it as already-done would strand the image: the checkpoint could never be
    reached again without --force, for a source that was never actually ingested.
    """
    _stub(monkeypatch)
    runner.invoke(app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="r\n", env=WIDE)
    assert any(n.endswith(".png") for n in _raw_files(world))

    second = runner.invoke(
        app, ["ingest", "-f", str(world["image"]), *_flags(world)], input="a\n", env=WIDE
    )
    assert second.exit_code == 0, _combined(second)
    assert "Already ingested" not in _combined(second)
    assert any(n.endswith(".txt") for n in _raw_files(world))


# --- PDF routing ---


def test_a_scanned_pdf_is_refused_with_the_next_step(world, monkeypatch) -> None:
    from test_ingest_pdf import _make_pdf

    scanned = _make_pdf(world["root"] / "scan.pdf", ["", "", ""])
    calls = _stub(monkeypatch)

    result = runner.invoke(app, ["ingest", "-f", str(scanned), *_flags(world)], env=WIDE)

    assert result.exit_code == 1
    out = _combined(result)
    assert "looks scanned" in out
    assert "gw ingest -f" in out  # points at the image path, which does OCR
    assert calls["vision"] == 0, "a refusal must not spend a vision call"


def test_a_text_layer_pdf_ingests_page_by_page_without_vision(world, monkeypatch) -> None:
    from test_ingest_pdf import _make_pdf

    pdf = _make_pdf(
        world["root"] / "kapitel.pdf",
        ["Erste Seite mit ausreichend Text fuer die Schwelle darin.",
         "Zweite Seite mit ausreichend Text fuer die Schwelle darin."],
    )
    calls = _stub(monkeypatch)

    result = runner.invoke(app, ["ingest", "-f", str(pdf), *_flags(world)], env=WIDE)

    assert result.exit_code == 0, _combined(result)
    assert calls["vision"] == 0, "a text layer costs nothing to read"
    assert calls["extraction"] == 2, "one extraction per page"
    assert "2 page(s) ingested as separate sources" in _combined(result)

    sidecars = sorted(world["raw"].glob("*.json"))
    assert len(sidecars) == 2
    pages = [json.loads(s.read_text(encoding="utf-8"))["page"] for s in sidecars]
    assert sorted(pages) == [1, 2]


def test_the_pdf_itself_is_stored_once_beside_its_pages(world, monkeypatch) -> None:
    """§12.1 still needs something to re-verify a drifted node against."""
    from test_ingest_pdf import _make_pdf

    pdf = _make_pdf(
        world["root"] / "kapitel.pdf",
        ["Erste Seite mit ausreichend Text fuer die Schwelle darin.",
         "Zweite Seite mit ausreichend Text fuer die Schwelle darin."],
    )
    _stub(monkeypatch)
    runner.invoke(app, ["ingest", "-f", str(pdf), *_flags(world)], env=WIDE)

    stored = [n for n in _raw_files(world) if n.endswith(".pdf")]
    assert len(stored) == 1

    sidecar = json.loads(min(world["raw"].glob("*.json")).read_text(encoding="utf-8"))
    assert sidecar["source_document"] == stored[0].removesuffix(".pdf")


def test_a_partial_pdf_reports_the_skipped_pages(world, monkeypatch) -> None:
    from test_ingest_pdf import _make_pdf

    pdf = _make_pdf(
        world["root"] / "gemischt.pdf",
        ["Erste Seite mit ausreichend Text fuer die Schwelle darin.",
         "",
         "Dritte Seite mit ausreichend Text fuer die Schwelle darin."],
    )
    _stub(monkeypatch)
    result = runner.invoke(app, ["ingest", "-f", str(pdf), *_flags(world)], env=WIDE)

    out = _combined(result)
    assert "1 page(s) had no text" in out
    assert "Export those as images" in out
