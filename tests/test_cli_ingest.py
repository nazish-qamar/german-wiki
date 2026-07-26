"""End-to-end gw ingest and gw queue, proving /nodes is never written (ADR-003)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import FakeChatClient
from typer.testing import CliRunner

from german_wiki import ingest
from german_wiki.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

TEXT = "Die Wechselpräpositionen stehen mit Akkusativ (wohin?) oder Dativ (wo?).\n"

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
    "steps": {"extraction": {"status": "active"}},
}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


def _payload(n: int = 2) -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "title_de": f"Konzept {i}",
                    "title_en": f"Concept {i}",
                    "type": "grammar",
                    "cefr": "A2",
                    "cefr_basis": "grammar:test",
                    "register": ["alltag"],
                    "themes": ["haushalt"],
                    "body_md": "Erklärung.",
                    "confidence": 0.8,
                }
                for i in range(n)
            ]
        },
        ensure_ascii=False,
    )


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text(TEXT, encoding="utf-8")
    return path


def _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab, *extra):
    return runner.invoke(
        app,
        [
            "ingest",
            "--file",
            str(source),
            "--raw-dir",
            str(tmp_raw),
            "--queue-dir",
            str(tmp_queue),
            "--nodes-dir",
            str(tmp_nodes),
            "--vocab-dir",
            str(tmp_vocab),
            *extra,
        ],
        env=WIDE,
    )


# --- the ADR-003 guarantee ---


def test_ingest_writes_the_queue_and_never_nodes(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    before = sorted(p.name for p in tmp_nodes.glob("*.md"))
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(2)))

    result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)

    assert result.exit_code == 0
    assert sorted(p.name for p in tmp_nodes.glob("*.md")) == before  # /nodes untouched
    assert len(list(tmp_queue.glob("*/*.md"))) == 2


def test_output_states_nothing_was_written_to_nodes(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(2)))
    result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    assert "Nothing written to /nodes" in result.stdout
    assert "gw promote" in result.stdout


def test_ingest_does_not_grow_the_vocabulary(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    """ADR-007: only promote learns."""
    before = (tmp_vocab / "themes.txt").read_text(encoding="utf-8")
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(1)))
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    assert (tmp_vocab / "themes.txt").read_text(encoding="utf-8") == before


# --- raw store ---


def test_raw_text_and_sidecar_are_written(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(2)))
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)

    txt = list(tmp_raw.glob("*.txt"))
    sidecars = list(tmp_raw.glob("*.json"))
    assert len(txt) == len(sidecars) == 1
    # Byte-identical to the file on disk, line endings included. Compared against
    # the source's own bytes, not the literal above: on Windows write_text turns
    # \n into \r\n, and preserving exactly what was ingested is the point.
    assert txt[0].read_bytes() == source.read_bytes()
    assert json.loads(sidecars[0].read_text(encoding="utf-8"))["candidate_count"] == 2


def test_reingest_is_reported_and_does_not_duplicate(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    fake = FakeChatClient(text=_payload(2))
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), fake)
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)

    assert result.exit_code == 0
    assert "Already ingested" in result.stdout
    assert len(list(tmp_queue.glob("*/*.md"))) == 2  # not doubled
    assert fake.call_count == 1


def test_force_reruns(monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab) -> None:
    fake = FakeChatClient(text=_payload(2))
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), fake)
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab, "--force")

    assert result.exit_code == 0
    assert "Already ingested" not in result.stdout
    assert fake.call_count == 1  # re-run served from the cache (ADR-005)


# --- failures ---


def test_missing_file_errors(cfg, tmp_path, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab) -> None:
    result = _ingest(tmp_path / "absent.txt", tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    assert result.exit_code == 1
    assert "No such file" in _combined(result)


def test_truncation_is_rendered_readably(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    """The slice-2 finding surfaced at the CLI, with the reasoning excerpt."""
    fake = FakeChatClient(
        text="", finish_reason="length", reasoning_content="Ich analysiere den Text..."
    )
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), fake)
    result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)

    output = _combined(result)
    assert result.exit_code == 1
    assert "Extraction failed" in output
    assert "max_tokens" in output
    assert "Ich analysiere" in output
    assert list(tmp_raw.glob("*.txt"))  # raw kept
    assert not list(tmp_raw.glob("*.json"))  # no sidecar: incomplete ingest
    assert not list(tmp_queue.glob("*/*.md"))


def test_zero_candidates_reports_and_keeps_raw(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    _patch_client(
        monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=json.dumps({"candidates": []}))
    )
    result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    assert result.exit_code == 0
    assert "No candidates extracted" in result.stdout


# --- gw queue ---


def test_queue_is_empty_initially(tmp_queue: Path) -> None:
    result = runner.invoke(app, ["queue", "--queue-dir", str(tmp_queue)], env=WIDE)
    assert result.exit_code == 0
    assert "Queue is empty" in result.stdout


def test_queue_lists_pending_sources(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab
) -> None:
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(3)))
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)

    result = runner.invoke(app, ["queue", "--queue-dir", str(tmp_queue)], env=WIDE)
    assert result.exit_code == 0
    assert "3 candidate(s) in 1 source(s)" in result.stdout
    assert "gw promote" in result.stdout


# --- gw promote: the only writer into /nodes ---


def _promote(source_id, tmp_queue, tmp_nodes, tmp_vocab, tmp_db):
    return runner.invoke(
        app,
        [
            "promote",
            source_id,
            "--queue-dir",
            str(tmp_queue),
            "--nodes-dir",
            str(tmp_nodes),
            "--vocab-dir",
            str(tmp_vocab),
            "--db",
            str(tmp_db),
        ],
        env=WIDE,
    )


def test_promote_writes_nodes_and_reindexes(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab, tmp_db
) -> None:
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(2)))
    ingest_result = _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    assert ingest_result.exit_code == 0
    source_id = next(tmp_queue.iterdir()).name
    before = len(list(tmp_nodes.glob("*.md")))

    result = _promote(source_id, tmp_queue, tmp_nodes, tmp_vocab, tmp_db)

    assert result.exit_code == 0
    assert "Promoted 2 node(s)" in result.stdout
    assert len(list(tmp_nodes.glob("*.md"))) == before + 2
    assert list(tmp_queue.glob("*/*.md")) == []
    assert tmp_db.exists()


def test_promote_reports_an_unknown_source(tmp_queue, tmp_nodes, tmp_vocab, tmp_db) -> None:
    result = _promote("nope", tmp_queue, tmp_nodes, tmp_vocab, tmp_db)
    assert result.exit_code == 1
    assert "nothing queued for source" in _combined(result)


def test_promote_refuses_to_overwrite_and_exits_nonzero(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab, tmp_db
) -> None:
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(1)))
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    source_id = next(tmp_queue.iterdir()).name
    queued = next((tmp_queue / source_id).glob("*.md"))
    clash = tmp_nodes / queued.name
    clash.write_text("handmade\n", encoding="utf-8")

    result = _promote(source_id, tmp_queue, tmp_nodes, tmp_vocab, tmp_db)

    assert result.exit_code == 1
    assert "Refused" in _combined(result)
    assert "Nothing was overwritten" in _combined(result)
    assert clash.read_text(encoding="utf-8") == "handmade\n"
    assert queued.exists()


def test_the_full_loop_ingest_review_promote(
    monkeypatch, cfg, source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab, tmp_db
) -> None:
    """Slice 3 end to end, with the manual review step in the middle."""
    _patch_client(monkeypatch, cfg, tmp_path_of(tmp_raw), FakeChatClient(text=_payload(3)))
    _ingest(source, tmp_raw, tmp_queue, tmp_nodes, tmp_vocab)
    source_id = next(tmp_queue.iterdir()).name

    # Review: reject one candidate by deleting its file.
    rejected = min((tmp_queue / source_id).glob("*.md"))
    rejected.unlink()

    result = _promote(source_id, tmp_queue, tmp_nodes, tmp_vocab, tmp_db)

    assert result.exit_code == 0
    assert "Promoted 2 node(s)" in result.stdout
    assert not (tmp_nodes / rejected.name).exists()
    assert not tmp_queue.exists() or list(tmp_queue.glob("*/*.md")) == []


# --- helpers -------------------------------------------------------------
# The CLI has no --client flag (a fake client is not a user-facing concept), so
# the seam is patched here. Everything else is passed as an explicit path.


def tmp_path_of(tmp_raw: Path) -> Path:
    return tmp_raw.parent


def _patch_client(monkeypatch, cfg: Path, tmp_path: Path, fake: FakeChatClient) -> None:
    real = ingest._ingest.extract

    def _extract(text, **kwargs):
        kwargs.update(
            client=fake,
            settings_path=cfg,
            cache_dir=tmp_path / "cache",
            usage_log=tmp_path / "llm_usage.jsonl",
            env={},
        )
        return real(text, **kwargs)

    monkeypatch.setattr(ingest._ingest, "extract", _extract)
