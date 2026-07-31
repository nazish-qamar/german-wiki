"""End-to-end gw adjudicate -> proposals -> review, proving /nodes moves only on approval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import FakeEmbedder
from typer.testing import CliRunner

from german_wiki import storage
from german_wiki.cli import app
from german_wiki.db import connect, rebuild_schema
from german_wiki.embed._text import embed_text
from german_wiki.models import Node

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

BODY = "Perfekt mit haben.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n"

CONFIG = {
    "version": 1,
    "providers": {
        "zai": {
            "kind": "api",
            "base_url": "https://example.invalid/v4",
            "api_key_env": "ZAI_API_KEY",
        },
        "local": {"kind": "local"},
    },
    "pricing": {"zai": {"free-model": {"input": 0.0, "cached_input": 0.0, "output": 0.0}}},
    "defaults": {"provider": "zai", "model": "free-model", "temperature": 0.0, "max_tokens": 4096},
    "steps": {
        "adjudication": {"status": "active"},
        "embeddings": {"status": "active", "provider": "local", "model": "fake-embedder"},
    },
}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


def _node(node_id: str, title: str, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": title,
        "title_en": title,
        "type": "grammar",
        "cefr": "A2",
        "status": "draft",
        "body_md": BODY,
        "source_ids": ["seed"],
    }
    data.update(overrides)
    return Node(**data)


@pytest.fixture
def workspace(tmp_path: Path, tmp_vocab: Path, monkeypatch):
    """A complete tmp world, with the embedder patched to the offline fake."""
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    queue = tmp_path / "queue" / "src-1"
    queue.mkdir(parents=True)
    (tmp_path / "raw").mkdir()

    existing = _node("perfekt-haben", "Perfekt mit haben")
    candidate = _node("perfekt-sein", "Perfekt mit sein", source_ids=["src-1"])
    storage.write_node(existing, nodes / "perfekt-haben.md", vocab_dir=tmp_vocab)
    storage.write_node(candidate, queue / "perfekt-sein.md", vocab_dir=tmp_vocab)

    db = tmp_path / "index.db"
    conn = connect(db)
    rebuild_schema(conn)
    conn.close()

    settings = tmp_path / "models.yaml"
    settings.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("GW_MODELS_CONFIG", str(settings))
    monkeypatch.setenv("ZAI_API_KEY", "test-key")

    # The offline suite must never load sentence-transformers (ADR-010), so the graph's
    # embedder is replaced with the deterministic fake, placing the two nodes adjacent.
    fake = FakeEmbedder(
        model="fake-embedder",
        similar_to={embed_text(candidate): embed_text(existing)},
    )
    monkeypatch.setattr("german_wiki.embed._model.load_embedder", lambda **kw: fake)

    return {
        "root": tmp_path,
        "nodes": nodes,
        "queue": tmp_path / "queue",
        "raw": tmp_path / "raw",
        "vocab": tmp_vocab,
        "proposals": tmp_path / "proposals",
        "merged": tmp_path / "_merged",
        "ledger": tmp_path / "decisions.jsonl",
        "cache": tmp_path / "cache",
        "db": db,
    }


def _flags(ws) -> list[str]:
    return [
        "--nodes-dir", str(ws["nodes"]),
        "--queue-dir", str(ws["queue"]),
        "--proposals-dir", str(ws["proposals"]),
        "--merged-dir", str(ws["merged"]),
        "--vocab-dir", str(ws["vocab"]),
        "--raw-dir", str(ws["raw"]),
        "--decisions-log", str(ws["ledger"]),
        "--cache-dir", str(ws["cache"]),
        "--db", str(ws["db"]),
    ]


def _stub_calls(monkeypatch, *payloads: dict) -> list[dict]:
    """Feed canned model responses without a client, recording what was asked."""
    from german_wiki.llm import ModelResponse, Usage

    seen: list[dict] = []
    queue = list(payloads)

    def _complete(step, prompt, **kwargs):
        seen.append({"step": step, "prompt": prompt})
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        return ModelResponse(
            text=json.dumps(body, ensure_ascii=False),
            step=step,
            provider="zai",
            model="free-model",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cached=False,
            cost_usd=0.0,
            saved_usd=0.0,
            cache_key="k",
            finish_reason="stop",
        )

    monkeypatch.setattr("german_wiki.merge._adjudicate.complete", _complete)
    monkeypatch.setattr("german_wiki.merge._regenerate.complete", _complete)
    return seen


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --- adjudicate ---


def test_adjudicate_writes_proposals_and_not_nodes(workspace, monkeypatch) -> None:
    _stub_calls(
        monkeypatch,
        {"outcome": "OVERLAP", "confidence": 0.9, "reason": "same tense", "b_adds": "sein"},
        {"body_md": "Zusammengeführt.\n", "changelog": "added sein"},
    )
    before = _snapshot(workspace["nodes"])

    result = runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)

    assert result.exit_code == 0, _combined(result)
    assert _snapshot(workspace["nodes"]) == before
    assert "Nothing written to /nodes" in _combined(result)
    assert list(workspace["proposals"].glob("*.md"))


def test_adjudicate_needs_a_source_or_all(workspace) -> None:
    result = runner.invoke(app, ["adjudicate", *_flags(workspace)], env=WIDE)
    assert result.exit_code == 1
    assert "--all" in _combined(result)


def test_proposals_lists_what_is_pending(workspace, monkeypatch) -> None:
    _stub_calls(monkeypatch, {"outcome": "DISTINCT", "confidence": 0.9, "reason": "different"})
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)

    result = runner.invoke(
        app, ["proposals", "--proposals-dir", str(workspace["proposals"])], env=WIDE
    )
    assert result.exit_code == 0
    assert "create" in result.stdout
    assert "perfekt-sein" in result.stdout


# --- review ---


def test_review_approves_a_merge_end_to_end(workspace, monkeypatch) -> None:
    _stub_calls(
        monkeypatch,
        {"outcome": "OVERLAP", "confidence": 0.9, "reason": "same tense", "b_adds": "sein"},
        {"body_md": "Perfekt mit haben und sein.\n", "changelog": "added sein"},
    )
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)

    result = runner.invoke(app, ["review", "--yes", *_flags(workspace)], env=WIDE)

    assert result.exit_code == 0, _combined(result)
    winner = storage.load_node(workspace["nodes"] / "perfekt-haben.md")
    assert winner.body_md.strip() == "Perfekt mit haben und sein."
    assert (workspace["merged"] / "perfekt-sein.md").is_file()
    assert (workspace["merged"] / "perfekt-sein.json").is_file()
    assert list(workspace["proposals"].glob("*.md")) == []  # resolved and removed
    assert workspace["ledger"].is_file()


def test_review_shows_a_before_after_diff(workspace, monkeypatch) -> None:
    """The reviewer's primary drift guard is seeing exactly what changed (SPEC §12.1)."""
    _stub_calls(
        monkeypatch,
        {"outcome": "OVERLAP", "confidence": 0.9, "reason": "same tense", "b_adds": "sein"},
        {"body_md": "Ein völlig neuer Text.\n", "changelog": "rewrote"},
    )
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)

    result = runner.invoke(app, ["review", "--yes", *_flags(workspace)], env=WIDE)
    out = _combined(result)
    assert "-Perfekt mit haben." in out
    assert "+Ein völlig neuer Text." in out


def test_review_rejects_without_writing(workspace, monkeypatch) -> None:
    _stub_calls(
        monkeypatch,
        {"outcome": "OVERLAP", "confidence": 0.9, "reason": "same tense", "b_adds": "sein"},
        {"body_md": "Zusammengeführt.\n", "changelog": "merged"},
    )
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)
    before = _snapshot(workspace["nodes"])

    result = runner.invoke(app, ["review", *_flags(workspace)], input="r\n", env=WIDE)

    assert result.exit_code == 0, _combined(result)
    assert _snapshot(workspace["nodes"]) == before
    assert list(workspace["proposals"].glob("*.md")) == []
    assert not workspace["merged"].exists()
    # ...but the rejection is on the record, so the pair is not asked about again.
    [record] = [json.loads(line) for line in workspace["ledger"].read_text().splitlines() if line]
    assert record["approved"] is False


def test_review_skips_by_default(workspace, monkeypatch) -> None:
    """The default answer must never be "yes" for a command that rewrites study notes."""
    _stub_calls(monkeypatch, {"outcome": "DISTINCT", "confidence": 0.9, "reason": "different"})
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)

    result = runner.invoke(app, ["review", *_flags(workspace)], input="\n", env=WIDE)
    assert result.exit_code == 0, _combined(result)
    assert "1 skipped" in _combined(result)
    assert list(workspace["proposals"].glob("*.md"))  # still pending


def test_a_link_is_gated_exactly_like_a_merge(workspace, monkeypatch) -> None:
    """ADR-010's load-bearing assumption: an edge never auto-accepts."""
    _stub_calls(
        monkeypatch,
        {
            "outcome": "DISTINCT_RELATED",
            "confidence": 0.9,
            "reason": "expression vs the rule it obeys",
            "relation": "governs",
            "direction": "a_to_b",
        },
    )
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)

    # The edge exists only as a proposal; nothing has been linked yet.
    assert storage.load_node(workspace["nodes"] / "perfekt-haben.md").links == []
    kinds = sorted(p.stem.split("-")[0] for p in workspace["proposals"].glob("*.md"))
    assert kinds == ["create", "link"]

    result = runner.invoke(app, ["review", "--yes", *_flags(workspace)], env=WIDE)
    assert result.exit_code == 0, _combined(result)

    linked = storage.load_node(workspace["nodes"] / "perfekt-haben.md")
    assert [(link.target, link.relation) for link in linked.links] == [
        ("perfekt-sein", "governs")
    ]
    assert (workspace["nodes"] / "perfekt-sein.md").is_file()  # create ran first


def test_review_reports_nothing_pending(workspace) -> None:
    result = runner.invoke(app, ["review", *_flags(workspace)], env=WIDE)
    assert result.exit_code == 0
    assert "No proposals pending" in result.stdout


def test_raw_is_untouched_by_the_whole_flow(workspace, monkeypatch) -> None:
    (workspace["raw"] / "src-1.txt").write_bytes(b"Die Quelle.\n")
    before = _snapshot(workspace["raw"])

    _stub_calls(
        monkeypatch,
        {"outcome": "OVERLAP", "confidence": 0.9, "reason": "same", "b_adds": "sein"},
        {"body_md": "Zusammengeführt.\n", "changelog": "merged"},
    )
    runner.invoke(app, ["adjudicate", "src-1", *_flags(workspace)], env=WIDE)
    runner.invoke(app, ["review", "--yes", *_flags(workspace)], env=WIDE)

    assert _snapshot(workspace["raw"]) == before


def test_a_lost_ledger_warns_before_review(workspace, monkeypatch) -> None:
    """ADR-011: the fix is offered before you sit down to decide anything."""
    merged_once = _node("perfekt-haben", "Perfekt mit haben", version=3)
    storage.write_node(
        merged_once, workspace["nodes"] / "perfekt-haben.md", vocab_dir=workspace["vocab"]
    )
    assert not workspace["ledger"].exists()

    result = runner.invoke(app, ["review", *_flags(workspace)], env=WIDE)
    assert "git restore" in _combined(result)
