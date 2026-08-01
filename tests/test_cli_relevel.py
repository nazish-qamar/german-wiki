"""End-to-end gw relevel -> gw review, proving a level change touches nothing else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from german_wiki import storage
from german_wiki.cli import app
from german_wiki.level import _lexical
from german_wiki.models import Node

runner = CliRunner()
WIDE = {"COLUMNS": "220"}

EMPTY_WORDLISTS = Path(__file__).parent / "fixtures" / "cefr-empty"

BODY = """\
Viele Verben verlangen eine feste Präposition.

## Examples
- Ich warte auf den Bus. (I am waiting for the bus.)
"""

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
    "steps": {"cefr_tiebreak": {"status": "active"}},
}


@pytest.fixture(autouse=True)
def _clear_cache():
    _lexical.clear_cache()
    yield
    _lexical.clear_cache()


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


@pytest.fixture
def workspace(tmp_path: Path, tmp_vocab: Path, monkeypatch):
    nodes = tmp_path / "nodes"
    nodes.mkdir()

    # A node whose title names a §5 structure: grammar decides, no model needed.
    storage.write_node(
        Node(
            id="wechselpräpositionen",
            title_de="Wechselpräpositionen",
            title_en="Two-way prepositions",
            type="grammar",
            cefr="B1",  # deliberately wrong; the anchor says A2
            cefr_basis="llm:extraction; a guess",
            status="stable",
            body_md=BODY,
            source_ids=["seed"],
            version=1,
        ),
        nodes / "wechselpräpositionen.md",
        vocab_dir=tmp_vocab,
    )

    settings = tmp_path / "models.yaml"
    settings.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("GW_MODELS_CONFIG", str(settings))
    monkeypatch.setenv("ZAI_API_KEY", "test-key")

    return {
        "nodes": nodes,
        "vocab": tmp_vocab,
        "proposals": tmp_path / "proposals",
        "merged": tmp_path / "_merged",
        "queue": tmp_path / "queue",
        "raw": tmp_path / "raw",
        "ledger": tmp_path / "decisions.jsonl",
        "cache": tmp_path / "cache",
        "db": tmp_path / "index.db",
    }


def _relevel_flags(ws) -> list[str]:
    return [
        "--nodes-dir", str(ws["nodes"]),
        "--proposals-dir", str(ws["proposals"]),
        "--cefr-dir", str(EMPTY_WORDLISTS),
        "--cache-dir", str(ws["cache"]),
    ]


def _review_flags(ws) -> list[str]:
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


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --- relevel proposes ---


def test_relevel_writes_proposals_and_not_nodes(workspace) -> None:
    before = _snapshot(workspace["nodes"])
    result = runner.invoke(app, ["relevel", *_relevel_flags(workspace)], env=WIDE)

    assert result.exit_code == 0, _combined(result)
    assert _snapshot(workspace["nodes"]) == before
    assert "Nothing written to /nodes" in _combined(result)
    assert list(workspace["proposals"].glob("*.md"))


def test_relevel_says_when_no_wordlist_is_installed(workspace) -> None:
    """The shipped state, stated rather than left to be discovered."""
    result = runner.invoke(app, ["relevel", *_relevel_flags(workspace)], env=WIDE)
    assert "No CEFR wordlist" in _combined(result)


def test_relevel_needs_no_model_for_a_title_anchored_node(workspace) -> None:
    """Rules-first, end to end: the config has no API key path that would work.

    The step is configured but the fake base URL would fail on a real call, so a passing
    run proves the grammar map decided without one.
    """
    result = runner.invoke(
        app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE
    )
    assert result.exit_code == 0, _combined(result)
    assert list(workspace["proposals"].glob("*.md"))


# --- review applies ---


def test_an_approved_relevel_changes_only_cefr_and_basis(workspace) -> None:
    before = storage.load_node(workspace["nodes"] / "wechselpräpositionen.md")
    runner.invoke(app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE)

    result = runner.invoke(app, ["review", "--yes", *_review_flags(workspace)], env=WIDE)
    assert result.exit_code == 0, _combined(result)

    after = storage.load_node(workspace["nodes"] / "wechselpräpositionen.md")
    assert (before.cefr, after.cefr) == ("B1", "A2")
    assert after.cefr_basis == "grammar:wechselpräposition(A2)"

    # Everything else is untouched -- a relevel is two frontmatter lines.
    assert after.body_md == before.body_md
    assert after.title_de == before.title_de
    assert after.register == before.register
    assert after.source_ids == before.source_ids


def test_an_approved_relevel_does_not_demote_status(workspace) -> None:
    """ADR-011 §7 demotes on OVERLAP because the BODY was re-encoded.

    Nothing was re-encoded here and the reviewer saw the whole change, so `stable` is
    still earned -- same reasoning as an approved link.
    """
    runner.invoke(app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE)
    runner.invoke(app, ["review", "--yes", *_review_flags(workspace)], env=WIDE)

    assert storage.load_node(workspace["nodes"] / "wechselpräpositionen.md").status == "stable"


def test_a_rejected_relevel_writes_nothing_but_is_recorded(workspace) -> None:
    runner.invoke(app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE)
    before = _snapshot(workspace["nodes"])

    result = runner.invoke(app, ["review", *_review_flags(workspace)], input="r\n", env=WIDE)

    assert result.exit_code == 0, _combined(result)
    assert _snapshot(workspace["nodes"]) == before
    assert list(workspace["proposals"].glob("*.md")) == []
    [record] = [
        json.loads(line) for line in workspace["ledger"].read_text().splitlines() if line
    ]
    assert record["approved"] is False
    assert record["kind"] == "relevel"


def test_the_review_diff_shows_the_level_moving(workspace) -> None:
    runner.invoke(app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE)
    result = runner.invoke(app, ["review", "--yes", *_review_flags(workspace)], env=WIDE)
    out = _combined(result)
    assert "B1" in out and "A2" in out


def test_a_second_relevel_finds_nothing_to_do(workspace) -> None:
    """Convergence: a derived basis is not a placeholder, so it is not re-targeted."""
    runner.invoke(app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE)
    runner.invoke(app, ["review", "--yes", *_review_flags(workspace)], env=WIDE)

    result = runner.invoke(app, ["relevel", "--no-llm", *_relevel_flags(workspace)], env=WIDE)
    assert "Nothing to re-level" in _combined(result)
