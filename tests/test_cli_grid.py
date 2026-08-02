"""gw grid / gaps / families — the study-facing output, and the review round trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from german_wiki import storage
from german_wiki.cli import app
from german_wiki.models import Link, Node

runner = CliRunner()
WIDE = {"COLUMNS": "200"}

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
    "defaults": {"provider": "zai", "model": "free-model", "temperature": 0.0, "max_tokens": 2048},
    "steps": {"transparency": {"status": "active"}},
}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


def _prefix(morpheme: str, *, separable: bool = True, targets: list[str] = ()) -> Node:
    return Node(
        id=f"prefix-{morpheme}",
        title_de=f"{morpheme}- (Präfix)",
        title_en=f"prefix {morpheme}-",
        type="pattern",
        cefr="A2",
        status="stable",
        separable=separable,
        links=[Link(target=t, relation="same_family") for t in targets],
        body_md="x",
    )


def _family(root: str, lemmas: list[str], **overrides) -> Node:
    data = {
        "id": f"familie-{root}",
        "title_de": root,
        "title_en": root,
        "type": "vocab",
        "cefr": "A2",
        "status": "stable",
        "root": root,
        "lemmas": lemmas,
        "family_transparency": "high",
        "body_md": "x",
    }
    data.update(overrides)
    return Node(**data)


@pytest.fixture
def world(tmp_path: Path, tmp_vocab: Path, monkeypatch):
    nodes = tmp_path / "nodes"
    nodes.mkdir()
    for node in (
        _prefix("an", targets=["ankommen"]),
        _prefix("ab"),
        # `aufkommen` motivates a prefix node that does NOT exist, so `gw families` has a
        # `create` to propose -- which is what the staged-node tests need.
        _family("kommen", ["kommen", "abkommen", "aufkommen"]),
    ):
        storage.write_node(node, nodes / f"{node.id}.md", vocab_dir=tmp_vocab)

    settings = tmp_path / "models.yaml"
    settings.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv("GW_MODELS_CONFIG", str(settings))
    monkeypatch.setenv("ZAI_API_KEY", "test-key")

    return {
        "nodes": nodes,
        "vocab": tmp_vocab,
        "proposals": tmp_path / "proposals",
        "cache": tmp_path / "cache",
        "ledger": tmp_path / "decisions.jsonl",
        "merged": tmp_path / "_merged",
        "queue": tmp_path / "queue",
        "raw": tmp_path / "raw",
        "db": tmp_path / "index.db",
    }


# --- gw grid ---


def test_grid_renders_the_matrix(world) -> None:
    result = runner.invoke(app, ["grid", "--nodes-dir", str(world["nodes"])], env=WIDE)
    assert result.exit_code == 0, _combined(result)
    assert "Root × prefix" in result.stdout
    assert "an-" in result.stdout and "ab-" in result.stdout
    assert "kommen" in result.stdout


def test_a_gap_cell_never_spells_out_a_word(world) -> None:
    """The load-bearing rendering rule.

    The grid computes a full cross-product and most of it is not German -- `an-` × `waschen`
    yields `anwaschen`. Printing that in a study tool reads as "learn this", and at
    10 prefixes × 20 roots it would bury the real signal under ~140 invented words. A cell
    spells a word only when something vouches for it.
    """
    storage.write_node(
        _family("waschen", ["waschen"]),
        world["nodes"] / "familie-waschen.md",
        vocab_dir=world["vocab"],
    )
    result = runner.invoke(app, ["grid", "--nodes-dir", str(world["nodes"])], env=WIDE)

    assert "anwaschen" not in result.stdout  # the non-word is never named
    assert "abkommen" in result.stdout  # ...but an attested one is
    assert "·" in result.stdout  # and the empty cell is still visible


def test_an_identified_cell_is_named_and_framed_as_next(world) -> None:
    """A dangling target is the one gap you can act on, so it says so."""
    result = runner.invoke(app, ["grid", "--nodes-dir", str(world["nodes"])], env=WIDE)
    assert "ankommen" in result.stdout
    assert "study next" in result.stdout


def test_an_empty_corpus_explains_what_the_grid_needs(tmp_path: Path) -> None:
    empty = tmp_path / "nodes"
    empty.mkdir()
    result = runner.invoke(app, ["grid", "--nodes-dir", str(empty)], env=WIDE)
    assert result.exit_code == 0
    assert "No grid yet" in result.stdout
    assert "separable" in result.stdout  # says what would make one


def test_glyphs_only_mode_drops_every_word(world) -> None:
    result = runner.invoke(
        app, ["grid", "--glyphs", "--nodes-dir", str(world["nodes"])], env=WIDE
    )
    assert "ankommen" not in result.stdout
    assert "◇" in result.stdout


# --- gw gaps ---


def test_gaps_lists_dangling_targets_as_intentions(world) -> None:
    result = runner.invoke(app, ["gaps", "--nodes-dir", str(world["nodes"])], env=WIDE)
    assert result.exit_code == 0, _combined(result)
    assert "ankommen" in result.stdout
    assert "intentions you wrote, not errors" in result.stdout


def test_gaps_reports_stress_ambiguity_separately(world) -> None:
    storage.write_node(
        _family("fahren", ["fahren", "umfahren"]),
        world["nodes"] / "familie-fahren.md",
        vocab_dir=world["vocab"],
    )
    result = runner.invoke(
        app, ["gaps", "--ambiguous", "--nodes-dir", str(world["nodes"])], env=WIDE
    )
    assert "umfahren" in result.stdout
    assert "stress" in result.stdout


# --- gw families -> gw review ---


def _flags(w) -> list[str]:
    return [
        "--nodes-dir", str(w["nodes"]),
        "--queue-dir", str(w["queue"]),
        "--proposals-dir", str(w["proposals"]),
        "--merged-dir", str(w["merged"]),
        "--vocab-dir", str(w["vocab"]),
        "--raw-dir", str(w["raw"]),
        "--decisions-log", str(w["ledger"]),
        "--cache-dir", str(w["cache"]),
        "--db", str(w["db"]),
    ]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_families_proposes_without_writing_nodes(world) -> None:
    before = _snapshot(world["nodes"])
    result = runner.invoke(
        app,
        [
            "families", "--no-judge",
            "--nodes-dir", str(world["nodes"]),
            "--proposals-dir", str(world["proposals"]),
            "--cache-dir", str(world["cache"]),
        ],
        env=WIDE,
    )
    assert result.exit_code == 0, _combined(result)
    assert _snapshot(world["nodes"]) == before
    assert "Nothing written to /nodes" in result.stdout


def test_a_hand_edit_to_a_staged_node_is_what_lands(world) -> None:
    """The regression test for a silent data-loss bug (ADR-011, amended).

    `apply_create` used to overwrite the loaded staged node's body with the proposal's
    duplicate copy, and then `unlink()` the staged file -- so a reviewer's edit was
    discarded AND destroyed, with no copy left anywhere. It was latent in slice 5 and
    first reachable in slice 7, the first code to stage a node and write a proposal
    carrying the same body.

    This edits ONLY the staged file, leaving the proposal untouched, which is the exact
    path that failed.
    """
    runner.invoke(
        app,
        [
            "families", "--no-judge",
            "--nodes-dir", str(world["nodes"]),
            "--proposals-dir", str(world["proposals"]),
            "--queue-dir", str(world["queue"]),
            "--vocab-dir", str(world["vocab"]),
            "--cache-dir", str(world["cache"]),
        ],
        env=WIDE,
    )
    staged = world["queue"] / "morph-analysis" / "prefix-auf.md"
    assert staged.is_file(), "gw families must stage the node it proposes creating"

    edited = "auf- means up/open. MY OWN WORDS."
    text = staged.read_text(encoding="utf-8")
    assert "TODO" in text
    staged.write_text(
        text.replace("**TODO — write what `auf-` does to a verb's sense.**", edited),
        encoding="utf-8",
    )

    # The proposal is deliberately NOT edited, and must not carry a competing copy.
    proposal = next(world["proposals"].glob("create-prefix-auf*.md"))
    assert edited not in proposal.read_text(encoding="utf-8")

    result = runner.invoke(app, ["review", "--yes", *_flags(world)], env=WIDE)
    assert result.exit_code == 0, _combined(result)

    written = storage.load_node(world["nodes"] / "prefix-auf.md")
    assert edited in written.body_md, "the hand-edit must be what lands"
    assert "TODO" not in written.body_md, "the stale proposal body must not win"


def test_review_displays_the_staged_file_not_the_proposal(world) -> None:
    """You must see what you are approving -- the same root cause, on the read side."""
    runner.invoke(
        app,
        [
            "families", "--no-judge",
            "--nodes-dir", str(world["nodes"]),
            "--proposals-dir", str(world["proposals"]),
            "--queue-dir", str(world["queue"]),
            "--vocab-dir", str(world["vocab"]),
            "--cache-dir", str(world["cache"]),
        ],
        env=WIDE,
    )
    staged = world["queue"] / "morph-analysis" / "prefix-auf.md"
    staged.write_text(
        staged.read_text(encoding="utf-8").replace(
            "**TODO — write what `auf-` does to a verb's sense.**", "VISIBLE-IN-REVIEW"
        ),
        encoding="utf-8",
    )

    # `s` skips without applying, so this asserts the DISPLAY alone.
    result = runner.invoke(app, ["review", *_flags(world)], input="s\ns\ns\n", env=WIDE)
    assert "VISIBLE-IN-REVIEW" in _combined(result)


def test_a_staged_create_proposal_carries_no_duplicate_body(world) -> None:
    """The duplicate is removed at the source, so the two artifacts cannot diverge."""
    runner.invoke(
        app,
        [
            "families", "--no-judge",
            "--nodes-dir", str(world["nodes"]),
            "--proposals-dir", str(world["proposals"]),
            "--queue-dir", str(world["queue"]),
            "--vocab-dir", str(world["vocab"]),
            "--cache-dir", str(world["cache"]),
        ],
        env=WIDE,
    )
    from german_wiki import merge

    for proposal in merge.list_proposals(proposals_dir=world["proposals"]):
        if proposal.kind == "create" and proposal.candidate_path:
            assert proposal.body_md == "", f"{proposal.id} duplicates the staged body"


def test_an_approved_morphology_changes_frontmatter_only(world, monkeypatch) -> None:
    """The same narrowness as relevel: two frontmatter fields, body byte-identical."""
    from german_wiki.llm import ModelResponse, Usage

    def _complete(step, prompt, **kwargs):
        return ModelResponse(
            text=json.dumps(
                {"transparency": "drifted", "reason": "meanings diverge", "confidence": 0.8}
            ),
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

    monkeypatch.setattr("german_wiki.morph._transparency.complete", _complete)

    # A family with no transparency yet, so it gets judged.
    target = _family("machen", ["machen", "abmachen"], family_transparency=None)
    storage.write_node(target, world["nodes"] / f"{target.id}.md", vocab_dir=world["vocab"])
    body_before = storage.load_node(world["nodes"] / f"{target.id}.md").body_md

    runner.invoke(
        app,
        [
            "families",
            "--nodes-dir", str(world["nodes"]),
            "--proposals-dir", str(world["proposals"]),
            "--cache-dir", str(world["cache"]),
        ],
        env=WIDE,
    )
    result = runner.invoke(app, ["review", "--yes", *_flags(world)], env=WIDE)
    assert result.exit_code == 0, _combined(result)

    after = storage.load_node(world["nodes"] / f"{target.id}.md")
    assert after.family_transparency == "drifted"
    assert after.body_md == body_before  # nothing else moved
