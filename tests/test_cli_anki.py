"""`gw anki` end to end, and the property the whole format choice rests on.

`.apkg` was chosen over a TSV export for exactly one reason: a note carries a GUID, so
re-importing a later export **updates** the cards you already have instead of inserting
duplicates and discarding their scheduling history. That claim is only worth making if it
is tested, so `test_re_export_*` below reads GUIDs back out of the written file rather than
off the in-memory objects -- the file is what Anki sees.
"""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from german_wiki import storage
from german_wiki.anki import BASIC_MODEL_ID, CLOZE_MODEL_ID, DECK_ID, DECK_NAME
from german_wiki.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


def _guids(apkg: Path, tmp_path: Path) -> set[str]:
    """The GUIDs Anki would actually see: an .apkg is a zip around a SQLite collection."""
    extract_to = tmp_path / f"_unpack-{apkg.stem}-{apkg.stat().st_mtime_ns}"
    with zipfile.ZipFile(apkg) as archive:
        name = (
            "collection.anki21" if "collection.anki21" in archive.namelist() else "collection.anki2"
        )
        archive.extract(name, extract_to)
    connection = sqlite3.connect(extract_to / name)
    try:
        return {row[0] for row in connection.execute("select guid from notes")}
    finally:
        connection.close()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _run(tmp_nodes: Path, out: Path, *extra: str):
    return runner.invoke(
        app, ["anki", "--nodes-dir", str(tmp_nodes), "-o", str(out), *extra], env=WIDE
    )


# --- writing ---------------------------------------------------------------------------


def test_it_writes_one_apkg_and_touches_nothing_else(tmp_nodes: Path, tmp_path: Path) -> None:
    before = _snapshot(tmp_nodes)
    out = tmp_path / "deck.apkg"

    result = _run(tmp_nodes, out)

    assert result.exit_code == 0, _combined(result)
    assert out.is_file()
    assert _snapshot(tmp_nodes) == before, "the export must never write to /nodes"


def test_it_creates_the_output_directory(tmp_nodes: Path, tmp_path: Path) -> None:
    out = tmp_path / "nested" / "deeper" / "deck.apkg"
    assert _run(tmp_nodes, out).exit_code == 0
    assert out.is_file()


def test_dry_run_reports_but_writes_nothing(tmp_nodes: Path, tmp_path: Path) -> None:
    out = tmp_path / "deck.apkg"

    result = _run(tmp_nodes, out, "--dry-run")

    assert result.exit_code == 0, _combined(result)
    assert not out.exists()
    assert "nothing written" in _combined(result)


def test_the_written_deck_contains_the_reported_cards(tmp_nodes: Path, tmp_path: Path) -> None:
    out = tmp_path / "deck.apkg"
    _run(tmp_nodes, out)

    from german_wiki.anki import build

    expected = build(storage.load_all_nodes(tmp_nodes))
    assert _guids(out, tmp_path) == expected.guids


# --- THE load-bearing property ----------------------------------------------------------


def test_re_export_updates_rather_than_duplicates(tmp_nodes: Path, tmp_path: Path) -> None:
    """Two assertions, because the second is the one that actually matters.

    (1) alone -- re-exporting an unchanged corpus -- would pass for a GUID scheme that
    re-derives note identity from anything corpus-wide. But a wiki is re-exported *as it
    grows*, and that is where such a scheme silently re-identifies every existing note.
    Anki then imports them as new cards, discarding the review history that is the entire
    reason for exporting to a scheduler at all.
    """
    first = tmp_path / "first.apkg"
    second = tmp_path / "second.apkg"

    # 1. Idempotent: the same corpus twice gives the same notes.
    assert _run(tmp_nodes, first).exit_code == 0
    assert _run(tmp_nodes, second).exit_code == 0
    original = _guids(first, tmp_path)
    assert original == _guids(second, tmp_path)
    assert original, "nothing was exported, so the comparison proved nothing"

    # 2. Grows without churn: adding a node leaves every existing note alone.
    new_node = storage.Node(
        id="neuer-knoten",
        title_de="Neuer Knoten",
        title_en="New node",
        type="vocab",
        cefr="A2",
        status="stable",
        body_md="Text.\n\n## Examples\n- Ein **neuer** Satz. (A new sentence.)\n",
        source_ids=["seed"],
    )
    storage.write_node(new_node, tmp_nodes / "neuer-knoten.md")

    third = tmp_path / "third.apkg"
    assert _run(tmp_nodes, third).exit_code == 0
    grown = _guids(third, tmp_path)

    assert original <= grown, "existing notes were re-identified; scheduling would be lost"
    assert len(grown - original) == 1, f"expected exactly one new note, got {grown - original}"


def test_editing_one_node_leaves_the_other_nodes_notes_untouched(
    tmp_nodes: Path, tmp_path: Path
) -> None:
    """The same property at finer grain: a node's notes are its own."""
    first = tmp_path / "a.apkg"
    _run(tmp_nodes, first)
    before = _guids(first, tmp_path)

    target = tmp_nodes / "familie-waschen.md"
    node = storage.load_node(target)
    edited = node.model_copy(
        update={"body_md": node.body_md + "\n- Ein **weiterer** Satz. (Another sentence.)\n"}
    )
    storage.write_node(edited, target)

    second = tmp_path / "b.apkg"
    _run(tmp_nodes, second)
    after = _guids(second, tmp_path)

    assert before <= after
    assert len(after - before) == 1


# --- the fixed ids ------------------------------------------------------------------------


def test_the_deck_and_model_ids_are_the_fixed_literals() -> None:
    """Named here on purpose: a change that regenerates them must fail loudly.

    Runtime-generated ids declare a new note type on every export, so Anki treats the
    cards as unrelated notes and every previous one becomes an orphan.
    """
    assert DECK_ID == 1928374650
    assert CLOZE_MODEL_ID == 1928374651
    assert BASIC_MODEL_ID == 1928374652
    assert DECK_NAME == "German Wiki"


@pytest.mark.parametrize("identifier", [DECK_ID, CLOZE_MODEL_ID, BASIC_MODEL_ID])
def test_the_ids_are_in_ankis_expected_range(identifier: int) -> None:
    assert 1 << 30 <= identifier < 1 << 31


# --- the report ---------------------------------------------------------------------------


def test_it_reports_each_node_and_the_total(tmp_nodes: Path, tmp_path: Path) -> None:
    output = _combined(_run(tmp_nodes, tmp_path / "d.apkg"))

    assert "familie-waschen" in output
    assert "card(s) from" in output


def test_draft_nodes_are_held_back_and_said_so(tmp_nodes: Path, tmp_path: Path) -> None:
    node = storage.load_node(tmp_nodes / "familie-waschen.md")
    storage.write_node(
        node.model_copy(update={"status": "draft"}), tmp_nodes / "familie-waschen.md"
    )

    output = _combined(_run(tmp_nodes, tmp_path / "d.apkg"))

    assert "Held back by status" in output
    assert "--include-draft" in output


def test_include_draft_adds_them_back(tmp_nodes: Path, tmp_path: Path) -> None:
    node = storage.load_node(tmp_nodes / "familie-waschen.md")
    storage.write_node(
        node.model_copy(update={"status": "draft"}), tmp_nodes / "familie-waschen.md"
    )

    without = tmp_path / "without.apkg"
    with_draft = tmp_path / "with.apkg"
    _run(tmp_nodes, without)
    _run(tmp_nodes, with_draft, "--include-draft")

    assert _guids(without, tmp_path) < _guids(with_draft, tmp_path)


def test_a_node_with_no_examples_is_reported_with_what_to_do(
    tmp_nodes: Path, tmp_path: Path
) -> None:
    """The skip report is the useful half of the output -- it names an edit only you can
    make, in the node."""
    bare = storage.Node(
        id="ohne-beispiele",
        title_de="Ohne Beispiele",
        title_en="Without examples",
        type="vocab",
        cefr="A2",
        status="stable",
        body_md="Nur Prosa.",
        source_ids=["seed"],
    )
    storage.write_node(bare, tmp_nodes / "ohne-beispiele.md")

    output = _combined(_run(tmp_nodes, tmp_path / "d.apkg"))

    assert "ohne-beispiele" in output
    assert "## Examples" in output


def test_an_empty_nodes_dir_exits_nonzero(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    result = runner.invoke(
        app, ["anki", "--nodes-dir", str(empty), "-o", str(tmp_path / "d.apkg")], env=WIDE
    )

    assert result.exit_code == 1
    assert "No nodes" in _combined(result)
