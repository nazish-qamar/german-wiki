"""The immutable raw store: source ids, byte-verbatim text, and the provenance sidecar."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from german_wiki.ingest import _raw

TODAY = date(2026, 7, 26)
TEXT = "Die Wechselpräpositionen stehen mit Akkusativ oder Dativ.\n"


def _store(text: str, name: str, tmp_raw: Path) -> str:
    source_id, _ = _raw.resolve_source_id(text, Path(name), raw_dir=tmp_raw, today=TODAY)
    _raw.store_raw(text, source_id, raw_dir=tmp_raw)
    return source_id


# --- slugify ---


@pytest.mark.parametrize(
    "text,expected",
    [
        ("notes", "notes"),
        ("Wechselpräpositionen", "wechselpraepositionen"),
        ("Straße", "strasse"),
        ("Übung 3", "uebung-3"),
        ("Café Notes!", "cafe-notes"),
        ("  spaced  out  ", "spaced-out"),
        ("---", "quelle"),
        ("", "quelle"),
    ],
)
def test_slugify(text, expected) -> None:
    """Source ids stay ASCII on purpose (ADR-012) -- ä->ae, ß->ss, é stripped."""
    assert _raw.slugify(text) == expected


def test_slugify_truncates_and_never_ends_in_a_hyphen() -> None:
    slug = _raw.slugify("wort " * 40)
    assert len(slug) <= 40
    assert not slug.endswith("-")


def test_the_two_slug_functions_deliberately_disagree() -> None:
    """ADR-012's whole point, in one assertion.

    A source id names an immutable ``/raw`` file (SPEC §1.2) and is an opaque handle
    carrying a content hash — nobody reads it as German and nobody renames it, so it
    stays ASCII. A node id is what Obsidian shows as the note's name, what appears in
    ``links: target:``, and what you type into ``gw review`` — so it carries the real
    word.

    This asserts they differ *because a future session will otherwise "fix" the
    inconsistency*, and doing so in either direction breaks something: transliterating
    node ids loses the German, while un-transliterating source ids would rename files
    in an append-only store.
    """
    assert _raw.slugify("Wechselpräpositionen") == "wechselpraepositionen"
    assert _raw.node_slug("Wechselpräpositionen") == "wechselpräpositionen"
    assert _raw.slugify("Straße") == "strasse"
    assert _raw.node_slug("Straße") == "straße"


def test_node_slug_folds_unicode_normalization_forms() -> None:
    """Precomposed U+00E4 and decomposed U+0061+U+0308 must not become two ids."""
    assert _raw.node_slug("Prüfung") == _raw.node_slug("Prüfung") == "prüfung"


# --- source ids ---


def test_id_shape(tmp_raw: Path) -> None:
    source_id, is_new = _raw.resolve_source_id(
        TEXT, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY
    )
    stamp, slug, short = source_id.split("-")
    assert (stamp, slug) == ("20260726", "notes")
    assert len(short) == 8
    assert set(short) <= set("0123456789abcdef")
    assert is_new is True


def test_same_content_under_a_different_filename_shares_the_hash(tmp_raw: Path) -> None:
    """The hash is over content, not filename -- that is what makes dedup possible."""
    a, _ = _raw.resolve_source_id(TEXT, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY)
    b, _ = _raw.resolve_source_id(TEXT, Path("kopie.txt"), raw_dir=tmp_raw, today=TODAY)
    assert a.split("-")[-1] == b.split("-")[-1]
    assert a != b  # slug still distinguishes them


def test_different_content_yields_a_different_id(tmp_raw: Path) -> None:
    a, _ = _raw.resolve_source_id(TEXT, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY)
    b, _ = _raw.resolve_source_id(
        "Etwas anderes.\n", Path("notes.txt"), raw_dir=tmp_raw, today=TODAY
    )
    assert a != b


def test_reingesting_identical_content_is_recognized(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    again, is_new = _raw.resolve_source_id(TEXT, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY)
    assert (again, is_new) == (source_id, False)


# --- short-hash collision ---


def test_short_hash_collision_extends_instead_of_overwriting(tmp_raw: Path) -> None:
    """Two different sources sharing date+slug+8-char prefix must not clobber.

    Forced by planting a file at the id the second text would resolve to.
    """
    other = "Ein völlig anderer Text.\n"
    taken, _ = _raw.resolve_source_id(other, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY)
    # Squat on that id with DIFFERENT bytes, simulating a prefix collision.
    tmp_raw.mkdir(parents=True, exist_ok=True)
    squatter = b"vorher da gewesen\n"
    (tmp_raw / f"{taken}.txt").write_bytes(squatter)

    resolved, is_new = _raw.resolve_source_id(
        other, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY
    )

    assert is_new is True
    assert resolved != taken
    assert len(resolved.split("-")[-1]) > 8  # prefix widened
    assert (tmp_raw / f"{taken}.txt").read_bytes() == squatter  # first source intact


def test_collision_resolution_works_without_a_sidecar(tmp_raw: Path) -> None:
    """Comparison is against stored bytes, so a failed prior ingest is handled."""
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    assert _raw.read_sidecar(source_id, raw_dir=tmp_raw) is None
    again, is_new = _raw.resolve_source_id(TEXT, Path("notes.txt"), raw_dir=tmp_raw, today=TODAY)
    assert (again, is_new) == (source_id, False)


# --- the raw text is immutable and verbatim ---


@pytest.mark.parametrize(
    "text",
    ["plain\n", "crlf\r\nlines\r\n", "no trailing newline", "  trailing spaces   \n\n\n", "äöüß\n"],
    ids=["lf", "crlf", "no-eol", "whitespace", "umlauts"],
)
def test_stored_text_is_byte_identical(tmp_raw: Path, text) -> None:
    source_id = _store(text, "notes.txt", tmp_raw)
    text_path, _ = _raw.raw_paths(source_id, raw_dir=tmp_raw)
    assert text_path.read_bytes() == text.encode("utf-8")


def test_store_raw_never_overwrites(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    text_path, _ = _raw.raw_paths(source_id, raw_dir=tmp_raw)
    text_path.write_bytes(b"hand-edited")
    _raw.store_raw(TEXT, source_id, raw_dir=tmp_raw)
    assert text_path.read_bytes() == b"hand-edited"


def test_read_source_rejects_non_utf8(tmp_path: Path) -> None:
    bad = tmp_path / "latin1.txt"
    bad.write_bytes("Küche".encode("latin-1"))
    with pytest.raises(ValueError, match="not valid UTF-8"):
        _raw.read_source(bad)


def test_read_source_round_trips_utf8(tmp_path: Path) -> None:
    src = tmp_path / "notes.txt"
    src.write_bytes(TEXT.encode("utf-8"))
    assert _raw.read_source(src) == TEXT


# --- sidecar ---


def _sidecar(tmp_raw: Path, source_id: str, **overrides) -> dict:
    kwargs = {
        "text": TEXT,
        "original": Path("notes.txt"),
        "model": "glm-4.5-flash",
        "provider": "zai",
        "candidate_count": 6,
        "raw_dir": tmp_raw,
        "ingested_at": datetime(2026, 7, 26, 9, 14, 3, tzinfo=UTC),
    }
    kwargs.update(overrides)
    _raw.write_sidecar(source_id, **kwargs)
    return _raw.read_sidecar(source_id, raw_dir=tmp_raw)


def test_sidecar_has_exactly_the_documented_keys(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    assert set(_sidecar(tmp_raw, source_id)) == set(_raw.SIDECAR_KEYS)


def test_sidecar_stores_the_full_content_sha256(tmp_raw: Path) -> None:
    """Slice 5's tier-1 dedup key (SPEC §3.1). Must NOT be the 8-char filename form."""
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    meta = _sidecar(tmp_raw, source_id)
    expected = hashlib.sha256(TEXT.encode("utf-8")).hexdigest()

    assert meta["content_sha256"] == expected
    assert len(meta["content_sha256"]) == 64
    assert meta["content_sha256"] != source_id.split("-")[-1]
    assert meta["content_sha256"].startswith(source_id.split("-")[-1])


def test_sidecar_records_the_extraction_outcome(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    meta = _sidecar(tmp_raw, source_id)
    assert meta["source_id"] == source_id
    assert meta["original_path"] == "notes.txt"
    assert meta["model"] == "glm-4.5-flash"
    assert meta["provider"] == "zai"
    assert meta["candidate_count"] == 6
    assert meta["chars"] == len(TEXT)
    assert meta["ingested_at"].startswith("2026-07-26T09:14:03")


def test_sidecar_keeps_umlauts_unescaped(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "übung.txt", tmp_raw)
    _sidecar(tmp_raw, source_id, original=Path("übung.txt"))
    _, sidecar_path = _raw.raw_paths(source_id, raw_dir=tmp_raw)
    assert "übung.txt" in sidecar_path.read_text(encoding="utf-8")


def test_missing_sidecar_reads_as_none(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    assert _raw.read_sidecar(source_id, raw_dir=tmp_raw) is None


def test_sidecar_is_valid_json_on_disk(tmp_raw: Path) -> None:
    source_id = _store(TEXT, "notes.txt", tmp_raw)
    _sidecar(tmp_raw, source_id)
    _, sidecar_path = _raw.raw_paths(source_id, raw_dir=tmp_raw)
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["source_id"] == source_id
