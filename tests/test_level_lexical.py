"""The lexical anchor, and its most important property: absence is not an error."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from german_wiki.level import _lexical

FIXTURE = Path(__file__).parent / "fixtures" / "cefr"


@pytest.fixture(autouse=True)
def _clear_cache():
    """Wordlists are cached per directory; tests must not see each other's."""
    _lexical.clear_cache()
    yield
    _lexical.clear_cache()


# --- the committed fixture list ---
#
# Deliberately NOT vocab/cefr/, which is empty in git by design (licensing). A test
# reading the real directory would pass only on a machine that happens to have a list
# installed, and fail on a fresh clone.


def test_a_listed_lemma_resolves() -> None:
    hit = _lexical.lookup("haus", cefr_dir=FIXTURE)
    assert (hit.lemma, hit.level, hit.source) == ("haus", "A1", "goethe:a1")


def test_lookup_is_case_and_whitespace_insensitive() -> None:
    assert _lexical.lookup("  HAUS  ", cefr_dir=FIXTURE).level == "A1"


def test_an_unlisted_lemma_is_none_not_a_default() -> None:
    """None means "no evidence", and must never be read as a level."""
    assert _lexical.lookup("kraftfahrzeugsteuer", cefr_dir=FIXTURE) is None


def test_a_lemma_in_two_lists_takes_the_lowest() -> None:
    """`waschen` is in both a1 and b1: you meet a word at the level that teaches it."""
    assert _lexical.lookup("waschen", cefr_dir=FIXTURE).level == "A1"


def test_comments_and_blank_lines_are_ignored() -> None:
    """The fixture's first line is a `#` comment and it contains a blank line."""
    assert _lexical.lookup("#", cefr_dir=FIXTURE) is None
    assert _lexical.lookup("", cefr_dir=FIXTURE) is None
    assert _lexical.available(FIXTURE) is True


# --- graceful absence, in every shape ---


def test_a_missing_directory_yields_no_signal(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert _lexical.available(missing) is False
    assert _lexical.lookup("haus", cefr_dir=missing) is None


def test_an_empty_file_yields_no_signal(tmp_path: Path) -> None:
    (tmp_path / "a1.txt").write_text("", encoding="utf-8")
    assert _lexical.available(tmp_path) is False
    assert _lexical.lookup("haus", cefr_dir=tmp_path) is None


def test_a_file_of_only_blanks_and_comments_yields_no_signal(tmp_path: Path) -> None:
    (tmp_path / "a1.txt").write_text("# nothing here\n\n   \n", encoding="utf-8")
    assert _lexical.available(tmp_path) is False


def test_the_shipped_directory_is_empty_and_that_is_fine() -> None:
    """The committed state: code present, data absent, no exception.

    If this ever fails, someone committed a wordlist that licensing does not allow --
    or the .gitignore stopped working.
    """
    from german_wiki import config

    assert _lexical.available(config.CEFR_DIR) is False
    assert _lexical.lookup("haus") is None


# --- the node-level anchor ---


def test_the_title_is_used_when_a_node_has_no_lemmas() -> None:
    hit = _lexical.lexical_anchor("Haus", cefr_dir=FIXTURE)
    assert hit.level == "A1"


def test_a_word_family_takes_its_hardest_member() -> None:
    """Opposite rule to `lookup`, and deliberately so.

    `lookup` resolves one word listed twice and takes the lowest. A *family* is a set of
    different words, and it is only as easy as its hardest member -- you have not learned
    the family until you can handle `abwaschen` too.
    """
    hit = _lexical.lexical_anchor(
        "waschen", lemmas=["waschen", "abwaschen"], cefr_dir=FIXTURE
    )
    assert (hit.lemma, hit.level) == ("abwaschen", "B1")


def test_unicode_normalisation_folds_on_both_sides(tmp_path: Path) -> None:
    """A list written on macOS (NFD) must match a title typed on Windows (NFC).

    Both forms are constructed rather than written as literals -- see the note in
    test_level_grammar.py; a literal pair is not reliably two different byte sequences.
    """
    precomposed = unicodedata.normalize("NFC", "Prüfung")
    decomposed = unicodedata.normalize("NFD", "Prüfung")
    assert precomposed != decomposed

    (tmp_path / "a1.txt").write_text(f"{decomposed}\n", encoding="utf-8")
    assert _lexical.lookup(precomposed, cefr_dir=tmp_path).level == "A1"
