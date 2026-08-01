"""The grammar anchor: SPEC §5's table, and the two ways matching it goes wrong."""

from __future__ import annotations

import unicodedata

import pytest

from german_wiki.level import _grammar


def _hits(text: str, where="title"):
    return [(h.structure, h.level) for h in _grammar.match(text, where)]


# --- SPEC §5's table, transcribed ---


@pytest.mark.parametrize(
    ("text", "structure", "level"),
    [
        ("Das Präsens", "präsens", "A1"),
        ("Der Nominativ", "nominativ", "A1"),
        ("Das Perfekt", "perfekt", "A2"),
        ("Der Akkusativ", "akkusativ", "A2"),
        ("Wechselpräpositionen", "wechselpräposition", "A2"),
        ("Das Passiv", "passiv", "B1"),
        ("Konjunktiv II", "konjunktiv-ii", "B1"),
        ("Relativsätze", "relativsatz", "B1"),
        ("Der Genitiv", "genitiv", "B2"),
        ("Erweiterte Infinitive", "erweiterter-infinitiv", "B2"),
        ("Partizipialattribute", "partizipialattribut", "C1"),
        ("Nominalstil", "nominalstil", "C1"),
        ("Konjunktiv I", "konjunktiv-i", "C1"),
    ],
)
def test_the_spec_table_maps(text, structure, level) -> None:
    assert (structure, level) in _hits(text)


# --- the two matching traps ---


@pytest.mark.parametrize(
    "text", ["Konjunktiv II", "konjunktiv ii", "Konjunktiv-II", "Konjunktiv 2", "Konjunktiv II bei Bitten"]
)
def test_konjunktiv_ii_is_never_read_as_konjunktiv_i(text) -> None:
    """C1's Konjunktiv I must not swallow B1's Konjunktiv II and over-level the node.

    Guarded by a negative lookahead rather than list ordering, so re-sorting the table
    cannot silently break it.
    """
    structures = {s for s, _ in _hits(text)}
    assert "konjunktiv-ii" in structures
    assert "konjunktiv-i" not in structures


def test_konjunktiv_i_still_matches_on_its_own() -> None:
    structures = {s for s, _ in _hits("Konjunktiv I in der indirekten Rede")}
    assert structures == {"konjunktiv-i"}


def test_german_compounds_match_their_head() -> None:
    """`Akkusativobjekt` is the same signal as `Akkusativ`; a trailing \\b would miss it."""
    assert ("akkusativ", "A2") in _hits("Das Akkusativobjekt")
    assert ("akkusativ", "A2") in _hits("Akkusativergänzung")


def test_a_compound_is_not_matched_by_its_tail() -> None:
    """`Plusquamperfekt` is not in §5's table, so it must contribute nothing at all.

    Matching it as `perfekt` would silently label a B1 tense A2.
    """
    assert _hits("Das Plusquamperfekt") == []


def test_wechselpraeposition_does_not_fire_on_bare_praeposition() -> None:
    assert _hits("Präpositionen mit Dativ") == []
    assert ("wechselpräposition", "A2") in _hits("Wechselpräpositionen")


def test_matching_survives_unicode_normalisation_forms() -> None:
    """Same ADR-012 hazard, third place: a decomposed ä must still match.

    The decomposed form is *constructed*, not written as a literal. Two literals that
    look different in a source file are not reliably different bytes -- editors and
    tooling normalize on save, which would silently turn this into a tautology that
    passes while testing nothing.
    """
    precomposed = unicodedata.normalize("NFC", "Wechselpräpositionen")
    decomposed = unicodedata.normalize("NFD", "Wechselpräpositionen")
    assert precomposed != decomposed, "the two forms must genuinely differ in bytes"
    assert _hits(precomposed) == _hits(decomposed) != []


# --- grading: where the hit landed ---


def test_a_title_hit_shadows_body_hits_entirely() -> None:
    """A node titled `Perfekt` is about the Perfekt, whatever its body mentions."""
    hits = _grammar.grammar_anchor("Das Perfekt", "Hier auch Genitiv und Nominalstil.")
    level, winners = _grammar.strongest(hits)
    assert level == "A2"
    assert {w.where for w in winners} == {"title"}
    assert _grammar.is_title_anchored(hits) is True


def test_the_highest_structure_wins_within_a_scope() -> None:
    """You cannot study a node until you can handle its hardest structure."""
    hits = _grammar.grammar_anchor("Passiv und Präsens", "")
    assert _grammar.strongest(hits)[0] == "B1"


def test_body_only_hits_are_reported_as_body() -> None:
    hits = _grammar.grammar_anchor("Verben mit Präpositionen", "… steht mit dem Akkusativ …")
    level, winners = _grammar.strongest(hits)
    assert level == "A2"
    assert {w.where for w in winners} == {"body"}
    assert _grammar.is_title_anchored(hits) is False


def test_the_verben_mit_praepositionen_mislevel_regression() -> None:
    """The live-corpus case that motivated grading hits by location.

    This node is B1. Its body mentions *Akkusativ*, which §5 puts at A2. A flat sweep
    would report A2 as authoritative and downgrade a B1 rule into beginner material --
    and since ``cefr`` drives §5.1's study order, that reorders what gets learned. The
    anchor may still *report* A2, but it must mark it body-only so ``_cefr`` routes it to
    the tiebreak instead of overriding the node.
    """
    hits = _grammar.grammar_anchor(
        "Verben mit Präpositionen",
        "Viele Verben verlangen eine feste Präposition mit dem Akkusativ oder Dativ.",
    )
    assert _grammar.is_title_anchored(hits) is False, (
        "a title hit here would let A2 override the node's B1 without review"
    )


def test_no_hits_is_none_not_a_default_level() -> None:
    assert _grammar.strongest([]) is None
    assert _grammar.grammar_anchor("Die Wochentage", "Montag, Dienstag …") == []
