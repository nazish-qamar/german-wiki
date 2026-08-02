"""Segmentation: what splits, what is withheld, and why the difference is not a warning."""

from __future__ import annotations

import pytest

from german_wiki.models import Node
from german_wiki.morph import _prefixes
from german_wiki.morph._segment import CorpusIndex, Segmentation, Withheld, segment


def _family(node_id: str, root: str, lemmas: list[str]) -> Node:
    return Node(
        id=node_id,
        title_de=root,
        title_en=root,
        type="vocab",
        cefr="A2",
        status="stable",
        root=root,
        lemmas=lemmas,
        body_md="",
    )


@pytest.fixture
def corpus() -> CorpusIndex:
    """A corpus that knows `waschen` and nothing else -- like the real one."""
    return CorpusIndex.build(
        [_family("familie-waschen", "waschen", ["waschen", "abwaschen", "die Wäsche"])]
    )


# --- the three inventories ---


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("an", "separable"),
        ("ab", "separable"),
        ("auf", "separable"),
        ("ver", "inseparable"),
        ("be", "inseparable"),
        ("um", "variable"),
        ("durch", "variable"),
        ("über", "variable"),
        ("wider", "variable"),
        ("xyz", None),
    ],
)
def test_prefixes_classify_into_three_inventories(prefix, expected) -> None:
    assert _prefixes.classify(prefix) == expected


def test_variable_beats_separable_for_the_same_string() -> None:
    """`um` is listed in SEPARABLE too -- it IS separable in one of its two readings.

    Answering "separable" would hand the segmenter a confidence the spelling does not
    support, so VARIABLE is tested first and wins.
    """
    assert "um" in _prefixes.SEPARABLE
    assert "um" in _prefixes.VARIABLE
    assert _prefixes.classify("um") == "variable"


def test_longest_prefix_is_tried_first() -> None:
    """Otherwise `vorbeikommen` segments as vor- + beikommen."""
    first = _prefixes.candidates("vorbeikommen")[0]
    assert first[0] == "vorbei"


def test_a_candidate_split_is_not_a_claim() -> None:
    """`candidates` is mechanical: it reports `verstehen` = ver- + stehen quite happily.

    That is the point of the corpus-evidence test in `segment` -- the mechanical layer is
    allowed to be naive because a later layer refuses to act on it.
    """
    assert ("ver", "stehen", "inseparable") in _prefixes.candidates("verstehen")


# --- corpus evidence (SPEC §7.4) ---


def test_a_stem_the_corpus_knows_splits(corpus) -> None:
    result = segment("abwaschen", corpus=corpus)
    assert isinstance(result, Segmentation)
    assert (result.prefix, result.stem) == ("ab", "waschen")
    assert result.stem_node_id == "familie-waschen"  # the evidence is cited


def test_verstehen_does_not_split(corpus) -> None:
    """The §7.4 trap: mechanically splittable, semantically a coincidence.

    The corpus has no `stehen`, so nothing is proposed. Note what this is *not*: it is
    not a judgment that `verstehen` is opaque -- it is a refusal to guess at all.
    """
    result = segment("verstehen", corpus=corpus)
    assert isinstance(result, Withheld)
    assert result.reason == "no-corpus-evidence"


def test_prefix_shaped_starts_are_never_blindly_stripped(corpus) -> None:
    for word in ("bekommen", "vergessen", "entstehen"):
        assert isinstance(segment(word, corpus=corpus), Withheld)


def test_no_corpus_evidence_resolves_itself_as_you_study(corpus) -> None:
    """`ankommen` waits for a `kommen` node rather than needing intervention."""
    assert segment("ankommen", corpus=corpus).needs_human is False

    grown = CorpusIndex.build(
        [
            _family("familie-waschen", "waschen", ["waschen"]),
            _family("familie-kommen", "kommen", ["kommen"]),
        ]
    )
    assert isinstance(segment("ankommen", corpus=grown), Segmentation)


def test_lemmas_vouch_for_a_stem_not_only_roots() -> None:
    """Families store members as lemmas, so the corpus knows more than its roots."""
    idx = CorpusIndex.build([_family("f", "waschen", ["waschen", "die Wäsche"])])
    assert idx.vouches_for("wäsche") == "f"  # article stripped on the way in
    assert idx.vouches_for("Wäsche") == "f"  # and lookup is folded


# --- the withheld claim (the load-bearing behaviour) ---


def test_a_stress_ambiguous_prefix_yields_no_segmentation(corpus) -> None:
    """úmfahren vs umfáhren are spelled identically; the text lacks the answer.

    Asserted as the ABSENCE of a split -- not the presence of a warning on one. A grid
    that says "here is a family, caveat" still teaches the family.
    """
    result = segment("umwaschen", corpus=corpus)
    assert isinstance(result, Withheld)
    assert result.reason == "variable-stress"
    assert not isinstance(result, Segmentation)


def test_stress_ambiguity_is_withheld_even_when_the_stem_is_known(corpus) -> None:
    """The refusal is about stress, so corpus evidence must not override it.

    This is the case that would regress if someone "improved" the ordering: `waschen` IS
    vouched for, so every other check passes and only the variable-stress rule stops it.
    """
    assert corpus.vouches_for("waschen") is not None
    assert segment("umwaschen", corpus=corpus).reason == "variable-stress"


def test_only_stress_ambiguity_asks_for_a_human(corpus) -> None:
    assert segment("umwaschen", corpus=corpus).needs_human is True
    assert segment("verstehen", corpus=corpus).needs_human is False


@pytest.mark.parametrize(
    ("declared", "expected"), [(True, "separable"), (False, "inseparable")]
)
def test_an_explicit_separable_on_the_node_unlocks_gridding(corpus, declared, expected) -> None:
    """SPEC §7.4: "The node holds the truth; the grid only predicts a guess."

    A person who knows which word this is has answered what the spelling cannot, so the
    inventory defers to them.
    """
    result = segment("umwaschen", corpus=corpus, declared_separable=declared)
    assert isinstance(result, Segmentation)
    assert result.separability == expected
    assert result.resolved_by_node is True


def test_a_declaration_does_not_invent_corpus_evidence(corpus) -> None:
    """The override answers the stress question only -- it is not a general bypass."""
    assert isinstance(segment("umstehen", corpus=corpus, declared_separable=True), Withheld)


def test_a_word_with_no_prefix_is_not_an_error(corpus) -> None:
    result = segment("waschen", corpus=corpus)
    assert isinstance(result, Withheld)
    assert result.reason == "no-prefix"
