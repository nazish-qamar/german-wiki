"""The cloze-target rungs: bold, then the node's own assertions, then refuse.

This is the one part of the export that cannot be fixed later by editing a constant. A
wrong deletion produces a card that looks real, enters the review rotation, and teaches
nothing or something false -- with no error and no visible gap. So the tests here are
mostly about what the module *declines* to do.

There is deliberately no test asserting that some arbitrary word got blanked, because that
behaviour must not exist.
"""

from __future__ import annotations

import pytest

from german_wiki import storage
from german_wiki.anki._targets import (
    PREPOSITIONS,
    bold_terms,
    cloze_targets,
    governed_prepositions,
    strip_emphasis,
    structural_terms,
)
from german_wiki.models import Node


def _node(**overrides) -> Node:
    data = {
        "id": "test-node",
        "title_de": "Test",
        "title_en": "Test",
        "type": "vocab",
        "cefr": "A2",
        "status": "stable",
        "body_md": "Body.",
        "source_ids": ["seed"],
    }
    data.update(overrides)
    return Node(**data)


# --- rung 1: bold wins --------------------------------------------------------------


def test_bold_is_the_target() -> None:
    node = _node()
    targets = cloze_targets(node, "Ich **wasche** mir die Hände.")

    assert targets is not None
    assert targets.terms == ["wasche"]
    assert targets.source == "bold"


def test_two_bold_spans_are_two_terms_of_one_deletion() -> None:
    node = _node()
    targets = cloze_targets(node, "Die **Wäsche** ist in der **Waschmaschine**.")

    assert targets is not None
    assert targets.terms == ["Wäsche", "Waschmaschine"]


def test_bold_beats_an_available_structural_target() -> None:
    """Precedence, not merger: the author's mark is an assertion and overrides the rule."""
    node = _node(lemmas=["Waschmaschine"])
    targets = cloze_targets(node, "Die **Wäsche** ist in der Waschmaschine.")

    assert targets is not None
    assert targets.source == "bold"
    assert targets.terms == ["Wäsche"]


# --- rung 2, vocab: the node's own lemmas -------------------------------------------


def test_a_vocab_node_blanks_its_own_lemma() -> None:
    node = _node(type="vocab", lemmas=["die Wäsche", "die Waschmaschine"])
    targets = cloze_targets(node, "Die Wäsche ist noch in der Waschmaschine.")

    assert targets is not None
    assert targets.source == "structure"
    assert targets.terms == ["Wäsche", "Waschmaschine"]


def test_the_root_counts_as_a_lemma() -> None:
    node = _node(type="vocab", root="waschen")
    targets = cloze_targets(node, "Wir waschen das Auto.")

    assert targets is not None
    assert targets.terms == ["waschen"]


def test_a_multi_word_lemma_matches_as_a_phrase() -> None:
    node = _node(type="vocab", lemmas=["guten Tag"])
    targets = cloze_targets(node, "Er sagt guten Tag zu mir.")

    assert targets is not None
    assert targets.terms == ["guten Tag"]


def test_an_inflected_form_is_not_matched_and_falls_through_cleanly() -> None:
    """The documented refusal. `wasche` is not `waschen`, and guessing that it is would
    open the door to `wäschst` -> `waschen`, which needs ablaut folding and starts
    producing wrong deletions silently."""
    node = _node(type="vocab", lemmas=["waschen"])

    assert cloze_targets(node, "Ich wasche mir die Hände.") is None


def test_a_vocab_node_with_no_lemmas_refuses() -> None:
    assert cloze_targets(_node(type="vocab"), "Ein Satz ohne alles.") is None


# --- rung 2, grammar: only prepositions the node ANNOTATES ---------------------------

GOVERNING_BODY = (
    "- teilnehmen an (+ Dativ): to participate in\n"
    "- denken an (+ Akkusativ): to think about\n"
    "- warten auf (+ Akkusativ): to wait for\n"
)


def test_governed_prepositions_reads_the_case_annotation() -> None:
    assert governed_prepositions(GOVERNING_BODY) == {"an", "auf"}


@pytest.mark.parametrize(
    "body",
    [
        "denken an (+ Akkusativ)",
        "denken an (Akkusativ)",
        "denken an + Akkusativ",
        "denken an (+ Akk.)",
    ],
)
def test_the_common_spellings_of_the_assertion_are_all_read(body: str) -> None:
    assert "an" in governed_prepositions(body)


def test_a_grammar_node_blanks_only_the_preposition_it_names() -> None:
    """`vom` is a preposition and is in the sentence -- and is left alone, because the
    node never claimed anything about it."""
    node = _node(type="grammar", body_md=GOVERNING_BODY)
    targets = cloze_targets(node, "Ich warte auf die Entscheidung vom Chef.")

    assert targets is not None
    assert targets.source == "structure"
    assert targets.terms == ["auf"]
    assert "vom" not in targets.terms


def test_a_bare_preposition_in_prose_is_not_an_assertion() -> None:
    """The guard against blanking `in` on a node that is not about prepositions at all."""
    node = _node(
        type="grammar",
        body_md="Konjunktiv II wird in der höflichen Bitte verwendet. Man nutzt es oft.",
    )

    assert governed_prepositions(node.body_md) == set()
    assert cloze_targets(node, "Ich würde gern in die Stadt gehen.") is None


def test_a_word_outside_the_inventory_is_never_selected() -> None:
    """The closed set is what makes rung 2 safe rather than a guess."""
    assert governed_prepositions("teilnehmen banane (+ Dativ)") == set()
    assert "banane" not in PREPOSITIONS


# --- rung 3: refusal -----------------------------------------------------------------


@pytest.mark.parametrize("node_type", ["phrase", "pattern", "culture"])
def test_types_with_no_structural_rule_refuse_without_bold(node_type: str) -> None:
    node = _node(type=node_type, body_md=GOVERNING_BODY, lemmas=["waschen"])

    assert structural_terms(node, "Ein Satz ohne Fettdruck.") == []
    assert cloze_targets(node, "Ein Satz ohne Fettdruck.") is None


def test_refusal_is_none_not_an_empty_deletion() -> None:
    """An empty terms list would build a Cloze note with no deletion, which Anki treats
    as an error -- so the refusal has to be visible to the caller, not falsy-but-present."""
    assert cloze_targets(_node(type="phrase"), "Nichts hier.") is None


# --- helpers -------------------------------------------------------------------------


def test_strip_emphasis_leaves_the_plain_sentence() -> None:
    assert strip_emphasis("Ich **wasche** *mir* die Hände.") == "Ich wasche mir die Hände."


def test_bold_terms_is_empty_for_an_unmarked_sentence() -> None:
    assert bold_terms("Ich denke an die neuen Projekte.") == []


# --- against the real corpus ---------------------------------------------------------


def test_every_real_seed_example_resolves_or_refuses_without_error(seed_paths) -> None:
    """Whatever the answer, the rungs must not raise on real material."""
    from german_wiki.bodytext import parse_examples

    for path in seed_paths.values():
        node = storage.load_node(path)
        for example in parse_examples(node.body_md):
            result = cloze_targets(node, example.german)
            assert result is None or result.terms, path.name
