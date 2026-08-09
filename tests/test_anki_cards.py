"""Card building: the dials, the cloze shape, the tags, and note identity.

Pure logic, no genanki and no filesystem -- that separation is why these tests can be
specific about note identity, which is the property the whole `.apkg` choice rests on.
"""

from __future__ import annotations

import ast
from typing import get_args

import pytest

from german_wiki import config, storage
from german_wiki.anki._cards import (
    Card,
    card_tags,
    cards_for,
    cloze_cards,
    guid_fold,
    guid_for,
    rule_card,
)
from german_wiki.anki._format import CARD_FORMATS, EXPORT_STATUSES, generators_for
from german_wiki.models import Node, NodeType

EXAMPLES = "\n## Examples\n- Ich **wasche** mir die Hände. (I'm washing my hands.) [alltag]\n"


def _node(**overrides) -> Node:
    data = {
        "id": "test-node",
        "title_de": "Test",
        "title_en": "Test EN",
        "type": "vocab",
        "cefr": "A2",
        "status": "stable",
        "body_md": f"Ein Körper.{EXAMPLES}",
        "source_ids": ["seed"],
    }
    data.update(overrides)
    return Node(**data)


# --- the dials ------------------------------------------------------------------------


def test_card_formats_covers_every_node_type() -> None:
    """Adding a sixth NodeType must fail HERE, not KeyError mid-export.

    Read off `models.NodeType` rather than a copied list, so the two cannot drift.
    """
    assert set(CARD_FORMATS) == set(get_args(NodeType))


def test_every_generator_named_in_the_table_is_one_cards_for_can_run() -> None:
    """AST, not a string search: `_cards.py`'s docstrings discuss both generator names."""
    source = (config.PROJECT_ROOT / "src" / "german_wiki" / "anki" / "_cards.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    handled = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in {"rule_basic", "examples_cloze"}
    }
    named = {generator for generators in CARD_FORMATS.values() for generator in generators}
    assert named <= handled


def test_draft_is_excluded_by_default() -> None:
    assert "draft" not in EXPORT_STATUSES
    assert set(EXPORT_STATUSES) == {"reviewed", "stable"}


def test_grammar_produces_both_a_rule_and_its_examples() -> None:
    assert generators_for("grammar") == ("rule_basic", "examples_cloze")


def test_pattern_nodes_produce_a_rule_card_even_with_no_examples() -> None:
    """`prefix-ab` has no Examples section at all; under a cloze-only design it exported
    nothing."""
    node = _node(
        type="pattern", body_md="| Verb | Meaning |\n|---|---|\n| abfahren | to depart |\n"
    )
    cards, _ = cards_for(node)

    assert [card.kind for card in cards] == ["basic"]


# --- cloze shape ----------------------------------------------------------------------


def test_the_bold_span_becomes_the_deletion() -> None:
    cards, skips = cloze_cards(_node())

    assert skips == []
    assert cards[0].fields[0] == "Ich {{c1::wasche}} mir die Hände."


def test_two_spans_share_c1_and_make_one_card() -> None:
    """Splitting a construction across c1/c2 would test half a Konjunktiv II frame."""
    body = "Text.\n\n## Examples\n- Die **Wäsche** ist in der **Waschmaschine**.\n"
    cards, _ = cloze_cards(_node(body_md=body))

    assert len(cards) == 1
    assert cards[0].fields[0].count("{{c1::") == 2
    assert "{{c2::" not in cards[0].fields[0]


def test_a_partial_cloze_keeps_its_gloss_off_the_front() -> None:
    """Prompt is front-side; showing the English next to a partial deletion leaks it."""
    (card,), _ = cloze_cards(_node())

    assert card.fields[1] == ""  # Prompt
    assert card.fields[2] == "I'm washing my hands."  # Note, back only


def test_a_fully_bold_sentence_gets_its_gloss_as_a_front_side_prompt() -> None:
    body = "Text.\n\n## Examples\n- **Kannst du mir mal kurz helfen?** (Can you help me a sec?)\n"
    (card,), _ = cloze_cards(_node(type="phrase", body_md=body))

    assert card.fields[1] == "Can you help me a sec?"
    assert card.fields[0].startswith("{{c1::")


def test_a_fully_bold_sentence_with_no_gloss_is_skipped_not_shipped_blank() -> None:
    """Its front would be a bare `[...]` -- nothing to recall from."""
    body = "Text.\n\n## Examples\n- **Kannst du mir helfen?**\n"
    cards, skips = cloze_cards(_node(type="phrase", body_md=body))

    assert cards == []
    assert [skip.reason for skip in skips] == ["no-prompt-for-full-cloze"]


def test_an_example_with_no_honest_target_is_skipped_and_reported() -> None:
    body = "Text.\n\n## Examples\n- Ich nehme an der Besprechung teil.\n"
    cards, skips = cloze_cards(_node(type="phrase", body_md=body))

    assert cards == []
    assert [skip.reason for skip in skips] == ["no-cloze-target"]
    assert skips[0].detail == "Ich nehme an der Besprechung teil."


def test_a_node_with_no_examples_section_is_reported_once() -> None:
    cards, skips = cloze_cards(_node(body_md="Nur Prosa, keine Beispiele."))

    assert cards == []
    assert [skip.reason for skip in skips] == ["no-examples"]


def test_emphasis_markers_never_reach_the_card() -> None:
    (card,), _ = cloze_cards(_node())

    assert "**" not in card.fields[0]


# --- tags -----------------------------------------------------------------------------


def test_tags_are_namespaced_and_carry_provenance() -> None:
    node = _node(register=["alltag", "büro"], themes=["küche"])
    tags = card_tags(node)

    assert "cefr::A2" in tags
    assert "type::vocab" in tags
    assert "node::test-node" in tags
    assert "register::alltag" in tags
    assert "theme::küche" in tags


def test_tags_never_contain_whitespace() -> None:
    """Anki tags are space-delimited, so a space would silently split one tag into two."""
    node = _node(register=["sehr formell"])
    assert all(" " not in tag for tag in card_tags(node))


def test_tags_are_sorted_and_deduplicated_so_re_export_is_byte_identical() -> None:
    node = _node(register=["alltag", "alltag"])
    tags = card_tags(node)

    assert tags == sorted(set(tags))


# --- note identity: the property .apkg was chosen for ----------------------------------


def test_the_guid_is_stable_across_reordering_examples() -> None:
    """`merge.new_examples()` appends on every SAME merge, so insertion is routine."""
    one = "A.\n\n## Examples\n- **Eins** hier.\n- **Zwei** dort.\n"
    two = "A.\n\n## Examples\n- **Zwei** dort.\n- **Eins** hier.\n"

    first = {card.guid for card in cloze_cards(_node(body_md=one))[0]}
    second = {card.guid for card in cloze_cards(_node(body_md=two))[0]}
    assert first == second


def test_the_guid_survives_a_tag_edit() -> None:
    plain = "A.\n\n## Examples\n- **Eins** hier.\n"
    tagged = "A.\n\n## Examples\n- **Eins** hier. [alltag, büro]\n"

    assert (
        cloze_cards(_node(body_md=plain))[0][0].guid
        == cloze_cards(_node(body_md=tagged))[0][0].guid
    )


def test_the_guid_survives_moving_which_word_is_bolded() -> None:
    """Same sentence, refined deletion -- an update, not a new card."""
    first = "A.\n\n## Examples\n- **Eins** hier.\n"
    second = "A.\n\n## Examples\n- Eins **hier**.\n"

    assert (
        cloze_cards(_node(body_md=first))[0][0].guid
        == cloze_cards(_node(body_md=second))[0][0].guid
    )


def test_the_guid_changes_when_the_sentence_changes() -> None:
    first = "A.\n\n## Examples\n- **Eins** hier.\n"
    second = "A.\n\n## Examples\n- **Eins** dort.\n"

    assert (
        cloze_cards(_node(body_md=first))[0][0].guid
        != cloze_cards(_node(body_md=second))[0][0].guid
    )


def test_the_rule_card_guid_survives_editing_the_rule() -> None:
    """Editing a rule must keep its scheduling, not restart the interval."""
    first = rule_card(_node(type="grammar", body_md="Erste Fassung."))
    second = rule_card(_node(type="grammar", body_md="Zweite, längere Fassung."))

    assert first.guid == second.guid


def test_two_nodes_never_share_a_guid() -> None:
    body = "A.\n\n## Examples\n- **Eins** hier.\n"
    a = cloze_cards(_node(id="node-a", body_md=body))[0][0]
    b = cloze_cards(_node(id="node-b", body_md=body))[0][0]

    assert a.guid != b.guid


def test_guid_fold_ignores_punctuation_and_case_but_not_words() -> None:
    assert guid_fold("Ich wasche mir die Hände.") == guid_fold("ich  wasche, mir die hände!")
    assert guid_fold("Ich wasche") != guid_fold("Ich wusch")


def test_guid_for_is_deterministic_across_calls() -> None:
    assert guid_for("n", "x") == guid_for("n", "x")


# --- the rule card --------------------------------------------------------------------


def test_the_rule_card_renders_its_body_as_html() -> None:
    node = _node(
        type="pattern", body_md="| Verb | Meaning |\n|---|---|\n| abfahren | to depart |\n"
    )
    card = rule_card(node)

    assert card.fields[0] == "Test"  # Front = title_de
    assert "<table>" in card.fields[1]


def test_the_rule_card_body_is_not_truncated() -> None:
    """Silently cutting a rule card drops the rule."""
    long_body = "Zeile.\n" * 200
    card = rule_card(_node(type="grammar", body_md=long_body))

    assert card.fields[1].count("Zeile.") == 200


# --- against the real corpus ------------------------------------------------------------


def test_every_seed_node_produces_cards_or_an_explained_skip(seed_paths) -> None:
    for path in seed_paths.values():
        node = storage.load_node(path)
        cards, skips = cards_for(node)
        assert cards or skips, path.name
        for card in cards:
            assert all(isinstance(field, str) for field in card.fields)
            assert card.guid


@pytest.mark.parametrize("node_type", sorted(get_args(NodeType)))
def test_cards_for_never_raises_on_any_node_type(node_type: str) -> None:
    cards, _ = cards_for(_node(type=node_type))
    assert isinstance(cards, list)
    assert all(isinstance(card, Card) for card in cards)
