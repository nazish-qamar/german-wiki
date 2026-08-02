"""The grid: how axes derive, what the four states mean, and that dangling is computed."""

from __future__ import annotations

import pytest

from german_wiki.models import Link, Node
from german_wiki.morph._grid import build_grid, dangling_targets, is_prefix_node, morpheme_of


def _prefix(node_id: str, title: str, *, separable: bool = True, targets: list[str] = ()) -> Node:
    return Node(
        id=node_id,
        title_de=title,
        title_en=title,
        type="pattern",
        cefr="A2",
        status="stable",
        separable=separable,
        links=[Link(target=t, relation="same_family") for t in targets],
        body_md="",
    )


def _family(node_id: str, root: str, lemmas: list[str], transparency: str = "high") -> Node:
    return Node(
        id=node_id,
        title_de=root,
        title_en=root,
        type="vocab",
        cefr="A2",
        status="stable",
        root=root,
        lemmas=lemmas,
        family_transparency=transparency,
        body_md="",
    )


# --- axes ---


def test_a_pattern_node_with_a_separability_is_a_prefix_row() -> None:
    assert is_prefix_node(_prefix("prefix-an", "an- (Präfix)")) is True


def test_a_register_pattern_is_not_mistaken_for_a_prefix() -> None:
    """SPEC §6.3 register pairs are also `type: pattern` -- committing to `separable:` is
    what makes a pattern node a *morphological* one."""
    register_pair = Node(
        id="hoefliche-bitte",
        title_de="Höfliche Bitte über Register",
        title_en="Polite request across registers",
        type="pattern",
        cefr="A2",
        status="stable",
        body_md="",
    )
    assert register_pair.separable is None
    assert is_prefix_node(register_pair) is False
    assert build_grid([register_pair]).prefixes == []


@pytest.mark.parametrize(
    ("title", "root", "expected"),
    [("an- (Präfix)", None, "an"), ("auf- (Präfix)", None, "auf"), ("whatever", "ab", "ab")],
)
def test_the_morpheme_comes_from_root_or_the_title(title, root, expected) -> None:
    node = _prefix("p", title)
    node = node.model_copy(update={"root": root})
    assert morpheme_of(node) == expected


def test_columns_come_from_family_roots() -> None:
    grid = build_grid([_family("familie-waschen", "waschen", ["waschen"])])
    assert [r.root for r in grid.roots] == ["waschen"]
    assert grid.roots[0].node_id == "familie-waschen"


def test_columns_are_also_implied_by_what_a_prefix_node_claims() -> None:
    """Reading the human's assertion, not segmenting.

    `_segment` refuses `ankommen` for lack of a `kommen` node, so a grid built on
    segmentation could never display the links you actually wrote. The prefix node
    already claims `ankommen` is in its family; stripping its own morpheme lays that out.
    """
    grid = build_grid([_prefix("prefix-an", "an- (Präfix)", targets=["ankommen", "anmachen"])])
    implied = {r.root for r in grid.roots if r.node_id is None}
    assert implied == {"kommen", "machen"}


# --- the four states ---


def test_a_lemma_in_a_family_reads_as_learned() -> None:
    grid = build_grid(
        [
            _prefix("prefix-ab", "ab- (Präfix)"),
            _family("familie-waschen", "waschen", ["waschen", "abwaschen"]),
        ]
    )
    assert grid.cell("ab", "waschen").state == "learned"


def test_a_dangling_link_target_reads_as_identified() -> None:
    """SPEC §7.3's gap signal: written down as an intention, not yet written up."""
    grid = build_grid([_prefix("prefix-an", "an- (Präfix)", targets=["ankommen"])])
    assert grid.cell("an", "kommen").state == "identified"


def test_an_unattested_combination_reads_as_a_gap() -> None:
    grid = build_grid(
        [_prefix("prefix-an", "an- (Präfix)"), _family("f", "waschen", ["waschen"])]
    )
    assert grid.cell("an", "waschen").state == "gap"


@pytest.mark.parametrize("transparency", ["drifted", "opaque"])
def test_a_drifted_family_predicts_irregular_not_learnable(transparency) -> None:
    """§7.4: the grid only predicts a guess, so a drifted family's cells are watch-outs."""
    grid = build_grid(
        [
            _prefix("prefix-an", "an- (Präfix)"),
            _family("f", "stehen", ["stehen"], transparency=transparency),
        ]
    )
    assert grid.cell("an", "stehen").state == "irregular"


def test_a_variable_prefix_withholds_predictions_but_still_shows_what_is_attested() -> None:
    """An `um-` row may report attested words; it must not invent them.

    Predicting `umwaschen` as vocabulary asserts a separability the spelling does not
    carry -- the same withholding `_segment` applies, at the display layer.
    """
    grid = build_grid(
        [
            _prefix("prefix-um", "um- (Präfix)"),
            _family("f", "waschen", ["waschen"]),
            _family("g", "fahren", ["fahren", "umfahren"]),
        ]
    )
    assert grid.cell("um", "waschen").state == "withheld"  # predicted -> withheld
    assert grid.cell("um", "fahren").state == "learned"  # attested -> shown


# --- dangling is computed, never stored (the load-bearing one) ---


def test_dangling_targets_are_computed_from_the_current_corpus() -> None:
    nodes = [_prefix("prefix-an", "an- (Präfix)", targets=["ankommen"])]
    assert dangling_targets(nodes) == [("prefix-an", "same_family", "ankommen")]


def test_writing_the_target_flips_the_cell_with_no_migration() -> None:
    """The roadmap fills in as you study, and nothing has to be maintained.

    There is no `dangling` flag anywhere -- the state is recomputed from "does this node
    exist right now?". So creating the node is the *entire* fixup: no flag to clear, no
    migration step, no reindex-and-repair. This test is the reason the design refused to
    store the flag.
    """
    corpus = [_prefix("prefix-an", "an- (Präfix)", targets=["ankommen"])]
    assert build_grid(corpus).cell("an", "kommen").state == "identified"
    assert dangling_targets(corpus)

    # The only action taken: the node now exists.
    corpus.append(_family("ankommen", "ankommen", ["ankommen"]))

    grid = build_grid(corpus)
    assert grid.cell("an", "kommen").state == "learned"
    assert dangling_targets(corpus) == []


def test_a_derived_verb_does_not_become_a_column() -> None:
    """Found by looking at real output, not by reasoning about it.

    Any `type: vocab` node with `root:` would otherwise become a column — including a
    *prefixed* verb. Crossing `ankommen` with every prefix yields `anankommen`,
    `abankommen`, nonsense at every cell. SPEC §3.4 says the family earns the node and
    derived forms live inside it, so this only excludes mis-shaped input; the grid should
    still not produce garbage when given some.
    """
    corpus = [
        _prefix("prefix-an", "an- (Präfix)"),
        _family("familie-kommen", "kommen", ["kommen"]),
        _family("familie-ankommen", "ankommen", ["ankommen"]),  # a derived form
    ]
    roots = {r.root for r in build_grid(corpus).roots}
    assert "kommen" in roots
    assert "ankommen" not in roots


def test_an_empty_corpus_is_an_empty_grid_not_an_error() -> None:
    grid = build_grid([])
    assert grid.is_empty
    assert grid.cells == []
