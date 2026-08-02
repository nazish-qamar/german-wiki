"""The root × prefix grid, and the gaps in it (SPEC §7, §7.3, §7.4).

SPEC §7's worked table is the target::

    | Prefix | + kommen        | + machen         | + stellen           |
    | an-    | ankommen        | anmachen         | anstellen           |
    | auf-   | aufkommen       | aufmachen        | aufstellen          |

and §7.3 says the empty cells are the point: "**Empty cells are words you haven't learned
yet** -- the view doubles as gap detection."

**Where the columns come from, and why it is not a guess.** Corpus roots alone are not
enough: today that yields one column (``waschen``) crossed with one row (``an-``), whose
only cell is the non-word ``anwaschen``. The columns that make the seed data render like
SPEC's table come from the prefix node's *own* ``same_family`` links -- ``prefix-an`` says
``ankommen`` belongs to it, so stripping ``an`` yields the column ``kommen``.

That is **reading a human's assertion, not segmenting**. It sidesteps the circularity that
would otherwise bite: ``_segment`` refuses ``ankommen`` for lack of a ``kommen`` node, so
a grid built on segmentation could never show the very links you wrote. Here the node
already made the claim; the grid only lays it out.

**Dangling-ness is computed at read time and never stored.** A cell is "identified"
because a link points at a node id that does not exist *right now*. There is no flag, so
the day you write ``nodes/ankommen.md`` the same cell reads "learned" with no migration and
nothing to clear -- the roadmap fills in as you study.

**Variable-stress rows withhold their predictions** (see ``_prefixes``). In an ``um-`` row,
attested cells still show, but *predicted* ones do not become gaps: proposing ``umwaschen``
as a word to learn asserts a separability the spelling does not carry.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..logutil import get_logger
from ..models import Node
from ._prefixes import Separability, classify
from ._segment import CorpusIndex, Segmentation, fold, segment

logger = get_logger(__name__)

CellState = Literal["learned", "identified", "gap", "irregular", "withheld"]

# `an- (Präfix)` -> `an`. Only a fallback: a prefix node may state its morpheme in `root:`,
# which is unambiguous and is preferred.
_TITLE_MORPHEME = re.compile(r"^\s*([^\s(-]+)\s*-")

# §7.4's safeguard. A family whose meanings have drifted is not a grid to learn from, so
# its *predicted* cells are marked rather than presented as vocabulary.
_UNTRUSTWORTHY_FAMILIES = frozenset({"drifted", "opaque"})


class PrefixAxis(BaseModel):
    """One row: a prefix node that has committed to a separability."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    morpheme: str
    separability: Separability
    declared_separable: bool

    @property
    def label(self) -> str:
        return f"{self.morpheme}-"

    @property
    def withholds_predictions(self) -> bool:
        """A stress-ambiguous prefix may show what is attested, but must not predict.

        The node's ``separable:`` resolved the prefix *in general*; it cannot resolve
        which reading an individual unattested verb would take.
        """
        return self.separability == "variable"


class RootAxis(BaseModel):
    """One column. ``node_id`` is None when the root is implied by a link, not written."""

    model_config = ConfigDict(extra="forbid")

    root: str
    node_id: str | None = None
    transparency: str | None = None

    @property
    def trusted(self) -> bool:
        return self.transparency not in _UNTRUSTWORTHY_FAMILIES


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefix: str
    root: str
    word: str
    state: CellState
    node_id: str | None = None


class Grid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prefixes: list[PrefixAxis] = Field(default_factory=list)
    roots: list[RootAxis] = Field(default_factory=list)
    cells: list[Cell] = Field(default_factory=list)

    def cell(self, prefix: str, root: str) -> Cell | None:
        for c in self.cells:
            if c.prefix == prefix and c.root == root:
                return c
        return None

    def by_state(self, state: CellState) -> list[Cell]:
        return [c for c in self.cells if c.state == state]

    @property
    def is_empty(self) -> bool:
        return not self.prefixes or not self.roots


def morpheme_of(node: Node) -> str | None:
    """The prefix a prefix-node is about: ``root:`` if stated, else parsed from the title."""
    if node.root:
        return fold(node.root).rstrip("-")
    match = _TITLE_MORPHEME.match(node.title_de)
    return fold(match.group(1)) if match else None


def is_prefix_node(node: Node) -> bool:
    """``type: pattern`` **with a separability**.

    The second half matters: SPEC §6.3's register pairs are also ``type: pattern`` and
    would otherwise be mistaken for prefixes. Committing to ``separable:`` is what makes a
    pattern node a *morphological* one.
    """
    return node.type == "pattern" and node.separable is not None


def is_family_node(node: Node) -> bool:
    return node.type == "vocab" and bool(node.root)


def dangling_targets(nodes: list[Node]) -> list[tuple[str, str, str]]:
    """``(source_id, relation, target)`` for every link whose target does not exist.

    Computed, never stored -- which is the whole reason a target appearing later needs no
    fixup. These are SPEC §7.3's gap signal: a human wrote them as an intention.
    """
    ids = {n.id for n in nodes}
    return [
        (n.id, lk.relation, lk.target)
        for n in nodes
        for lk in n.links
        if lk.target not in ids
    ]


def _known_words(nodes: list[Node]) -> dict[str, str | None]:
    """Every word the corpus can be said to hold: node ids, and family lemmas."""
    known: dict[str, str | None] = {}
    for node in nodes:
        for lemma in node.lemmas or []:
            known.setdefault(fold(lemma), node.id)
    for node in nodes:
        known[fold(node.id)] = node.id
    return known


def build_grid(nodes: list[Node]) -> Grid:
    """Assemble the matrix. Pure read: nothing is written, proposed, or inferred."""
    ids = {n.id for n in nodes}
    link_targets = {lk.target for n in nodes for lk in n.links}
    known = _known_words(nodes)

    prefixes: list[PrefixAxis] = []
    for node in nodes:
        if not is_prefix_node(node):
            continue
        morpheme = morpheme_of(node)
        if not morpheme:
            continue
        prefixes.append(
            PrefixAxis(
                node_id=node.id,
                morpheme=morpheme,
                separability=classify(morpheme) or "separable",
                declared_separable=bool(node.separable),
            )
        )
    prefixes.sort(key=lambda p: p.morpheme)

    # A column must be a *base* root. A node whose own root is itself a prefixed verb is a
    # derived form, and crossing it with every prefix yields `anankommen`, `abankommen`
    # and so on -- nonsense at every cell. SPEC §3.4 says the family (shared stem) earns
    # the node and derived forms live inside it, so this excludes only mis-shaped input --
    # but the grid should not produce garbage when given some.
    corpus_index = CorpusIndex.build(nodes)
    roots: dict[str, RootAxis] = {}
    for node in nodes:
        if not is_family_node(node):
            continue
        key = fold(node.root or "")
        if isinstance(segment(key, corpus=corpus_index), Segmentation):
            logger.info(
                "%s has root %r, which is itself %s -- not a grid column",
                node.id,
                node.root,
                "a prefixed form",
            )
            continue
        roots[key] = RootAxis(
            root=key, node_id=node.id, transparency=node.family_transparency
        )

    # Columns implied by what the prefix nodes themselves claim (see module docstring).
    for node in nodes:
        if not is_prefix_node(node):
            continue
        morpheme = morpheme_of(node)
        if not morpheme:
            continue
        for link in node.links:
            if link.relation != "same_family":
                continue
            target = fold(link.target)
            if not target.startswith(morpheme):
                continue
            implied = target[len(morpheme) :]
            if len(implied) >= 3:
                roots.setdefault(implied, RootAxis(root=implied))

    ordered_roots = sorted(roots.values(), key=lambda r: (r.node_id is None, r.root))

    cells: list[Cell] = []
    for prefix in prefixes:
        for root in ordered_roots:
            word = f"{prefix.morpheme}{root.root}"
            if word in known:
                state: CellState = "learned"
                node_id = known[word]
            elif word in link_targets and word not in ids:
                # Written down as an intention, not yet written up. §7.3's gap signal.
                state, node_id = "identified", None
            elif prefix.withholds_predictions:
                state, node_id = "withheld", None
            elif not root.trusted:
                # §7.4: the grid only predicts a guess, and this family's meanings have
                # drifted -- so the cell is a watch-out, not a suggestion.
                state, node_id = "irregular", None
            else:
                state, node_id = "gap", None
            cells.append(
                Cell(prefix=prefix.morpheme, root=root.root, word=word, state=state, node_id=node_id)
            )

    return Grid(prefixes=prefixes, roots=ordered_roots, cells=cells)
