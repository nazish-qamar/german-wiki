"""German verb prefixes, in **three** inventories (SPEC §7.1, §7.4).

Two would be the obvious design — separable and inseparable — and it would be wrong. A
third class exists whose separability is not recoverable from the text at all:

    úmfahren   run over      separable    "ich fahre den Baum um"
    umfáhren   drive around  inseparable  "ich umfahre den Baum"

**Identical spelling.** The distinction is stress, and stress is not written. So for these
prefixes the segmenter cannot know which verb it is looking at, and neither can the grid.

The consequence is the whole reason this module splits three ways rather than annotating
a boolean: a variable-prefix verb yields **no segmentation and no grid cell at all** --
not a cell carrying a caveat. A grid that says "here is a family, caveat" still teaches
the family; a grid that says nothing prompts you to decide. Stress-homographs are exactly
where a silent segmentation does the §7.4 damage (`verstehen` is not `ver-` + `stehen`),
so the correct behaviour is to withhold the claim, not to qualify it.

That is structural here rather than a flag downstream, because a flag is something a
later change can quietly start ignoring. Membership in ``VARIABLE`` *is* the withholding.

**One escape hatch, and it is the human.** SPEC §7.4: "The node holds the truth; the grid
only predicts a guess." When a verb node carries an explicit ``separable: true|false``,
that statement overrides this inventory and the verb grids normally -- the ambiguity was
resolved by someone who knows which word it is. See ``_segment.segment``.
"""

from __future__ import annotations

from typing import Literal

Separability = Literal["separable", "inseparable", "variable"]

# Always separable, always stressed on the prefix. These carry the directional logic
# SPEC §7 is built around -- an- = toward/on, aus- = out/off, auf- = up/open, ab- = away.
#
# Ordered longest-first for matching: "vorbei" must be tried before "vor", or
# `vorbeikommen` segments as `vor` + `beikommen`.
SEPARABLE: tuple[str, ...] = (
    "auseinander",
    "gegenüber",
    "entgegen",
    "herunter",
    "herüber",
    "hinunter",
    "hinüber",
    "zusammen",
    "zurecht",
    "zurück",
    "vorbei",
    "voraus",
    "vorher",
    "herein",
    "heraus",
    "herauf",
    "herab",
    "hinein",
    "hinaus",
    "hinauf",
    "davon",
    "dazu",
    "empor",
    "fest",
    "statt",
    "teil",
    "weiter",
    "wieder",  # see VARIABLE note below -- listed there too, deliberately
    "nach",
    "über",  # variable; see below
    "unter",  # variable; see below
    "durch",  # variable; see below
    "hinter",  # variable; see below
    "voll",  # variable; see below
    "wider",  # variable; see below
    "auf",
    "aus",
    "bei",
    "ein",
    "fort",
    "her",
    "hin",
    "los",
    "mit",
    "vor",
    "weg",
    "zu",
    "ab",
    "an",
    "um",  # variable; see below
)

# Never separable, never stressed on the prefix. Short, closed, and genuinely fixed --
# this is the one list in German morphology that behaves.
INSEPARABLE: tuple[str, ...] = (
    "miss",
    "emp",
    "ent",
    "ver",
    "zer",
    "be",
    "ge",
    "er",
)

# **Stress-dependent: the same spelling is both.** Membership here withholds the grid
# claim entirely (see the module docstring). These also appear in SEPARABLE above,
# because they *are* separable in one of their two readings -- ``classify`` checks this
# set first, so the variable answer wins and the SEPARABLE entry never decides anything
# on its own.
VARIABLE: frozenset[str] = frozenset(
    {"um", "durch", "über", "unter", "wider", "wieder", "hinter", "voll"}
)


def classify(prefix: str) -> Separability | None:
    """Which inventory ``prefix`` belongs to, or ``None`` if it is not a verb prefix.

    ``VARIABLE`` is tested first on purpose: several of its members are also listed in
    ``SEPARABLE`` (they are separable in one reading), and answering "separable" for
    ``um`` would hand the segmenter a confidence the spelling does not support.
    """
    p = prefix.casefold()
    if p in VARIABLE:
        return "variable"
    if p in INSEPARABLE:
        return "inseparable"
    if p in SEPARABLE:
        return "separable"
    return None


def candidates(word: str) -> list[tuple[str, str, Separability]]:
    """Every ``(prefix, stem, separability)`` split of ``word``, longest prefix first.

    Purely mechanical: it reports what *could* be a prefix, and says nothing about
    whether the split is real. ``_segment`` applies the corpus-evidence test that turns a
    candidate into a proposal, which is what keeps `verstehen` from becoming a `ver-`
    family (SPEC §7.4).

    A one- or two-letter remainder is not a German verb stem, so those are dropped here
    rather than burdening the caller.
    """
    lowered = word.casefold().strip()
    found = []
    for prefix in SEPARABLE + INSEPARABLE:
        if not lowered.startswith(prefix):
            continue
        stem = lowered[len(prefix) :]
        if len(stem) < 3:
            continue
        kind = classify(prefix)
        if kind is not None:
            found.append((prefix, stem, kind))
    # Longest prefix first, and de-duplicated: `um` appears in SEPARABLE and is caught by
    # VARIABLE, so without this `umfahren` would report the same split twice.
    seen: set[str] = set()
    ordered = []
    for prefix, stem, kind in sorted(found, key=lambda t: -len(t[0])):
        if prefix in seen:
            continue
        seen.add(prefix)
        ordered.append((prefix, stem, kind))
    return ordered
