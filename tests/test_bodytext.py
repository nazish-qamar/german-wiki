"""Parsing the ``## Examples`` section, and proving the extraction changed nothing.

``parse_examples`` was moved here out of ``merge/_regenerate.py`` so the Anki export could
read the gloss and tags that the merge drift-check discards. A move like that is only safe
if the old projection is bit-identical afterwards, so the first test in this file is a
**characterization test against the previous implementation**, inlined verbatim below
rather than against frozen expected strings.

That distinction matters. Pinning to hardcoded output would only prove the parser agrees
with whatever it produced the day someone wrote the strings down, and it would have to be
re-pinned by hand every time a fixture body gains a line -- at which point the pin no
longer proves anything about the refactor. Comparing against the old algorithm proves the
actual claim ("this move preserved behaviour") over any input at all.
"""

from __future__ import annotations

import re

import pytest

from german_wiki import storage
from german_wiki.bodytext import example_lines, parse_examples

# --- the previous implementation, copied verbatim from merge/_regenerate.py -------------
#
# Do not "clean this up" -- its value is being a frozen record of the behaviour that
# shipped in slices 5-8. If it and `bodytext` ever disagree, the parser regressed.

_LEGACY_EXAMPLES_HEADING = re.compile(r"^#{1,6}\s*(examples|beispiele)\b", re.IGNORECASE)
_LEGACY_HEADING = re.compile(r"^#{1,6}\s")
_LEGACY_BULLET = re.compile(r"^\s*[-*+]\s+")
_LEGACY_TRAILING_TAGS = re.compile(r"\s*\[[^\[\]]*\]\s*$")
_LEGACY_TRAILING_GLOSS = re.compile(r"\s*\([^()]*\)\s*$")


def _legacy_example_lines(body_md: str) -> list[str]:
    out: list[str] = []
    in_section = False
    for line in body_md.splitlines():
        if _LEGACY_EXAMPLES_HEADING.match(line.strip()):
            in_section = True
            continue
        if in_section and _LEGACY_HEADING.match(line.strip()):
            in_section = False
            continue
        if not in_section or not _LEGACY_BULLET.match(line):
            continue

        text = _LEGACY_BULLET.sub("", line).strip()
        text = _LEGACY_TRAILING_TAGS.sub("", text)
        text = _LEGACY_TRAILING_GLOSS.sub("", text)
        if text.strip():
            out.append(text.strip())
    return out


# Shapes chosen to exercise the branches, including ones the corpus does not have yet.
EDGE_CASES = [
    "",
    "No examples section at all.\n",
    "## Examples\n",  # heading, no bullets
    "## Examples\n- Eins.\n",
    "## Beispiele\n- Zwei.\n",  # the German heading
    "### examples\n- Drei.\n",  # different depth, lowercase
    "## Examples\n- Vier.\n\n## Notes\n- Not an example.\n",  # stops at next heading
    "## Examples\n* Stern.\n+ Plus.\n",  # other bullet markers
    "## Examples\n-   Viel   Abstand.   \n",  # whitespace
    "## Examples\n- Nur ein Gloss. (only a gloss)\n",
    "## Examples\n- Nur Tags. [alltag]\n",
    "## Examples\n- Beides. (gloss) [alltag, büro]\n",
    "## Examples\n- (nur ein Gloss)\n",  # strips to empty -> dropped entirely
    "## Examples\n- []\n",  # ditto
    "## Examples\n- Klammern (mitten) im Satz.\n",  # non-trailing parens are kept
    "## Examples\n- **Ganz fett.**\n",
    "Prose first.\n\n## Examples\n- Nach Prosa.\n",
]


@pytest.mark.parametrize("body", EDGE_CASES)
def test_the_move_preserved_example_lines_exactly(body: str) -> None:
    assert example_lines(body) == _legacy_example_lines(body)


def test_the_move_preserved_example_lines_on_every_real_node(seed_paths) -> None:
    """The same claim against real bodies, which is where the shapes actually vary."""
    for path in seed_paths.values():
        body = storage.load_node(path).body_md
        assert example_lines(body) == _legacy_example_lines(body), path.name


# --- what the new parser adds -----------------------------------------------------------


def test_it_splits_german_gloss_and_tags() -> None:
    body = "## Examples\n- Ich **wasche** mir die Hände. (I'm washing my hands.) [alltag, gesprochen]\n"
    (example,) = parse_examples(body)

    assert example.german == "Ich **wasche** mir die Hände."
    assert example.note == "I'm washing my hands."
    assert example.tags == ["alltag", "gesprochen"]


def test_the_parenthetical_is_not_assumed_to_be_english() -> None:
    """`wechselpräpositionen` annotates case, not translation -- hence `note`, not `english`.

    A card that labelled this "English" would put a German case label on the wrong side.
    """
    body = "## Examples\n- Ich stelle die Flasche **auf den** Tisch. (Akk — movement) [alltag]\n"
    (example,) = parse_examples(body)

    assert example.note == "Akk — movement"
    assert example.german.endswith("Tisch.")


def test_emphasis_survives_the_parse() -> None:
    """`**bold**` is the author's cloze assertion; stripping it here would destroy the
    only non-guessed target the export has."""
    body = "## Examples\n- Die **Wäsche** ist in der **Waschmaschine**.\n"
    (example,) = parse_examples(body)

    assert example.german.count("**") == 4


def test_missing_gloss_and_tags_are_absent_not_empty_strings() -> None:
    body = "## Examples\n- Ich denke an die neuen Projekte.\n"
    (example,) = parse_examples(body)

    assert example.note is None
    assert example.tags == []


def test_every_real_example_round_trips_to_a_non_empty_german_sentence(seed_paths) -> None:
    for path in seed_paths.values():
        for example in parse_examples(storage.load_node(path).body_md):
            assert example.german.strip(), path.name
            assert "[" not in example.german, f"tags leaked into german: {path.name}"
