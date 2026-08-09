"""Reading structure back out of a node's Markdown body.

A node body is Markdown by design (SPEC §1.2: files are the source of truth, and a file
you cannot read in a text editor is not one). That makes the body the *canonical* form and
everything else -- the SQLite index, the embedding, an Anki card -- a derived view of it.
This module is where those views get their structure from.

Today it reads one construct: the ``## Examples`` section. That section is the highest-value
part of a body to parse, because SPEC §8.1 wants the example sentences cloze-deleted onto
flashcards and SPEC §12.1 wants them checked against ``/raw`` for drift -- two consumers,
two different projections of the same lines.

The line format is fixed by the extraction prompt::

    - German sentence (gloss or note) [tag, tag]

and the two trailing parts are optional. Both are captured here rather than discarded,
because *which* parts a consumer wants differs: the drift check in ``merge`` wants the
German alone, while a flashcard wants the note on its back and the tags on the card. A
parser that threw them away forced the second consumer to re-implement the format, which is
how one line shape ends up with two definitions that quietly disagree.

**The parenthetical is not reliably a translation.** ``wechselpräpositionen`` writes
``(Akk -- movement)`` and ``(Dat -- location)``, which are grammatical annotations, not
English. Hence the neutral field name ``note``: anything that renders it as "English" will
eventually put a German case label on the wrong side of a card.

Nothing here writes, calls a model, or validates. It answers one question about one body.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

# The section this module can find. `beispiele` is accepted because a hand-written node may
# reasonably use the German heading, and a body that used it would otherwise silently
# contribute no examples to either consumer.
_EXAMPLES_HEADING = re.compile(r"^#{1,6}\s*(examples|beispiele)\b", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s")
_BULLET = re.compile(r"^\s*[-*+]\s+")

# Anchored at the end and stripped in this order -- tags sit outside the gloss:
#     Ich **wasche** mir die Hände. (I'm washing my hands.) [alltag, gesprochen]
#                                   \__________ gloss ____/ \______ tags ______/
# The capture groups are the only change from the originals in `merge/_regenerate.py`;
# adding a group does not alter what the pattern matches, so `example_lines` still sees
# exactly the text it saw before.
_TRAILING_TAGS = re.compile(r"\s*\[([^\[\]]*)\]\s*$")
_TRAILING_GLOSS = re.compile(r"\s*\(([^()]*)\)\s*$")


class Example(BaseModel):
    """One bullet from a ``## Examples`` section, split into its parts."""

    model_config = ConfigDict(extra="forbid")

    # Emphasis markers are kept. `**bold**` is the author's assertion about which part of
    # the sentence is being taught, and it is the only non-guessed cloze target there is --
    # stripping it here would destroy that signal before any consumer saw it.
    german: str
    # The trailing `(...)`: a gloss OR a grammatical note. Never assume it is English.
    note: str | None = None
    # The trailing `[...]`, split on commas. Register/context labels per SPEC §6.2.
    tags: list[str] = []


def parse_examples(body_md: str) -> list[Example]:
    """The example bullets in ``body_md``'s ``## Examples`` section, parsed.

    Scoped to that section on purpose. Prose elsewhere in a body is explanation, which is
    legitimately paraphrased on merge; example sentences are where invention actually hurts,
    because a fabricated sentence is a fact you would memorize.
    """
    out: list[Example] = []
    in_section = False
    for line in body_md.splitlines():
        if _EXAMPLES_HEADING.match(line.strip()):
            in_section = True
            continue
        if in_section and _HEADING.match(line.strip()):
            in_section = False
            continue
        if not in_section or not _BULLET.match(line):
            continue

        text = _BULLET.sub("", line).strip()

        tags: list[str] = []
        if match := _TRAILING_TAGS.search(text):
            tags = [tag.strip() for tag in match.group(1).split(",") if tag.strip()]
            text = text[: match.start()]

        note: str | None = None
        if match := _TRAILING_GLOSS.search(text):
            note = match.group(1).strip() or None
            text = text[: match.start()]

        if german := text.strip():
            out.append(Example(german=german, note=note, tags=tags))
    return out


def example_lines(body_md: str) -> list[str]:
    """The German example sentences alone -- the projection ``merge`` wants.

    Kept as a named function rather than inlined at its call sites: SPEC §12.1's drift
    check is about *sentences*, and spelling that as an attribute access would make the
    next reader wonder whether the note and tags were meant to be included.
    """
    return [example.german for example in parse_examples(body_md)]
