"""The card-back renderer, and the one property that actually matters: nothing is lost.

`_html` converts four Markdown constructs and passes everything else through. The
passthrough is the point, not a shortcut: a rule card exists to teach you the rule, so an
unhandled construct that silently vanished would be the same failure as truncating the
body -- and worse than truncation, because there would be no visible gap to notice.

So the central test here is not "does bold render". It is a **token-level no-drop
invariant**, parametrized over constructs the corpus does not contain yet, because the
bodies will grow new ones and the untested case is exactly where a parser-shaped
implementation would start losing text.
"""

from __future__ import annotations

import html
import re
from collections import Counter

import pytest

from german_wiki import storage
from german_wiki.anki._html import to_html

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"\w+", re.UNICODE)


def _visible(rendered: str) -> str:
    """The text a reader sees: tags removed, entities decoded."""
    return html.unescape(_TAG.sub(" ", rendered))


def _words(text: str) -> Counter[str]:
    return Counter(_WORD.findall(text))


def _assert_nothing_dropped(body: str) -> None:
    """Every word in the source appears at least as often in the rendered output.

    A multiset rather than a set: a word written twice that renders once has still lost
    content. Markdown markers (`*`, `|`, `#`, `>`) contain no word characters, so their
    removal never trips this -- which is what lets the same invariant cover both handled
    and unhandled constructs.
    """
    missing = _words(body) - _words(_visible(to_html(body)))
    assert not missing, f"dropped {dict(missing)} from: {body!r}"


# Constructs with NO rule in `_html`. Each must survive as literal text.
UNHANDLED = [
    pytest.param("See [Duden](https://duden.de) for more.", id="link"),
    pytest.param("## Ablaut\nDer Vokal wechselt.", id="heading"),
    pytest.param("- outer\n  - nested item\n    - deeper", id="nested-list"),
    pytest.param("> Zitat aus dem Buch.", id="blockquote"),
    pytest.param("```\ncode block bleibt\n```", id="code-fence"),
    pytest.param("![Bild](scan.png)", id="image"),
    pytest.param("<!-- ein Kommentar -->", id="html-comment"),
    pytest.param("1. erstens\n2. zweitens", id="ordered-list"),
    pytest.param("Ein `inline code` Beispiel.", id="inline-code"),
    pytest.param("---\nnach der Linie", id="thematic-break"),
    pytest.param("Ein **unmatched marker", id="unmatched-bold"),
    pytest.param("Fußnote[^1] im Satz.\n\n[^1]: Die Note.", id="footnote"),
]


@pytest.mark.parametrize("body", UNHANDLED)
def test_unhandled_constructs_are_passed_through_never_dropped(body: str) -> None:
    _assert_nothing_dropped(body)


@pytest.mark.parametrize("body", UNHANDLED)
def test_unhandled_constructs_keep_their_literal_markdown(body: str) -> None:
    """Passthrough means *literal*, so the reader can see what the source said."""
    rendered = _visible(to_html(body))
    for word in _WORD.findall(body):
        assert word in rendered


def test_nothing_is_dropped_from_any_real_node_body(seed_paths) -> None:
    for path in seed_paths.values():
        _assert_nothing_dropped(storage.load_node(path).body_md)


# --- the four constructs that do have rules ---------------------------------------------


def test_bold_and_italic_render() -> None:
    assert "<b>waschen</b>" in to_html("Das Verb **waschen** ist stark.")
    assert "<i>to wash</i>" in to_html("Core sense is *to wash*.")


def test_two_bold_spans_on_one_line_stay_two_spans() -> None:
    rendered = to_html("Die **Wäsche** in der **Waschmaschine**.")
    assert rendered.count("<b>") == 2


def test_a_pipe_table_becomes_a_table() -> None:
    body = "| Verb | Meaning |\n|---|---|\n| abfahren | to depart |\n"
    rendered = to_html(body)

    assert "<table>" in rendered
    assert "<th>Verb</th>" in rendered
    assert "<td>abfahren</td>" in rendered
    assert "|" not in _visible(rendered), "raw pipes leaked into the card"


def test_a_table_without_a_separator_row_still_renders_every_row() -> None:
    """Defensive: a hand-written table missing its `|---|` line must not lose row one."""
    body = "| a | b |\n| c | d |\n"
    rendered = to_html(body)

    assert rendered.count("<tr>") == 2
    _assert_nothing_dropped(body)


def test_emphasis_inside_a_table_cell_renders() -> None:
    body = "| Wort | Note |\n|---|---|\n| waschen | takes **Akk.** |\n"
    assert "<b>Akk.</b>" in to_html(body)


def test_the_table_separator_row_is_the_only_thing_removed() -> None:
    body = "| Verb | Meaning |\n|:---|---:|\n| abfahren | to depart |\n"
    assert "---" not in _visible(to_html(body))
    _assert_nothing_dropped(body)


def test_html_in_a_body_is_escaped_not_executed() -> None:
    rendered = to_html("Ein <script>alert(1)</script> Satz.")
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_prose_lines_keep_their_breaks() -> None:
    assert to_html("Erste Zeile.\nZweite Zeile.").count("<br>") == 1
