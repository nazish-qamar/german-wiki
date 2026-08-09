"""Markdown -> HTML for card backs, over the subset this project's bodies actually use.

Anki fields are HTML, node bodies are Markdown, so something has to convert. A full
Markdown package would be a fourth dependency for four constructs -- the same judgment that
kept PyMuPDF out in slice 8 -- so this handles bold, italic, pipe tables and line breaks,
and nothing else.

**The safety property is that "nothing else" means passed through, never dropped.**

That is structural rather than careful. This is a *transformation pipeline over the whole
text*, not a parser that walks a syntax tree and emits the nodes it recognises:

    escape everything  ->  rewrite the known constructs in place  ->  join the lines

Text is never *selected for output*, only ever rewritten where a pattern matches. So a
construct with no rule -- a link, a heading, a nested list, a blockquote, a code fence --
survives verbatim as escaped literal text. It renders as ugly Markdown source on the card,
which is the correct side of the trade: SPEC-wise, a rule card exists to teach you the rule,
and silently dropping a paragraph of it is the same failure as truncating it. Ugly and
complete beats pretty and lossy.

The alternative -- a real parser -- fails in exactly the dangerous direction: an
unrecognised construct produces no output node, so the content vanishes with no error and
no visible gap. ``test_anki_html.py`` pins the no-drop property with a token-level invariant
over constructs the corpus does not contain yet, because the bodies will grow new ones.
"""

from __future__ import annotations

import html
import re

# Bold before italic: `**x**` would otherwise be eaten as an italic `*` wrapping `*x*`.
# Non-greedy so two bold spans on one line stay two spans. An UNMATCHED marker (`**oops`)
# matches neither pattern and is left exactly as it is -- passthrough, by omission.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

# A pipe-table separator carries no content -- only `-`, `:`, `|` and spaces -- so dropping
# it drops nothing. Every other line reaches the output.
_SEPARATOR_CELL = re.compile(r"^[\s:-]+$")


def _inline(text: str) -> str:
    """Emphasis only. Runs on already-escaped text, so it can only ever add tags."""
    text = _BOLD.sub(r"<b>\1</b>", text)
    return _ITALIC.sub(r"<i>\1</i>", text)


def _is_table_row(line: str) -> bool:
    return line.strip().startswith("|")


def _cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _is_separator(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(_SEPARATOR_CELL.match(cell) for cell in cells)


def _render_table(rows: list[str]) -> str:
    """A run of consecutive `|`-leading lines as one <table>."""
    header: list[str] | None = None
    body: list[list[str]] = []

    # A header exists only when row 2 is a separator; otherwise every row is data, so a
    # table written without the separator line still renders all of its content.
    start = 0
    if len(rows) >= 2 and _is_separator(rows[1]):
        header = _cells(rows[0])
        start = 2
    for row in rows[start:]:
        if not _is_separator(row):
            body.append(_cells(row))

    out = ["<table>"]
    if header:
        out.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>")
    for cells in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def to_html(body_md: str) -> str:
    """Render ``body_md`` for an Anki field. Unhandled Markdown survives as literal text."""
    escaped = html.escape(body_md, quote=False)

    out: list[str] = []
    table: list[str] = []
    for line in escaped.splitlines():
        if _is_table_row(line):
            table.append(line)
            continue
        if table:
            out.append(_render_table(table))
            table = []
        out.append(_inline(line))
    if table:
        out.append(_render_table(table))

    # <br> rather than <p>: bodies use single newlines meaningfully (a rule, then a
    # "Watch out:" line), and paragraph-wrapping would have to guess where blocks end.
    return "<br>".join(out).strip()
