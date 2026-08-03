"""``strip_fences`` — the one provider artifact this layer undoes."""

from __future__ import annotations

import ast

import pytest

from german_wiki import config
from german_wiki.llm import strip_fences

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),  # no fence at all
        ('```json\n{"a": 1}\n```', '{"a": 1}'),  # the common case
        ('```JSON\n{"a": 1}\n```', '{"a": 1}'),  # tag case does not matter
        ('```\n{"a": 1}\n```', '{"a": 1}'),  # bare fence, no language tag
        ('  ```json\n{"a": 1}\n```  ', '{"a": 1}'),  # surrounding whitespace
        ('```json\n{"a": 1}', '{"a": 1}'),  # unterminated -- still recoverable
        ('```json\n{"a": "x```y"}\n```', '{"a": "x```y"}'),  # fence inside a string
    ],
)
def test_fences_come_off(raw, expected) -> None:
    assert strip_fences(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "```", "```json"])
def test_degenerate_input_yields_empty_rather_than_raising(raw) -> None:
    """Callers already treat empty content as their own error, with a message naming the
    step and the finish_reason. Raising here would replace that with a worse one."""
    assert strip_fences(raw) == ""


def test_multiline_content_survives() -> None:
    raw = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
    assert strip_fences(raw) == '{\n  "a": 1,\n  "b": 2\n}'


def test_there_is_exactly_one_implementation() -> None:
    """The rule-of-three extraction, pinned (ADR-010's policy).

    ``ingest/_extract``, ``merge/_adjudicate`` and ``morph/_transparency`` each carried a
    byte-identical private copy. They were verified identical at the AST level before
    collapsing — had one diverged, the extraction would have needed reconciling rather
    than picking a winner. This fails if a fourth copy is ever pasted in.
    """
    definitions = []
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.endswith("strip_fences"):
                definitions.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert definitions == ["llm/_parse.py"]
