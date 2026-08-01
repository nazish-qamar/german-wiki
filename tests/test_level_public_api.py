"""The level package exposes one public interface, and CEFR stays rules-first."""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import config, level

EXPECTED = {
    "CEFR_ORDER",
    "FLAG_BASIS_ONLY",
    "FLAG_TIEBREAK",
    "GRAMMAR_MAP",
    "HUMAN_SEED_MARKER",
    "TIEBREAK_MARKER",
    "GrammarHit",
    "GrammarRule",
    "LevelResult",
    "LexicalHit",
    "RelevelResult",
    "Tiebreak",
    "TiebreakError",
    "available",
    "clear_cache",
    "derive_level",
    "grammar_anchor",
    "is_absent",
    "is_placeholder",
    "is_title_anchored",
    "lexical_anchor",
    "lookup",
    "relevel",
    "strongest",
    "targets",
    "tiebreak",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*level\._\w+", re.MULTILINE)


def test_all_matches_the_documented_interface() -> None:
    assert set(level.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(level, name) is not None


def test_no_module_outside_the_package_imports_a_private_level_module() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "level":
            continue
        if _PRIVATE_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    modules = {p.stem for p in (PACKAGE_DIR / "level").glob("*.py") if p.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules


def _call_sites(predicate) -> list[str]:
    """Files containing a matching call, via AST rather than grep (CLAUDE.md)."""
    found = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and predicate(node):
                found.add(path.relative_to(PACKAGE_DIR).as_posix())
    return sorted(found)


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_the_level_package_never_writes_a_node() -> None:
    """Re-levelling proposes; only ``gw review`` writes (ADR-003).

    If a future edit made ``gw relevel`` write directly, this is what fails -- and the
    plan rejected exactly that shortcut, since a ``--write`` flag is the slice-3 pattern
    ADR-009 already turned down.
    """
    writers = _call_sites(lambda c: _called_name(c) in {"write_node", "write_approved"})
    assert not any(f.startswith("level/") for f in writers), writers


def test_only_the_tiebreak_module_calls_a_model() -> None:
    """SPEC §5 is rules-first; the grammar map and wordlist are pure lookups.

    A ``complete()`` appearing in ``_grammar`` or ``_lexical`` would mean an anchor had
    quietly become a model call, which is the thing §5 exists to avoid.
    """
    callers = [f for f in _call_sites(lambda c: _called_name(c) == "complete") if f.startswith("level/")]
    assert callers == ["level/_tiebreak.py"]


def test_the_tiebreak_marker_is_greppable() -> None:
    """ADR-009's habit, one slice on: the least-grounded levels stay findable."""
    assert level.TIEBREAK_MARKER == "llm:tiebreak"
    assert not level.is_placeholder(f"{level.TIEBREAK_MARKER}(B1); grammar:none")


def test_the_spec_table_covers_every_cefr_level_it_names() -> None:
    """Guards against a row being dropped from GRAMMAR_MAP in a refactor."""
    levels = {rule.level for rule in level.GRAMMAR_MAP}
    assert levels == {"A1", "A2", "B1", "B2", "C1"}
