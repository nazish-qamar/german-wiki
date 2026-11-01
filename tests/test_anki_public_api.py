"""The anki package exposes one public interface; nothing outside crosses it.

Same policing as `test_ingest_public_api.py` and `test_morph_public_api.py`, including the
tightened two-form private-import pattern that slice 8 needed after `cli.py` reached into
`ingest._vision` the spelling the original regex did not catch.
"""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import anki, config

EXPECTED = {
    "BASIC_MODEL_ID",
    # The dials, on the public surface by the same convention as `embed.GRAY_LOW` and
    # `ingest.MIN_PAGE_CHARS`. Both were chosen before the deck was studied even once and
    # are expected to move -- which is exactly why they must be findable.
    "CARD_FORMATS",
    "CEFR_ORDER",
    "CLOZE_MODEL_ID",
    "DECK_ID",
    "DECK_NAME",
    "DEFAULT_EXPORT_PATH",
    "EXPORT_STATUSES",
    "PREPOSITIONS",
    "Card",
    "ClozeTargets",
    "ExcludedNode",
    "ExportResult",
    "Generator",
    "Skip",
    "build",
    "card_tags",
    "cards_for",
    "cloze_targets",
    "export",
    "generators_for",
    "guid_for",
    "rule_card",
    "strip_emphasis",
    "to_html",
    "write_deck",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

_PRIVATE_IMPORT = re.compile(
    r"""^\s*(?:
        (?:from|import)\s+[\w.]*anki\._\w+
      | from\s+[\w.]*anki\s+import\s+[^\n]*\b_\w+
    )""",
    re.MULTILINE | re.VERBOSE,
)


def test_all_matches_the_documented_interface() -> None:
    assert set(anki.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(anki, name) is not None


def test_no_module_outside_the_package_imports_a_private_anki_module() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "anki":
            continue
        if _PRIVATE_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    modules = {p.stem for p in (PACKAGE_DIR / "anki").glob("*.py") if p.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules


def _imports_genanki(path) -> bool:
    """AST, not a text search: several modules here *discuss* genanki in prose."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name.split(".")[0] == "genanki" for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "genanki":
            return True
    return False


def test_genanki_is_imported_by_exactly_one_module() -> None:
    """The new dependency gets one edge, so the card logic stays testable without it and
    swapping the writer later touches one file."""
    importers = sorted(
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*.py")
        if _imports_genanki(path)
    )
    assert importers == ["anki/_deck.py"]


def test_the_export_never_writes_to_nodes() -> None:
    """`storage.write_node` is how /nodes changes; no module in this package may call it.

    ADR-003 is enforced elsewhere by a review gate. Here it is enforced by the package
    simply having no write path, which is a stronger guarantee and worth pinning so it is
    not casually added later.
    """
    offenders = []
    for path in (PACKAGE_DIR / "anki").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr in {
                    "write_node",
                    "write_approved",
                }:
                    offenders.append(path.name)
    assert offenders == []
