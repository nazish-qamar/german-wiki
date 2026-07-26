"""The ingest package exposes one public interface; nothing outside crosses it."""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import config, ingest

EXPECTED = {
    "Candidate",
    "ExtractionError",
    "IngestResult",
    "PromoteResult",
    "Refusal",
    "ingest_file",
    "list_queue",
    "promote_source",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*ingest\._\w+", re.MULTILINE)


def test_all_matches_the_documented_interface() -> None:
    assert set(ingest.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(ingest, name) is not None


def test_the_cli_can_do_its_job_through_the_public_interface_only() -> None:
    for name in ("ingest_file", "promote_source", "list_queue"):
        assert callable(getattr(ingest, name))


def test_no_module_outside_the_package_imports_a_private_ingest_module() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "ingest":
            continue  # the package's own internals
        if _PRIVATE_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    modules = {p.stem for p in (PACKAGE_DIR / "ingest").glob("*.py") if p.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules


def _call_sites(predicate) -> list[str]:
    """Files containing a call matching `predicate`, found via AST.

    Deliberately not a text search: docstrings and comments discussing ``learn=True``
    are not call sites, and a grep would flag every file that explains the rule.
    """
    found = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and predicate(node):
                found.add(path.relative_to(PACKAGE_DIR).as_posix())
    return sorted(found)


def _passes_learn_true(call: ast.Call) -> bool:
    return any(
        kw.arg == "learn" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in call.keywords
    )


def test_promote_is_the_only_learn_true_caller() -> None:
    """ADR-007: the tag vocabulary grows at exactly one gate, and this proves it."""
    assert _call_sites(_passes_learn_true) == ["ingest/_promote.py"]


def test_every_normalize_call_states_learn_explicitly() -> None:
    """vocab.normalize defaults to learn=True, so omitting it would learn by accident."""

    def _implicit_normalize(call: ast.Call) -> bool:
        # `vocab.normalize` specifically -- unicodedata.normalize is unrelated.
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "normalize"):
            return False
        if not (isinstance(func.value, ast.Name) and func.value.id == "vocab"):
            return False
        return not any(kw.arg == "learn" for kw in call.keywords)

    assert _call_sites(_implicit_normalize) == []
