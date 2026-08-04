"""The ingest package exposes one public interface; nothing outside crosses it."""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import config, ingest

EXPECTED = {
    # The tunable dials, on the public surface by the same convention as
    # `embed.GRAY_LOW` and `merge.MAX_REGENERATIONS`. Slice 8's two PDF thresholds were
    # guessed before seeing real German textbook PDFs, so they are the likeliest in the
    # codebase to need moving -- which is precisely why they must be findable.
    "MAX_CANDIDATES",
    "MAX_IMAGE_BYTES",
    "MIN_PAGE_CHARS",
    "MIN_TEXT_PAGE_RATIO",
    "Candidate",
    "ExtractionError",
    "IngestResult",
    "PdfError",
    "PdfExtraction",
    "PromoteResult",
    "Refusal",
    "VisionError",
    "extract_pages",
    "ingest_file",
    "list_queue",
    "promote_source",
    "read_raw_text",
    "transcribe",
    "write_approved",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*ingest\._\w+", re.MULTILINE)


def test_all_matches_the_documented_interface() -> None:
    assert set(ingest.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(ingest, name) is not None


def test_the_cli_can_do_its_job_through_the_public_interface_only() -> None:
    for name in ("ingest_file", "promote_source", "list_queue", "write_approved"):
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


def _enclosing_functions(predicate) -> list[str]:
    """``file::function`` for every call matching ``predicate``, via AST.

    Attributing a call to its *enclosing function* is what the file-level check
    above cannot do. Nested defs report the innermost one, which is the tightest
    true statement about where the call lives.
    """
    found = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for scope in ast.walk(tree):
            if not isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for node in ast.walk(scope):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node is not scope:
                    continue  # a nested def owns its own calls
                if isinstance(node, ast.Call) and predicate(node):
                    found.add(f"{path.relative_to(PACKAGE_DIR).as_posix()}::{scope.name}")
    return sorted(found)


def test_write_approved_is_the_only_function_that_learns() -> None:
    """ADR-007/ADR-011, tightened: the right *function*, not merely the right file.

    Slice 5 moved the ``learn=True`` call out of ``promote_source`` and into
    ``write_approved`` so the merge pipeline could share one writer. The
    file-level assertion above would have stayed green either way -- it cannot
    see the move at all -- so it no longer pins what it exists to pin. This does.
    """
    assert _enclosing_functions(_passes_learn_true) == ["ingest/_promote.py::write_approved"]


def test_promote_source_does_not_write_nodes_itself() -> None:
    """The seam only holds while every caller goes *through* write_approved."""

    def _write_node_call(call: ast.Call) -> bool:
        func = call.func
        return isinstance(func, ast.Attribute) and func.attr == "write_node"

    assert "ingest/_promote.py::promote_source" not in _enclosing_functions(_write_node_call)


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
