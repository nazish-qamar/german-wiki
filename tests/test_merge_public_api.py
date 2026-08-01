"""The merge package exposes one public interface, and one path reaches a write."""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import config, merge

EXPECTED = {
    "FLAG_CAP",
    "FLAG_LEDGER_MISSING",
    "FLAG_LOW_CONFIDENCE",
    "FLAG_UNSOURCED",
    "MAX_REGENERATIONS",
    "Adjudication",
    "AdjudicationError",
    "ApplyError",
    "ApplyResult",
    "CapCheck",
    "Context",
    "Decision",
    "LedgerUnreadable",
    "MergedBody",
    "Proposal",
    "ProposeResult",
    "adjudicate",
    "apply_decision",
    "build_graph",
    "check_cap",
    "decided_pairs",
    "delete_proposal",
    "list_proposals",
    "load_proposal",
    "merge_count",
    "now_iso",
    "proposal_id",
    "propose_for_source",
    "read_all",
    "regenerate",
    "review_order",
    "write_proposal",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*merge\._\w+", re.MULTILINE)

# The single implementation of the routing (ADR-011). Reaching one of these IS the write.
# `apply_relevel` joined in slice 6 and is listed here deliberately: a new apply function
# that nobody added to this set would be a write path outside the invariant, which is the
# exact hole this test exists to close.
APPLY_FUNCTIONS = {
    "apply_merge",
    "apply_link",
    "apply_create",
    "apply_discard",
    "apply_relevel",
}


def test_all_matches_the_documented_interface() -> None:
    assert set(merge.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(merge, name) is not None


def test_no_module_outside_the_package_imports_a_private_merge_module() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "merge":
            continue  # the package's own internals
        if _PRIVATE_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    modules = {p.stem for p in (PACKAGE_DIR / "merge").glob("*.py") if p.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules


def _call_sites(predicate) -> list[str]:
    """Files containing a call matching ``predicate``, found via AST.

    Deliberately not a text search (CLAUDE.md, learned in slices 3 and 4): every one of
    these rules is *explained* in a docstring somewhere, and grep would flag the
    explanation as a violation.
    """
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


def test_only_the_graph_drives_the_apply_functions() -> None:
    """One implementation, one caller: the routing is not duplicated beside the graph.

    ADR-011: ``gw review`` drives the graph rather than a parallel copy of the routing,
    because two implementations are how one of them quietly stops honouring the gate.
    If the CLI ever reached past ``apply_decision`` straight into ``_apply``, this fails.
    """
    assert _call_sites(lambda c: _called_name(c) in APPLY_FUNCTIONS) == ["merge/_graph.py"]


def test_the_merge_package_never_writes_a_node_directly() -> None:
    """Every write goes through ingest.write_approved -- still the one door into /nodes."""
    writers = [f for f in _call_sites(lambda c: _called_name(c) == "write_node")]
    assert not any(f.startswith("merge/") for f in writers), writers


def test_write_approved_is_reached_only_from_the_apply_layer() -> None:
    callers = _call_sites(lambda c: _called_name(c) == "write_approved")
    assert callers == ["ingest/_promote.py", "merge/_apply.py"]


def test_the_graph_has_no_path_from_adjudication_to_a_write() -> None:
    """The gate again, from the compiled graph rather than the source (see test_merge_graph)."""
    graph = merge.build_graph().get_graph()
    assert {e.target for e in graph.edges if e.source == "adjudicate"} == {"__end__"}


def test_interrupt_is_called_exactly_once_and_only_in_adjudicate() -> None:
    """ADR-003 names ``interrupt()`` at the adjudication node; this pins it there."""
    tree = ast.parse((PACKAGE_DIR / "merge" / "_graph.py").read_text(encoding="utf-8"))
    enclosing = []
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.FunctionDef):
            continue
        for node in ast.walk(scope):
            if isinstance(node, ast.FunctionDef) and node is not scope:
                continue
            if isinstance(node, ast.Call) and _called_name(node) == "interrupt":
                enclosing.append(scope.name)
    assert enclosing == ["adjudicate_node"]
