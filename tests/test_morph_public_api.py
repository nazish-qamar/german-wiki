"""The morph package exposes one public interface, and it never writes a node."""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import config, morph

EXPECTED = {
    "INSEPARABLE",
    "MORPH_SOURCE_ID",
    "SEPARABLE",
    "TRUSTED",
    "VARIABLE",
    "Analysis",
    "Cell",
    "CellState",
    "CorpusIndex",
    "Grid",
    "PrefixAxis",
    "RootAxis",
    "Segmentation",
    "Separability",
    "Transparency",
    "TransparencyError",
    "Withheld",
    "analyse",
    "build_grid",
    "candidates",
    "classify",
    "dangling_targets",
    "is_family_node",
    "is_prefix_node",
    "judge",
    "morpheme_of",
    "segment",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"
_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*morph\._\w+", re.MULTILINE)


def test_all_matches_the_documented_interface() -> None:
    assert set(morph.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(morph, name) is not None


def test_no_module_outside_the_package_imports_a_private_morph_module() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "morph":
            continue
        if _PRIVATE_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    modules = {p.stem for p in (PACKAGE_DIR / "morph").glob("*.py") if p.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules


def _calls_named(name: str) -> list[str]:
    """Files containing a call to ``name``, via AST rather than grep (CLAUDE.md).

    Every rule here is *explained* in a docstring somewhere, so a text search would flag
    the explanation as a violation.
    """
    found = set()
    for path in PACKAGE_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if called == name:
                found.add(path.relative_to(PACKAGE_DIR).as_posix())
    return sorted(found)


def test_morph_never_writes_a_node() -> None:
    """Slice 7 proposes; it does not write. SPEC §7.2's ingest-time auto-creation is
    deliberately deferred (ADR-014), so no path from here reaches /nodes."""
    for writer in ("write_node", "write_approved"):
        assert not any(f.startswith("morph/") for f in _calls_named(writer))


def test_morph_does_not_reach_into_the_merge_internals() -> None:
    """It builds `Proposal`s through the public surface, like any other consumer."""
    private_merge = re.compile(r"^\s*(?:from|import)\s+[\w.]*merge\._\w+", re.MULTILINE)
    for path in (PACKAGE_DIR / "morph").glob("*.py"):
        assert not private_merge.search(path.read_text(encoding="utf-8")), path.name


def test_the_transparency_call_is_the_only_model_use_in_morph() -> None:
    """ADR-014's rules/judgment split, pinned.

    Segmentation, the grid and the analysis are all mechanical; exactly one module in
    this package is allowed to reach a model. If `complete` appears in a second morph
    module, the split has quietly eroded and a rule has started asking a model.

    Scoped to ``morph/`` deliberately -- ingest, level and merge each have their own
    legitimate call sites, and asserting across the whole package would only measure how
    many slices exist.
    """
    in_morph = [f for f in _calls_named("complete") if f.startswith("morph/")]
    assert in_morph == ["morph/_transparency.py"]
