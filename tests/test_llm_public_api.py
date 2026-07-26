"""The llm package exposes one public interface; the rest of the app uses only that."""

from __future__ import annotations

import re

import pytest

from german_wiki import config, llm

EXPECTED = {
    "JSON_OBJECT",
    "ChatClient",
    "ModelResponse",
    "Prompt",
    "ShotPair",
    "Usage",
    "cache_clear",
    "cache_stats",
    "complete",
    "cost_totals",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

# `from german_wiki.llm._x import`, `from .llm._x import`, `from german_wiki import llm._x`
_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*llm\._\w+", re.MULTILINE)


def test_all_matches_the_documented_interface() -> None:
    assert set(llm.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(llm, name) is not None


def test_the_cli_can_do_its_job_through_the_public_interface_only() -> None:
    """gw cost / gw cache must not need to reach into _usage or _cache."""
    for name in ("complete", "cost_totals", "cache_stats", "cache_clear"):
        assert callable(getattr(llm, name))


def test_no_module_outside_the_package_imports_a_private_llm_module() -> None:
    """The boundary is only real if nothing crosses it."""
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "llm":
            continue  # the package's own internals
        source = path.read_text(encoding="utf-8")
        if _PRIVATE_IMPORT.search(source):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    """Anything public-looking in llm/ would invite an import that bypasses __init__."""
    modules = {path.stem for path in (PACKAGE_DIR / "llm").glob("*.py") if path.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules
