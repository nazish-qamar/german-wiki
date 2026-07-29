"""The embed package exposes one public interface; nothing outside crosses it."""

from __future__ import annotations

import ast
import re

import pytest

from german_wiki import config, embed

EXPECTED = {
    "DEFAULT_K",
    "GRAY_HIGH",
    "GRAY_LOW",
    "NEAR_EXACT_JACCARD",
    "DuplicateReport",
    "EmbedResult",
    "Embedder",
    "Match",
    "cache_clear",
    "cache_stats",
    "embed_nodes",
    "embedding_model_name",
    "find_duplicates",
    "load_embedder",
}

PACKAGE_DIR = config.PROJECT_ROOT / "src" / "german_wiki"

_PRIVATE_IMPORT = re.compile(r"^\s*(?:from|import)\s+[\w.]*embed\._\w+", re.MULTILINE)


def test_all_matches_the_documented_interface() -> None:
    assert set(embed.__all__) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_exported_name_is_importable(name) -> None:
    assert getattr(embed, name) is not None


def test_the_cli_can_do_its_job_through_the_public_interface_only() -> None:
    for name in ("embed_nodes", "find_duplicates", "cache_stats", "cache_clear"):
        assert callable(getattr(embed, name))


def test_no_module_outside_the_package_imports_a_private_embed_module() -> None:
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.parent.name == "embed":
            continue  # the package's own internals
        if _PRIVATE_IMPORT.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(PACKAGE_DIR).as_posix())
    assert offenders == []


def test_private_modules_are_underscore_prefixed() -> None:
    modules = {p.stem for p in (PACKAGE_DIR / "embed").glob("*.py") if p.stem != "__init__"}
    assert all(name.startswith("_") for name in modules), modules


# Measured on the seed corpus plus a real ingested pair (see embed_text). These are
# the facts the thresholds have to sit between, not arbitrary constants.
MEASURED_UNRELATED_CEILING = 0.8635
MEASURED_WEAKEST_DUPLICATE = 0.9194


def test_the_gray_zone_brackets_the_measured_distribution() -> None:
    """SPEC §3.1 proposes 0.75–0.92, but multilingual-e5's scores compress into a
    narrow high band: at 0.75 every pair is gray and slice 5 gets a meaningless
    queue. The thresholds must sit between what was actually measured.

    Asserting the relationship rather than a literal pair, so recalibrating on more
    material updates one comment instead of breaking this test spuriously.
    """
    assert 0.0 < embed.GRAY_LOW < embed.GRAY_HIGH < 1.0
    assert embed.GRAY_LOW > MEASURED_UNRELATED_CEILING, "would flag unrelated nodes"
    assert embed.GRAY_LOW < MEASURED_WEAKEST_DUPLICATE, "would miss real duplicates"
    # Auto-classifying as a duplicate must be stricter than a genuine paraphrase,
    # so paraphrases reach the LLM instead of being decided by a constant.
    assert embed.GRAY_HIGH > MEASURED_WEAKEST_DUPLICATE


def test_embed_never_routes_through_the_chat_client() -> None:
    """ADR-004: embeddings are in-process. complete() refuses kind=local anyway,
    but this proves the package does not even try.

    AST, not a text search: the module docstrings *explain* that llm.complete()
    refuses this step, and a grep would flag exactly the files that document the
    rule most carefully.
    """
    callers = []
    for path in (PACKAGE_DIR / "embed").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name == "complete":
                callers.append(path.name)
    assert callers == []
