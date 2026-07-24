"""Shared fixtures. The four seed nodes in /nodes are the primary fixtures.

Tests only READ the real /nodes and /vocab; anything mutable (DB, appended
vocab, touched files) is done against tmp copies so the repo is never altered.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from german_wiki import config, storage

SEED_IDS = [
    "familie-waschen",
    "prefix-an",
    "um-hilfe-bitten",
    "wechselpraepositionen",
]


@pytest.fixture
def nodes_dir() -> Path:
    """The real /nodes directory (read-only in tests)."""
    return config.NODES_DIR


@pytest.fixture
def seed_paths(nodes_dir: Path) -> dict[str, Path]:
    return {sid: nodes_dir / f"{sid}.md" for sid in SEED_IDS}


@pytest.fixture
def seed_nodes(seed_paths: dict[str, Path]) -> dict[str, "storage.Node"]:
    return {sid: storage.load_node(p) for sid, p in seed_paths.items()}


@pytest.fixture
def tmp_nodes(tmp_path: Path, nodes_dir: Path) -> Path:
    """A writable copy of /nodes so tests can touch files for staleness."""
    dst = tmp_path / "nodes"
    dst.mkdir()
    for md in nodes_dir.glob("*.md"):
        shutil.copy2(md, dst / md.name)
    return dst


@pytest.fixture
def tmp_vocab(tmp_path: Path) -> Path:
    """A writable copy of /vocab so normalization appends don't touch the repo."""
    dst = tmp_path / "vocab"
    dst.mkdir()
    for name in ("themes.txt", "registers.txt", "aliases.yaml"):
        shutil.copy2(config.VOCAB_DIR / name, dst / name)
    return dst


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "index.db"
