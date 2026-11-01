"""The repo is public: nothing git-tracked may carry this machine's absolute path.

Writers that land in tracked directories (``/nodes``, ``/_merged``, the decision ledger)
record paths project-relative. This scans what is actually on disk, so a regression in
any writer -- or a hand-copied file -- fails here before it is committed. Gitignored
directories (``/raw``, ``/queue``, ``/proposals``, ``logs/gw.log``) are free to hold
absolute paths; they never leave the machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from german_wiki import config

TRACKED = ["nodes", "_merged", "config", "docs", "vocab", "logs/decisions.jsonl", "README.md"]


def _forms(root: Path) -> list[str]:
    """The root as it can appear in a file: native, forward-slashed, and JSON-escaped."""
    native = str(root)
    return sorted({native, root.as_posix(), json.dumps(native)[1:-1]})


def _tracked_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for entry in TRACKED:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return files


@pytest.mark.parametrize("path", _tracked_files(config.PROJECT_ROOT), ids=str)
def test_tracked_file_holds_no_absolute_project_path(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    leaked = [form for form in _forms(config.PROJECT_ROOT.resolve()) if form in text]
    assert not leaked, f"{path} contains the local project path {leaked[0]!r}"
