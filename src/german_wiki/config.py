"""Default paths, resolvable relative to the project root and overridable via env.

The project root is located by walking up from this file until a directory
containing ``pyproject.toml`` is found (falls back to cwd). Every path can be
overridden with an environment variable so tests and alternate checkouts don't
have to touch these constants.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def _path_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


NODES_DIR = _path_env("GW_NODES_DIR", PROJECT_ROOT / "nodes")
DATA_DIR = _path_env("GW_DATA_DIR", PROJECT_ROOT / "data")
DB_PATH = _path_env("GW_DB_PATH", DATA_DIR / "index.db")
VOCAB_DIR = _path_env("GW_VOCAB_DIR", PROJECT_ROOT / "vocab")
LOGS_DIR = _path_env("GW_LOGS_DIR", PROJECT_ROOT / "logs")
