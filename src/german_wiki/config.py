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

# --- slice 2: model layer ---
CACHE_DIR = _path_env("GW_CACHE_DIR", PROJECT_ROOT / ".cache")
CONFIG_DIR = _path_env("GW_CONFIG_DIR", PROJECT_ROOT / "config")
MODELS_CONFIG_PATH = _path_env("GW_MODELS_CONFIG", CONFIG_DIR / "models.yaml")
DOTENV_PATH = _path_env("GW_DOTENV", PROJECT_ROOT / ".env")
USAGE_LOG_PATH = _path_env("GW_USAGE_LOG", LOGS_DIR / "llm_usage.jsonl")

# --- slice 3: ingestion ---
RAW_DIR = _path_env("GW_RAW_DIR", PROJECT_ROOT / "raw")
QUEUE_DIR = _path_env("GW_QUEUE_DIR", PROJECT_ROOT / "queue")

# --- slice 5: merge pipeline + review ---
# Pending adjudication results awaiting `gw review`. Gitignored like QUEUE_DIR:
# transient staging, resolved by approval-or-rejection (ADR-011).
PROPOSALS_DIR = _path_env("GW_PROPOSALS_DIR", PROJECT_ROOT / "proposals")

# Archived losing nodes from approved merges (SPEC §3.2: never silently delete).
# Git-TRACKED, unlike the two staging dirs -- it is the audit trail.
MERGED_DIR = _path_env("GW_MERGED_DIR", PROJECT_ROOT / "_merged")

# One record per human decision. Git-tracked and load-bearing: it is the
# authoritative regeneration count for the SPEC §12.1 drift cap, not just a
# statistic (ADR-011).
DECISIONS_LOG_PATH = _path_env("GW_DECISIONS_LOG", LOGS_DIR / "decisions.jsonl")
