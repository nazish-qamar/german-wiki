"""Shared logging setup.

Warnings and info from the pipeline go to ``logs/gw.log`` (per CLAUDE.md) and,
at WARNING and above, also to stderr so they're visible when running the CLI.
"""

from __future__ import annotations

import logging

from . import config

_ROOT = "german_wiki"
_configured = False


def get_logger(name: str = _ROOT) -> logging.Logger:
    global _configured
    if not _configured:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger(_ROOT)
        root.setLevel(logging.INFO)
        root.propagate = False

        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler = logging.FileHandler(config.LOGS_DIR / "gw.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(stderr_handler)

        _configured = True
    return logging.getLogger(name)
