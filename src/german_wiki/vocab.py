"""Tag normalization for the open ``register`` and ``themes`` axes.

These two axes are *discovered from material*, not enumerated. On write we
normalize each value (strip -> lowercase -> alias-map). An unknown canonical
value is **appended** to the field's known-set (append-only, no rewrite) and a
warning is logged; it never raises. Read/query paths pass ``learn=False`` so a
filter value is normalized but the known-set is left untouched.

Vocab store (git-tracked, in ``vocab/``):
- ``themes.txt`` / ``registers.txt`` — one normalized tag per line.
- ``aliases.yaml`` — human-curated ``field -> {alias: canonical}``; read-only here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import config
from .logutil import get_logger

logger = get_logger(__name__)

# field -> known-set filename
KNOWN_SET_FILES = {"themes": "themes.txt", "register": "registers.txt"}


def _norm(value: str) -> str:
    """strip + collapse internal whitespace + lowercase."""
    return " ".join(value.split()).lower()


class Vocab:
    """Known-sets + alias maps loaded from a ``vocab/`` directory."""

    def __init__(self, vocab_dir: Path | str | None = None):
        self.dir = Path(vocab_dir) if vocab_dir is not None else config.VOCAB_DIR
        self._aliases = self._load_aliases()
        self._known = {field: self._load_known(field) for field in KNOWN_SET_FILES}

    # --- loading ---
    def _known_path(self, field: str) -> Path:
        return self.dir / KNOWN_SET_FILES[field]

    def _load_known(self, field: str) -> set[str]:
        path = self._known_path(field)
        if not path.exists():
            return set()
        return {
            _norm(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    def _load_aliases(self) -> dict[str, dict[str, str]]:
        path = self.dir / "aliases.yaml"
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            str(field): {
                _norm(str(alias)): _norm(str(canonical))
                for alias, canonical in (mapping or {}).items()
            }
            for field, mapping in data.items()
        }

    # --- the one public operation ---
    def normalize(self, field: str, value: str, *, learn: bool = True) -> str:
        """Return the canonical tag for ``value`` in ``field``.

        ``learn=True`` (write path): an unknown value is appended to the
        known-set and warned. ``learn=False`` (query path): normalize + alias
        only, never touch the store. Empty values normalize to ``""`` and are
        never learned.
        """
        if field not in KNOWN_SET_FILES:
            raise ValueError(f"unknown vocab field: {field!r}")

        canonical = _norm(value)
        canonical = self._aliases.get(field, {}).get(canonical, canonical)
        if not canonical:
            return canonical

        known = self._known[field]
        if canonical not in known and learn:
            self._append_known(field, canonical)
            known.add(canonical)
            logger.warning(
                "unknown %s tag %r -> appended to %s",
                field,
                canonical,
                KNOWN_SET_FILES[field],
            )
        return canonical

    def _append_known(self, field: str, value: str) -> None:
        path = self._known_path(field)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(value + "\n")


# --- module-level convenience over a per-directory cached instance ---
_cache: dict[Path, Vocab] = {}


def get_vocab(vocab_dir: Path | str | None = None) -> Vocab:
    key = Path(vocab_dir) if vocab_dir is not None else config.VOCAB_DIR
    if key not in _cache:
        _cache[key] = Vocab(key)
    return _cache[key]


def normalize(
    field: str,
    value: str,
    *,
    learn: bool = True,
    vocab_dir: Path | str | None = None,
) -> str:
    return get_vocab(vocab_dir).normalize(field, value, learn=learn)
