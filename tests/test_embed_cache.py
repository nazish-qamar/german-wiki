"""The embedding vector cache: keying by model, and degrading to a miss on damage."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

from german_wiki.embed import _cache

MODEL = "intfloat/multilingual-e5-small"
OTHER_MODEL = "intfloat/multilingual-e5-base"
TEXT = "query: Wechselpräpositionen — Two-way prepositions\n\nAkkusativ bei Bewegung."
VECTOR = [0.1, 0.2, 0.3, 0.4]


def _records() -> list[logging.LogRecord]:
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger("german_wiki.embed._cache").addHandler(_Collector())
    return collected


def _store(tmp_cache: Path, *, model=MODEL, text=TEXT, vector=None) -> str:
    key = _cache.cache_key(model, text)
    _cache.write(model, key, vector if vector is not None else VECTOR, cache_dir=tmp_cache)
    return key


# --- keying ---


def test_key_is_a_full_sha256() -> None:
    key = _cache.cache_key(MODEL, TEXT)
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_identical_inputs_share_a_key() -> None:
    assert _cache.cache_key(MODEL, TEXT) == _cache.cache_key(MODEL, TEXT)


def test_the_model_is_part_of_the_key(tmp_cache: Path) -> None:
    """Two embedding models must never collide, so they can be A/B'd."""
    assert _cache.cache_key(MODEL, TEXT) != _cache.cache_key(OTHER_MODEL, TEXT)

    _store(tmp_cache, model=MODEL, vector=[1.0, 0.0])
    _store(tmp_cache, model=OTHER_MODEL, vector=[0.0, 1.0])

    assert _cache.read(MODEL, _cache.cache_key(MODEL, TEXT), cache_dir=tmp_cache) == [1.0, 0.0]
    assert _cache.read(OTHER_MODEL, _cache.cache_key(OTHER_MODEL, TEXT), cache_dir=tmp_cache) == [
        0.0,
        1.0,
    ]
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 2


def test_different_text_yields_a_different_key() -> None:
    assert _cache.cache_key(MODEL, TEXT) != _cache.cache_key(MODEL, TEXT + " mehr")


def test_the_e5_prefix_is_part_of_the_hashed_input() -> None:
    """The key covers the EXACT model input, prefix included."""
    assert _cache.cache_key(MODEL, "query: x") != _cache.cache_key(MODEL, "x")


def test_entries_are_grouped_by_model_and_sharded(tmp_cache: Path) -> None:
    key = _cache.cache_key(MODEL, TEXT)
    path = _cache.entry_path(MODEL, key, cache_dir=tmp_cache)
    assert path.parent.name == key[:2]
    assert path.parent.parent.name == "intfloat-multilingual-e5-small"


# --- roundtrip ---


def test_miss_then_hit(tmp_cache: Path) -> None:
    key = _cache.cache_key(MODEL, TEXT)
    assert _cache.read(MODEL, key, cache_dir=tmp_cache) is None

    _cache.write(MODEL, key, VECTOR, cache_dir=tmp_cache)
    assert _cache.read(MODEL, key, cache_dir=tmp_cache) == VECTOR


def test_payload_shape(tmp_cache: Path) -> None:
    key = _store(tmp_cache)
    payload = json.loads(
        _cache.entry_path(MODEL, key, cache_dir=tmp_cache).read_text(encoding="utf-8")
    )
    assert payload["model"] == MODEL
    assert payload["dim"] == len(VECTOR)
    assert payload["vector"] == VECTOR
    assert payload["created_at"]


def test_write_leaves_no_temp_files(tmp_cache: Path) -> None:
    _store(tmp_cache)
    assert list(tmp_cache.rglob("*.tmp")) == []


def test_write_overwrites(tmp_cache: Path) -> None:
    key = _store(tmp_cache, vector=[1.0])
    _cache.write(MODEL, key, [2.0], cache_dir=tmp_cache)
    assert _cache.read(MODEL, key, cache_dir=tmp_cache) == [2.0]


# --- degrade to a miss, never break the run ---


def test_dimension_mismatch_is_a_miss_and_is_removed(tmp_cache: Path) -> None:
    """A stale entry from a differently-shaped model must never reach a vec0 column."""
    records = _records()
    key = _store(tmp_cache, vector=[0.1, 0.2, 0.3, 0.4])

    assert _cache.read(MODEL, key, cache_dir=tmp_cache, expect_dim=384) is None
    assert not _cache.entry_path(MODEL, key, cache_dir=tmp_cache).exists()
    assert any("expected 384" in r.getMessage() for r in records)


def test_matching_dimension_is_still_a_hit(tmp_cache: Path) -> None:
    key = _store(tmp_cache)
    assert _cache.read(MODEL, key, cache_dir=tmp_cache, expect_dim=len(VECTOR)) == VECTOR


@pytest.mark.parametrize(
    "content", ["{not json", '{"key": "x"}', "[]"], ids=["unparseable", "missing-keys", "not-dict"]
)
def test_damaged_entry_is_a_miss_and_is_removed(tmp_cache: Path, content) -> None:
    records = _records()
    key = _store(tmp_cache)
    _cache.entry_path(MODEL, key, cache_dir=tmp_cache).write_text(content, encoding="utf-8")

    assert _cache.read(MODEL, key, cache_dir=tmp_cache) is None
    assert not _cache.entry_path(MODEL, key, cache_dir=tmp_cache).exists()
    assert any(r.levelno == logging.WARNING for r in records)


def test_write_failure_warns_but_does_not_raise(tmp_cache: Path) -> None:
    records = _records()
    key = _cache.cache_key(MODEL, TEXT)
    shard = _cache.entry_path(MODEL, key, cache_dir=tmp_cache).parent
    shard.parent.mkdir(parents=True)
    shard.write_text("blocking file", encoding="utf-8")  # mkdir will fail

    _cache.write(MODEL, key, VECTOR, cache_dir=tmp_cache)
    assert any("cache write failed" in r.getMessage() for r in records)


# --- stats and clear ---


def _seed(tmp_cache: Path, count: int) -> list[str]:
    return [_store(tmp_cache, text=f"query: Quelle {n}") for n in range(count)]


def test_stats_on_an_empty_cache(tmp_cache: Path) -> None:
    assert _cache.stats(cache_dir=tmp_cache) == {
        "entries": 0,
        "bytes": 0,
        "oldest": None,
        "newest": None,
    }


def test_stats_counts_entries(tmp_cache: Path) -> None:
    _seed(tmp_cache, 3)
    result = _cache.stats(cache_dir=tmp_cache)
    assert result["entries"] == 3
    assert result["bytes"] > 0


def test_clear_removes_everything(tmp_cache: Path) -> None:
    _seed(tmp_cache, 3)
    assert _cache.clear(cache_dir=tmp_cache) == 3
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 0


def test_clear_older_than_days_keeps_recent(tmp_cache: Path) -> None:
    old, recent = _seed(tmp_cache, 2)
    stale = time.time() - 40 * 86400
    os.utime(_cache.entry_path(MODEL, old, cache_dir=tmp_cache), (stale, stale))

    assert _cache.clear(cache_dir=tmp_cache, older_than_days=30) == 1
    assert _cache.read(MODEL, old, cache_dir=tmp_cache) is None
    assert _cache.read(MODEL, recent, cache_dir=tmp_cache) is not None


def test_clear_on_a_missing_dir_is_a_noop(tmp_path: Path) -> None:
    assert _cache.clear(cache_dir=tmp_path / "absent") == 0
