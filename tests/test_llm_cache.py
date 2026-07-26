"""Cache keying, roundtrip, and the degrade-to-miss behavior on bad entries (ADR-005)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

from german_wiki.llm import _cache

MESSAGES = [
    {"role": "system", "content": "Du bist ein Lehrer."},
    {"role": "user", "content": "Der Text."},
]


def _material(**overrides):
    kwargs = {
        "provider": "zai",
        "model": "glm-4.5-flash",
        "messages": MESSAGES,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    kwargs.update(overrides)
    return _cache.key_material(**kwargs)


def _payload(material, *, text="Antwort") -> dict:
    return {
        "key": _cache.cache_key(material),
        "created_at": "2026-07-26T09:14:03+00:00",
        "step": "extraction",
        "provider": material["provider"],
        "model": material["model"],
        "request": material,
        "text": text,
        "finish_reason": "stop",
        "response_id": "chatcmpl-test",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cached_prompt_tokens": 0,
        },
        "cost_usd": 0.0,
    }


def _records(logger_name: str) -> list[logging.LogRecord]:
    """Collect records directly: logutil sets propagate=False, so caplog is blind."""
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger(logger_name).addHandler(_Collector())
    return collected


# --- keying ---


def test_key_is_stable_across_dict_insertion_order() -> None:
    a = {"v": 1, "provider": "zai", "model": "m", "messages": MESSAGES}
    b = {"messages": MESSAGES, "model": "m", "provider": "zai", "v": 1}
    assert _cache.cache_key(a) == _cache.cache_key(b)


def test_key_is_a_full_sha256_hexdigest() -> None:
    key = _cache.cache_key(_material())
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


def test_identical_material_yields_the_same_key() -> None:
    assert _cache.cache_key(_material()) == _cache.cache_key(_material())


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "deepseek"},
        {"model": "glm-4.6"},
        {"temperature": 0.7},
        {"max_tokens": 128},
        {"response_format": {"type": "json_object"}},
        {"seed": 7},
        {"prompt_version": "extract@2"},
        {"messages": [*MESSAGES, {"role": "user", "content": "mehr"}]},
    ],
)
def test_key_changes_when_any_hashed_field_changes(overrides) -> None:
    assert _cache.cache_key(_material()) != _cache.cache_key(_material(**overrides))


def test_message_order_is_significant() -> None:
    reversed_messages = list(reversed(MESSAGES))
    assert _cache.cache_key(_material()) != _cache.cache_key(_material(messages=reversed_messages))


def test_key_material_covers_exactly_the_documented_fields() -> None:
    """Guards the deliberate exclusions: no step, no credentials, no base_url."""
    assert set(_material()) == {
        "v",
        "provider",
        "model",
        "messages",
        "temperature",
        "max_tokens",
        "response_format",
        "seed",
        "prompt_version",
    }


def test_umlauts_are_hashed_as_utf8_not_escapes() -> None:
    """ensure_ascii=False: German text hashes as its own bytes."""
    material = _material(messages=[{"role": "user", "content": "Küche"}])
    assert _cache.cache_key(material) != _cache.cache_key(
        _material(messages=[{"role": "user", "content": "K\\u00fcche"}])
    )


# --- roundtrip ---


def test_miss_then_hit_roundtrip(tmp_cache: Path) -> None:
    material = _material()
    key = _cache.cache_key(material)
    assert _cache.read(key, cache_dir=tmp_cache) is None

    _cache.write(key, _payload(material), cache_dir=tmp_cache)
    hit = _cache.read(key, cache_dir=tmp_cache, expect_request=material)
    assert hit is not None
    assert hit["text"] == "Antwort"
    assert hit["usage"]["prompt_tokens"] == 10


def test_entry_path_is_sharded_by_key_prefix(tmp_cache: Path) -> None:
    key = _cache.cache_key(_material())
    path = _cache.entry_path(key, cache_dir=tmp_cache)
    assert path == tmp_cache / "llm" / key[:2] / f"{key}.json"


def test_write_leaves_no_temp_files(tmp_cache: Path) -> None:
    material = _material()
    _cache.write(_cache.cache_key(material), _payload(material), cache_dir=tmp_cache)
    assert list(tmp_cache.rglob("*.tmp")) == []


def test_write_overwrites_an_existing_entry(tmp_cache: Path) -> None:
    material = _material()
    key = _cache.cache_key(material)
    _cache.write(key, _payload(material, text="alt"), cache_dir=tmp_cache)
    _cache.write(key, _payload(material, text="neu"), cache_dir=tmp_cache)
    assert _cache.read(key, cache_dir=tmp_cache)["text"] == "neu"


def test_stored_umlauts_are_not_escaped(tmp_cache: Path) -> None:
    material = _material()
    key = _cache.cache_key(material)
    _cache.write(key, _payload(material, text="Küche"), cache_dir=tmp_cache)
    raw = _cache.entry_path(key, cache_dir=tmp_cache).read_text(encoding="utf-8")
    assert "Küche" in raw


# --- degrade to a miss, never break the run ---


def _seed_then_corrupt(tmp_cache: Path, content: str) -> str:
    material = _material()
    key = _cache.cache_key(material)
    _cache.write(key, _payload(material), cache_dir=tmp_cache)
    _cache.entry_path(key, cache_dir=tmp_cache).write_text(content, encoding="utf-8")
    return key


@pytest.mark.parametrize(
    "content",
    ["{not json at all", '{"key": "x"}', "[]"],
    ids=["unparseable", "missing-required-keys", "not-an-object"],
)
def test_bad_entry_is_a_miss_and_is_removed(tmp_cache: Path, content) -> None:
    records = _records("german_wiki.llm._cache")
    key = _seed_then_corrupt(tmp_cache, content)

    assert _cache.read(key, cache_dir=tmp_cache) is None
    assert not _cache.entry_path(key, cache_dir=tmp_cache).exists()
    assert any(r.levelno == logging.WARNING for r in records)


def test_request_mismatch_is_a_miss_and_is_removed(tmp_cache: Path) -> None:
    """A stored request that disagrees with its key means a collision or a hand-edit."""
    material = _material()
    key = _cache.cache_key(material)
    _cache.write(key, _payload(_material(model="something-else")), cache_dir=tmp_cache)

    assert _cache.read(key, cache_dir=tmp_cache, expect_request=material) is None
    assert not _cache.entry_path(key, cache_dir=tmp_cache).exists()


def test_matching_request_is_still_a_hit(tmp_cache: Path) -> None:
    material = _material()
    key = _cache.cache_key(material)
    _cache.write(key, _payload(material), cache_dir=tmp_cache)
    assert _cache.read(key, cache_dir=tmp_cache, expect_request=material) is not None


def test_write_failure_warns_but_does_not_raise(tmp_cache: Path) -> None:
    records = _records("german_wiki.llm._cache")
    material = _material()
    key = _cache.cache_key(material)
    # A file where the shard directory needs to be makes mkdir fail.
    (tmp_cache / "llm").mkdir(parents=True)
    (tmp_cache / "llm" / key[:2]).write_text("blocked", encoding="utf-8")

    _cache.write(key, _payload(material), cache_dir=tmp_cache)
    assert any("cache write failed" in r.getMessage() for r in records)


# --- stats and clear ---


def _seed(tmp_cache: Path, count: int) -> list[str]:
    keys = []
    for n in range(count):
        material = _material(messages=[{"role": "user", "content": f"Quelle {n}"}])
        key = _cache.cache_key(material)
        _cache.write(key, _payload(material), cache_dir=tmp_cache)
        keys.append(key)
    return keys


def test_stats_on_an_empty_cache(tmp_cache: Path) -> None:
    assert _cache.stats(cache_dir=tmp_cache) == {
        "entries": 0,
        "bytes": 0,
        "oldest": None,
        "newest": None,
    }


def test_stats_counts_entries_and_bytes(tmp_cache: Path) -> None:
    _seed(tmp_cache, 3)
    result = _cache.stats(cache_dir=tmp_cache)
    assert result["entries"] == 3
    assert result["bytes"] > 0
    assert result["oldest"] <= result["newest"]


def test_clear_removes_every_entry(tmp_cache: Path) -> None:
    keys = _seed(tmp_cache, 3)
    assert _cache.clear(cache_dir=tmp_cache) == 3
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 0
    assert all(_cache.read(k, cache_dir=tmp_cache) is None for k in keys)


def test_clear_older_than_days_keeps_recent_entries(tmp_cache: Path) -> None:
    old, recent = _seed(tmp_cache, 2)
    stale = time.time() - 40 * 86400
    os.utime(_cache.entry_path(old, cache_dir=tmp_cache), (stale, stale))

    assert _cache.clear(cache_dir=tmp_cache, older_than_days=30) == 1
    assert _cache.read(old, cache_dir=tmp_cache) is None
    assert _cache.read(recent, cache_dir=tmp_cache) is not None


def test_clear_on_a_missing_cache_dir_is_a_noop(tmp_path: Path) -> None:
    assert _cache.clear(cache_dir=tmp_path / "never-created") == 0


def test_entries_are_valid_json_on_disk(tmp_cache: Path) -> None:
    key = _seed(tmp_cache, 1)[0]
    raw = _cache.entry_path(key, cache_dir=tmp_cache).read_text(encoding="utf-8")
    assert json.loads(raw)["key"] == key
