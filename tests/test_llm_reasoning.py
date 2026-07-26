"""reasoning_content is captured for debugging: in the cache payload, out of the key."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from conftest import FakeChatClient

from german_wiki.llm import _cache
from german_wiki.llm._client import complete
from german_wiki.llm._prompt import Prompt

CONFIG = {
    "version": 1,
    "providers": {
        "zai": {
            "kind": "api",
            "base_url": "https://example.invalid/v4",
            "api_key_env": "ZAI_API_KEY",
        }
    },
    "pricing": {"zai": {"free-model": {"input": 0.0, "cached_input": 0.0, "output": 0.0}}},
    "defaults": {"provider": "zai", "model": "free-model", "temperature": 0.0, "max_tokens": 4096},
    "steps": {"extraction": {"status": "active"}},
}

REASONING = "Ich analysiere den Text und finde zuerst die Grammatikregel..."


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    return path


def _call(cfg: Path, tmp_cache: Path, tmp_usage_log: Path, client, **overrides):
    kwargs = {
        "client": client,
        "settings_path": cfg,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
        "env": {},
    }
    kwargs.update(overrides)
    return complete("extraction", Prompt(system="s", variable="v"), **kwargs)


# --- capture ---


def test_captured_when_the_provider_returns_it(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient(text="Antwort", reasoning_content=REASONING)
    assert _call(cfg, tmp_cache, tmp_usage_log, fake).reasoning_content == REASONING


def test_none_when_the_provider_omits_it(cfg, tmp_cache, tmp_usage_log) -> None:
    """Most providers never send it; absence is normal, not an error."""
    fake = FakeChatClient(text="Antwort")
    assert _call(cfg, tmp_cache, tmp_usage_log, fake).reasoning_content is None


def test_captured_even_when_content_is_empty(cfg, tmp_cache, tmp_usage_log) -> None:
    """The truncation case: all tokens went to reasoning, none to content."""
    fake = FakeChatClient(text="", finish_reason="length", reasoning_content=REASONING)
    response = _call(cfg, tmp_cache, tmp_usage_log, fake)
    assert response.text == ""
    assert response.finish_reason == "length"
    assert response.reasoning_content == REASONING


# --- out of the key, in the payload ---


def test_not_in_the_cache_key(cfg, tmp_cache, tmp_usage_log) -> None:
    """Reasoning is an output, not an input -- it must not change the key."""
    with_reasoning = _call(
        cfg, tmp_cache, tmp_usage_log, FakeChatClient(reasoning_content=REASONING)
    )
    _cache.clear(cache_dir=tmp_cache)
    without = _call(cfg, tmp_cache, tmp_usage_log, FakeChatClient())
    assert with_reasoning.cache_key == without.cache_key


def test_stored_in_the_cache_payload(cfg, tmp_cache, tmp_usage_log) -> None:
    response = _call(cfg, tmp_cache, tmp_usage_log, FakeChatClient(reasoning_content=REASONING))
    entry = json.loads(
        _cache.entry_path(response.cache_key, cache_dir=tmp_cache).read_text(encoding="utf-8")
    )
    assert entry["reasoning_content"] == REASONING
    # ...and not smuggled into the hashed request material.
    assert "reasoning_content" not in entry["request"]


def test_a_cached_truncated_call_still_reports_why_on_the_hit(
    cfg, tmp_cache, tmp_usage_log
) -> None:
    """The subtle part: complete() caches regardless of finish_reason, so without
    the payload field a re-run would hit the cache and report the failure with an
    empty context -- exactly when the reasoning is needed."""
    fake = FakeChatClient(text="", finish_reason="length", reasoning_content=REASONING)
    first = _call(cfg, tmp_cache, tmp_usage_log, fake)
    second = _call(cfg, tmp_cache, tmp_usage_log, fake)

    assert fake.call_count == 1  # genuinely served from cache
    assert second.cached is True
    assert second.finish_reason == "length"
    assert second.reasoning_content == REASONING == first.reasoning_content


def test_entries_written_before_the_field_existed_read_back_as_none(
    cfg, tmp_cache, tmp_usage_log
) -> None:
    """Backwards compatibility: an old entry lacks the key entirely."""
    response = _call(cfg, tmp_cache, tmp_usage_log, FakeChatClient(reasoning_content=REASONING))
    path = _cache.entry_path(response.cache_key, cache_dir=tmp_cache)
    entry = json.loads(path.read_text(encoding="utf-8"))
    del entry["reasoning_content"]
    path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

    hit = _call(cfg, tmp_cache, tmp_usage_log, FakeChatClient())
    assert hit.cached is True
    assert hit.reasoning_content is None


def test_reasoning_is_not_written_to_the_usage_ledger(cfg, tmp_cache, tmp_usage_log) -> None:
    """It is debugging context, not accounting -- the ledger shape is unchanged."""
    _call(cfg, tmp_cache, tmp_usage_log, FakeChatClient(reasoning_content=REASONING))
    record = json.loads(tmp_usage_log.read_text(encoding="utf-8").splitlines()[0])
    assert "reasoning_content" not in record
