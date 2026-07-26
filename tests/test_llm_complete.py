"""complete(): cache hit/miss against a fake client, and the refusal paths."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml
from conftest import FakeChatClient

from german_wiki.llm import _cache, _client, _settings
from german_wiki.llm._client import complete
from german_wiki.llm._prompt import Prompt

CONFIG = {
    "version": 1,
    "providers": {
        "zai": {
            "kind": "api",
            "base_url": "https://example.invalid/v4",
            "api_key_env": "ZAI_API_KEY",
        },
        "local": {"kind": "local"},
    },
    "pricing": {"zai": {"free-model": {"input": 0.0, "cached_input": 0.0, "output": 0.0}}},
    "defaults": {
        "provider": "zai",
        "model": "free-model",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "steps": {
        "extraction": {"status": "active"},
        "paid": {"status": "active", "model": "paid-model"},  # deliberately unpriced
        "later": {"status": "planned"},
        # ACTIVE and local: the only shape that exercises the ADR-004 guard.
        # If this were `planned`, the planned gate would fire first and the kind
        # check would never run -- the test would pass for the wrong reason.
        "vectors": {"status": "active", "provider": "local", "model": "e5-small"},
    },
}


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    return path


def _prompt(variable: str = "Der Text.", **overrides) -> Prompt:
    return Prompt(system="Du bist ein Lehrer.", variable=variable, **overrides)


def _call(cfg: Path, tmp_cache: Path, tmp_usage_log: Path, client, **overrides):
    kwargs = {
        "client": client,
        "settings_path": cfg,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
        "env": {},
    }
    kwargs.update(overrides)
    step = kwargs.pop("step", "extraction")
    prompt = kwargs.pop("prompt", _prompt())
    return complete(step, prompt, **kwargs)


def _records(tmp_usage_log: Path) -> list[dict]:
    return [json.loads(line) for line in tmp_usage_log.read_text(encoding="utf-8").splitlines()]


# --- the load-bearing behavior: identical input never re-calls (ADR-005) ---


def test_miss_calls_the_client_once_and_returns_its_text(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient(text="Antwort")
    response = _call(cfg, tmp_cache, tmp_usage_log, fake)
    assert response.text == "Antwort"
    assert response.cached is False
    assert fake.call_count == 1


def test_identical_call_is_served_from_cache_without_calling_the_client(
    cfg, tmp_cache, tmp_usage_log
) -> None:
    # Distinct bodies per call: a genuine hit must return the FIRST one.
    fake = FakeChatClient(text=["erste", "zweite"])
    first = _call(cfg, tmp_cache, tmp_usage_log, fake)
    second = _call(cfg, tmp_cache, tmp_usage_log, fake)

    assert fake.call_count == 1
    assert second.cached is True
    assert second.text == first.text == "erste"
    assert second.cost_usd == 0.0
    assert second.saved_usd == first.cost_usd
    assert second.cache_key == first.cache_key


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": _prompt("Ein anderer Text.")},
        {"prompt": _prompt(version="extract@2")},
        {"model": "other-model"},
        {"temperature": 0.9},
        {"response_format": {"type": "json_object"}},
        {"seed": 7},
    ],
    ids=["variable", "prompt-version", "model", "temperature", "response-format", "seed"],
)
def test_changing_any_hashed_input_is_a_miss(cfg, tmp_cache, tmp_usage_log, overrides) -> None:
    fake = FakeChatClient()
    _call(cfg, tmp_cache, tmp_usage_log, fake)
    _call(cfg, tmp_cache, tmp_usage_log, fake, **overrides)
    assert fake.call_count == 2


def test_use_cache_false_always_calls_and_stores_nothing(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    _call(cfg, tmp_cache, tmp_usage_log, fake, use_cache=False)
    _call(cfg, tmp_cache, tmp_usage_log, fake, use_cache=False)
    assert fake.call_count == 2
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 0


def test_env_kill_switch_disables_the_cache(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    env = {"GW_LLM_NO_CACHE": "1"}
    _call(cfg, tmp_cache, tmp_usage_log, fake, env=env)
    _call(cfg, tmp_cache, tmp_usage_log, fake, env=env)
    assert fake.call_count == 2


def test_refresh_skips_the_read_and_overwrites_the_entry(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient(text=["erste", "zweite"])
    _call(cfg, tmp_cache, tmp_usage_log, fake)
    refreshed = _call(cfg, tmp_cache, tmp_usage_log, fake, refresh=True)

    assert fake.call_count == 2
    assert refreshed.text == "zweite"
    assert refreshed.cached is False
    # The stored entry now holds the fresh body.
    assert _call(cfg, tmp_cache, tmp_usage_log, fake).text == "zweite"
    assert fake.call_count == 2


# --- the request that goes on the wire ---


def test_request_kwargs_shape(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    _call(cfg, tmp_cache, tmp_usage_log, fake)
    sent = fake.calls[0]

    assert sent["model"] == "free-model"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 4096
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][-1]["content"] == "Der Text."
    # Unset optionals are omitted, not sent as None.
    assert "response_format" not in sent
    assert "seed" not in sent
    assert "stream" not in sent


def test_response_format_and_seed_are_passed_through(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    _call(
        cfg,
        tmp_cache,
        tmp_usage_log,
        fake,
        response_format={"type": "json_object"},
        seed=7,
    )
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert fake.calls[0]["seed"] == 7


def test_explicit_model_overrides_config(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    response = _call(cfg, tmp_cache, tmp_usage_log, fake, model="override-model")
    assert fake.calls[0]["model"] == "override-model"
    assert response.model == "override-model"


# --- the ledger ---


def test_every_call_including_hits_lands_one_record(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    _call(cfg, tmp_cache, tmp_usage_log, fake)
    _call(cfg, tmp_cache, tmp_usage_log, fake)

    records = _records(tmp_usage_log)
    assert len(records) == 2
    assert records[0]["cached"] is False
    assert records[1]["cached"] is True
    assert records[1]["cost_usd"] == 0.0
    assert records[1]["latency_ms"] == 0


def test_cached_prompt_tokens_are_recorded(cfg, tmp_cache, tmp_usage_log) -> None:
    """SPEC §10 feedback loop: provider prompt-cache hits must be observable."""
    fake = FakeChatClient(prompt_tokens=100, cached_tokens=90)
    response = _call(cfg, tmp_cache, tmp_usage_log, fake)
    assert response.usage.cached_prompt_tokens == 90
    assert _records(tmp_usage_log)[0]["cached_prompt_tokens"] == 90


def test_unpriced_model_records_null_cost_and_warns_once(cfg, tmp_cache, tmp_usage_log) -> None:
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger("german_wiki.llm._pricing").addHandler(_Collector())
    from german_wiki.llm import _pricing

    _pricing._warned.discard(("zai", "paid-model"))

    fake = FakeChatClient()
    response = _call(cfg, tmp_cache, tmp_usage_log, fake, step="paid")

    assert response.cost_usd is None
    record = _records(tmp_usage_log)[0]
    assert record["cost_usd"] is None
    assert record["priced"] is False
    assert record["prompt_tokens"] == 100  # tokens still counted
    assert len([r for r in collected if "paid-model" in r.getMessage()]) == 1


def test_cache_hit_on_an_unpriced_model_stays_labelled_unpriced(
    cfg, tmp_cache, tmp_usage_log
) -> None:
    """A hit costs 0.0, but that must not relabel the model as priced."""
    fake = FakeChatClient()
    _call(cfg, tmp_cache, tmp_usage_log, fake, step="paid")
    _call(cfg, tmp_cache, tmp_usage_log, fake, step="paid")

    hit = _records(tmp_usage_log)[1]
    assert hit["cached"] is True
    assert hit["cost_usd"] == 0.0
    assert hit["priced"] is False


def test_client_error_is_logged_and_reraised(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _call(cfg, tmp_cache, tmp_usage_log, fake)

    record = _records(tmp_usage_log)[0]
    assert record["error"] == "RuntimeError: boom"
    assert record["total_tokens"] == 0
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 0


# --- refusals: each must raise BEFORE any network call ---


def test_unknown_step_raises_before_calling_the_client(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient()
    with pytest.raises(ValueError, match="unknown pipeline step"):
        _call(cfg, tmp_cache, tmp_usage_log, fake, step="nope")
    assert fake.call_count == 0


def test_planned_step_raises_before_calling_the_client(cfg, tmp_cache, tmp_usage_log) -> None:
    """The planned gate: reserved routing must not be reachable."""
    fake = FakeChatClient()
    with pytest.raises(ValueError, match="is planned, not active") as excinfo:
        _call(cfg, tmp_cache, tmp_usage_log, fake, step="later")

    # Identify WHICH guard fired, not merely that something raised.
    assert "ADR-004" not in str(excinfo.value)
    assert fake.call_count == 0
    assert not tmp_usage_log.exists()


def test_active_local_step_is_refused_as_an_api_call(cfg, tmp_cache, tmp_usage_log) -> None:
    """ADR-004, proven on an ACTIVE local step so the kind check is what fires.

    Were the step `planned`, the planned gate would raise first and this guard
    would never be exercised -- the refusal path nothing else touches.
    """
    fake = FakeChatClient()
    with pytest.raises(ValueError, match=r"kind=local.*ADR-004") as excinfo:
        _call(cfg, tmp_cache, tmp_usage_log, fake, step="vectors")

    message = str(excinfo.value)
    assert "'vectors'" in message and "'local'" in message
    # The step is active, so the planned gate must NOT be what stopped this.
    assert "planned" not in message
    assert fake.call_count == 0
    assert not tmp_usage_log.exists()


def test_the_local_step_used_by_the_adr004_test_is_genuinely_active(cfg) -> None:
    """Guards the guard: if `vectors` ever became planned, the ADR-004 test above
    would start passing for the wrong reason and silently stop testing anything."""
    settings = _settings.load_settings(cfg)
    assert settings.steps["vectors"].status == "active"
    assert settings.providers["local"].kind == "local"


def test_shipped_embeddings_step_is_refused_even_when_forced_active(
    models_config, tmp_cache, tmp_usage_log
) -> None:
    """The real embeddings step, activated as slice 4 will: still never an API call."""
    settings = _settings.load_settings(models_config)
    settings.steps["embeddings"] = settings.steps["embeddings"].model_copy(
        update={"status": "active"}
    )
    assert settings.steps["embeddings"].status == "active"  # the gate under test is bypassed
    fake = FakeChatClient()

    with pytest.raises(ValueError, match=r"kind=local.*ADR-004") as excinfo:
        complete(
            "embeddings",
            _prompt(),
            client=fake,
            settings=settings,
            cache_dir=tmp_cache,
            usage_log=tmp_usage_log,
            env={},
        )
    assert "planned" not in str(excinfo.value)
    assert fake.call_count == 0


def test_local_step_is_refused_even_with_an_explicit_client(cfg, tmp_cache, tmp_usage_log) -> None:
    """Injecting a client must not be a way around the ADR-004 guard."""
    with pytest.raises(ValueError, match=r"kind=local.*ADR-004"):
        complete(
            "vectors",
            _prompt(),
            client=FakeChatClient(),
            settings_path=cfg,
            cache_dir=tmp_cache,
            usage_log=tmp_usage_log,
            env={},
        )


# --- credentials ---


def test_missing_api_key_raises_and_names_the_env_var(cfg, tmp_cache, tmp_usage_log) -> None:
    """No client is constructed and no socket opened."""
    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        complete(
            "extraction",
            _prompt(),
            settings_path=cfg,
            cache_dir=tmp_cache,
            usage_log=tmp_usage_log,
            env={},
        )


def test_an_explicit_client_needs_no_api_key(cfg, tmp_cache, tmp_usage_log) -> None:
    response = _call(cfg, tmp_cache, tmp_usage_log, FakeChatClient())
    assert response.text == "Antwort"


def test_build_client_is_memoized_per_endpoint(cfg) -> None:
    resolved = _settings.resolve_step("extraction", settings_path=cfg, env={})
    env = {"ZAI_API_KEY": "test-key"}
    assert _client.build_client(resolved, env=env) is _client.build_client(resolved, env=env)
