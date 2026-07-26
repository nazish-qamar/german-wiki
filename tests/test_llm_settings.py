"""Per-step routing: inheritance, override precedence, and the active/planned gate."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from german_wiki.llm import _settings

# A small purpose-built config. Deliberately exercises inheritance (``inherits``
# declares only a status), an explicit override, a fallback, a planned step and
# a local-provider step.
BASE: dict = {
    "version": 1,
    "providers": {
        "zai": {
            "kind": "api",
            "base_url": "https://example.invalid/v4",
            "api_key_env": "ZAI_API_KEY",
            "timeout_s": 30,
            "max_retries": 1,
        },
        "deepseek": {
            "kind": "api",
            "base_url": "https://deepseek.invalid",
            "api_key_env": "DEEPSEEK_API_KEY",
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
        "inherits": {"status": "active"},
        "overrides": {
            "status": "active",
            "provider": "deepseek",
            "model": "paid-model",
            "temperature": 0.7,
            "max_tokens": 128,
            "fallback": {"provider": "zai", "model": "free-model"},
        },
        "later": {"status": "planned", "model": "paid-model"},
        "vectors": {"status": "active", "provider": "local", "model": "e5-small"},
    },
}


def _write(tmp_path: Path, data: dict | None = None) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(data if data is not None else BASE, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _mutate(**changes) -> dict:
    """A deep copy of BASE with top-level keys replaced."""
    data = copy.deepcopy(BASE)
    data.update(changes)
    return data


# --- the shipped config ---


def test_shipped_config_parses(models_config: Path) -> None:
    settings = _settings.load_settings(models_config)
    assert "extraction" in settings.steps
    assert settings.defaults.provider == "zai"


def test_shipped_active_steps_all_resolve(models_config: Path) -> None:
    settings = _settings.load_settings(models_config)
    active = [name for name, step in settings.steps.items() if step.status == "active"]
    assert active, "at least one step must be callable"
    for name in active:
        resolved = _settings.resolve_step(name, settings=settings, env={})
        assert resolved.model
        assert resolved.provider in settings.providers


def test_shipped_embeddings_is_local_and_planned(models_config: Path) -> None:
    """ADR-004: embeddings are never an API step."""
    settings = _settings.load_settings(models_config)
    step = settings.steps["embeddings"]
    assert step.provider == "local"
    assert settings.providers["local"].kind == "local"


def test_shipped_only_free_models_are_priced(models_config: Path) -> None:
    """No guessed rates: a paid model must be absent, not estimated."""
    settings = _settings.load_settings(models_config)
    priced = {model for models in settings.pricing.values() for model in models}
    assert priced == {"glm-4.5-flash"}
    for models in settings.pricing.values():
        for pricing in models.values():
            assert (pricing.input, pricing.output) == (0.0, 0.0)


# --- inheritance ---


def test_step_inherits_unspecified_fields_from_defaults(tmp_path: Path) -> None:
    resolved = _settings.resolve_step("inherits", settings_path=_write(tmp_path), env={})
    assert resolved.provider == "zai"
    assert resolved.model == "free-model"
    assert resolved.temperature == 0.0
    assert resolved.max_tokens == 4096


def test_step_block_overrides_defaults(tmp_path: Path) -> None:
    resolved = _settings.resolve_step("overrides", settings_path=_write(tmp_path), env={})
    assert resolved.provider == "deepseek"
    assert resolved.model == "paid-model"
    assert resolved.temperature == 0.7
    assert resolved.max_tokens == 128


def test_provider_block_supplies_endpoint_and_retries(tmp_path: Path) -> None:
    resolved = _settings.resolve_step("inherits", settings_path=_write(tmp_path), env={})
    assert resolved.base_url == "https://example.invalid/v4"
    assert resolved.api_key_env == "ZAI_API_KEY"
    assert resolved.timeout_s == 30
    assert resolved.max_retries == 1


def test_local_provider_resolves_without_endpoint(tmp_path: Path) -> None:
    resolved = _settings.resolve_step("vectors", settings_path=_write(tmp_path), env={})
    assert resolved.kind == "local"
    assert resolved.base_url is None
    assert resolved.api_key_env is None


# --- override precedence ---


@pytest.mark.parametrize(
    "kwargs,attr,expected",
    [
        ({"provider": "deepseek"}, "provider", "deepseek"),
        ({"model": "other"}, "model", "other"),
        ({"temperature": 0.9}, "temperature", 0.9),
        ({"max_tokens": 7}, "max_tokens", 7),
    ],
)
def test_explicit_args_beat_config(tmp_path: Path, kwargs, attr, expected) -> None:
    resolved = _settings.resolve_step("inherits", settings_path=_write(tmp_path), env={}, **kwargs)
    assert getattr(resolved, attr) == expected


def test_env_override_applies_to_every_step(tmp_path: Path) -> None:
    path = _write(tmp_path)
    env = {"GW_LLM_PROVIDER": "deepseek", "GW_LLM_MODEL": "env-model"}
    for step in ("inherits", "overrides"):
        resolved = _settings.resolve_step(step, settings_path=path, env=env)
        assert (resolved.provider, resolved.model) == ("deepseek", "env-model")


def test_explicit_arg_beats_env_override(tmp_path: Path) -> None:
    resolved = _settings.resolve_step(
        "inherits",
        settings_path=_write(tmp_path),
        env={"GW_LLM_MODEL": "env-model"},
        model="explicit-model",
    )
    assert resolved.model == "explicit-model"


def test_fallback_selects_the_configured_swap(tmp_path: Path) -> None:
    resolved = _settings.resolve_step(
        "overrides", settings_path=_write(tmp_path), env={}, fallback=True
    )
    assert (resolved.provider, resolved.model) == ("zai", "free-model")


def test_fallback_without_a_configured_swap_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no fallback configured"):
        _settings.resolve_step("inherits", settings_path=_write(tmp_path), env={}, fallback=True)


# --- the gates ---


def test_unknown_step_raises_and_lists_known_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown pipeline step: 'nope'") as excinfo:
        _settings.resolve_step("nope", settings_path=_write(tmp_path), env={})
    assert "inherits" in str(excinfo.value)


def test_planned_step_raises_and_names_the_config_file(tmp_path: Path) -> None:
    path = _write(tmp_path)
    with pytest.raises(ValueError, match="is planned, not active") as excinfo:
        _settings.resolve_step("later", settings_path=path, env={})
    assert str(path) in str(excinfo.value)


def test_status_defaults_to_planned_when_omitted(tmp_path: Path) -> None:
    """A step must opt in to being callable."""
    data = _mutate(steps={"bare": {"model": "x"}})
    with pytest.raises(ValueError, match="is planned, not active"):
        _settings.resolve_step("bare", settings_path=_write(tmp_path, data), env={})


# --- load-time validation ---


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no model config at"):
        _settings.load_settings(tmp_path / "absent.yaml")


def test_unknown_key_raises(tmp_path: Path) -> None:
    data = copy.deepcopy(BASE)
    data["steps"]["inherits"]["temprature"] = 0.5
    with pytest.raises(ValueError):
        _settings.load_settings(_write(tmp_path, data))


@pytest.mark.parametrize(
    "data,message",
    [
        (
            _mutate(steps={"x": {"status": "active", "provider": "ghost"}}),
            "unknown provider 'ghost'",
        ),
        (
            _mutate(defaults={"provider": "ghost", "model": "m"}),
            "unknown provider 'ghost'",
        ),
        (
            _mutate(pricing={"ghost": {"m": {"input": 1.0, "output": 1.0}}}),
            "unknown provider 'ghost'",
        ),
        (
            _mutate(
                providers={"zai": {"kind": "api", "base_url": "https://x.invalid"}},
                pricing={},
                steps={},
            ),
            "missing base_url",
        ),
    ],
)
def test_cross_reference_errors(tmp_path: Path, data, message) -> None:
    with pytest.raises(ValueError, match=message):
        _settings.load_settings(_write(tmp_path, data))


def test_step_fallback_to_unknown_provider_raises(tmp_path: Path) -> None:
    data = copy.deepcopy(BASE)
    data["steps"]["overrides"]["fallback"] = {"provider": "ghost", "model": "m"}
    with pytest.raises(ValueError, match="unknown provider 'ghost'"):
        _settings.load_settings(_write(tmp_path, data))


def test_env_override_to_unknown_provider_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolved to unknown provider 'ghost'"):
        _settings.resolve_step(
            "inherits", settings_path=_write(tmp_path), env={"GW_LLM_PROVIDER": "ghost"}
        )


# --- pricing lookup ---


def test_priced_model_carries_rates(tmp_path: Path) -> None:
    resolved = _settings.resolve_step("inherits", settings_path=_write(tmp_path), env={})
    assert resolved.pricing is not None
    assert resolved.pricing.input == 0.0


def test_unpriced_model_resolves_with_pricing_none(tmp_path: Path) -> None:
    """Absent from the table means unpriced -- never a guessed rate."""
    resolved = _settings.resolve_step("overrides", settings_path=_write(tmp_path), env={})
    assert resolved.pricing is None


# --- caching ---


def test_get_settings_returns_a_cached_instance(tmp_path: Path) -> None:
    path = _write(tmp_path)
    assert _settings.get_settings(path) is _settings.get_settings(path)
