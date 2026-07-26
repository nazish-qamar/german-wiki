"""Per-step model routing, parsed from ``config/models.yaml`` (SPEC §9).

Provider and model are configured *per pipeline step*, never hardcoded at a call
site. A step declares only what differs from ``defaults``; everything else is
inherited.

Two gates live here, and both fail loudly rather than guessing:

- An **unknown** step raises. There is no fall-through to ``defaults``, because a
  typo'd step name silently running the wrong model is exactly the failure this
  config exists to prevent.
- A **planned** step raises. The file documents the whole routing plan from SPEC
  §9, but only ``status: active`` steps are callable, so a later slice's config
  can sit in the repo without being reachable.

The third gate -- refusing to route a ``kind: local`` provider through an HTTP
call (ADR-004) -- lives in ``_client.complete()``, where the API call actually
happens, so it holds regardless of what ``status`` says.

Pricing is deliberately sparse: see ``_pricing`` for why an absent model is
"unpriced" rather than assigned a guessed rate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .. import config

ProviderKind = Literal["api", "local"]
StepStatus = Literal["active", "planned"]

ENV_PROVIDER = "GW_LLM_PROVIDER"
ENV_MODEL = "GW_LLM_MODEL"


class ProviderSettings(BaseModel):
    """One OpenAI-compatible endpoint, or the in-process local runner."""

    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind = "api"
    base_url: str | None = None  # required when kind == "api"
    api_key_env: str | None = None  # required when kind == "api"
    timeout_s: float = 60.0
    max_retries: int = 2


class ModelPricing(BaseModel):
    """USD per 1M tokens."""

    model_config = ConfigDict(extra="forbid")

    input: float
    output: float
    cached_input: float | None = None  # None -> bill cached tokens at ``input``


class StepFallback(BaseModel):
    """The manual swap target for a step (``complete(..., fallback=True)``)."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str


class DefaultSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


class StepSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Default to ``planned``: a step becomes callable only by opting in.
    status: StepStatus = "planned"
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    fallback: StepFallback | None = None


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = 1
    providers: dict[str, ProviderSettings]
    pricing: dict[str, dict[str, ModelPricing]] = Field(default_factory=dict)
    defaults: DefaultSettings
    steps: dict[str, StepSettings] = Field(default_factory=dict)


class ResolvedStep(BaseModel):
    """Everything one call needs, with no further lookups downstream."""

    model_config = ConfigDict(extra="forbid")

    step: str
    provider: str
    kind: ProviderKind
    model: str
    base_url: str | None
    api_key_env: str | None
    timeout_s: float
    max_retries: int
    temperature: float
    max_tokens: int | None
    pricing: ModelPricing | None


# --- loading ---


def _validate(settings: LLMSettings, path: Path) -> None:
    """Cross-reference checks Pydantic can't express field-locally."""
    known = set(settings.providers)

    for name, provider in settings.providers.items():
        if provider.kind == "api" and not (provider.base_url and provider.api_key_env):
            raise ValueError(
                f"provider {name!r} has kind: api but is missing "
                f"base_url and/or api_key_env in {path}"
            )

    def _check(provider: str | None, where: str) -> None:
        if provider is not None and provider not in known:
            raise ValueError(
                f"{where} references unknown provider {provider!r}; "
                f"known providers: {sorted(known)}"
            )

    _check(settings.defaults.provider, "defaults")
    for name, step in settings.steps.items():
        _check(step.provider, f"step {name!r}")
        if step.fallback is not None:
            _check(step.fallback.provider, f"step {name!r} fallback")
    for provider in settings.pricing:
        _check(provider, "pricing")


def load_settings(path: Path | str | None = None) -> LLMSettings:
    """Parse ``config/models.yaml``. Raises ``ValueError`` on anything unusable."""
    path = Path(path) if path is not None else config.MODELS_CONFIG_PATH
    if not path.is_file():
        raise ValueError(f"no model config at {path}; expected config/models.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    settings = LLMSettings.model_validate(data)
    _validate(settings, path)
    return settings


# --- module-level convenience over a per-path cached instance ---
_cache: dict[Path, LLMSettings] = {}


def get_settings(path: Path | str | None = None) -> LLMSettings:
    key = Path(path) if path is not None else config.MODELS_CONFIG_PATH
    if key not in _cache:
        _cache[key] = load_settings(key)
    return _cache[key]


# --- resolution ---


def _first(*candidates: object) -> object:
    for value in candidates:
        if value is not None:
            return value
    return None


def resolve_step(
    step: str,
    *,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    fallback: bool = False,
    env: Mapping[str, str] | None = None,
    settings_path: Path | str | None = None,
) -> ResolvedStep:
    """Resolve one pipeline step into a fully-specified call configuration.

    Precedence, first non-``None`` wins per field: explicit argument, then the
    ``GW_LLM_PROVIDER`` / ``GW_LLM_MODEL`` env overrides, then the step's
    ``fallback`` block when ``fallback=True``, then the step block, then
    ``defaults``.

    ``env`` is an injectable mapping (defaults to ``os.environ``) so callers and
    tests can override without touching process state.
    """
    path = Path(settings_path) if settings_path is not None else config.MODELS_CONFIG_PATH
    if settings is None:
        settings = get_settings(settings_path)
    env = os.environ if env is None else env

    if step not in settings.steps:
        raise ValueError(f"unknown pipeline step: {step!r}; known steps: {sorted(settings.steps)}")

    cfg = settings.steps[step]
    if cfg.status != "active":
        raise ValueError(
            f"pipeline step {step!r} is {cfg.status}, not active; "
            f"set status: active in {path} once the slice implementing it lands"
        )

    if fallback and cfg.fallback is None:
        raise ValueError(f"pipeline step {step!r} has no fallback configured in {path}")
    swap = cfg.fallback if fallback else None

    provider_name = _first(
        provider,
        env.get(ENV_PROVIDER),
        swap.provider if swap else None,
        cfg.provider,
        settings.defaults.provider,
    )
    model_name = _first(
        model,
        env.get(ENV_MODEL),
        swap.model if swap else None,
        cfg.model,
        settings.defaults.model,
    )

    if provider_name not in settings.providers:
        raise ValueError(
            f"step {step!r} resolved to unknown provider {provider_name!r}; "
            f"known providers: {sorted(settings.providers)}"
        )
    endpoint = settings.providers[provider_name]

    return ResolvedStep(
        step=step,
        provider=provider_name,
        kind=endpoint.kind,
        model=model_name,
        base_url=endpoint.base_url,
        api_key_env=endpoint.api_key_env,
        timeout_s=endpoint.timeout_s,
        max_retries=endpoint.max_retries,
        temperature=_first(temperature, cfg.temperature, settings.defaults.temperature),
        max_tokens=_first(max_tokens, cfg.max_tokens, settings.defaults.max_tokens),
        pricing=settings.pricing.get(provider_name, {}).get(model_name),
    )
