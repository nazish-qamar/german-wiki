"""``complete()`` -- the one model-call interface every later slice uses.

One OpenAI-compatible client, with provider and model resolved per pipeline step
from config (CLAUDE.md). Z.AI and DeepSeek differ only by base URL and key.

Every call is wrapped by the content-hash cache (ADR-005) and lands one record in
the usage ledger (SPEC §10) -- cache hits included, so the hit rate and the money
the cache saved are both visible.

Three refusals, each raising before any network call:

- an unknown step, and
- a ``planned`` step (both from ``_settings.resolve_step``), and
- a step whose provider is ``kind: local``.

That last one is ADR-004's enforcement: embeddings run in-process via
sentence-transformers and must never become an API call. It is checked *here*,
where the HTTP request would happen, and independently of ``status`` -- so
flipping ``embeddings`` to active in slice 4 still cannot turn it into one.

Deferred to a later slice: automatic cross-provider failover. Swapping is manual
(``provider=``, ``fallback=True``, ``GW_LLM_PROVIDER``, or a config edit) because
auto-failover raises questions -- shared cache key or not, double-logging, what
counts as fatal -- better answered against a real failure than guessed at now.
Timeouts and retries are the SDK's, configured per provider.

No response parsing lives here. ``response_format`` is passed through and hashed,
but interpreting what comes back belongs to slice 3.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from .. import config
from . import _cache
from ._pricing import estimate_cost, warn_if_unpriced
from ._prompt import Prompt
from ._settings import LLMSettings, ResolvedStep, resolve_step
from ._usage import Usage, log_call

# Convenience for callers asking for JSON back (slice 3 will want this).
JSON_OBJECT: dict[str, str] = {"type": "json_object"}

ENV_NO_CACHE = "GW_LLM_NO_CACHE"


class ChatClient(Protocol):
    """The only seam this slice needs: ``.chat.completions.create(**kwargs)``.

    ``openai.OpenAI`` satisfies it; tests pass a fake, injected explicitly rather
    than monkeypatched.
    """

    chat: Any


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    step: str
    provider: str
    model: str
    usage: Usage
    cached: bool
    cost_usd: float | None  # 0.0 on a hit; None when the model is unpriced
    saved_usd: float  # what a hit avoided; 0.0 on a miss
    cache_key: str
    finish_reason: str | None = None
    # Provider-specific reasoning trace (GLM/DeepSeek return it; absence is normal).
    # Capture-only: never parsed, never in the cache key, never written to /nodes or
    # /raw. It exists so a truncated or garbage response is inspectable -- see
    # `finish_reason == "length"`, which slice 3's extraction treats as a failure.
    reasoning_content: str | None = None


# --- client construction ---

_dotenv_loaded = False
_clients: dict[tuple[str | None, str | None], OpenAI] = {}


def _ensure_dotenv(path: Path | str | None = None) -> None:
    """Load .env once, lazily. Never at import time, and never overriding a real env var."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    load_dotenv(Path(path) if path is not None else config.DOTENV_PATH, override=False)
    _dotenv_loaded = True


def build_client(
    step: ResolvedStep,
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = None,
) -> OpenAI:
    """An OpenAI-compatible client for this step's provider, memoized per endpoint."""
    if env is None:
        # Only touch process state when we are actually reading process state.
        _ensure_dotenv(dotenv_path)
        env = os.environ

    api_key = env.get(step.api_key_env) if step.api_key_env else None
    if not api_key:
        raise ValueError(
            f"missing API key for provider {step.provider!r}: set {step.api_key_env} in .env"
        )

    endpoint = (step.base_url, step.api_key_env)
    if endpoint not in _clients:
        _clients[endpoint] = OpenAI(
            api_key=api_key,
            base_url=step.base_url,
            timeout=step.timeout_s,
            max_retries=step.max_retries,
        )
    return _clients[endpoint]


# --- response unpacking ---


def _usage_of(response: Any) -> Usage:
    raw = getattr(response, "usage", None)
    if raw is None:
        return Usage()
    details = getattr(raw, "prompt_tokens_details", None)
    return Usage(
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        total_tokens=getattr(raw, "total_tokens", 0) or 0,
        cached_prompt_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
    )


def _text_of(response: Any) -> tuple[str, str | None, str | None]:
    """Return ``(content, finish_reason, reasoning_content)``.

    ``reasoning_content`` is non-standard -- GLM and DeepSeek return it, others do
    not -- so its absence is normal, not an error. It is captured for debugging
    only: nothing here or downstream parses it or branches on its value.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "", None, None
    message = choices[0].message
    return (
        getattr(message, "content", None) or "",
        choices[0].finish_reason,
        getattr(message, "reasoning_content", None) or None,
    )


# --- the public call ---


def complete(
    step: str,
    prompt: Prompt,
    *,
    client: ChatClient | None = None,
    settings: LLMSettings | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    seed: int | None = None,
    fallback: bool = False,
    use_cache: bool = True,
    refresh: bool = False,
    settings_path: Path | str | None = None,
    cache_dir: Path | str | None = None,
    usage_log: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = None,
) -> ModelResponse:
    """Run one model call for ``step``, served from cache when identical input recurs.

    ``use_cache=False`` skips both read and write; ``refresh=True`` skips the read
    but stores the fresh result. ``GW_LLM_NO_CACHE=1`` disables the cache globally.
    """
    lookup = os.environ if env is None else env

    resolved = resolve_step(
        step,
        settings=settings,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        fallback=fallback,
        env=lookup,
        settings_path=settings_path,
    )

    # ADR-004. Checked independently of `status`, so activating a local step in a
    # later slice still cannot route it through HTTP.
    if resolved.kind != "api":
        raise ValueError(
            f"step {step!r} uses provider {resolved.provider!r} (kind={resolved.kind}) "
            "and cannot be an API call; embeddings are always local (ADR-004)"
        )

    messages = prompt.to_messages()
    material = _cache.key_material(
        provider=resolved.provider,
        model=resolved.model,
        messages=messages,
        temperature=resolved.temperature,
        max_tokens=resolved.max_tokens,
        response_format=response_format,
        seed=seed,
        prompt_version=prompt.version,
    )
    key = _cache.cache_key(material)
    caching = use_cache and lookup.get(ENV_NO_CACHE) not in ("1", "true", "True")
    priced = resolved.pricing is not None

    def _record(**fields: Any) -> None:
        log_call(
            step=step,
            provider=resolved.provider,
            model=resolved.model,
            cache_key=key,
            usage_log=usage_log,
            **fields,
        )

    if caching and not refresh:
        hit = _cache.read(key, cache_dir=cache_dir, expect_request=material)
        if hit is not None:
            usage = Usage(**hit["usage"])
            # A hit spends nothing, priced or not; saved_usd carries what it avoided.
            _record(
                cached=True,
                usage=usage,
                cost_usd=0.0,
                priced=priced,
                saved_usd=hit.get("cost_usd") or 0.0,
                latency_ms=0,
                finish_reason=hit.get("finish_reason"),
            )
            return ModelResponse(
                text=hit["text"],
                step=step,
                provider=resolved.provider,
                model=resolved.model,
                usage=usage,
                cached=True,
                cost_usd=0.0,
                saved_usd=hit.get("cost_usd") or 0.0,
                cache_key=key,
                finish_reason=hit.get("finish_reason"),
                # Read back from the payload so a re-run of a truncated call can
                # still show WHY it truncated. Entries written before this field
                # existed simply read back as None.
                reasoning_content=hit.get("reasoning_content"),
            )

    if client is None:
        client = build_client(resolved, env=env, dotenv_path=dotenv_path)

    request: dict[str, Any] = {
        "model": resolved.model,
        "messages": messages,
        "temperature": resolved.temperature,
    }
    # Omitted rather than sent as None, so the wire request matches what was hashed.
    if resolved.max_tokens is not None:
        request["max_tokens"] = resolved.max_tokens
    if response_format is not None:
        request["response_format"] = response_format
    if seed is not None:
        request["seed"] = seed

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        _record(
            cached=False,
            usage=Usage(),
            cost_usd=None,
            priced=priced,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    latency_ms = int((time.perf_counter() - started) * 1000)

    text, finish_reason, reasoning_content = _text_of(response)
    usage = _usage_of(response)
    warn_if_unpriced(resolved.provider, resolved.model, resolved.pricing)
    cost_usd = estimate_cost(resolved.pricing, usage)

    if caching:
        _cache.write(
            key,
            {
                "key": key,
                "created_at": datetime.now(UTC).isoformat(),
                "step": step,
                "provider": resolved.provider,
                "model": resolved.model,
                "request": material,
                "text": text,
                "finish_reason": finish_reason,
                # In the payload but NOT in the key (see key_material): storing it
                # keeps a cached truncated call inspectable on re-run, while the
                # key stays exactly what it was before this field existed.
                "reasoning_content": reasoning_content,
                "response_id": getattr(response, "id", None),
                "usage": usage.model_dump(),
                "cost_usd": cost_usd,
            },
            cache_dir=cache_dir,
        )

    _record(
        cached=False,
        usage=usage,
        cost_usd=cost_usd,
        priced=priced,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
    )
    return ModelResponse(
        text=text,
        step=step,
        provider=resolved.provider,
        model=resolved.model,
        usage=usage,
        cached=False,
        cost_usd=cost_usd,
        saved_usd=0.0,
        cache_key=key,
        finish_reason=finish_reason,
        reasoning_content=reasoning_content,
    )
