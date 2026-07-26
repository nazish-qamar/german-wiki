"""Estimated cost from token counts, using rates configured in ``config/models.yaml``.

Rates are per 1M tokens, which is how both providers publish them.

**A model with no configured rate is "unpriced", never estimated.** Token counts
are still recorded from the API response; only the dollar figure is withheld, as
``None``. An invented rate would go stale silently and corrupt the running total,
which is worse than an honest gap -- so the gap is made visible instead, by
warning once per ``(provider, model)``.

``glm-4.5-flash`` carries explicit zeros rather than being omitted, so "known
free" stays distinguishable from "price unknown".
"""

from __future__ import annotations

from ..logutil import get_logger
from ._settings import ModelPricing
from ._usage import Usage

logger = get_logger(__name__)

# One warning per (provider, model) per process, not one per call.
_warned: set[tuple[str, str]] = set()


def estimate_cost(pricing: ModelPricing | None, usage: Usage) -> float | None:
    """USD for one call, or ``None`` when the model has no configured rate.

    Cached input tokens bill at ``cached_input`` when the provider publishes a
    discounted rate; absent that, they fall back to the full input rate, which
    over-reports rather than under-reports.
    """
    if pricing is None:
        return None

    cached_rate = pricing.cached_input if pricing.cached_input is not None else pricing.input
    # max(): guards a provider reporting more cached tokens than prompt tokens.
    fresh_input = max(usage.prompt_tokens - usage.cached_prompt_tokens, 0)
    return (
        fresh_input * pricing.input
        + usage.cached_prompt_tokens * cached_rate
        + usage.completion_tokens * pricing.output
    ) / 1_000_000


def warn_if_unpriced(provider: str, model: str, pricing: ModelPricing | None) -> None:
    """Surface the gap once, so an untracked model doesn't stay invisible."""
    if pricing is not None:
        return
    if (provider, model) in _warned:
        return
    _warned.add((provider, model))
    logger.warning(
        "no pricing entry for %s/%s; tokens logged, cost not tracked. "
        "Add a rate under pricing in config/models.yaml to track spend.",
        provider,
        model,
    )
