"""Cost math from per-1M rates, and the refusal to guess an unconfigured one."""

from __future__ import annotations

import logging

import pytest

from german_wiki.llm import _pricing
from german_wiki.llm._settings import ModelPricing
from german_wiki.llm._usage import Usage

FREE = ModelPricing(input=0.0, cached_input=0.0, output=0.0)
PAID = ModelPricing(input=0.6, cached_input=0.11, output=2.2)


@pytest.mark.parametrize(
    "pricing,usage,expected",
    [
        (FREE, Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000), 0.0),
        (PAID, Usage(prompt_tokens=1_000_000), 0.6),
        (PAID, Usage(completion_tokens=1_000_000), 2.2),
        (PAID, Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000), 2.8),
        # Half the prompt served from the provider's own cache, billed cheaper.
        (
            PAID,
            Usage(prompt_tokens=1_000_000, cached_prompt_tokens=500_000),
            0.5 * 0.6 + 0.5 * 0.11,
        ),
        (PAID, Usage(), 0.0),
    ],
    ids=["free", "input", "output", "both", "cached-split", "no-tokens"],
)
def test_estimate_cost(pricing, usage, expected) -> None:
    assert _pricing.estimate_cost(pricing, usage) == pytest.approx(expected)


def test_cached_input_falls_back_to_the_full_input_rate() -> None:
    """Absent a discounted rate, over-report rather than under-report."""
    pricing = ModelPricing(input=1.0, output=0.0)
    usage = Usage(prompt_tokens=1_000_000, cached_prompt_tokens=1_000_000)
    assert _pricing.estimate_cost(pricing, usage) == pytest.approx(1.0)


def test_more_cached_than_prompt_tokens_does_not_go_negative() -> None:
    usage = Usage(prompt_tokens=100, cached_prompt_tokens=500)
    assert _pricing.estimate_cost(PAID, usage) >= 0.0


def test_unpriced_model_returns_none_not_zero() -> None:
    """A missing rate is unknown, not free -- those must stay distinguishable."""
    assert _pricing.estimate_cost(None, Usage(prompt_tokens=1_000)) is None


def test_known_free_model_returns_zero_not_none() -> None:
    assert _pricing.estimate_cost(FREE, Usage(prompt_tokens=1_000)) == 0.0


def test_unpriced_model_warns_once_per_model() -> None:
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger("german_wiki.llm._pricing").addHandler(_Collector())
    _pricing._warned.discard(("zai", "test-only-model"))

    for _ in range(3):
        _pricing.warn_if_unpriced("zai", "test-only-model", None)

    messages = [r.getMessage() for r in collected if "test-only-model" in r.getMessage()]
    assert len(messages) == 1
    assert "tokens logged, cost not tracked" in messages[0]


def test_priced_model_does_not_warn() -> None:
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger("german_wiki.llm._pricing").addHandler(_Collector())
    _pricing.warn_if_unpriced("zai", "priced-test-model", FREE)

    assert not [r for r in collected if "priced-test-model" in r.getMessage()]
