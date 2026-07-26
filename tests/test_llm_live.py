"""Optional live smoke test against a real provider. Skipped unless GW_LIVE_TESTS=1.

The rest of the suite is offline and free by construction. This one file is the
only thing that opens a socket, and it stays skipped by default so ``pytest``
never costs money or needs a key.

    GW_LIVE_TESTS=1 pytest -m live

It uses the ``extraction`` step, which routes to glm-4.5-flash (free tier), and
writes its cache and ledger into tmp_path so the repo's .cache/ and the tracked
logs/llm_usage.jsonl are untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from german_wiki.llm import Prompt, complete
from german_wiki.llm._usage import read_records

LIVE = os.environ.get("GW_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE, reason="set GW_LIVE_TESTS=1 to run live model API tests"),
]

PROMPT = Prompt(
    system="Du bist ein knapper Deutschlehrer. Antworte mit einem einzigen Wort.",
    variable="Was ist der bestimmte Artikel von 'Küche'?",
    version="smoke@1",
)


# GLM-4.5 is a reasoning model: it spends completion tokens on internal reasoning
# BEFORE emitting any visible content. A tight cap (32 was the first try here)
# returns finish_reason="length" with content="" -- the budget ran out mid-thought.
# Leave room for reasoning plus the answer.
MAX_TOKENS = 512


def test_live_call_then_cache_hit(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    usage_log = tmp_path / "llm_usage.jsonl"

    first = complete(
        "extraction", PROMPT, cache_dir=cache_dir, usage_log=usage_log, max_tokens=MAX_TOKENS
    )
    assert first.cached is False
    # Assert this before the text, so a truncated response fails with a readable
    # reason rather than an opaque empty string.
    assert first.finish_reason == "stop", (
        f"truncated at {first.usage.completion_tokens} completion tokens; "
        f"raise MAX_TOKENS (reasoning tokens count toward the cap)"
    )
    assert first.text.strip()
    assert first.usage.prompt_tokens > 0
    # glm-4.5-flash is priced at explicit zeros: known free, not unpriced.
    assert first.cost_usd == 0.0

    second = complete(
        "extraction", PROMPT, cache_dir=cache_dir, usage_log=usage_log, max_tokens=MAX_TOKENS
    )
    assert second.cached is True
    assert second.text == first.text

    records = read_records(usage_log=usage_log)
    assert len(records) == 2
    assert [r["cached"] for r in records] == [False, True]
