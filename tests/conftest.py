"""Shared fixtures. The four seed nodes in /nodes are the primary fixtures.

Tests only READ the real /nodes and /vocab; anything mutable (DB, appended
vocab, touched files) is done against tmp copies so the repo is never altered.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import PromptTokensDetails

from german_wiki import config, storage

SEED_IDS = [
    "familie-waschen",
    "prefix-an",
    "um-hilfe-bitten",
    "wechselpraepositionen",
]


@pytest.fixture
def nodes_dir() -> Path:
    """The real /nodes directory (read-only in tests)."""
    return config.NODES_DIR


@pytest.fixture
def seed_paths(nodes_dir: Path) -> dict[str, Path]:
    return {sid: nodes_dir / f"{sid}.md" for sid in SEED_IDS}


@pytest.fixture
def seed_nodes(seed_paths: dict[str, Path]) -> dict[str, storage.Node]:
    return {sid: storage.load_node(p) for sid, p in seed_paths.items()}


@pytest.fixture
def tmp_nodes(tmp_path: Path, nodes_dir: Path) -> Path:
    """A writable copy of /nodes so tests can touch files for staleness."""
    dst = tmp_path / "nodes"
    dst.mkdir()
    for md in nodes_dir.glob("*.md"):
        shutil.copy2(md, dst / md.name)
    return dst


@pytest.fixture
def tmp_vocab(tmp_path: Path) -> Path:
    """A writable copy of /vocab so normalization appends don't touch the repo."""
    dst = tmp_path / "vocab"
    dst.mkdir()
    for name in ("themes.txt", "registers.txt", "aliases.yaml"):
        shutil.copy2(config.VOCAB_DIR / name, dst / name)
    return dst


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "index.db"


# --- slice 2: model layer ---


@pytest.fixture
def models_config() -> Path:
    """The real config/models.yaml (read-only in tests)."""
    return config.MODELS_CONFIG_PATH


@pytest.fixture
def tmp_cache(tmp_path: Path) -> Path:
    """A throwaway model-call cache; the repo's .cache/ is never touched."""
    return tmp_path / "cache"


@pytest.fixture
def tmp_usage_log(tmp_path: Path) -> Path:
    """A throwaway usage ledger; the tracked logs/llm_usage.jsonl is never touched."""
    return tmp_path / "llm_usage.jsonl"


class FakeChatClient:
    """Stand-in for ``openai.OpenAI`` exposing only ``.chat.completions.create``.

    It returns *real* openai SDK response types, so attribute access in
    ``_client.py`` is exercised against the same shapes the live SDK produces.

    ``calls`` records the kwargs of every request; ``call_count`` is how the
    suite proves a cache hit issued zero requests, which is the load-bearing
    assertion for ADR-005. Pass ``text`` as a list to return a different body per
    call, which distinguishes a genuine hit from a coincidental re-fetch.
    """

    def __init__(
        self,
        *,
        text: str | list[str] = "Antwort",
        prompt_tokens: int = 100,
        completion_tokens: int = 10,
        cached_tokens: int = 0,
        finish_reason: str = "stop",
        reasoning_content: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self._texts = [text] if isinstance(text, str) else list(text)
        self._reasoning_content = reasoning_content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._cached_tokens = cached_tokens
        self._finish_reason = finish_reason
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _create(self, **kwargs) -> ChatCompletion:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self._texts) - 1)
        # reasoning_content is a provider extension, not an SDK field; the SDK
        # models allow extras, so omitting it reproduces a provider that never
        # sends one (attribute reads back as None).
        extra = (
            {"reasoning_content": self._reasoning_content}
            if self._reasoning_content is not None
            else {}
        )
        return ChatCompletion(
            id=f"chatcmpl-fake-{len(self.calls)}",
            created=0,
            model=kwargs["model"],
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason=self._finish_reason,
                    message=ChatCompletionMessage(
                        role="assistant", content=self._texts[index], **extra
                    ),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                total_tokens=self._prompt_tokens + self._completion_tokens,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=self._cached_tokens),
            ),
        )


@pytest.fixture
def fake_client() -> FakeChatClient:
    return FakeChatClient()


# --- slice 3: ingestion ---


@pytest.fixture
def tmp_raw(tmp_path: Path) -> Path:
    """A throwaway /raw; the repo's tracked raw/ is never written by tests."""
    return tmp_path / "raw"


@pytest.fixture
def tmp_queue(tmp_path: Path) -> Path:
    """A throwaway /queue."""
    return tmp_path / "queue"
