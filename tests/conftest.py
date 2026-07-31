"""Shared fixtures. The four hand-authored seed nodes are the primary fixtures.

They live in ``tests/fixtures/nodes``, **not** in the repo's ``/nodes``, and that
separation became load-bearing in slice 5.

Slices 1-4 could safely treat ``/nodes`` as a fixed four-node corpus, because ADR-003
meant nothing was allowed to write there. ``gw review`` is the first code that does, so
``/nodes`` is now a *living wiki* that grows every time a proposal is approved. Tests
asserting on its exact contents would break on every study session -- and re-pinning
them to a fifth node, then a sixth, only defers the breakage by one.

So the split is: **the app reads the real ``/nodes``; the tests read a frozen copy.**
ADR-010's threshold calibration ("the four seeds report zero pairs") keeps its meaning
precisely because the set can no longer shift underneath it.

One deliberate exception: ``test_merge_live.py`` reaches for the live ``/nodes`` and
``/queue`` on purpose -- it asserts against real ingested material, and is gated behind
``GW_LIVE_TESTS=1`` for exactly that reason.

Tests only READ the fixtures and /vocab; anything mutable (DB, appended vocab,
touched files) is done against tmp copies so the repo is never altered.
"""

from __future__ import annotations

import hashlib
import math
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
    "wechselpräpositionen",
]

# The frozen corpus. Copied from the hand-authored seeds at the point slice 5 made
# /nodes mutable; it is a test asset now and should change only when a test needs it to.
SEED_DIR = Path(__file__).parent / "fixtures" / "nodes"


@pytest.fixture
def nodes_dir() -> Path:
    """The frozen seed corpus (read-only in tests) -- NOT the live ``/nodes``.

    Everything derived from this fixture (``seed_paths``, ``seed_nodes``,
    ``tmp_nodes``) inherits the isolation, which is why repointing it here was the
    whole change.
    """
    return SEED_DIR


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


# --- slice 4: embeddings ---


class FakeEmbedder:
    """Deterministic stand-in for the local embedding model.

    Mirrors ``FakeChatClient``: real output shape, a call counter, and no network.
    The offline suite must never load sentence-transformers or download the ~470MB
    e5 weights, so every non-slow test injects one of these.

    Vectors are derived from a hash of the text, so they are stable across runs and
    unrelated texts land far apart -- but ``similar_to`` lets a test place two texts
    deliberately close, which is what makes the semantic-tier bands testable without
    a real model.

    ``encode_count`` counts *calls*, ``encoded`` counts texts; both are how the
    suite proves a cache hit did no work.
    """

    def __init__(
        self,
        *,
        dimension: int = 384,
        model: str = "fake-embedder",
        similar_to: dict[str, str] | None = None,
    ) -> None:
        self.dimension = dimension
        self.model = model
        # text -> the text whose vector it should sit near (plus a small nudge)
        self._similar_to = similar_to or {}
        self.calls: list[list[str]] = []
        self.encoded: list[str] = []

    @property
    def encode_count(self) -> int:
        return len(self.calls)

    def _base_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimension)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.encoded.extend(texts)

        out = []
        for text in texts:
            anchor = self._similar_to.get(text)
            if anchor is None:
                out.append(self._base_vector(text))
                continue
            # Nudge a copy of the anchor's vector so cosine lands just below 1.0.
            base = self._base_vector(anchor)
            nudged = [v + (0.05 if i % 7 == 0 else 0.0) for i, v in enumerate(base)]
            norm = math.sqrt(sum(v * v for v in nudged)) or 1.0
            out.append([v / norm for v in nudged])
        return out


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --- slice 5: merge pipeline + review ---


@pytest.fixture
def tmp_proposals(tmp_path: Path) -> Path:
    """A throwaway /proposals; the repo's proposals/ is never written by tests."""
    return tmp_path / "proposals"


@pytest.fixture
def tmp_merged(tmp_path: Path) -> Path:
    """A throwaway /_merged; the tracked archive is never written by tests."""
    return tmp_path / "_merged"


@pytest.fixture
def tmp_decisions(tmp_path: Path) -> Path:
    """A throwaway decision ledger; the tracked logs/decisions.jsonl is never touched."""
    return tmp_path / "decisions.jsonl"
