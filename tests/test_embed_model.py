"""The local embedder: config-driven model id, and the dimension guard's timing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import FakeEmbedder

from german_wiki.db import EMBEDDING_DIM
from german_wiki.embed import _model

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
    "pricing": {},
    "defaults": {"provider": "zai", "model": "glm-4.5-flash", "temperature": 0.0},
    "steps": {
        "embeddings": {
            "status": "active",
            "provider": "local",
            "model": "intfloat/multilingual-e5-small",
        },
        "later": {"status": "planned", "provider": "local", "model": "some-model"},
    },
}


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    return path


# --- the protocol ---


def test_fake_embedder_satisfies_the_protocol(fake_embedder: FakeEmbedder) -> None:
    vectors = fake_embedder.encode(["query: eins", "query: zwei"])
    assert len(vectors) == 2
    assert all(len(v) == fake_embedder.dimension for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])


def test_fake_embedder_is_deterministic() -> None:
    a, b = FakeEmbedder(), FakeEmbedder()
    assert a.encode(["query: gleich"]) == b.encode(["query: gleich"])


def test_fake_embedder_counts_work(fake_embedder: FakeEmbedder) -> None:
    fake_embedder.encode(["a", "b"])
    fake_embedder.encode(["c"])
    assert fake_embedder.encode_count == 2
    assert fake_embedder.encoded == ["a", "b", "c"]


def test_fake_embedder_vectors_are_normalized(fake_embedder: FakeEmbedder) -> None:
    """Real encoding normalizes, so cosine == 1 - distance; the fake must match."""
    (vector,) = fake_embedder.encode(["query: Wechselpräpositionen"])
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-6)


# --- the model id comes from config, never a literal ---


def test_model_name_comes_from_config(cfg: Path) -> None:
    assert _model.embedding_model_name(settings_path=cfg) == "intfloat/multilingual-e5-small"


def test_a_planned_embeddings_step_still_raises(tmp_path: Path) -> None:
    """The active/planned gate applies to the local runner too."""
    data = {**CONFIG, "steps": {"embeddings": {"status": "planned", "provider": "local"}}}
    path = tmp_path / "planned.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="is planned, not active"):
        _model.embedding_model_name(settings_path=path)


# --- the dimension guard: it must fire at LOAD, before any encode ---


def test_dimension_mismatch_raises_naming_both_numbers() -> None:
    with pytest.raises(ValueError, match=r"768-dimensional.*float\[384\]") as excinfo:
        _model.check_dimension(768, "some/other-model", expect_dim=384)
    message = str(excinfo.value)
    assert "some/other-model" in message
    assert "EMBEDDING_DIM" in message  # tells you how to fix it


def test_matching_dimension_passes() -> None:
    assert _model.check_dimension(384, "intfloat/multilingual-e5-small", expect_dim=384) is None


def test_the_guard_fires_before_any_encode() -> None:
    """A wrong-width model must not produce a single vector.

    The failure this prevents is a wrong-width value reaching a fixed-width vec0
    column, where sqlite-vec may reject it loudly or accept it quietly. So the
    check has to happen at load time, not at first use.
    """
    wrong = FakeEmbedder(dimension=768)

    with pytest.raises(ValueError, match="768-dimensional"):
        _model.check_dimension(wrong.dimension, wrong.model, expect_dim=EMBEDDING_DIM)

    assert wrong.encode_count == 0
    assert wrong.encoded == []


def test_a_mismatched_embedder_never_reaches_the_store(tmp_db: Path) -> None:
    """End-to-end ordering: guard raises, so store_vectors is never called."""
    from german_wiki.db import connect, rebuild_schema
    from german_wiki.embed import _store

    conn = connect(tmp_db)
    try:
        rebuild_schema(conn)
        wrong = FakeEmbedder(dimension=8)

        with pytest.raises(ValueError):
            _model.check_dimension(wrong.dimension, wrong.model, expect_dim=EMBEDDING_DIM)
            # Unreachable -- but if the guard ever stopped raising, this would
            # write an 8-wide vector into a 384-wide column.
            _store.store_vectors(conn, {"x": wrong.encode(["query: x"])[0]})

        assert _store.stored_ids(conn) == set()
        assert wrong.encode_count == 0
    finally:
        conn.close()


# --- torch must not be dragged in by an ordinary import ---


def test_importing_the_package_does_not_import_sentence_transformers() -> None:
    """`gw list` must stay instant; sentence-transformers is imported lazily."""
    code = (
        "import sys; import german_wiki.embed; "
        "print('sentence_transformers' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"
