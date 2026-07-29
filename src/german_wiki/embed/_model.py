"""The local embedding model (ADR-004: always in-process, never an API).

``sentence_transformers`` is imported **inside** ``load_embedder``, never at module
scope. Importing it pulls in torch, which costs seconds -- and ``gw list`` must
stay instant. Nothing about ``import german_wiki.embed`` should cost that.

The model id comes from ``resolve_step("embeddings")``, so it is configured in
``config/models.yaml`` like every other step and never hardcoded here (CLAUDE.md).
The one thing that is pinned in code is the *dimension*, in ``db.EMBEDDING_DIM``,
because the vec0 column needs its width at CREATE time.

**The dimension guard fires at load, before a single encode.** A model whose width
disagrees with the table can never produce a vector at all, so a wrong-width value
never reaches ``store_vectors`` -- where sqlite-vec would either reject it noisily
or, worse, accept it quietly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..db import EMBEDDING_DIM
from ..llm import resolve_step
from ..logutil import get_logger

logger = get_logger(__name__)

STEP = "embeddings"


class Embedder(Protocol):
    """The seam: anything that turns texts into fixed-width vectors.

    Injected explicitly, exactly like ``ChatClient`` in the model layer, so the
    offline test suite never loads sentence-transformers or downloads weights.
    """

    model: str
    dimension: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model, normalizing on the way out."""

    def __init__(self, model_name: str, *, expect_dim: int = EMBEDDING_DIM) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model_name
        self._st = SentenceTransformer(model_name)
        # get_embedding_dimension, not get_sentence_embedding_dimension: the latter
        # is deprecated in sentence-transformers 5.6 (the pinned version) and warns.
        self.dimension = int(self._st.get_embedding_dimension())
        check_dimension(self.dimension, model_name, expect_dim=expect_dim)

    def encode(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True so cosine distance is directly meaningful and
        # similarity is exactly 1 - distance downstream.
        vectors = self._st.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        )
        # .tolist() rather than importing numpy: it is only a transitive dependency
        # (via sentence-transformers) and is not declared in pyproject.
        return [list(map(float, v)) for v in vectors.tolist()]


def check_dimension(actual: int, model_name: str, *, expect_dim: int = EMBEDDING_DIM) -> None:
    """Raise unless the model's width matches the vector column's."""
    if actual != expect_dim:
        raise ValueError(
            f"embedding model {model_name!r} produces {actual}-dimensional vectors, "
            f"but vec_nodes is declared float[{expect_dim}]. Update db.EMBEDDING_DIM "
            "and rebuild the index (gw reindex), or configure a matching model."
        )


def embedding_model_name(*, settings_path: Path | str | None = None) -> str:
    """The configured model id for the ``embeddings`` step.

    Goes through the same routing config as every other step, so the model is
    swappable in YAML. ``resolve_step`` also enforces the active/planned gate, so
    this raises a clear error until the step is switched on.
    """
    return resolve_step(STEP, settings_path=settings_path).model


def load_embedder(
    *,
    settings_path: Path | str | None = None,
    expect_dim: int = EMBEDDING_DIM,
) -> Embedder:
    """Load the configured local embedding model. Downloads weights on first use."""
    model_name = embedding_model_name(settings_path=settings_path)
    logger.info("loading local embedding model %s", model_name)
    return SentenceTransformerEmbedder(model_name, expect_dim=expect_dim)
