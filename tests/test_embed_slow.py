"""Optional test against the real local embedding model. Skipped unless GW_MODEL_TESTS=1.

The rest of the suite injects a FakeEmbedder, so it is offline and instant. This
file is the only place the actual multilingual-e5-small weights are loaded.

    GW_MODEL_TESTS=1 pytest -m slow

Distinct from `live`: that marker means network + an API key + money. This one is
offline and free -- it just downloads ~470MB once and then runs on CPU. If the
weights cannot be fetched, these **skip with a reason** rather than failing: an
unavailable download is an environment fact, not a defect in the code.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from german_wiki.db import EMBEDDING_DIM
from german_wiki.embed import _model
from german_wiki.embed._text import E5_PREFIX

RUN = os.environ.get("GW_MODEL_TESTS") == "1"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not RUN, reason="set GW_MODEL_TESTS=1 to run local model tests"),
]


@pytest.fixture(scope="module")
def embedder():
    """The real model, loaded once. Skips cleanly if the weights are unavailable."""
    try:
        return _model.load_embedder()
    except Exception as exc:  # noqa: BLE001 - any failure to obtain weights
        pytest.skip(f"local embedding model unavailable ({type(exc).__name__}: {exc})")


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_the_real_model_matches_the_pinned_dimension(embedder) -> None:
    """If this fails, db.EMBEDDING_DIM and the vec0 column are wrong, not the test."""
    assert embedder.dimension == EMBEDDING_DIM


def test_loading_emits_no_deprecation_warnings(embedder) -> None:
    """A renamed upstream method should fail here, not print noise on every run.

    Deliberately outside the `embedder` fixture: that fixture skips on any load
    failure, which would turn a real deprecation into a silent skip.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model.SentenceTransformerEmbedder(embedder.model)

    offenders = [
        f"{w.category.__name__}: {w.message}"
        for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning))
    ]
    assert offenders == []


def test_it_embeds_german_and_returns_unit_vectors(embedder) -> None:
    vectors = embedder.encode([f"{E5_PREFIX}Die Küche ist sauber."])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM
    # normalize_embeddings=True, which is what makes cosine == 1 - distance.
    assert sum(v * v for v in vectors[0]) == pytest.approx(1.0, abs=1e-4)


def test_related_german_scores_higher_than_unrelated(embedder) -> None:
    """The property the whole semantic tier rests on, checked against real weights."""
    anchor, related, unrelated = embedder.encode(
        [
            f"{E5_PREFIX}Wechselpräpositionen stehen mit Akkusativ oder Dativ.",
            f"{E5_PREFIX}Nach zweiseitigen Präpositionen steht der Akkusativ bei Bewegung.",
            f"{E5_PREFIX}Die Waschmaschine steht in der Küche und wäscht die Wäsche.",
        ]
    )
    assert _cos(anchor, related) > _cos(anchor, unrelated)


def test_identical_text_is_near_perfectly_similar(embedder) -> None:
    text = f"{E5_PREFIX}Der Dativ folgt auf 'mit'."
    a, b = embedder.encode([text, text])
    assert _cos(a, b) == pytest.approx(1.0, abs=1e-5)


def test_a_real_embedding_round_trips_through_the_index(embedder, tmp_db: Path) -> None:
    """Real vectors, real vec0 column: width and cosine agree end to end."""
    from german_wiki.db import connect, rebuild_schema
    from german_wiki.embed import _store

    vectors = embedder.encode(
        [f"{E5_PREFIX}Die Küche", f"{E5_PREFIX}Das Badezimmer", f"{E5_PREFIX}Der Akkusativ"]
    )
    conn = connect(tmp_db)
    try:
        rebuild_schema(conn)
        _store.store_vectors(conn, dict(zip(["kueche", "bad", "akkusativ"], vectors, strict=True)))

        results = _store.knn(conn, vectors[0], k=3)
        assert results[0][0] == "kueche"
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)
        # Two rooms should sit closer than a room and a grammatical case.
        by_id = dict(results)
        assert by_id["bad"] > by_id["akkusativ"]
    finally:
        conn.close()
