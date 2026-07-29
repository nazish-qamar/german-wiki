"""Text preparation: the two normalizations, shingles, exact Jaccard, prefilter."""

from __future__ import annotations

import random

import pytest

from german_wiki.embed import _text
from german_wiki.models import Node


def _node(**overrides) -> Node:
    data = {
        "id": "x",
        "title_de": "Wechselpräpositionen",
        "title_en": "Two-way prepositions",
        "type": "grammar",
        "cefr": "A2",
        "status": "draft",
        "body_md": "Akkusativ bei Bewegung, Dativ bei Ort.",
    }
    data.update(overrides)
    return Node(**data)


# --- normalize_for_hash: aggressive, tier-1 identity only ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Die Küche", "die küche"),
        ("  collapsed   \n whitespace ", "collapsed whitespace"),
        ("MiXeD CaSe", "mixed case"),
        ("", ""),
    ],
)
def test_normalize_for_hash(raw, expected) -> None:
    assert _text.normalize_for_hash(raw) == expected


def test_normalize_is_idempotent() -> None:
    once = _text.normalize_for_hash("  Die  KÜCHE\n\n ist  schön ")
    assert _text.normalize_for_hash(once) == once


def test_normalize_applies_nfkc() -> None:
    """Compatibility forms collapse, so visually identical text hashes identically."""
    assert _text.normalize_for_hash("ﬁnden") == _text.normalize_for_hash("finden")


def test_differently_cased_text_hashes_the_same() -> None:
    a, b = "Der Dativ", "der   dativ"
    assert _text.normalize_for_hash(a) == _text.normalize_for_hash(b)


# --- embed_text: natural case, e5 prefix ---


def test_embed_text_carries_the_e5_prefix() -> None:
    """multilingual-e5 requires it; symmetric comparison uses `query:` on both sides."""
    assert _text.embed_text(_node()).startswith("query: ")


def test_embed_text_preserves_german_capitalization() -> None:
    """Lowercasing would discard a signal the model was trained on."""
    text = _text.embed_text(_node())
    assert "Wechselpräpositionen" in text
    assert "Akkusativ" in text


def test_embed_text_includes_the_german_title_and_body() -> None:
    text = _text.embed_text(_node())
    assert "Wechselpräpositionen" in text
    assert "Akkusativ bei Bewegung" in text


def test_embed_text_excludes_the_english_title() -> None:
    """Measured: the German-English pairing is shared by every node, so including
    it pulls unrelated nodes together. Dropping it widened the margin by ~1/3."""
    assert "Two-way prepositions" not in _text.embed_text(_node())


def test_embed_text_keeps_the_whole_body() -> None:
    """Truncating measurably shrank the margin -- the body carries real signal."""
    long_body = "Ein Satz über Präpositionen. " * 40
    text = _text.embed_text(_node(body_md=long_body))
    assert text.endswith("Präpositionen.")
    assert len(text) > 500


def test_embed_text_tolerates_an_empty_body() -> None:
    """body_md defaults to "" on the model, so this is a real shape."""
    assert _text.embed_text(_node(body_md="")) == "query: Wechselpräpositionen"


def test_embed_text_is_deterministic() -> None:
    assert _text.embed_text(_node()) == _text.embed_text(_node())


# --- strip_markdown: the scaffolding every node shares ---


SCAFFOLDED = """\
Strong verb: **waschen – wusch – gewaschen**, du wäschst.

| Word | Meaning |
|---|---|
| waschen | to wash |

## Examples
- Ich **wasche** mir die Hände. (I'm washing my hands.) [alltag, gesprochen]
"""


def test_strip_markdown_drops_headings_tables_and_tags() -> None:
    result = _text.strip_markdown(SCAFFOLDED)
    assert "## Examples" not in result
    assert "|" not in result
    assert "[alltag, gesprochen]" not in result
    assert "**" not in result


def test_strip_markdown_keeps_the_prose() -> None:
    result = _text.strip_markdown(SCAFFOLDED)
    assert "waschen – wusch – gewaschen" in result
    assert "Ich wasche mir die Hände." in result


def test_strip_markdown_collapses_to_one_line() -> None:
    assert "\n" not in _text.strip_markdown(SCAFFOLDED)


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", "## nur eine Überschrift"])
def test_strip_markdown_of_content_free_text(raw) -> None:
    assert _text.strip_markdown(raw) == ""


def test_strip_markdown_leaves_plain_prose_alone() -> None:
    plain = "Akkusativ bei Bewegung, Dativ bei Ort."
    assert _text.strip_markdown(plain) == plain


def test_scaffolding_is_absent_from_the_embed_text() -> None:
    """The end-to-end property: no node's vector spends capacity on table pipes."""
    text = _text.embed_text(_node(body_md=SCAFFOLDED))
    assert "|" not in text
    assert "## " not in text


# --- shingles ---


def test_shingles_of_text_longer_than_the_window() -> None:
    result = _text.shingles("abcdefg", size=5)
    assert result == {"abcde", "bcdef", "cdefg"}


def test_shingles_of_text_shorter_than_the_window() -> None:
    assert _text.shingles("abc", size=5) == {"abc"}


def test_shingles_of_empty_text() -> None:
    assert _text.shingles("", size=5) == set()


def test_shingles_normalize_first() -> None:
    assert _text.shingles("ABC DEF") == _text.shingles("  abc   def  ")


# --- jaccard: exact, not estimated ---


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ({"x"}, {"x"}, 1.0),
        ({"x"}, {"y"}, 0.0),
        ({"a", "b"}, {"b", "c"}, 1 / 3),
        ({"a", "b", "c", "d"}, {"a", "b", "c"}, 3 / 4),
        (set(), set(), 1.0),
        ({"a"}, set(), 0.0),
    ],
)
def test_jaccard(a, b, expected) -> None:
    assert _text.jaccard(a, b) == pytest.approx(expected)


def test_jaccard_is_symmetric() -> None:
    a, b = _text.shingles("Der Dativ steht hier"), _text.shingles("Der Dativ steht dort")
    assert _text.jaccard(a, b) == _text.jaccard(b, a)


def test_identical_text_scores_one() -> None:
    body = "Akkusativ bei Bewegung, Dativ bei Ort."
    assert _text.jaccard(_text.shingles(body), _text.shingles(body)) == 1.0


def test_a_small_edit_scores_high_but_not_one() -> None:
    a = _text.shingles("Akkusativ bei Bewegung, Dativ bei Ort.")
    b = _text.shingles("Akkusativ bei Bewegung, Dativ bei Orten.")
    score = _text.jaccard(a, b)
    assert 0.8 < score < 1.0


def test_unrelated_german_text_scores_low() -> None:
    a = _text.shingles("Akkusativ bei Bewegung, Dativ bei Ort.")
    b = _text.shingles("Die Waschmaschine steht in der Küche.")
    assert _text.jaccard(a, b) < 0.2


# --- the prefilter must be SOUND: never skip a pair that would pass ---


def test_prefilter_admits_identical_sets() -> None:
    s = _text.shingles("Der Dativ")
    assert _text.could_reach(s, s, 0.85) is True


def test_prefilter_rejects_wildly_different_lengths() -> None:
    short = _text.shingles("kurz")
    long = _text.shingles("ein deutlich viel längerer Text über Präpositionen" * 5)
    assert _text.could_reach(short, long, 0.85) is False


def test_prefilter_rejects_empty_sets() -> None:
    assert _text.could_reach(set(), {"a"}, 0.5) is False


@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.85, 0.95])
def test_prefilter_never_excludes_a_pair_that_would_pass(threshold) -> None:
    """Soundness: could_reach may only skip pairs whose true Jaccard is below
    the threshold. A false negative here would silently lose real duplicates."""
    rng = random.Random(1234)
    universe = [f"s{i}" for i in range(60)]

    for _ in range(400):
        a = set(rng.sample(universe, rng.randint(1, 40)))
        b = set(rng.sample(universe, rng.randint(1, 40)))
        if not _text.could_reach(a, b, threshold):
            assert _text.jaccard(a, b) < threshold
