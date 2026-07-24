"""Tag normalization: strip/lower/alias, append-on-unknown, learn=False."""

from __future__ import annotations

import logging

from german_wiki.vocab import Vocab


def _known(vocab_dir, filename):
    return (vocab_dir / filename).read_text(encoding="utf-8").splitlines()


def test_strip_and_lowercase(tmp_vocab):
    v = Vocab(tmp_vocab)
    assert v.normalize("themes", "  KÜCHE ") == "küche"
    assert v.normalize("register", "Alltag") == "alltag"


def test_alias_mapping(tmp_vocab):
    v = Vocab(tmp_vocab)
    assert v.normalize("themes", "kitchen") == "küche"
    assert v.normalize("register", "Umgangssprache") == "umgangssprachlich"


def test_known_value_not_reappended(tmp_vocab):
    v = Vocab(tmp_vocab)
    before = _known(tmp_vocab, "themes.txt")
    v.normalize("themes", "küche")
    assert _known(tmp_vocab, "themes.txt") == before


def test_unknown_appends_once_and_warns(tmp_vocab, caplog):
    v = Vocab(tmp_vocab)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logging.getLogger("german_wiki.vocab").addHandler(_Capture())

    assert v.normalize("themes", "garten") == "garten"
    assert v.normalize("themes", "garten") == "garten"  # idempotent

    lines = _known(tmp_vocab, "themes.txt")
    assert lines.count("garten") == 1  # appended exactly once
    assert any("garten" in r.getMessage() for r in records)


def test_learn_false_does_not_append(tmp_vocab):
    v = Vocab(tmp_vocab)
    before = _known(tmp_vocab, "themes.txt")
    assert v.normalize("themes", "keller", learn=False) == "keller"
    assert _known(tmp_vocab, "themes.txt") == before


def test_empty_value_not_learned(tmp_vocab):
    v = Vocab(tmp_vocab)
    before = _known(tmp_vocab, "themes.txt")
    assert v.normalize("themes", "   ") == ""
    assert _known(tmp_vocab, "themes.txt") == before
