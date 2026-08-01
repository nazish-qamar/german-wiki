"""Signal precedence: rules decide, and the model is reached only as a last resort.

The assertion that matters throughout is ``fake_client.call_count == 0`` — SPEC §5's
ordering is a precedence rule, not a vote, and the way it fails in practice is the
tiebreak quietly becoming the default path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.level import _cefr, _lexical
from german_wiki.models import Node

FIXTURE = Path(__file__).parent / "fixtures" / "cefr"
EMPTY = Path(__file__).parent / "fixtures" / "cefr-empty"


@pytest.fixture(autouse=True)
def _clear_cache():
    _lexical.clear_cache()
    yield
    _lexical.clear_cache()


def _node(**overrides) -> Node:
    data = {
        "id": "n",
        "title_de": "Die Wochentage",
        "title_en": "Weekdays",
        "type": "vocab",
        "cefr": "A2",
        "status": "draft",
        "body_md": "Montag, Dienstag, Mittwoch.",
    }
    data.update(overrides)
    return Node(**data)


def _client(level: str = "B1") -> FakeChatClient:
    import json

    return FakeChatClient(text=json.dumps({"cefr": level, "reason": "because"}))


def _derive(node, client, models_config, tmp_path, **kw):
    return _cefr.derive_level(
        node,
        client=client,
        settings_path=models_config,
        cache_dir=tmp_path / "cache",
        usage_log=tmp_path / "usage.jsonl",
        **kw,
    )


# --- rules decide, model untouched ---


def test_a_title_anchored_node_never_calls_the_model(models_config, tmp_path) -> None:
    client = _client()
    node = _node(title_de="Wechselpräpositionen", type="grammar")
    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 0
    assert result.cefr == "A2"
    assert result.basis == "grammar:wechselpräposition(A2)"
    assert result.used_tiebreak is False
    assert result.rules_grounded is True


def test_lexical_alone_decides_without_the_model(models_config, tmp_path) -> None:
    client = _client()
    result = _derive(_node(title_de="Haus"), client, models_config, tmp_path, cefr_dir=FIXTURE)

    assert client.call_count == 0
    assert (result.cefr, result.basis) == ("A1", "goethe:a1(haus)")


def test_agreeing_signals_are_both_recorded(models_config, tmp_path) -> None:
    """Both contributed, so the basis says so -- that is what makes a level auditable."""
    client = _client()
    node = _node(title_de="Wechselpräpositionen", type="grammar", lemmas=["helfen"])
    result = _derive(node, client, models_config, tmp_path, cefr_dir=FIXTURE)

    assert client.call_count == 0
    assert result.basis == "grammar:wechselpräposition(A2); goethe:a2(helfen)"


def test_a_body_only_hit_that_agrees_with_the_node_is_kept(models_config, tmp_path) -> None:
    """Corroboration, not a change -- so nothing moves on the strength of a mention."""
    client = _client()
    node = _node(title_de="Höflichkeit", cefr="B1", body_md="Man nutzt Konjunktiv II.")
    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 0
    assert result.cefr == "B1"
    assert result.basis == "grammar:konjunktiv-ii(B1,body)"


# --- the model, only when the rules cannot answer ---


def test_a_body_only_hit_that_disagrees_goes_to_the_tiebreak(models_config, tmp_path) -> None:
    """The `verben-mit-präpositionen` case: never silently downgrade on a mention."""
    client = _client("B1")
    node = _node(
        title_de="Verben mit Präpositionen",
        type="grammar",
        cefr="B1",
        body_md="… mit dem Akkusativ …",
    )
    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 1
    assert result.cefr == "B1"  # preserved, not dragged to the body's A2
    assert result.used_tiebreak is True
    assert result.basis.startswith("llm:tiebreak(B1)")
    # The evidence the model was shown is recorded, not just its answer.
    assert "grammar:akkusativ(A2,body)" in result.basis
    assert "lexical:none" in result.basis


def test_conflicting_title_grammar_and_wordlist_go_to_the_tiebreak(
    models_config, tmp_path
) -> None:
    """SPEC §5's literal 'when they conflict' case."""
    client = _client("A2")
    # Title says Perfekt (A2); the fixture lists `abwaschen` at B1.
    node = _node(title_de="Perfekt", type="grammar", lemmas=["abwaschen"])
    result = _derive(node, client, models_config, tmp_path, cefr_dir=FIXTURE)

    assert client.call_count == 1
    assert result.used_tiebreak is True


def test_no_signals_at_all_goes_to_the_tiebreak(models_config, tmp_path) -> None:
    """…but only when the level was itself a machine guess -- see the pair below."""
    client = _client("A1")
    node = _node(cefr_basis="llm:extraction; a guess")
    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 1
    assert result.cefr == "A1"
    assert result.basis == "llm:tiebreak(A1); grammar:none; lexical:none"


# --- absent basis is not a placeholder basis ---


def test_an_ungrounded_tiebreak_never_moves_a_hand_authored_level(
    models_config, tmp_path
) -> None:
    """The `prefix-an` regression, caught on the live corpus.

    A hand-authored seed with no ``cefr_basis`` and no anchors: the tiebreak has strictly
    LESS information than whoever set the level, so it must explain the level rather than
    replace it. Letting it move A2 -> B1 would overwrite a human judgment purely because
    the *explanation* field was empty -- inverting the point of the slice.
    """
    client = _client("B1")  # what an ungrounded tiebreak would have said
    node = _node(cefr="A2", cefr_basis=None)

    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 0, "the model must not even be asked"
    assert result.cefr == "A2", "the human's level stands"
    assert result.basis == _cefr.HUMAN_SEED_MARKER
    assert result.used_tiebreak is False


def test_a_placeholder_basis_may_still_be_moved(models_config, tmp_path) -> None:
    """The other half of the distinction: that level was a machine guess to begin with."""
    client = _client("B1")
    node = _node(cefr="A2", cefr_basis="llm:extraction; a guess")

    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 1
    assert result.cefr == "B1"


def test_a_hand_authored_level_still_yields_to_a_real_anchor(models_config, tmp_path) -> None:
    """Protection applies only when the rules are silent, not as a blanket veto."""
    client = _client("C1")
    node = _node(title_de="Wechselpräpositionen", type="grammar", cefr="B2", cefr_basis=None)

    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY)

    assert client.call_count == 0
    assert result.cefr == "A2"  # the grammar map outranks an unexplained level
    assert result.basis == "grammar:wechselpräposition(A2)"


@pytest.mark.parametrize(
    ("basis", "absent"),
    [(None, True), ("", True), ("  ", True), ("llm:extraction", False), ("human:seed", False)],
)
def test_absent_is_narrower_than_placeholder(basis, absent) -> None:
    assert _cefr.is_absent(basis) is absent
    # Everything absent is still *targeted* -- it just cannot have its level moved.
    if absent:
        assert _cefr.is_placeholder(basis) is True


def test_the_tiebreak_marker_stays_greppable(models_config, tmp_path) -> None:
    """The successor to slice 3's `llm:extraction` habit (ADR-009).

    `grep -l 'cefr_basis: llm:tiebreak' nodes/` is how the least-grounded levels stay
    findable once a real wordlist arrives.
    """
    node = _node(cefr_basis="llm:extraction; a guess")
    result = _derive(node, _client(), models_config, tmp_path, cefr_dir=EMPTY)
    assert result.basis.startswith(_cefr.TIEBREAK_MARKER)
    assert result.rules_grounded is False


# --- the offline mode the test suite depends on ---


def test_allow_llm_false_reports_instead_of_guessing(models_config, tmp_path) -> None:
    client = _client()
    node = _node(cefr_basis="llm:extraction; a guess")
    result = _derive(node, client, models_config, tmp_path, cefr_dir=EMPTY, allow_llm=False)

    assert client.call_count == 0
    assert result.cefr is None  # not a default, not the current level
    assert "tiebreak was disabled" in result.basis


def test_the_human_seed_rule_applies_in_rules_only_mode_too(models_config, tmp_path) -> None:
    """It is a conclusion the *rules* reach, not a substitute for the tiebreak.

    So `--no-llm` still records ``human:seed`` rather than reporting the node unresolved:
    "the rules cannot ground this, and a human already chose" is a complete answer.
    """
    node = _node(cefr="A2", cefr_basis=None)
    result = _derive(node, _client(), models_config, tmp_path, cefr_dir=EMPTY, allow_llm=False)

    assert (result.cefr, result.basis) == ("A2", _cefr.HUMAN_SEED_MARKER)


# --- placeholder detection ---


@pytest.mark.parametrize(
    ("basis", "expected"),
    [
        (None, True),
        ("", True),
        ("   ", True),
        ("llm:extraction", True),
        ("llm:extraction; verb-preposition combinations", True),
        ("grammar:wechselpraeposition", False),
        ("freq:high; goethe:A1(waschen)", False),
        ("llm:tiebreak(B1); grammar:none", False),
    ],
)
def test_placeholder_detection(basis, expected) -> None:
    """A missing basis counts: SPEC §5 says always store one, so absent is ungrounded.

    `llm:tiebreak` is deliberately NOT a placeholder -- it is a derived level with its
    evidence recorded, so re-running would loop forever re-deriving it.
    """
    assert _cefr.is_placeholder(basis) is expected
