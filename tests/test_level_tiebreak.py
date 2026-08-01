"""The tiebreak: what it parses, what it refuses, and what it is shown."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.level import TiebreakError, _tiebreak
from german_wiki.llm import ModelResponse, Usage


def _response(text: str, *, finish_reason: str = "stop", reasoning: str | None = None):
    return ModelResponse(
        text=text,
        step="cefr_tiebreak",
        provider="zai",
        model="glm-4.5-flash",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cached=False,
        cost_usd=0.0,
        saved_usd=0.0,
        cache_key="k",
        finish_reason=finish_reason,
        reasoning_content=reasoning,
    )


def _payload(cefr: str = "B1", reason: str = "because") -> str:
    return json.dumps({"cefr": cefr, "reason": reason}, ensure_ascii=False)


# --- parsing ---


@pytest.mark.parametrize("cefr", ["A1", "A2", "B1", "B2", "C1", "C2"])
def test_every_valid_level_parses(cefr) -> None:
    assert _tiebreak.parse(_response(_payload(cefr))).cefr == cefr


def test_an_invented_level_is_rejected() -> None:
    """CEFR is a strict enum (ADR-007) precisely so a made-up level fails loudly."""
    with pytest.raises(TiebreakError, match="schema"):
        _tiebreak.parse(_response(_payload("B3")))


def test_a_prose_level_is_rejected() -> None:
    with pytest.raises(TiebreakError, match="schema"):
        _tiebreak.parse(_response(_payload("beginner")))


def test_truncation_is_a_failure_not_a_missing_level() -> None:
    """Same reasoning-token guard as extraction and adjudication."""
    with pytest.raises(TiebreakError) as exc:
        _tiebreak.parse(_response("", finish_reason="length", reasoning="Weighing A2…"))
    assert "max_tokens" in str(exc.value)
    assert exc.value.reasoning_content == "Weighing A2…"


def test_truncation_is_checked_before_parsing() -> None:
    with pytest.raises(TiebreakError, match="truncated"):
        _tiebreak.parse(_response(_payload(), finish_reason="length"))


def test_empty_content_fails() -> None:
    with pytest.raises(TiebreakError, match="no content"):
        _tiebreak.parse(_response("  "))


def test_bad_json_fails() -> None:
    with pytest.raises(TiebreakError, match="valid JSON"):
        _tiebreak.parse(_response("A2, probably"))


def test_code_fences_are_stripped() -> None:
    assert _tiebreak.parse(_response(f"```json\n{_payload('C1')}\n```")).cefr == "C1"


# --- the prompt ---


def test_signals_one_and_two_are_in_context() -> None:
    """SPEC §5 point 3: the tiebreak runs "with signals 1 and 2 already in context".

    Asking cold would make it the zero-shot judgment §5 opens by rejecting.
    """
    variable = _tiebreak.build_prompt(
        title_de="Verben mit Präpositionen",
        title_en="Verbs with prepositions",
        node_type="grammar",
        body_md="… Akkusativ …",
        grammar="akkusativ (A2) matched in the BODY",
        lexical="no wordlist installed",
    ).to_messages()[-1]["content"]

    assert "GRAMMAR: akkusativ (A2) matched in the BODY" in variable
    assert "LEXICAL: no wordlist installed" in variable


def test_fixed_content_comes_first_and_variable_last() -> None:
    messages = _tiebreak.build_prompt(
        title_de="Titel",
        title_en="Title",
        node_type="vocab",
        body_md="Körper",
        grammar="none",
        lexical="none",
    ).to_messages()
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Titel" not in messages[0]["content"]


def test_the_prompt_tells_the_model_to_prefer_the_lower_level() -> None:
    """Under-levelling surfaces and gets corrected; over-levelling hides a node forever."""
    assert "LOWER" in _tiebreak.SYSTEM


# --- the call ---


def test_tiebreak_routes_to_the_free_step(models_config: Path, tmp_path: Path) -> None:
    """It must not ride the paid adjudication model -- it can fire on every node."""
    client = FakeChatClient(text=_payload("A1"))
    verdict, response = _tiebreak.tiebreak(
        title_de="Die Wochentage",
        title_en="Weekdays",
        node_type="vocab",
        body_md="Montag …",
        grammar="none",
        lexical="none",
        client=client,
        settings_path=models_config,
        cache_dir=tmp_path / "cache",
        usage_log=tmp_path / "usage.jsonl",
    )
    assert verdict.cefr == "A1"
    assert response.model == "glm-4.5-flash"
    assert response.cost_usd == 0.0  # explicitly zero-priced, not merely unpriced
    assert client.call_count == 1


def test_a_repeated_tiebreak_is_free(models_config: Path, tmp_path: Path) -> None:
    client = FakeChatClient(text=[_payload("A1"), _payload("C2")])
    common = {
        "title_de": "Die Wochentage",
        "title_en": "Weekdays",
        "node_type": "vocab",
        "body_md": "Montag …",
        "grammar": "none",
        "lexical": "none",
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_path / "cache",
        "usage_log": tmp_path / "usage.jsonl",
    }
    first, _ = _tiebreak.tiebreak(**common)
    second, response = _tiebreak.tiebreak(**common)

    assert client.call_count == 1
    assert second.cefr == first.cefr == "A1"
    assert response.cached is True
