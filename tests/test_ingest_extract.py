"""Extraction parsing and its failure modes, chiefly the length-truncation guard."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml
from conftest import FakeChatClient

from german_wiki.ingest import _extract
from german_wiki.ingest._extract import Candidate, ExtractionError
from german_wiki.llm._client import ModelResponse
from german_wiki.llm._usage import Usage

CONFIG = {
    "version": 1,
    "providers": {
        "zai": {
            "kind": "api",
            "base_url": "https://example.invalid/v4",
            "api_key_env": "ZAI_API_KEY",
        }
    },
    "pricing": {"zai": {"free-model": {"input": 0.0, "cached_input": 0.0, "output": 0.0}}},
    "defaults": {"provider": "zai", "model": "free-model", "temperature": 0.0, "max_tokens": 4096},
    "steps": {"extraction": {"status": "active"}},
}

TEXT = "Die Wechselpräpositionen stehen mit Akkusativ (wohin?) oder Dativ (wo?)."


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(CONFIG, allow_unicode=True), encoding="utf-8")
    return path


def _candidate(**overrides) -> dict:
    data = {
        "title_de": "Wechselpräpositionen",
        "title_en": "Two-way prepositions",
        "type": "grammar",
        "cefr": "A2",
        "cefr_basis": "grammar:wechselpräpositionen",
        "register": ["alltag"],
        "themes": ["haushalt"],
        "body_md": "Akkusativ bei Bewegung, Dativ bei Ort.",
        "confidence": 0.9,
    }
    data.update(overrides)
    return data


def _payload(n: int = 1, **overrides) -> str:
    fields = [{"title_de": f"Konzept {i}", **overrides} for i in range(n)]
    return json.dumps(
        {"candidates": [_candidate(**f) for f in fields]},
        ensure_ascii=False,
    )


def _response(text: str, *, finish_reason: str = "stop", reasoning: str | None = None):
    return ModelResponse(
        text=text,
        step="extraction",
        provider="zai",
        model="free-model",
        usage=Usage(prompt_tokens=100, completion_tokens=4096),
        cached=False,
        cost_usd=0.0,
        saved_usd=0.0,
        cache_key="k",
        finish_reason=finish_reason,
        reasoning_content=reasoning,
    )


def _records(logger_name: str) -> list[logging.LogRecord]:
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    logging.getLogger(logger_name).addHandler(_Collector())
    return collected


# --- the prompt (SPEC §10) ---


def test_raw_text_is_the_last_message_and_the_prefix_is_shared() -> None:
    a = _extract.build_prompt("Quelle A").to_messages()
    b = _extract.build_prompt("Quelle B").to_messages()
    assert a[-1] == {"role": "user", "content": "Quelle A"}
    assert a[:-1] == b[:-1]


def test_prompt_carries_the_cap_and_the_granularity_rules() -> None:
    system = _extract.build_prompt(TEXT).to_messages()[0]["content"]
    assert str(_extract.MAX_CANDIDATES) in system
    assert "word FAMILY" in system
    assert "Do NOT add grammar rules" in system  # SPEC §4.1 no-invention constraint


def test_prompt_version_is_set() -> None:
    """It enters the cache key, so a parser change can invalidate cached responses."""
    assert _extract.build_prompt(TEXT).version == _extract.PROMPT_VERSION


# --- happy path ---


def test_parses_candidates() -> None:
    candidates = _extract.parse(_response(_payload(2)))
    assert len(candidates) == 2
    assert isinstance(candidates[0], Candidate)
    assert candidates[0].type == "grammar"
    assert candidates[0].cefr == "A2"


def test_tolerates_json_code_fences() -> None:
    """Providers emit them even under response_format=json_object."""
    fenced = f"```json\n{_payload(1)}\n```"
    assert len(_extract.parse(_response(fenced))) == 1


def test_umlauts_survive_parsing() -> None:
    candidates = _extract.parse(_response(_payload(1, title_de="Küche und Bad")))
    assert "ü" in candidates[0].title_de


def test_optional_fields_default() -> None:
    minimal = json.dumps(
        {
            "candidates": [
                {
                    "title_de": "X",
                    "title_en": "X",
                    "type": "vocab",
                    "cefr": "A1",
                    "body_md": "b",
                }
            ]
        }
    )
    candidate = _extract.parse(_response(minimal))[0]
    assert candidate.register == [] and candidate.themes == []
    assert candidate.cefr_basis is None
    assert candidate.confidence == 0.5


def test_zero_candidates_is_not_an_error() -> None:
    """A source may legitimately hold nothing learnable."""
    assert _extract.parse(_response(json.dumps({"candidates": []}))) == []


# --- the truncation guard (the slice-2 finding) ---


def test_length_finish_reason_raises_even_with_parseable_content() -> None:
    """Checked BEFORE parsing: a truncated response can still look parseable."""
    with pytest.raises(ExtractionError, match="truncated at 4096 completion tokens"):
        _extract.parse(_response(_payload(1), finish_reason="length"))


def test_truncation_error_names_the_fix() -> None:
    with pytest.raises(ExtractionError) as excinfo:
        _extract.parse(_response("", finish_reason="length"))
    message = str(excinfo.value)
    assert "max_tokens" in message
    assert "'extraction'" in message
    assert "Reasoning tokens count toward the cap" in message


def test_truncation_error_carries_the_reasoning_trace() -> None:
    """The whole point of capturing reasoning_content in slice 3's terms."""
    reasoning = "Ich analysiere den Text und finde zuerst..."
    with pytest.raises(ExtractionError) as excinfo:
        _extract.parse(_response("", finish_reason="length", reasoning=reasoning))
    assert excinfo.value.reasoning_content == reasoning
    assert excinfo.value.response is not None


def test_empty_content_without_truncation_also_raises() -> None:
    with pytest.raises(ExtractionError, match="returned no content"):
        _extract.parse(_response("   "))


# --- schema enforcement ---


def test_malformed_json_raises() -> None:
    with pytest.raises(ExtractionError, match="did not return valid JSON"):
        _extract.parse(_response("{not json"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"type": "noun"},
        {"cefr": "B3"},
        {"confidence": "high"},
    ],
    ids=["bad-type", "bad-cefr", "bad-confidence"],
)
def test_values_outside_the_node_vocabularies_raise(overrides) -> None:
    """Reusing models.py Literals means bad values die here, not in a node file."""
    with pytest.raises(ExtractionError, match="did not match the candidate schema"):
        _extract.parse(_response(_payload(1, **overrides)))


def test_unknown_candidate_key_raises() -> None:
    payload = json.dumps({"candidates": [{**_candidate(), "nonsense": 1}]})
    with pytest.raises(ExtractionError, match="did not match the candidate schema"):
        _extract.parse(_response(payload))


def test_missing_required_field_raises() -> None:
    payload = json.dumps({"candidates": [{"title_de": "X"}]})
    with pytest.raises(ExtractionError, match="did not match the candidate schema"):
        _extract.parse(_response(payload))


# --- the 5-8 cap (SPEC §2, ADR-006) ---


def test_over_the_cap_truncates_and_warns() -> None:
    records = _records("german_wiki.ingest._extract")
    candidates = _extract.parse(_response(_payload(20)))

    assert len(candidates) == _extract.MAX_CANDIDATES
    assert candidates[0].title_de == "Konzept 0"  # the model's own ordering
    warning = next(r for r in records if r.levelno == logging.WARNING)
    assert "atomizing" in warning.getMessage()


def test_exactly_at_the_cap_is_untouched() -> None:
    assert len(_extract.parse(_response(_payload(_extract.MAX_CANDIDATES)))) == 8


# --- end to end through complete() ---


def test_extract_calls_the_model_and_requests_json(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient(text=_payload(3))
    candidates, response = _extract.extract(
        TEXT, client=fake, settings_path=cfg, cache_dir=tmp_cache, usage_log=tmp_usage_log, env={}
    )
    assert len(candidates) == 3
    assert response.cached is False
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_extract_is_cached(cfg, tmp_cache, tmp_usage_log) -> None:
    """Re-running the pipeline on the same source must be free (ADR-005)."""
    fake = FakeChatClient(text=_payload(2))
    for _ in range(2):
        _extract.extract(
            TEXT,
            client=fake,
            settings_path=cfg,
            cache_dir=tmp_cache,
            usage_log=tmp_usage_log,
            env={},
        )
    assert fake.call_count == 1


def test_extract_propagates_truncation(cfg, tmp_cache, tmp_usage_log) -> None:
    fake = FakeChatClient(text="", finish_reason="length", reasoning_content="denke...")
    with pytest.raises(ExtractionError) as excinfo:
        _extract.extract(
            TEXT,
            client=fake,
            settings_path=cfg,
            cache_dir=tmp_cache,
            usage_log=tmp_usage_log,
            env={},
        )
    assert excinfo.value.reasoning_content == "denke..."
