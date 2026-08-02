"""Transparency: the one semantic judgment, on the free step, with its failure modes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.llm import ModelResponse, Usage, resolve_step
from german_wiki.morph import _transparency
from german_wiki.morph._transparency import TransparencyError, judge


def _response(text: str, *, finish_reason: str = "stop", reasoning: str | None = None):
    return ModelResponse(
        text=text,
        step="transparency",
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


def _payload(transparency: str, **extra) -> str:
    body = {"transparency": transparency, "reason": "because", "confidence": 0.9}
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


# --- the verdict ---


@pytest.mark.parametrize("value", ["high", "drifted", "opaque"])
def test_the_three_ratings_parse(value) -> None:
    assert _transparency.parse(_response(_payload(value))).transparency == value


def test_an_invented_rating_is_rejected() -> None:
    """`family_transparency` is a strict enum on Node -- a typo must not reach a file."""
    with pytest.raises(TransparencyError, match="schema"):
        _transparency.parse(_response(_payload("mostly-fine")))


def test_only_high_is_trusted_by_the_grid() -> None:
    """The link to `_grid`: anything else marks predicted cells irregular (§7.4)."""
    assert _transparency.TRUSTED == "high"


# --- failure modes this module owns ---


def test_truncation_is_a_failure_not_an_empty_verdict() -> None:
    with pytest.raises(TransparencyError) as exc:
        _transparency.parse(_response("", finish_reason="length", reasoning="Comparing..."))
    assert "max_tokens" in str(exc.value)
    assert exc.value.reasoning_content == "Comparing..."


def test_truncation_is_checked_before_parsing() -> None:
    with pytest.raises(TransparencyError, match="truncated"):
        _transparency.parse(_response(_payload("high"), finish_reason="length"))


def test_bad_json_fails_with_the_body_quoted() -> None:
    with pytest.raises(TransparencyError, match="valid JSON"):
        _transparency.parse(_response("not json"))


def test_empty_content_fails() -> None:
    with pytest.raises(TransparencyError, match="no content"):
        _transparency.parse(_response("  "))


def test_code_fences_are_stripped() -> None:
    fenced = f"```json\n{_payload('opaque')}\n```"
    assert _transparency.parse(_response(fenced)).transparency == "opaque"


# --- the prompt ---


def test_fixed_content_first_and_the_family_last() -> None:
    """SPEC §10's provider-cache lever: the prefix is invariant, the family is not.

    The probe word is deliberately one the prompt does not already use as an exemplar --
    `abwaschen` would false-positive, since it IS in the system prompt as the worked
    example of `high`.
    """
    messages = _transparency.build_prompt(
        root="spielen", prefix="mit", word="mitspielen"
    ).to_messages()
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("root: spielen")
    assert "mitspielen" not in messages[0]["content"]


def test_the_prompt_teaches_spec_74s_worked_examples() -> None:
    """§7.4 names `verstehen` and `bekommen` as the trap; the few-shot must show one."""
    shots = "\n".join(s.user + s.assistant for s in _transparency.FEW_SHOT)
    assert "verstehen" in shots
    assert "opaque" in shots
    assert "high" in shots and "drifted" in shots  # all three ratings demonstrated


def test_the_split_is_presented_as_settled_not_up_for_debate() -> None:
    """Segmentation is `_segment`'s job, from evidence. Asking the model to re-litigate
    it is how a model's fluency gets mistaken for morphological fact."""
    assert "do not question it" in _transparency.SYSTEM.lower()


# --- routing ---


def test_transparency_runs_on_the_free_step() -> None:
    """ADR-011 §5's rule applied again: a per-candidate call that fires often stays free.

    Asserted against config rather than a literal, so switching adjudication to a paid
    model (as slice 6 did) cannot silently drag this along with it.
    """
    step = resolve_step("transparency")
    assert step.model == "glm-4.5-flash"
    assert step.pricing is not None
    assert (step.pricing.input, step.pricing.output) == (0.0, 0.0)
    assert step.model != resolve_step("adjudication").model


def test_judge_calls_the_model_once_and_returns_the_verdict(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text=_payload("drifted"))
    verdict, response = judge(
        root="stellen",
        prefix="an",
        word="anstellen",
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert verdict.transparency == "drifted"
    assert client.call_count == 1
    assert response.step == "transparency"


def test_a_second_identical_judgment_is_free(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text=[_payload("high"), _payload("opaque")])
    common = {
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
    }
    first, _ = judge(root="waschen", prefix="ab", word="abwaschen", **common)
    second, response = judge(root="waschen", prefix="ab", word="abwaschen", **common)

    assert client.call_count == 1
    assert second.transparency == first.transparency == "high"
    assert response.cached is True
