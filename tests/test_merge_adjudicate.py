"""Four-outcome adjudication: what parses, what fails, and what the prompt teaches."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.llm import ModelResponse, Usage, resolve_step
from german_wiki.merge import Adjudication, AdjudicationError, _adjudicate, adjudicate
from german_wiki.models import Node


def _node(node_id: str, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": "Wechselpräpositionen",
        "title_en": "Two-way prepositions",
        "type": "grammar",
        "cefr": "A2",
        "status": "draft",
        "body_md": "Akkusativ bei Bewegung, Dativ bei Ort.",
    }
    data.update(overrides)
    return Node(**data)


def _response(text: str, *, finish_reason: str = "stop", reasoning: str | None = None):
    return ModelResponse(
        text=text,
        step="adjudication",
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


def _payload(outcome: str, **extra) -> str:
    body = {
        "outcome": outcome,
        "confidence": 0.9,
        "reason": "because",
        "b_adds": None,
        "relation": None,
        "direction": None,
    }
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


# --- the four outcomes ---


@pytest.mark.parametrize("outcome", ["SAME", "OVERLAP", "DISTINCT"])
def test_the_simple_outcomes_parse(outcome) -> None:
    verdict = _adjudicate.parse(_response(_payload(outcome)))
    assert verdict.outcome == outcome
    assert verdict.relation is None


def test_distinct_related_carries_its_edge() -> None:
    """ADR-010's fourth branch: the outcome is only useful with a relation attached."""
    verdict = _adjudicate.parse(
        _response(_payload("DISTINCT_RELATED", relation="governs", direction="a_to_b"))
    )
    assert (verdict.outcome, verdict.relation, verdict.direction) == (
        "DISTINCT_RELATED",
        "governs",
        "a_to_b",
    )


def test_overlap_keeps_what_b_adds() -> None:
    """SPEC §3.1 asks for it explicitly; the merge prompt consumes it."""
    verdict = _adjudicate.parse(_payload_response("OVERLAP", b_adds="the sein auxiliary"))
    assert verdict.b_adds == "the sein auxiliary"


def _payload_response(outcome: str, **extra):
    return _response(_payload(outcome, **extra))


# --- failure modes this module owns ---


def test_truncation_is_a_failure_not_an_empty_verdict() -> None:
    """The reasoning-token guard from slice 2/3: empty content with finish_reason=length."""
    with pytest.raises(AdjudicationError) as exc:
        _adjudicate.parse(_response("", finish_reason="length", reasoning="Let me compare..."))
    assert "max_tokens" in str(exc.value)
    assert exc.value.reasoning_content == "Let me compare..."
    assert exc.value.response is not None


def test_truncation_is_checked_before_parsing() -> None:
    """A truncated response can still contain parseable-looking content."""
    with pytest.raises(AdjudicationError, match="truncated"):
        _adjudicate.parse(_response(_payload("SAME"), finish_reason="length"))


def test_empty_content_fails() -> None:
    with pytest.raises(AdjudicationError, match="no content"):
        _adjudicate.parse(_response("   "))


def test_bad_json_fails_with_the_body_quoted() -> None:
    with pytest.raises(AdjudicationError, match="valid JSON"):
        _adjudicate.parse(_response("not json at all"))


def test_an_invented_relation_is_rejected() -> None:
    """SPEC §4.2's seven are closed in the schema so `related_to` cannot slip through."""
    with pytest.raises(AdjudicationError, match="schema"):
        _adjudicate.parse(
            _response(_payload("DISTINCT_RELATED", relation="related_to", direction="a_to_b"))
        )


def test_an_invented_outcome_is_rejected() -> None:
    with pytest.raises(AdjudicationError, match="schema"):
        _adjudicate.parse(_response(_payload("MAYBE")))


def test_a_null_outcome_says_so_plainly() -> None:
    """Providers emit `null` for the "otherwise null" fields; on `outcome` that is a bug."""
    body = json.dumps({"outcome": None, "confidence": 0.5, "reason": ""})
    with pytest.raises(AdjudicationError, match="no outcome"):
        _adjudicate.parse(_response(body))


def test_distinct_related_without_a_relation_fails() -> None:
    """Unusable: there is no edge to propose, so it cannot become a proposal."""
    with pytest.raises(AdjudicationError, match="needs both a relation and a direction"):
        _adjudicate.parse(_response(_payload("DISTINCT_RELATED")))


def test_a_stray_relation_on_another_outcome_is_dropped_not_fatal() -> None:
    """Asymmetric on purpose: a meaningless field is cheaper to discard than to re-run."""
    verdict = _adjudicate.parse(
        _response(_payload("DISTINCT", relation="governs", direction="a_to_b"))
    )
    assert verdict.outcome == "DISTINCT"
    assert verdict.relation is None and verdict.direction is None


def test_code_fences_are_stripped() -> None:
    fenced = f"```json\n{_payload('SAME')}\n```"
    assert _adjudicate.parse(_response(fenced)).outcome == "SAME"


# --- the prompt ---


def test_fixed_content_comes_first_and_variable_last() -> None:
    """SPEC §10's provider-cache lever, enforced by Prompt's shape."""
    messages = _adjudicate.build_prompt(_node("a"), _node("b", title_de="Andere")).to_messages()
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "Andere" in messages[-1]["content"]
    # The few-shot block sits between them, so the whole prefix is invariant per pair.
    assert len(messages) > 2
    assert "Andere" not in messages[0]["content"]


def test_both_sides_are_labelled_a_and_b() -> None:
    """SPEC §3.2 keys off the orientation: B's body is discarded, A keeps its id."""
    variable = _adjudicate.build_prompt(_node("left"), _node("right")).to_messages()[-1]["content"]
    assert variable.startswith("A: ")
    assert "\n\nB: " in variable


def test_the_few_shot_does_not_teach_the_live_test_pair() -> None:
    """Test integrity: the live test asserts generalization, not prompt recall.

    ADR-010's worked example is ``um-hilfe-bitten`` ↔ ``verben-mit-präpositionen``
    resolving to ``governs``, and tests/test_merge_live.py asserts exactly that against
    the real provider. Few-shotting the model on that pair would make the live test
    measure whether it can copy an answer out of its own prompt. Shot 3 is deliberately
    a *structurally analogous* pair (fixed expression vs the rule it obeys) with
    different content, so `governs` is still demonstrated.
    """
    prompt = "\n".join(shot.user + shot.assistant for shot in _adjudicate.FEW_SHOT)
    lowered = prompt.lower()
    for leaked in ("um hilfe", "bitten um", "hilfe bitten", "verben mit präpositionen"):
        assert leaked not in lowered, f"few-shot leaks the live-test pair: {leaked!r}"
    # ...but it must still demonstrate the fourth outcome, or nothing teaches it.
    assert "DISTINCT_RELATED" in prompt
    assert "governs" in prompt


# --- the call ---


def test_adjudicate_calls_the_model_and_returns_the_verdict(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    client = FakeChatClient(text=_payload("OVERLAP", b_adds="the dative case"))
    verdict, response = adjudicate(
        _node("a"),
        _node("b"),
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert isinstance(verdict, Adjudication)
    assert verdict.outcome == "OVERLAP"
    assert client.call_count == 1
    # The response reports whatever the step is routed to, rather than a model this
    # test pins -- CLAUDE.md puts routing in config, so hardcoding one here would make
    # a config edit fail an unrelated test (it did, on the glm-4.6 switch).
    assert response.model == resolve_step("adjudication", settings_path=models_config).model


def test_a_second_identical_adjudication_is_free(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    """ADR-005 at this layer: re-running the pipeline must not re-spend tokens.

    This is also what makes ADR-011's design work -- ``gw review`` can re-derive a
    proposal's context at zero cost, so no paused graph has to survive the process.
    """
    client = FakeChatClient(text=[_payload("SAME"), _payload("DISTINCT")])
    common = {
        "client": client,
        "settings_path": models_config,
        "cache_dir": tmp_cache,
        "usage_log": tmp_usage_log,
    }
    first, _ = adjudicate(_node("a"), _node("b"), **common)
    second, response = adjudicate(_node("a"), _node("b"), **common)

    assert client.call_count == 1  # the second call never reached the client
    assert second.outcome == first.outcome == "SAME"
    assert response.cached is True
