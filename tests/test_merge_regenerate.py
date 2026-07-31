"""Merge regeneration and the SPEC §12.1 drift guards: one flags, one refuses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeChatClient

from german_wiki.llm import ModelResponse, Usage
from german_wiki.merge import (
    MAX_REGENERATIONS,
    AdjudicationError,
    _ledger,
    _regenerate,
    check_cap,
    regenerate,
)
from german_wiki.models import Node

BODY_A = """\
Das Perfekt bildet man mit *haben* + Partizip II.

## Examples
- Ich habe gearbeitet. (I worked.)
"""

BODY_B = """\
Verben der Bewegung nehmen *sein*.

## Examples
- Ich bin nach Berlin gefahren. (I drove to Berlin.) [alltag]
"""


def _node(node_id: str, body: str = BODY_A, **overrides) -> Node:
    data = {
        "id": node_id,
        "title_de": "Perfekt",
        "title_en": "Perfect tense",
        "type": "grammar",
        "cefr": "A2",
        "status": "draft",
        "body_md": body,
    }
    data.update(overrides)
    return Node(**data)


def _response(text: str, *, finish_reason: str = "stop"):
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
    )


def _merge_payload(body: str, changelog: str = "merged") -> str:
    return json.dumps({"body_md": body, "changelog": changelog}, ensure_ascii=False)


def _decision(winner: str, outcome: str = "OVERLAP", approved: bool = True) -> _ledger.Decision:
    return _ledger.Decision(
        decision_id=f"d-{winner}-{outcome}-{approved}",
        proposal_id="p",
        decided_at="2026-07-31T00:00:00+00:00",
        approved=approved,
        kind="merge",
        outcome=outcome,
        winner=winner,
        loser="other",
    )


# --- example sentences ---


def test_example_lines_reads_the_examples_section_only() -> None:
    body = "Prose that is not an example.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n"
    assert _regenerate.example_lines(body) == ["Ich habe gearbeitet."]


def test_example_lines_strips_glosses_and_register_tags() -> None:
    body = "## Examples\n- **Kannst du mir helfen?** (Can you help?) [alltag, du-Ebene]\n"
    assert _regenerate.example_lines(body) == ["**Kannst du mir helfen?**"]


def test_example_lines_stops_at_the_next_heading() -> None:
    body = "## Examples\n- Eins. (One.)\n\n## Notes\n- Not an example.\n"
    assert _regenerate.example_lines(body) == ["Eins."]


def test_a_preserved_example_is_not_flagged() -> None:
    merged = "## Examples\n- Ich habe gearbeitet. (I worked.)\n"
    assert _regenerate.unsourced_examples(merged, [BODY_A]) == []


def test_reformatting_an_example_does_not_trip_the_flag() -> None:
    """The check must survive emphasis and punctuation, or it cries wolf on every merge."""
    merged = "## Examples\n- **Ich habe gearbeitet!** (I worked.)\n"
    assert _regenerate.unsourced_examples(merged, [BODY_A]) == []


def test_an_invented_example_is_flagged() -> None:
    """SPEC §4.1's 'do not add': a fabricated sentence is a fact you would memorize."""
    merged = "## Examples\n- Ich habe das Auto repariert. (I repaired the car.)\n"
    assert _regenerate.unsourced_examples(merged, [BODY_A, BODY_B]) == [
        "Ich habe das Auto repariert."
    ]


def test_raw_text_can_source_an_example_the_bodies_lack() -> None:
    """§12.1's anchor: /raw is what a derived body is checked against."""
    merged = "## Examples\n- Er ist eingeschlafen. (He fell asleep.)\n"
    raw = "Beispiel aus dem Buch: Er ist eingeschlafen."
    assert _regenerate.unsourced_examples(merged, [BODY_A, raw]) == []


def test_new_examples_finds_only_what_the_winner_lacks() -> None:
    winner, loser = _node("a", BODY_A), _node("b", BODY_B)
    assert _regenerate.new_examples(winner, loser) == ["Ich bin nach Berlin gefahren."]
    assert _regenerate.new_examples(winner, winner) == []


# --- regeneration ---


def test_regenerate_returns_a_body_and_a_changelog(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    merged_body = "Beide Hilfsverben.\n\n## Examples\n- Ich habe gearbeitet. (I worked.)\n"
    client = FakeChatClient(text=_merge_payload(merged_body, "added sein"))
    merged, _ = regenerate(
        _node("a", BODY_A),
        _node("b", BODY_B),
        sources=[BODY_A, BODY_B],
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert merged.changelog == "added sein"
    assert "Hilfsverben" in merged.body_md
    assert merged.unsourced == []


def test_regeneration_flags_but_does_not_refuse_an_invented_example(
    models_config: Path, tmp_cache: Path, tmp_usage_log: Path
) -> None:
    """Flag, don't veto: legitimate paraphrase is not drift, and the human is the gate."""
    body = "Text.\n\n## Examples\n- Der Hund frisst den Knochen. (The dog eats the bone.)\n"
    client = FakeChatClient(text=_merge_payload(body))
    merged, _ = regenerate(
        _node("a", BODY_A),
        _node("b", BODY_B),
        sources=[BODY_A, BODY_B],
        client=client,
        settings_path=models_config,
        cache_dir=tmp_cache,
        usage_log=tmp_usage_log,
    )
    assert merged.unsourced == ["Der Hund frisst den Knochen."]
    assert merged.body_md == body  # returned anyway; review decides


def test_a_truncated_merge_is_an_error_not_a_half_node() -> None:
    with pytest.raises(AdjudicationError, match="truncated"):
        _regenerate.parse(_response(_merge_payload("half a bo"), finish_reason="length"))


def test_an_empty_merged_body_is_refused() -> None:
    with pytest.raises(AdjudicationError, match="empty body"):
        _regenerate.parse(_response(_merge_payload("   ")))


def test_merge_prompt_puts_the_pair_last(models_config: Path) -> None:
    messages = _regenerate.build_prompt(
        _node("a", BODY_A), _node("b", BODY_B), b_adds="the sein auxiliary"
    ).to_messages()
    assert messages[0]["role"] == "system"
    # SPEC §4.1's load-bearing sentence must actually be in the prompt.
    assert "not present in either source" in messages[0]["content"]
    assert messages[-1]["content"].startswith("A: ")
    assert "B adds: the sein auxiliary" in messages[-1]["content"]


def test_raw_sources_are_not_sent_to_the_model() -> None:
    """§12.1 makes /raw the re-verification anchor -- for the *check*, not the prompt.

    Feeding it in would bloat every merge call and hand the model more material to blend,
    which is the opposite of SPEC §4.1's constraint.
    """
    variable = _regenerate.build_prompt(_node("a", BODY_A), _node("b", BODY_B)).to_messages()[-1]
    assert "Beispiel aus dem Buch" not in variable["content"]


# --- the regeneration cap, and its three ledger states ---


def test_a_readable_ledger_is_authoritative(tmp_decisions: Path) -> None:
    for _ in range(2):
        _ledger.append(_decision("perfekt"), decisions_log=tmp_decisions)
    check = check_cap(_node("perfekt", version=99), decisions_log=tmp_decisions)
    assert check.allowed is True
    assert check.count == 2  # version 99 is ignored entirely when the ledger can be read
    assert check.ledger_readable is True


def test_the_cap_refuses_at_the_limit(tmp_decisions: Path) -> None:
    for _ in range(MAX_REGENERATIONS):
        _ledger.append(_decision("perfekt"), decisions_log=tmp_decisions)
    check = check_cap(_node("perfekt"), decisions_log=tmp_decisions)
    assert check.allowed is False
    assert check.count == MAX_REGENERATIONS
    assert "/raw" in check.reason  # points at the re-derive path, not just "no"


def test_same_outcomes_do_not_count_toward_the_cap(tmp_decisions: Path) -> None:
    """SAME appends mechanically -- no model call, no re-encoding, so no drift."""
    for _ in range(MAX_REGENERATIONS + 3):
        _ledger.append(_decision("perfekt", outcome="SAME"), decisions_log=tmp_decisions)
    check = check_cap(_node("perfekt"), decisions_log=tmp_decisions)
    assert (check.allowed, check.count) == (True, 0)


def test_rejected_merges_do_not_count(tmp_decisions: Path) -> None:
    for _ in range(MAX_REGENERATIONS + 3):
        _ledger.append(_decision("perfekt", approved=False), decisions_log=tmp_decisions)
    assert check_cap(_node("perfekt"), decisions_log=tmp_decisions).count == 0


def test_a_missing_ledger_on_a_never_merged_node_proceeds(tmp_decisions: Path) -> None:
    """The fresh-clone case: no ledger yet because nothing has ever been merged."""
    assert not tmp_decisions.exists()
    check = check_cap(_node("perfekt"), decisions_log=tmp_decisions)
    assert check.allowed is True
    assert check.count is None  # unknown, and honestly reported as such
    assert check.ledger_readable is False


def test_a_missing_ledger_on_a_merged_node_refuses(tmp_decisions: Path) -> None:
    """The wipe case, and the reason the cap reads three states instead of two.

    If a lost ledger read as "count = 0, proceed", the one guard protecting against
    drift the reviewer cannot see would silently disarm. `version > 1` is the tripwire:
    hand-editable, so it can never *permit* a merge, but sound enough to forbid one.
    """
    check = check_cap(_node("perfekt", version=4), decisions_log=tmp_decisions)
    assert check.allowed is False
    assert check.count is None
    assert check.ledger_readable is False
    assert "git restore" in check.reason


def test_a_corrupt_line_poisons_the_whole_read(tmp_decisions: Path) -> None:
    """Skipping the bad line would undercount, which lands back in the same fail-open."""
    _ledger.append(_decision("perfekt"), decisions_log=tmp_decisions)
    with tmp_decisions.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")

    with pytest.raises(_ledger.LedgerUnreadable, match="Refusing to skip"):
        _ledger.merge_count("perfekt", decisions_log=tmp_decisions)
    assert check_cap(_node("perfekt", version=2), decisions_log=tmp_decisions).allowed is False


def test_an_empty_ledger_is_not_a_missing_one(tmp_decisions: Path) -> None:
    """"Nothing decided yet" and "the record is gone" are different facts."""
    tmp_decisions.write_text("", encoding="utf-8")
    check = check_cap(_node("perfekt", version=4), decisions_log=tmp_decisions)
    assert (check.allowed, check.count, check.ledger_readable) == (True, 0, True)
