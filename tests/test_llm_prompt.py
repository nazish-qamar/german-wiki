"""Prompt assembly puts fixed content first and variable content last (SPEC §10)."""

from __future__ import annotations

import pytest

from german_wiki.llm._prompt import Prompt, ShotPair

SHOTS = [
    ShotPair(user="Beispiel eins", assistant="Antwort eins"),
    ShotPair(user="Beispiel zwei", assistant="Antwort zwei"),
]


def _prompt(**overrides) -> Prompt:
    kwargs = {"system": "Du bist ein Lehrer.", "variable": "Der Text."}
    kwargs.update(overrides)
    return Prompt(**kwargs)


@pytest.mark.parametrize(
    "few_shot,expected_roles",
    [
        ([], ["system", "user"]),
        (SHOTS[:1], ["system", "user", "assistant", "user"]),
        (SHOTS, ["system", "user", "assistant", "user", "assistant", "user"]),
    ],
)
def test_role_order_is_fixed_prefix_then_variable(few_shot, expected_roles) -> None:
    messages = _prompt(few_shot=few_shot).to_messages()
    assert [m["role"] for m in messages] == expected_roles


def test_variable_is_always_the_last_message() -> None:
    messages = _prompt(few_shot=SHOTS).to_messages()
    assert messages[-1] == {"role": "user", "content": "Der Text."}


def test_only_the_last_message_changes_when_the_variable_changes() -> None:
    """The SPEC §10 property: two calls share a cacheable prefix."""
    a = _prompt(few_shot=SHOTS, variable="Quelle A").to_messages()
    b = _prompt(few_shot=SHOTS, variable="Quelle B").to_messages()
    assert a[:-1] == b[:-1]
    assert a[-1] != b[-1]


def test_output_schema_is_folded_into_the_system_message() -> None:
    messages = _prompt(output_schema='{"type": "object"}').to_messages()
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == 'Du bist ein Lehrer.\n\n{"type": "object"}'
    assert [m["role"] for m in messages].count("system") == 1


def test_absent_output_schema_leaves_the_system_message_untouched() -> None:
    assert _prompt().to_messages()[0]["content"] == "Du bist ein Lehrer."


def test_few_shot_pairs_keep_their_order() -> None:
    messages = _prompt(few_shot=SHOTS).to_messages()
    assert [m["content"] for m in messages[1:5]] == [
        "Beispiel eins",
        "Antwort eins",
        "Beispiel zwei",
        "Antwort zwei",
    ]


def test_version_is_not_sent_to_the_provider() -> None:
    """version only ever affects the cache key."""
    assert _prompt(version="extract@1").to_messages() == _prompt().to_messages()


def test_umlauts_survive_assembly() -> None:
    messages = _prompt(variable="Küche, Büro, Behörde").to_messages()
    assert messages[-1]["content"] == "Küche, Büro, Behörde"


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        Prompt(system="s", variable="v", scheme="typo")
