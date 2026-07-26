"""The JSONL ledger: record shape, append semantics, and derived totals."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from german_wiki.llm import _usage
from german_wiki.llm._usage import Usage


def _log(tmp_usage_log: Path, **overrides) -> dict:
    kwargs = {
        "step": "extraction",
        "provider": "zai",
        "model": "glm-4.5-flash",
        "cache_key": "abc123",
        "cached": False,
        "usage": Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        "cost_usd": 0.0,
        "usage_log": tmp_usage_log,
    }
    kwargs.update(overrides)
    return _usage.log_call(**kwargs)


# --- record shape ---


def test_record_has_exactly_the_documented_keys(tmp_usage_log: Path) -> None:
    assert set(_log(tmp_usage_log)) == set(_usage.RECORD_KEYS)


def test_record_is_one_json_line(tmp_usage_log: Path) -> None:
    _log(tmp_usage_log)
    lines = tmp_usage_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["step"] == "extraction"


def test_records_append_without_clobbering(tmp_usage_log: Path) -> None:
    for step in ("extraction", "adjudication", "vision"):
        _log(tmp_usage_log, step=step)
    assert len(_usage.read_records(usage_log=tmp_usage_log)) == 3


def test_priced_is_derived_from_cost(tmp_usage_log: Path) -> None:
    assert _log(tmp_usage_log, cost_usd=0.0)["priced"] is True
    assert _log(tmp_usage_log, cost_usd=None)["priced"] is False


def test_token_counts_come_from_the_usage_object(tmp_usage_log: Path) -> None:
    usage = Usage(
        prompt_tokens=1832,
        cached_prompt_tokens=1536,
        completion_tokens=214,
        total_tokens=2046,
    )
    record = _log(tmp_usage_log, usage=usage)
    assert record["prompt_tokens"] == 1832
    assert record["cached_prompt_tokens"] == 1536
    assert record["completion_tokens"] == 214
    assert record["total_tokens"] == 2046


def test_non_ascii_is_stored_unescaped(tmp_usage_log: Path) -> None:
    _log(tmp_usage_log, model="modell-küche")
    assert "modell-küche" in tmp_usage_log.read_text(encoding="utf-8")


def test_error_records_are_kept(tmp_usage_log: Path) -> None:
    record = _log(tmp_usage_log, error="APIError: boom", cost_usd=None)
    assert record["error"] == "APIError: boom"
    assert _usage.read_records(usage_log=tmp_usage_log)[0]["error"] == "APIError: boom"


def test_ledger_directory_is_created_on_demand(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "logs" / "llm_usage.jsonl"
    _log(nested)
    assert nested.is_file()


# --- reading ---


def test_missing_ledger_reads_as_empty(tmp_path: Path) -> None:
    assert _usage.read_records(usage_log=tmp_path / "absent.jsonl") == []


def test_blank_lines_are_skipped(tmp_usage_log: Path) -> None:
    _log(tmp_usage_log)
    with tmp_usage_log.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert len(_usage.read_records(usage_log=tmp_usage_log)) == 1


# --- totals ---


def test_totals_on_a_missing_ledger_are_zero(tmp_path: Path) -> None:
    result = _usage.totals(usage_log=tmp_path / "absent.jsonl")
    assert result["calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["hit_rate"] == 0.0


def test_totals_sum_calls_tokens_and_cost(tmp_usage_log: Path) -> None:
    _log(tmp_usage_log, cost_usd=0.001)
    _log(tmp_usage_log, cost_usd=0.002)
    result = _usage.totals(usage_log=tmp_usage_log)
    assert result["calls"] == 2
    assert result["prompt_tokens"] == 200
    assert result["completion_tokens"] == 20
    assert result["cost_usd"] == pytest.approx(0.003)


def test_totals_count_hits_and_hit_rate(tmp_usage_log: Path) -> None:
    _log(tmp_usage_log, cached=False)
    _log(tmp_usage_log, cached=True, saved_usd=0.004)
    _log(tmp_usage_log, cached=True, saved_usd=0.004)
    result = _usage.totals(usage_log=tmp_usage_log)
    assert (result["calls"], result["hits"]) == (3, 2)
    assert result["hit_rate"] == pytest.approx(2 / 3)
    assert result["saved_usd"] == pytest.approx(0.008)


def test_unpriced_calls_are_counted_separately_from_zero_cost(tmp_usage_log: Path) -> None:
    """A null cost must not silently read as $0.00 spent."""
    _log(tmp_usage_log, cost_usd=0.0)
    _log(tmp_usage_log, cost_usd=None)
    _log(tmp_usage_log, cost_usd=None)
    result = _usage.totals(usage_log=tmp_usage_log)
    assert result["calls"] == 3
    assert result["unpriced_calls"] == 2
    assert result["cost_usd"] == 0.0


@pytest.mark.parametrize(
    "group_by,expected",
    [
        ("step", {"adjudication", "extraction"}),
        ("model", {"glm-4.5-flash", "glm-4.6"}),
        ("provider", {"deepseek", "zai"}),
    ],
)
def test_totals_group_by(tmp_usage_log: Path, group_by, expected) -> None:
    _log(tmp_usage_log, step="extraction", model="glm-4.5-flash", provider="zai")
    _log(tmp_usage_log, step="extraction", model="glm-4.5-flash", provider="zai")
    _log(tmp_usage_log, step="adjudication", model="glm-4.6", provider="deepseek")

    result = _usage.totals(usage_log=tmp_usage_log, group_by=group_by)
    assert set(result["groups"]) == expected
    assert sum(g["calls"] for g in result["groups"].values()) == result["calls"]


def test_groups_absent_unless_requested(tmp_usage_log: Path) -> None:
    _log(tmp_usage_log)
    assert "groups" not in _usage.totals(usage_log=tmp_usage_log)


def test_totals_group_by_day(tmp_usage_log: Path) -> None:
    records = [
        {**_log(tmp_usage_log), "ts": "2026-07-01T10:00:00+00:00"},
        {**_log(tmp_usage_log), "ts": "2026-07-01T11:00:00+00:00"},
        {**_log(tmp_usage_log), "ts": "2026-07-05T10:00:00+00:00"},
    ]
    result = _usage.totals(records=records, group_by="day")
    assert set(result["groups"]) == {"2026-07-01", "2026-07-05"}
    assert result["groups"]["2026-07-01"]["calls"] == 2


def test_since_filters_by_date(tmp_usage_log: Path) -> None:
    records = [
        {**_log(tmp_usage_log), "ts": "2026-07-01T10:00:00+00:00"},
        {**_log(tmp_usage_log), "ts": "2026-07-20T10:00:00+00:00"},
    ]
    assert _usage.totals(records=records, since=date(2026, 7, 10))["calls"] == 1
    assert _usage.totals(records=records, since=date(2026, 1, 1))["calls"] == 2
