"""End-to-end CLI for the model layer: gw cost and gw cache stats|clear."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from german_wiki.cli import app
from german_wiki.llm import _cache, _usage
from german_wiki.llm._usage import Usage

runner = CliRunner()
WIDE = {"COLUMNS": "220"}


def _combined(result) -> str:
    return result.stdout + (result.stderr or "")


def _seed_ledger(path: Path) -> None:
    _usage.log_call(
        step="extraction",
        provider="zai",
        model="glm-4.5-flash",
        cache_key="k1",
        cached=False,
        usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        cost_usd=0.001,
        usage_log=path,
    )
    _usage.log_call(
        step="extraction",
        provider="zai",
        model="glm-4.5-flash",
        cache_key="k1",
        cached=True,
        usage=Usage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        cost_usd=0.0,
        saved_usd=0.001,
        usage_log=path,
    )
    _usage.log_call(
        step="adjudication",
        provider="deepseek",
        model="deepseek-chat",
        cache_key="k2",
        cached=False,
        usage=Usage(prompt_tokens=50, completion_tokens=5, total_tokens=55),
        cost_usd=None,
        usage_log=path,
    )


def _seed_cache(cache_dir: Path, count: int = 3) -> None:
    for n in range(count):
        material = _cache.key_material(
            provider="zai",
            model="glm-4.5-flash",
            messages=[{"role": "user", "content": f"Quelle {n}"}],
            temperature=0.0,
            max_tokens=4096,
        )
        key = _cache.cache_key(material)
        _cache.write(
            key,
            {
                "key": key,
                "request": material,
                "text": "Antwort",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            cache_dir=cache_dir,
        )


# --- gw cost ---


def test_cost_on_an_empty_ledger_reports_no_calls(tmp_path: Path) -> None:
    result = runner.invoke(app, ["cost", "--log", str(tmp_path / "absent.jsonl")], env=WIDE)
    assert result.exit_code == 0
    assert "No model calls logged yet" in result.stdout


def test_cost_reports_totals(tmp_usage_log: Path) -> None:
    _seed_ledger(tmp_usage_log)
    result = runner.invoke(app, ["cost", "--log", str(tmp_usage_log)], env=WIDE)
    assert result.exit_code == 0
    assert "TOTAL" in result.stdout
    assert "3 call(s)" in result.stdout
    assert "1 cache hit(s)" in result.stdout


def test_cost_surfaces_unpriced_calls_separately(tmp_usage_log: Path) -> None:
    """An unknown rate must not read as $0.00 spent."""
    _seed_ledger(tmp_usage_log)
    result = runner.invoke(app, ["cost", "--log", str(tmp_usage_log)], env=WIDE)
    assert "1 call(s) on unpriced models" in result.stdout


def test_cost_by_step_groups_rows(tmp_usage_log: Path) -> None:
    _seed_ledger(tmp_usage_log)
    result = runner.invoke(app, ["cost", "--log", str(tmp_usage_log), "--by", "step"], env=WIDE)
    assert result.exit_code == 0
    assert "extraction" in result.stdout
    assert "adjudication" in result.stdout


def test_cost_rejects_an_unknown_group_by(tmp_usage_log: Path) -> None:
    _seed_ledger(tmp_usage_log)
    result = runner.invoke(app, ["cost", "--log", str(tmp_usage_log), "--by", "nope"], env=WIDE)
    assert result.exit_code == 1
    assert "Unknown --by value" in _combined(result)


def test_cost_rejects_a_malformed_since(tmp_usage_log: Path) -> None:
    _seed_ledger(tmp_usage_log)
    result = runner.invoke(
        app, ["cost", "--log", str(tmp_usage_log), "--since", "last-tuesday"], env=WIDE
    )
    assert result.exit_code == 1
    assert "Not an ISO date" in _combined(result)


def test_cost_since_filters(tmp_usage_log: Path) -> None:
    _seed_ledger(tmp_usage_log)
    result = runner.invoke(
        app, ["cost", "--log", str(tmp_usage_log), "--since", "2099-01-01"], env=WIDE
    )
    assert result.exit_code == 0
    assert "No model calls logged yet" in result.stdout


# --- gw cache ---


def test_cache_stats_on_an_empty_cache(tmp_cache: Path) -> None:
    result = runner.invoke(app, ["cache", "stats", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert result.exit_code == 0
    assert "Cache is empty" in result.stdout


def test_cache_stats_counts_entries(tmp_cache: Path) -> None:
    _seed_cache(tmp_cache, 3)
    result = runner.invoke(app, ["cache", "stats", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert result.exit_code == 0
    assert "entries" in result.stdout
    assert "3" in result.stdout


def test_cache_clear_without_yes_errors_and_keeps_entries(tmp_cache: Path) -> None:
    _seed_cache(tmp_cache, 2)
    result = runner.invoke(app, ["cache", "clear", "--cache-dir", str(tmp_cache)], env=WIDE)
    assert result.exit_code == 1
    assert "--yes" in _combined(result)
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 2


def test_cache_clear_with_yes_removes_entries(tmp_cache: Path) -> None:
    _seed_cache(tmp_cache, 2)
    result = runner.invoke(
        app, ["cache", "clear", "--yes", "--cache-dir", str(tmp_cache)], env=WIDE
    )
    assert result.exit_code == 0
    assert "Removed" in result.stdout
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 0


def test_cache_clear_older_than_days_keeps_recent(tmp_cache: Path) -> None:
    _seed_cache(tmp_cache, 2)
    result = runner.invoke(
        app,
        ["cache", "clear", "--yes", "--older-than-days", "30", "--cache-dir", str(tmp_cache)],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert _cache.stats(cache_dir=tmp_cache)["entries"] == 2
