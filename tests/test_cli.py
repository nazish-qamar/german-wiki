"""End-to-end CLI: reindex then list, filters, missing-DB, staleness warning."""

from __future__ import annotations

import os
import time

from typer.testing import CliRunner

from german_wiki.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "200"}


def _combined(result):
    return result.stdout + (result.stderr or "")


def _reindex(tmp_nodes, tmp_db):
    return runner.invoke(
        app, ["reindex", "--nodes-dir", str(tmp_nodes), "--db", str(tmp_db)], env=WIDE
    )


def test_reindex_reports_counts(tmp_nodes, tmp_db):
    result = _reindex(tmp_nodes, tmp_db)
    assert result.exit_code == 0
    assert "Indexed 4 nodes" in result.stdout
    assert tmp_db.exists()


def test_list_type_filter(tmp_nodes, tmp_db):
    _reindex(tmp_nodes, tmp_db)
    result = runner.invoke(
        app,
        ["list", "--type", "vocab", "--db", str(tmp_db), "--nodes-dir", str(tmp_nodes)],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "familie" in result.stdout
    for other in ("prefix", "um-hilfe", "wechsel"):
        assert other not in result.stdout
    assert "1 node(s)" in result.stdout


def test_list_theme_filter_normalizes_alias(tmp_nodes, tmp_db):
    _reindex(tmp_nodes, tmp_db)
    # 'kitchen' is an alias of 'küche' in vocab/aliases.yaml
    result = runner.invoke(
        app,
        ["list", "--theme", "kitchen", "--db", str(tmp_db), "--nodes-dir", str(tmp_nodes)],
        env=WIDE,
    )
    assert result.exit_code == 0
    assert "familie" in result.stdout


def test_list_missing_db_errors(tmp_path, tmp_nodes):
    missing = tmp_path / "nope.db"
    result = runner.invoke(
        app, ["list", "--db", str(missing), "--nodes-dir", str(tmp_nodes)], env=WIDE
    )
    assert result.exit_code == 1
    assert "reindex" in _combined(result).lower()


def test_list_warns_when_stale(tmp_nodes, tmp_db):
    _reindex(tmp_nodes, tmp_db)
    target = next(tmp_nodes.glob("*.md"))
    future = time.time() + 1000
    os.utime(target, (future, future))
    result = runner.invoke(
        app, ["list", "--db", str(tmp_db), "--nodes-dir", str(tmp_nodes)], env=WIDE
    )
    assert result.exit_code == 0
    assert "stale" in _combined(result).lower()
