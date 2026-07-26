"""Optional live ingestion against a real provider. Skipped unless GW_LIVE_TESTS=1.

The rest of the suite is offline and free by construction. This file is the only
ingestion test that opens a socket.

    GW_LIVE_TESTS=1 pytest -m live

Everything is redirected into tmp_path -- raw, queue, nodes, vocab, cache and
ledger -- so the repo's tracked /raw and /nodes are never touched, and the
run cannot grow the real tag vocabulary.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from german_wiki import config, storage
from german_wiki.ingest import ingest_file, promote_source

LIVE = os.environ.get("GW_LIVE_TESTS") == "1"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not LIVE, reason="set GW_LIVE_TESTS=1 to run live model API tests"),
]

# Short and dense on purpose: a few genuinely distinct concepts, so the
# granularity rules in the prompt have something to work with.
SOURCE = """\
Wechselpräpositionen wie "in", "auf" und "unter" stehen mit dem Akkusativ, wenn
eine Bewegung gemeint ist (wohin?), und mit dem Dativ, wenn ein Ort gemeint ist
(wo?). Beispiel: Ich stelle das Glas auf den Tisch. Das Glas steht auf dem Tisch.

Im Büro sagt man höflich "Könnten Sie mir bitte helfen?" statt "Hilf mir".
"""


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    vocab = tmp_path / "vocab"
    vocab.mkdir()
    for name in ("themes.txt", "registers.txt", "aliases.yaml"):
        shutil.copy2(config.VOCAB_DIR / name, vocab / name)

    nodes = tmp_path / "nodes"
    nodes.mkdir()
    source = tmp_path / "notizen.txt"
    source.write_text(SOURCE, encoding="utf-8")

    return {
        "source": source,
        "raw_dir": tmp_path / "raw",
        "queue_dir": tmp_path / "queue",
        "nodes_dir": nodes,
        "vocab_dir": vocab,
        "cache_dir": tmp_path / "cache",
        "usage_log": tmp_path / "llm_usage.jsonl",
    }


def test_live_ingest_then_promote(workspace) -> None:
    result = ingest_file(
        workspace["source"],
        raw_dir=workspace["raw_dir"],
        queue_dir=workspace["queue_dir"],
        nodes_dir=workspace["nodes_dir"],
        vocab_dir=workspace["vocab_dir"],
        cache_dir=workspace["cache_dir"],
        usage_log=workspace["usage_log"],
    )

    # Extraction produced something, within the SPEC §2 cap.
    assert 1 <= len(result.nodes) <= 8
    assert not result.already_ingested

    # Raw provenance is complete.
    assert workspace["raw_dir"].joinpath(f"{result.source_id}.txt").read_bytes() == (
        workspace["source"].read_bytes()
    )
    assert workspace["raw_dir"].joinpath(f"{result.source_id}.json").is_file()

    # ADR-003: nothing reached /nodes yet.
    assert list(workspace["nodes_dir"].glob("*.md")) == []

    # Every queued candidate is a real, loadable node with provenance.
    for path in result.queue_paths:
        node = storage.load_node(path)
        assert node.status == "draft"
        assert node.source_ids == [result.source_id]
        assert node.body_md.strip()
        assert node.cefr_basis.startswith("llm:extraction")

    promoted = promote_source(
        result.source_id,
        queue_dir=workspace["queue_dir"],
        nodes_dir=workspace["nodes_dir"],
        vocab_dir=workspace["vocab_dir"],
        db_path=workspace["cache_dir"] / "index.db",
    )

    assert sorted(promoted.promoted) == sorted(n.id for n in result.nodes)
    assert promoted.refused == []
    assert len(list(workspace["nodes_dir"].glob("*.md"))) == len(result.nodes)
    assert list(workspace["queue_dir"].glob("*/*.md")) == []


def test_live_reingest_is_free(workspace) -> None:
    """The second run must be served from the cache (ADR-005)."""
    common = {
        "raw_dir": workspace["raw_dir"],
        "queue_dir": workspace["queue_dir"],
        "nodes_dir": workspace["nodes_dir"],
        "vocab_dir": workspace["vocab_dir"],
        "cache_dir": workspace["cache_dir"],
        "usage_log": workspace["usage_log"],
    }
    first = ingest_file(workspace["source"], **common)
    second = ingest_file(workspace["source"], force=True, **common)

    assert first.cached is False
    assert second.cached is True
    assert second.source_id == first.source_id
