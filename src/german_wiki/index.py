"""Build and query the derived SQLite index.

``reindex`` rebuilds the whole DB from ``/nodes`` (the source of truth) — it never
writes back to files. ``query_nodes`` powers ``gw list``. ``is_stale`` compares the
newest node file's mtime against the last reindex so the CLI can nudge the user to
re-sync. The file-reading path stays in ``storage`` so a later ``--verify`` flag can
cross-check index rows against files without going through this module.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import config
from .db import connect, rebuild_schema
from .storage import load_node

REINDEXED_AT_KEY = "reindexed_at"
EMBEDDING_MODEL_KEY = "embedding_model"
EMBEDDING_DIM_KEY = "embedding_dim"


def _json_or_none(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert one key in the generic ``meta`` table."""
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", (key, value))


def reindex(
    nodes_dir: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, int]:
    """Rebuild the index from scratch. Returns counts of what was written."""
    nodes_dir = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    conn = connect(db_path)
    try:
        rebuild_schema(conn)
        n_nodes = n_links = n_themes = 0
        for path in sorted(nodes_dir.glob("*.md")):
            node = load_node(path)
            conn.execute(
                """
                INSERT INTO nodes (
                    id, title_de, title_en, type, cefr, cefr_basis, status,
                    confidence, register, themes, source_ids, separable,
                    family_transparency, root, lemmas, version, updated_at,
                    body_md, path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    node.id,
                    node.title_de,
                    node.title_en,
                    node.type,
                    node.cefr,
                    node.cefr_basis,
                    node.status,
                    node.confidence,
                    json.dumps(node.register, ensure_ascii=False),
                    _json_or_none(node.themes),
                    json.dumps(node.source_ids, ensure_ascii=False),
                    None if node.separable is None else int(node.separable),
                    node.family_transparency,
                    node.root,
                    _json_or_none(node.lemmas),
                    node.version,
                    node.updated_at.isoformat() if node.updated_at else None,
                    node.body_md,
                    str(path),
                ),
            )
            n_nodes += 1
            for link in node.links:
                conn.execute(
                    "INSERT INTO links (source_id, target, relation, confidence) VALUES (?,?,?,?)",
                    (node.id, link.target, link.relation, link.confidence),
                )
                n_links += 1
            for theme in node.themes or []:
                conn.execute(
                    "INSERT INTO node_themes (node_id, theme) VALUES (?,?)",
                    (node.id, theme),
                )
                n_themes += 1

        set_meta(conn, REINDEXED_AT_KEY, repr(time.time()))
        conn.commit()
        return {"nodes": n_nodes, "links": n_links, "themes": n_themes}
    finally:
        conn.close()


def query_nodes(
    conn: sqlite3.Connection,
    *,
    cefr: str | None = None,
    type: str | None = None,
    theme: str | None = None,
) -> list[sqlite3.Row]:
    """Filtered node listing. Filters combine with AND. ``theme`` must already
    be normalized by the caller."""
    sql = ["SELECT n.* FROM nodes n"]
    params: list[str] = []
    if theme:
        sql.append("JOIN node_themes t ON t.node_id = n.id AND t.theme = ?")
        params.append(theme)
    where = []
    if cefr:
        where.append("n.cefr = ?")
        params.append(cefr)
    if type:
        where.append("n.type = ?")
        params.append(type)
    if where:
        sql.append("WHERE " + " AND ".join(where))
    sql.append("ORDER BY n.id")
    return conn.execute(" ".join(sql), params).fetchall()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def is_stale(conn: sqlite3.Connection, nodes_dir: Path | str | None = None) -> bool:
    """True if any node file is newer than the last reindex (or never indexed)."""
    nodes_dir = Path(nodes_dir) if nodes_dir is not None else config.NODES_DIR
    raw = get_meta(conn, REINDEXED_AT_KEY)
    if raw is None:
        return True
    reindexed_at = float(raw)
    mtimes = [p.stat().st_mtime for p in nodes_dir.glob("*.md")]
    return any(m > reindexed_at for m in mtimes)
