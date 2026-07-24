"""SQLite (+ sqlite-vec) connection and schema for the DERIVED index.

This DB is never authoritative — it is rebuilt from ``/nodes`` by ``reindex``.
The sqlite-vec extension is loaded on every connection (verified working), but no
vector table is created in slice 1: the embedding table (``vec_nodes``, float[384]
for multilingual-e5-small) arrives in slice 4. The schema is expressed as an
ordered list of named DDL statements so that slice 4 adds ``vec_nodes`` as one
more entry rather than editing existing DDL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from . import config

# (table_name, create_sql). rebuild_schema drops each table (reverse order) then
# creates them (forward order). Slice 4 appends a ("vec_nodes", <vec0 DDL>) entry.
SCHEMA: list[tuple[str, str]] = [
    (
        "nodes",
        """
        CREATE TABLE nodes (
            id                  TEXT PRIMARY KEY,
            title_de            TEXT NOT NULL,
            title_en            TEXT NOT NULL,
            type                TEXT NOT NULL,
            cefr                TEXT NOT NULL,
            cefr_basis          TEXT,
            status              TEXT NOT NULL,
            confidence          REAL,
            register            TEXT NOT NULL,   -- JSON array
            themes              TEXT,            -- JSON array or NULL
            source_ids          TEXT NOT NULL,   -- JSON array
            separable           INTEGER,         -- bool or NULL
            family_transparency TEXT,
            root                TEXT,
            lemmas              TEXT,            -- JSON array or NULL
            version             INTEGER,
            updated_at          TEXT,
            body_md             TEXT NOT NULL,
            path                TEXT NOT NULL
        )
        """,
    ),
    (
        "links",
        """
        CREATE TABLE links (
            source_id  TEXT NOT NULL,
            target     TEXT NOT NULL,
            relation   TEXT NOT NULL,
            confidence REAL
        )
        """,
    ),
    (
        "node_themes",
        """
        CREATE TABLE node_themes (
            node_id TEXT NOT NULL,
            theme   TEXT NOT NULL
        )
        """,
    ),
    (
        "meta",
        """
        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        """,
    ),
]

INDEXES: list[str] = [
    "CREATE INDEX idx_nodes_type ON nodes(type)",
    "CREATE INDEX idx_nodes_cefr ON nodes(cefr)",
    "CREATE INDEX idx_node_themes_theme ON node_themes(theme)",
    "CREATE INDEX idx_node_themes_node ON node_themes(node_id)",
    "CREATE INDEX idx_links_source ON links(source_id)",
]


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded and ``Row`` factory."""
    db_path = Path(db_path) if db_path is not None else config.DB_PATH
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def rebuild_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate all tables + indexes from scratch."""
    for name, _ in reversed(SCHEMA):
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    for _, create_sql in SCHEMA:
        conn.execute(create_sql)
    for index_sql in INDEXES:
        conn.execute(index_sql)
    conn.commit()
