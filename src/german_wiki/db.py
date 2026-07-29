"""SQLite (+ sqlite-vec) connection and schema for the DERIVED index.

This DB is never authoritative — it is rebuilt from ``/nodes`` by ``reindex``.
The sqlite-vec extension is loaded on every connection, and slice 4 adds the
embedding table ``vec_nodes`` as one more entry in ``SCHEMA`` rather than editing
existing DDL, exactly as slice 1 set it up to.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

from . import config

# Width of one embedding vector: multilingual-e5-small emits 384 dimensions.
#
# It is pinned HERE, next to the DDL that consumes it, because a vec0 table needs
# its width at CREATE time -- not in models.yaml, whose StepSettings/ResolvedStep
# are both extra="forbid" and describe routing, not storage layout.
# ``embed/_model.py`` asserts the loaded model actually reports this many
# dimensions, so swapping the model fails loudly at load rather than writing
# wrong-width vectors that sqlite-vec might reject noisily or accept quietly.
EMBEDDING_DIM = 384

# (table_name, create_sql). rebuild_schema drops each table (reverse order) then
# creates them (forward order).
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
    (
        # Vectors are stored normalized, so cosine distance is directly meaningful
        # and similarity is simply ``1 - distance`` (see embed/_store.py).
        #
        # DROP TABLE removes vec0's five shadow tables (vec_nodes_chunks, _info,
        # _rowids, _vector_chunks00) along with it, so rebuild_schema's existing
        # drop/create loop needs no special handling. Verified on sqlite-vec 0.1.9.
        "vec_nodes",
        f"""
        CREATE VIRTUAL TABLE vec_nodes USING vec0(
            node_id   TEXT PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}] distance_metric=cosine
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
