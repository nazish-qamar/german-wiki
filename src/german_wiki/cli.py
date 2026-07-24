"""``gw`` command-line interface (slice 1: ``reindex`` and ``list``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import config, index, vocab
from .db import connect

app = typer.Typer(add_completion=False, help="German Wiki — node layer CLI.")
console = Console()
err_console = Console(stderr=True)


@app.command()
def reindex(
    nodes_dir: Path = typer.Option(
        config.NODES_DIR, "--nodes-dir", help="Directory of node Markdown files."
    ),
    db: Path = typer.Option(config.DB_PATH, "--db", help="SQLite index path."),
) -> None:
    """Rebuild the SQLite index from /nodes (from scratch)."""
    counts = index.reindex(nodes_dir=nodes_dir, db_path=db)
    console.print(
        f"Indexed [bold]{counts['nodes']}[/] nodes, "
        f"{counts['links']} links, {counts['themes']} theme tags "
        f"from [dim]{nodes_dir}[/] -> [dim]{db}[/]"
    )


@app.command("list")
def list_nodes(
    cefr: Optional[str] = typer.Option(None, "--cefr", help="Filter by CEFR level, e.g. A2."),
    type: Optional[str] = typer.Option(
        None, "--type", help="Filter by type: grammar|vocab|phrase|pattern|culture."
    ),
    theme: Optional[str] = typer.Option(None, "--theme", help="Filter by theme tag, e.g. küche."),
    db: Path = typer.Option(config.DB_PATH, "--db", help="SQLite index path."),
    nodes_dir: Path = typer.Option(
        config.NODES_DIR, "--nodes-dir", help="Node dir (for the staleness check)."
    ),
) -> None:
    """List nodes from the index, with optional filters (combined with AND)."""
    if not Path(db).exists():
        err_console.print(f"[red]No index at {db}.[/] Run [bold]gw reindex[/] first.")
        raise typer.Exit(code=1)

    # Forgiving normalization of filter values (never mutates the vocab store).
    cefr = cefr.upper() if cefr else None
    type = type.lower() if type else None
    theme = vocab.normalize("themes", theme, learn=False) if theme else None

    conn = connect(db)
    try:
        if index.is_stale(conn, nodes_dir=nodes_dir):
            err_console.print(
                "[yellow]⚠ Index is stale[/] (a node file is newer than the last "
                "reindex). Run [bold]gw reindex[/] to sync."
            )
        rows = index.query_nodes(conn, cefr=cefr, type=type, theme=theme)
    finally:
        conn.close()

    table = Table(show_lines=False)
    for col in ("id", "type", "cefr", "title_de", "title_en", "themes", "status"):
        table.add_column(col)
    for row in rows:
        themes = json.loads(row["themes"]) if row["themes"] else []
        table.add_row(
            row["id"],
            row["type"],
            row["cefr"],
            row["title_de"],
            row["title_en"],
            ", ".join(themes),
            row["status"],
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} node(s)[/]")


if __name__ == "__main__":
    app()
