"""``gw`` command-line interface.

Slice 1: ``reindex`` and ``list``. Slice 2: ``cost`` and ``cache``.

The model layer is reached only through ``german_wiki.llm``'s public interface --
never its private modules -- so the CLI stays on the same side of that boundary
as the rest of the app.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import config, index, llm, vocab
from .db import connect

app = typer.Typer(add_completion=False, help="German Wiki — node layer CLI.")
cache_app = typer.Typer(help="Inspect and clear the model-call cache (ADR-005).")
app.add_typer(cache_app, name="cache")
console = Console()
err_console = Console(stderr=True)

GROUP_BY = ("step", "model", "provider", "day")


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


def _money(value: float | None) -> str:
    """Six decimals: lifetime spend is expected in single dollars (SPEC §10)."""
    return "—" if value is None else f"${value:.6f}"


def _row(label: str, bucket: dict) -> list[str]:
    return [
        label,
        str(bucket["calls"]),
        str(bucket["hits"]),
        f"{bucket['hit_rate']:.0%}",
        f"{bucket['prompt_tokens']:,}",
        f"{bucket['cached_prompt_tokens']:,}",
        f"{bucket['completion_tokens']:,}",
        _money(bucket["cost_usd"]),
        _money(bucket["saved_usd"]),
    ]


@app.command()
def cost(
    since: Optional[str] = typer.Option(None, "--since", help="ISO date, e.g. 2026-07-01."),
    by: Optional[str] = typer.Option(None, "--by", help=f"Group by: {'|'.join(GROUP_BY)}."),
    log: Path = typer.Option(config.USAGE_LOG_PATH, "--log", help="Usage ledger (JSONL) path."),
) -> None:
    """Token counts and estimated cost for model calls, with a running total."""
    cutoff: date | None = None
    if since:
        try:
            cutoff = date.fromisoformat(since)
        except ValueError:
            err_console.print(f"[red]Not an ISO date:[/] {since}. Use e.g. 2026-07-01.")
            raise typer.Exit(code=1) from None
    if by and by not in GROUP_BY:
        err_console.print(f"[red]Unknown --by value:[/] {by}. Choose one of {', '.join(GROUP_BY)}.")
        raise typer.Exit(code=1)

    totals = llm.cost_totals(usage_log=log, since=cutoff, group_by=by or None)
    if totals["calls"] == 0:
        console.print("No model calls logged yet.")
        return

    table = Table(show_lines=False)
    for col in (
        by or "total",
        "calls",
        "hits",
        "hit%",
        "prompt",
        "cached",
        "output",
        "cost",
        "saved",
    ):
        table.add_column(col)
    for name, bucket in totals.get("groups", {}).items():
        table.add_row(*_row(name, bucket))
    total_row = _row("TOTAL", totals)
    table.add_row(*(f"[bold]{cell}[/]" for cell in total_row))
    console.print(table)

    console.print(
        f"[dim]{totals['calls']} call(s), {totals['hits']} cache hit(s) "
        f"({totals['hit_rate']:.0%}), {_money(totals['cost_usd'])} spent, "
        f"{_money(totals['saved_usd'])} saved by cache[/]"
    )
    if totals["unpriced_calls"]:
        # Never silently fold an unknown rate into "$0.00 spent".
        console.print(
            f"[dim]{totals['unpriced_calls']} call(s) on unpriced models "
            f"(tokens counted, cost unknown)[/]"
        )


@cache_app.command("stats")
def cache_stats(
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
) -> None:
    """Show how much the model-call cache is holding."""
    stats = llm.cache_stats(cache_dir=cache_dir)
    if not stats["entries"]:
        console.print(f"Cache is empty [dim]({cache_dir})[/]")
        return

    def _stamp(value: float | None) -> str:
        # mtimes are shown in local time; that is what "when did I last run this" means.
        return (
            datetime.fromtimestamp(value, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
            if value
            else "—"
        )

    table = Table(show_header=False, show_lines=False)
    table.add_row("entries", f"{stats['entries']:,}")
    table.add_row("size", f"{stats['bytes'] / 1_048_576:.2f} MB")
    table.add_row("oldest", _stamp(stats["oldest"]))
    table.add_row("newest", _stamp(stats["newest"]))
    console.print(table)
    console.print(f"[dim]{cache_dir}[/]")


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", help="Required. Deletes cache entries."),
    older_than_days: Optional[int] = typer.Option(
        None, "--older-than-days", help="Only remove entries older than N days."
    ),
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
) -> None:
    """Delete cached model responses. They are regenerable, but re-running costs tokens."""
    if not yes:
        err_console.print(
            "[red]Refusing to clear the cache without [bold]--yes[/].[/] "
            "Re-running uncached calls costs tokens (ADR-005)."
        )
        raise typer.Exit(code=1)

    removed = llm.cache_clear(cache_dir=cache_dir, older_than_days=older_than_days)
    console.print(f"Removed [bold]{removed}[/] cache entry(ies) from [dim]{cache_dir}[/]")


if __name__ == "__main__":
    app()
