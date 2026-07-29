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

from . import config, embed, index, ingest, llm, storage, vocab
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
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
) -> None:
    """Rebuild the SQLite index from /nodes. Reloads cached vectors; never embeds."""
    counts = index.reindex(nodes_dir=nodes_dir, db_path=db)
    console.print(
        f"Indexed [bold]{counts['nodes']}[/] nodes, "
        f"{counts['links']} links, {counts['themes']} theme tags "
        f"from [dim]{nodes_dir}[/] -> [dim]{db}[/]"
    )

    # rebuild_schema drops vec_nodes, so restore whatever the embedding cache
    # already holds. compute=False means no model is ever loaded: reindex must
    # stay fast and must not pull in torch on a fresh clone.
    conn = connect(db)
    try:
        restored = embed.embed_nodes(
            storage.load_all_nodes(nodes_dir), conn=conn, cache_dir=cache_dir, compute=False
        )
    except ValueError as exc:
        # A broken or disabled embeddings config must never fail a reindex.
        err_console.print(f"[yellow]Vectors not restored:[/] {exc}")
        return
    finally:
        conn.close()

    if restored.stored:
        console.print(f"[dim]restored {restored.stored} cached vector(s)[/]")
    elif restored.total:
        console.print("[dim]no cached vectors — run [bold]gw embed[/] to compute them[/]")


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


# Both caches follow ADR-005's principle, so both are inspectable and clearable
# here. Otherwise .cache/embeddings/ grows with no CLI to see it.
CACHE_KINDS = ("llm", "embeddings")


def _cache_stats_for(kind: str, cache_dir: Path) -> dict:
    return (
        llm.cache_stats(cache_dir=cache_dir)
        if kind == "llm"
        else embed.cache_stats(cache_dir=cache_dir)
    )


@cache_app.command("stats")
def cache_stats(
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
) -> None:
    """Show what the model-call and embedding caches are holding."""

    def _stamp(value: float | None) -> str:
        # mtimes are shown in local time; that is what "when did I last run this" means.
        return (
            datetime.fromtimestamp(value, UTC).astimezone().strftime("%Y-%m-%d %H:%M")
            if value
            else "—"
        )

    rows = {kind: _cache_stats_for(kind, cache_dir) for kind in CACHE_KINDS}
    if not any(stats["entries"] for stats in rows.values()):
        console.print(f"Caches are empty [dim]({cache_dir})[/]")
        return

    table = Table(show_lines=False)
    for col in ("cache", "entries", "size", "oldest", "newest"):
        table.add_column(col)
    for kind, stats in rows.items():
        table.add_row(
            kind,
            f"{stats['entries']:,}",
            f"{stats['bytes'] / 1_048_576:.2f} MB",
            _stamp(stats["oldest"]),
            _stamp(stats["newest"]),
        )
    console.print(table)
    console.print(f"[dim]{cache_dir}[/]")


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", help="Required. Deletes cache entries."),
    kind: str = typer.Option("all", "--kind", help=f"Which cache: {'|'.join(CACHE_KINDS)}|all."),
    older_than_days: Optional[int] = typer.Option(
        None, "--older-than-days", help="Only remove entries older than N days."
    ),
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
) -> None:
    """Delete cached model responses and/or vectors. Both are regenerable — at a cost."""
    if kind not in (*CACHE_KINDS, "all"):
        err_console.print(
            f"[red]Unknown --kind value:[/] {kind}. Choose one of {', '.join(CACHE_KINDS)}, all."
        )
        raise typer.Exit(code=1)
    if not yes:
        err_console.print(
            "[red]Refusing to clear a cache without [bold]--yes[/].[/] "
            "Re-running uncached calls costs tokens (ADR-005); re-embedding costs CPU."
        )
        raise typer.Exit(code=1)

    targets = CACHE_KINDS if kind == "all" else (kind,)
    for target in targets:
        clear = llm.cache_clear if target == "llm" else embed.cache_clear
        removed = clear(cache_dir=cache_dir, older_than_days=older_than_days)
        console.print(f"Removed [bold]{removed}[/] {target} cache entry(ies)")
    console.print(f"[dim]{cache_dir}[/]")


@app.command("ingest")
def ingest_text(  # named to avoid shadowing the `ingest` package imported above
    file: Path = typer.Option(..., "--file", "-f", help="Plain-text source to ingest."),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if already in /raw."),
    raw_dir: Path = typer.Option(config.RAW_DIR, "--raw-dir", help="Immutable raw store."),
    queue_dir: Path = typer.Option(config.QUEUE_DIR, "--queue-dir", help="Review queue."),
    nodes_dir: Path = typer.Option(
        config.NODES_DIR, "--nodes-dir", help="Node dir (read-only here; for id collisions)."
    ),
    vocab_dir: Path = typer.Option(config.VOCAB_DIR, "--vocab-dir", help="Tag vocabulary."),
) -> None:
    """Extract concepts from a text file into the review queue. Writes NOTHING to /nodes."""
    if not Path(file).is_file():
        err_console.print(f"[red]No such file:[/] {file}")
        raise typer.Exit(code=1)

    try:
        result = ingest.ingest_file(
            file,
            force=force,
            raw_dir=raw_dir,
            queue_dir=queue_dir,
            nodes_dir=nodes_dir,
            vocab_dir=vocab_dir,
        )
    except ingest.ExtractionError as exc:
        err_console.print(f"[red]Extraction failed:[/] {exc}")
        if exc.reasoning_content:
            excerpt = exc.reasoning_content.strip().replace("\n", " ")[:300]
            err_console.print(f"[dim]Model reasoning began: {excerpt}…[/]")
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    if result.already_ingested:
        console.print(
            f"Already ingested as [bold]{result.source_id}[/] "
            f"({len(result.queue_paths)} still queued). Use [bold]--force[/] to re-run."
        )
        return

    if not result.nodes:
        console.print(
            f"[yellow]No candidates extracted[/] from {file}. Raw kept at [dim]{result.raw_path}[/]"
        )
        return

    table = Table(show_lines=False)
    for col in ("node id", "type", "cefr", "title_de", "title_en", "conf"):
        table.add_column(col)
    for node in result.nodes:
        table.add_row(
            node.id,
            node.type,
            node.cefr,
            node.title_de,
            node.title_en,
            f"{node.confidence:.2f}" if node.confidence is not None else "—",
        )
    console.print(table)

    queued = ingest.list_queue(queue_dir=queue_dir)
    console.print(
        f"[dim]raw -> {result.raw_path}[/]\n"
        f"[bold]{len(result.nodes)}[/] candidate(s) -> "
        f"[dim]{queue_dir / result.source_id}[/]"
    )
    # ADR-003: say plainly that nothing landed in /nodes, and name the gate.
    console.print(
        "[dim]Nothing written to /nodes.[/] Review the files above, delete any you "
        f"reject, then run [bold]gw promote {result.source_id}[/]"
    )
    if result.cached:
        console.print("[dim]served from cache — no tokens spent[/]")
    if len(queued) > 1:
        console.print(f"[dim]{len(queued)} source(s) now pending — see [bold]gw queue[/][/]")


@app.command()
def queue(
    queue_dir: Path = typer.Option(config.QUEUE_DIR, "--queue-dir", help="Review queue."),
) -> None:
    """Show ingested candidates awaiting review."""
    pending = ingest.list_queue(queue_dir=queue_dir)
    if not pending:
        console.print(f"Queue is empty [dim]({queue_dir})[/]")
        return

    table = Table(show_lines=False)
    for col in ("source id", "candidates", "node ids"):
        table.add_column(col)
    for source_id, paths in pending.items():
        ids = ", ".join(p.stem for p in paths)
        table.add_row(source_id, str(len(paths)), ids if len(ids) < 90 else ids[:87] + "…")
    console.print(table)
    console.print(
        f"[dim]{sum(len(p) for p in pending.values())} candidate(s) in "
        f"{len(pending)} source(s). Promote one with [bold]gw promote <source id>[/][/]"
    )


@app.command()
def promote(
    source_id: str = typer.Argument(..., help="Source id from `gw queue`."),
    queue_dir: Path = typer.Option(config.QUEUE_DIR, "--queue-dir", help="Review queue."),
    nodes_dir: Path = typer.Option(config.NODES_DIR, "--nodes-dir", help="Node directory."),
    vocab_dir: Path = typer.Option(config.VOCAB_DIR, "--vocab-dir", help="Tag vocabulary."),
    db: Path = typer.Option(config.DB_PATH, "--db", help="SQLite index path."),
) -> None:
    """Write a source's reviewed candidates into /nodes. The only command that does."""
    try:
        result = ingest.promote_source(
            source_id,
            queue_dir=queue_dir,
            nodes_dir=nodes_dir,
            vocab_dir=vocab_dir,
            db_path=db,
        )
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None

    if result.promoted:
        console.print(
            f"Promoted [bold]{len(result.promoted)}[/] node(s) -> [dim]{nodes_dir}[/]\n"
            f"[dim]{', '.join(result.promoted)}[/]"
        )
    for tag_field, tags in result.learned_tags.items():
        # ADR-007: the vocabulary grew, and it happened at the approved gate.
        console.print(f"[dim]learned {tag_field}: {', '.join(tags)}[/]")
    if result.reindexed:
        console.print(f"[dim]reindexed {result.reindexed['nodes']} node(s)[/]")

    for refusal in result.refused:
        err_console.print(f"[yellow]Refused[/] {refusal.node_id}: {refusal.reason}")
    if result.refused:
        err_console.print(
            f"[dim]{len(result.refused)} candidate(s) left in the queue. "
            "Nothing was overwritten.[/]"
        )
        raise typer.Exit(code=1)


def _require_index(db: Path):
    """Open the index, or explain which command builds it.

    Same convention as `gw list`: the derived DB is rebuildable, so the fix is
    always `gw reindex` rather than anything the user has to reason about.
    """
    if not Path(db).exists():
        err_console.print(f"[red]No index at {db}.[/] Run [bold]gw reindex[/] first.")
        raise typer.Exit(code=1)
    conn = connect(db)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_nodes'"
    ).fetchone()
    if not exists:
        conn.close()
        err_console.print(
            f"[red]Index at {db} has no vector table.[/] "
            "It predates slice 4 — run [bold]gw reindex[/] to rebuild it."
        )
        raise typer.Exit(code=1)
    return conn


def _queued_nodes(queue_dir: Path, source_id: Optional[str]) -> list:
    """Load queued candidates as Nodes -- they are complete node files (slice 3)."""
    pending = ingest.list_queue(queue_dir=queue_dir)
    if source_id is not None:
        if source_id not in pending:
            err_console.print(
                f"[red]Nothing queued for source {source_id}.[/] Run [bold]gw queue[/] to list."
            )
            raise typer.Exit(code=1)
        paths = pending[source_id]
    else:
        paths = [p for group in pending.values() for p in group]
    return [storage.load_node(p) for p in paths]


@app.command("embed")
def embed_cmd(  # named to avoid shadowing the `embed` package imported above
    nodes_dir: Path = typer.Option(config.NODES_DIR, "--nodes-dir", help="Node directory."),
    db: Path = typer.Option(config.DB_PATH, "--db", help="SQLite index path."),
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
) -> None:
    """Compute and store embeddings for /nodes. Loads the local model (ADR-004)."""
    nodes = storage.load_all_nodes(nodes_dir)
    if not nodes:
        console.print(f"No nodes to embed [dim]({nodes_dir})[/]")
        return

    conn = _require_index(db)
    try:
        result = embed.embed_nodes(nodes, conn=conn, cache_dir=cache_dir)
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None
    finally:
        conn.close()

    # The counts are the cache proof: a second run should say "0 new".
    console.print(
        f"Embedded [bold]{result.computed}[/] new, {result.from_cache} from cache "
        f"-> {result.stored} vector(s) in [dim]{db}[/]"
    )
    console.print(f"[dim]model {result.model}[/]")


@app.command()
def dupes(
    queue_source: Optional[str] = typer.Option(
        None, "--queue", help="Compare one queued source's candidates against /nodes."
    ),
    nodes_dir: Path = typer.Option(config.NODES_DIR, "--nodes-dir", help="Node directory."),
    queue_dir: Path = typer.Option(config.QUEUE_DIR, "--queue-dir", help="Review queue."),
    db: Path = typer.Option(config.DB_PATH, "--db", help="SQLite index path."),
    cache_dir: Path = typer.Option(config.CACHE_DIR, "--cache-dir", help="Cache root."),
    top: int = typer.Option(embed.DEFAULT_K, "--top", help="Neighbours considered per node."),
) -> None:
    """Report likely duplicates. Detection only — writes nothing to /nodes."""
    existing = storage.load_all_nodes(nodes_dir)
    if queue_source is not None:
        focus, against = _queued_nodes(queue_dir, queue_source), existing
    else:
        focus, against = existing, None

    if not focus:
        console.print("Nothing to compare.")
        return

    conn = _require_index(db)
    try:
        report = embed.find_duplicates(
            focus, against=against, conn=conn, cache_dir=cache_dir, k=top
        )
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from None
    finally:
        conn.close()

    if not report.matches:
        console.print(f"No duplicates found among {len(focus)} node(s).")
        _print_embedding_note(report)
        return

    table = Table(show_lines=False)
    for col in ("left", "right", "tier", "similarity", "band"):
        table.add_column(col)
    for match in report.matches:
        colour = "yellow" if match.band == "gray" else "red"
        table.add_row(
            match.left_id,
            match.right_id,
            match.tier,
            f"{match.similarity:.3f}",
            f"[{colour}]{match.band}[/]",
        )
    console.print(table)

    console.print(
        f"[dim]{len(report.matches)} pair(s): {len(report.duplicates)} duplicate, "
        f"{len(report.gray)} gray-zone. Report only — nothing was written to /nodes.[/]"
    )
    if report.gray:
        console.print(
            "[dim]Gray-zone pairs are what slice 5 sends to LLM adjudication (SPEC §3.1 tier 3).[/]"
        )
    _print_embedding_note(report)


def _print_embedding_note(report: embed.DuplicateReport) -> None:
    if report.embedding is not None and report.embedding.total:
        console.print(
            f"[dim]embeddings: {report.embedding.computed} new, "
            f"{report.embedding.from_cache} from cache[/]"
        )


if __name__ == "__main__":
    app()
