import click

from memask.db.connection import get_connection
from memask.db.migrate import migrate_up
from memask.repository import items, jobs
from memask.search.hybrid import hybrid_search
from memask.search.keyword import keyword_search
from memask.search.worker import enqueue_embedding, process_all_pending


def _lance_path(db_path):
    from pathlib import Path
    if db_path:
        return Path(db_path).parent / "vectors"
    return Path.home() / ".memask" / "vectors"


def _get_store(ctx: click.Context):
    if "store" not in ctx.obj:
        from memask.search.vector_store import VectorStore
        ctx.obj["store"] = VectorStore(_lance_path(ctx.obj.get("db_path")))
    return ctx.obj["store"]


def _get_embedder(ctx: click.Context):
    if "embedder" not in ctx.obj:
        from memask.search.embedding import EmbeddingService
        ctx.obj["embedder"] = EmbeddingService()
    return ctx.obj["embedder"]


@click.group()
@click.option("--db", default=None, help="Database path (default: ~/.memask/memask.db)")
@click.pass_context
def cli(ctx: click.Context, db: str | None) -> None:
    ctx.ensure_object(dict)
    conn = get_connection(db)
    migrate_up(conn)
    ctx.obj["conn"] = conn
    ctx.obj["db_path"] = db


@cli.command()
@click.argument("content")
@click.option("--type", "item_type", default="note")
@click.option("--title")
@click.option("--status")
@click.option("--priority", type=int)
@click.option("--due-date")
@click.option("--category")
@click.option("--tags")
@click.pass_context
def create(
    ctx: click.Context,
    content: str,
    item_type: str,
    title: str | None,
    status: str | None,
    priority: int | None,
    due_date: str | None,
    category: str | None,
    tags: str | None,
) -> None:
    kwargs = {
        k: v
        for k, v in {
            "title": title,
            "status": status,
            "priority": priority,
            "due_date": due_date,
            "category": category,
            "tags": tags,
        }.items()
        if v is not None
    }
    conn = ctx.obj["conn"]
    item = items.create_item(conn, content=content, type=item_type, **kwargs)
    enqueue_embedding(conn, item.id)
    click.echo(f"Created {item.type}: {item.id}")


@cli.command("list")
@click.option("--type", "item_type")
@click.option("--status")
@click.option("--category")
@click.option("--include-deleted", is_flag=True)
@click.pass_context
def list_cmd(
    ctx: click.Context,
    item_type: str | None,
    status: str | None,
    category: str | None,
    include_deleted: bool,
) -> None:
    results = items.list_items(
        ctx.obj["conn"],
        type=item_type,
        status=status,
        category=category,
        include_deleted=include_deleted,
    )
    if not results:
        click.echo("No items found.")
        return
    for item in results:
        prefix = f"[{item.type}]"
        if item.status:
            prefix += f" ({item.status})"
        click.echo(f"{prefix} {item.id}: {item.content}")


@cli.command()
@click.argument("query")
@click.option("--type", "item_type")
@click.option("--status")
@click.option("--category")
@click.option("--limit", default=20, type=int)
@click.option("--keyword-only", is_flag=True, help="Skip semantic search, keyword match only")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    item_type: str | None,
    status: str | None,
    category: str | None,
    limit: int,
    keyword_only: bool,
) -> None:
    conn = ctx.obj["conn"]

    if keyword_only:
        results = keyword_search(
            conn, query,
            type=item_type, status=status, category=category, limit=limit,
        )
    else:
        results = hybrid_search(
            conn, _get_store(ctx), _get_embedder(ctx), query,
            type=item_type, status=status, category=category, limit=limit,
        )

    if not results:
        click.echo("No results found.")
        return

    for result in results:
        item = result.item
        score = f"{result.score:.3f}"
        source = result.source
        prefix = f"[{item.type}]"
        if item.status:
            prefix += f" ({item.status})"
        click.echo(f"{prefix} [{source} {score}] {item.id}: {item.content}")


@cli.command("job-status")
@click.pass_context
def job_status(ctx: click.Context) -> None:
    stats = jobs.queue_status(ctx.obj["conn"])
    if not stats:
        click.echo("No jobs.")
        return
    for status, count in sorted(stats.items()):
        click.echo(f"{status}: {count}")


@cli.command("reindex")
@click.pass_context
def reindex(ctx: click.Context) -> None:
    """Process pending embedding jobs and reindex stale items."""
    from memask.search.startup import on_startup

    conn = ctx.obj["conn"]
    store = _get_store(ctx)
    embedder = _get_embedder(ctx)

    result = on_startup(conn, store, embedder)
    click.echo(f"Recovered {result['stalled_recovered']} stalled jobs")
    click.echo(f"Removed {result['orphans_removed']} orphaned vectors")
    click.echo(f"Enqueued {result['stale_reindex_enqueued']} items for reindexing")

    processed = process_all_pending(conn, store, embedder)
    click.echo(f"Processed {processed} jobs")
