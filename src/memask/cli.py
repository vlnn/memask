import click

from memask.db.connection import get_connection
from memask.db.migrate import migrate_up
from memask.repository import items, jobs
from memask.repository.search import search_items


@click.group()
@click.option("--db", default=None, help="Database path (default: ~/.memask/memask.db)")
@click.pass_context
def cli(ctx: click.Context, db: str | None) -> None:
    ctx.ensure_object(dict)
    conn = get_connection(db)
    migrate_up(conn)
    ctx.obj["conn"] = conn


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
    item = items.create_item(ctx.obj["conn"], content=content, type=item_type, **kwargs)
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
@click.option("--date-from")
@click.option("--date-to")
@click.option("--limit", default=20, type=int)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    item_type: str | None,
    status: str | None,
    category: str | None,
    date_from: str | None,
    date_to: str | None,
    limit: int,
) -> None:
    results = search_items(
        ctx.obj["conn"],
        query,
        type=item_type,
        status=status,
        category=category,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )
    if not results:
        click.echo("No results found.")
        return
    for result in results:
        item = result.item
        prefix = f"[{item.type}]"
        if item.status:
            prefix += f" ({item.status})"
        click.echo(f"{prefix} {item.id}: {item.content}")


@cli.command("job-status")
@click.pass_context
def job_status(ctx: click.Context) -> None:
    stats = jobs.queue_status(ctx.obj["conn"])
    if not stats:
        click.echo("No jobs.")
        return
    for status, count in sorted(stats.items()):
        click.echo(f"{status}: {count}")
