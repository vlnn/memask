import click

from memask.app import AppContext
from memask.context import ServiceContext
from memask.repository import items, jobs
from memask.router.dispatcher import dispatch
from memask.search.hybrid import hybrid_search
from memask.search.keyword import keyword_search
from memask.search.worker import enqueue_embedding, process_all_pending


def _get_app(ctx: click.Context) -> AppContext:
    return ctx.obj["app"]


def _get_svc(ctx: click.Context) -> ServiceContext:
    return _get_app(ctx).service_context()


@click.group()
@click.option(
    "--db",
    default=None,
    help="Database path (default: ~/.memask/memask.db)",
)
@click.pass_context
def cli(ctx: click.Context, db: str | None) -> None:
    ctx.ensure_object(dict)
    app = AppContext(db_path=db)
    ctx.obj["app"] = app
    ctx.call_on_close(app.shutdown)


@cli.command()
@click.argument("text", nargs=-1, required=True)
@click.pass_context
def input(ctx: click.Context, text: tuple[str, ...]) -> None:
    svc = _get_svc(ctx)
    full_text = " ".join(text)
    result = dispatch(svc, full_text)

    formatters = {
        "captured": _format_captured,
        "searched": _format_searched,
        "answered": _format_answered,
        "todo_created": _format_todo_created,
        "todo_listed": _format_todo_listed,
        "todo_completed": _format_todo_completed,
        "todo_not_found": _format_todo_not_found,
        "app_command": _format_app_command,
    }
    formatter = formatters.get(result.action, _format_default)
    formatter(result)


@cli.command()
@click.argument("content", nargs=-1, required=True)
@click.option("--type", "item_type", default="note")
@click.option("--status")
@click.option("--category")
@click.option("--tags")
@click.pass_context
def create(
    ctx: click.Context,
    content: tuple[str, ...],
    item_type: str,
    status: str | None,
    category: str | None,
    tags: str | None,
) -> None:
    svc = _get_svc(ctx)
    full_content = " ".join(content)
    item = items.create_item(
        svc.conn,
        full_content,
        type=item_type,
        status=status,
        category=category,
        tags=tags,
    )
    if svc.store is not None and svc.embedder is not None:
        enqueue_embedding(svc.conn, item.id)
        process_all_pending(svc.conn, svc.store, svc.embedder)
    click.echo(f"Created [{item.type}] {item.id}: {item.content}")


@cli.command("list")
@click.option("--type", "item_type")
@click.option("--status")
@click.option("--category")
@click.option("--limit", default=20, type=int)
@click.pass_context
def list_cmd(
    ctx: click.Context,
    item_type: str | None,
    status: str | None,
    category: str | None,
    limit: int,
) -> None:
    svc = _get_svc(ctx)
    results = items.list_items(
        svc.conn,
        type=item_type,
        status=status,
        category=category,
        limit=limit,
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
@click.option("--keyword-only", is_flag=True)
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
    svc = _get_svc(ctx)

    if keyword_only or svc.store is None or svc.embedder is None:
        results = keyword_search(
            svc.conn,
            query,
            type=item_type,
            status=status,
            category=category,
            limit=limit,
        )
    else:
        results = hybrid_search(
            svc.conn,
            svc.store,
            svc.embedder,
            query,
            type=item_type,
            status=status,
            category=category,
            limit=limit,
        )

    if not results:
        click.echo("No results found.")
        return
    for result in results:
        item = result.item
        score = f"{result.score:.3f}"
        source = result.source
        click.echo(f"[{item.type}] [{source} {score}] {item.id}: {item.content}")


@cli.command("job-status")
@click.pass_context
def job_status(ctx: click.Context) -> None:
    svc = _get_svc(ctx)
    stats = jobs.queue_status(svc.conn)
    if not stats:
        click.echo("No jobs.")
        return
    for status, count in sorted(stats.items()):
        click.echo(f"{status}: {count}")


@cli.command("reindex")
@click.pass_context
def reindex(ctx: click.Context) -> None:
    from memask.search.startup import on_startup

    svc = _get_svc(ctx)
    if svc.store is None or svc.embedder is None:
        click.echo("No embedding service configured.")
        return

    result = on_startup(svc.conn, svc.store, svc.embedder)
    click.echo(f"Recovered {result['stalled_recovered']} stalled jobs")
    click.echo(f"Removed {result['orphans_removed']} orphaned vectors")
    click.echo(f"Enqueued {result['stale_reindex_enqueued']} items for reindexing")

    processed = process_all_pending(svc.conn, svc.store, svc.embedder)
    click.echo(f"Processed {processed} jobs")


@cli.group()
def model():
    """Manage LLM model files."""


@model.command("status")
def model_status_cmd():
    """Show current model status."""
    from memask.rag.models import model_status as get_status

    status = get_status()
    click.echo(f"Model:  {status['model_name']}")
    click.echo(f"Path:   {status['path']}")
    if status["downloaded"]:
        click.echo(f"Size:   {status['size_mb']} MB")
        click.echo("Status: ready")
    else:
        click.echo("Status: not downloaded")
        click.echo("Run 'memask model download' to fetch from:")
        click.echo(f"  {status['url']}")


@model.command("download")
def model_download_cmd():
    """Download the default LLM model."""
    from memask.rag.models import download_model, model_status as get_status

    status = get_status()
    if status["downloaded"]:
        click.echo(f"Model already downloaded at {status['path']}")
        click.echo(f"Size: {status['size_mb']} MB")
        return

    click.echo(f"Downloading {status['model_name']}...")
    click.echo(f"From: {status['url']}")

    def progress(pct, downloaded, total):
        mb_done = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        click.echo(f"\r  {pct:.0f}% ({mb_done:.0f}/{mb_total:.0f} MB)", nl=False)

    try:
        path = download_model(progress_callback=progress)
        click.echo(f"\nSaved to {path}")
    except Exception as e:
        click.echo(f"\nDownload failed: {e}", err=True)
        raise SystemExit(1)


@model.command("path")
def model_path_cmd():
    """Print the model file path (for scripting)."""
    from memask.rag.models import model_path

    click.echo(str(model_path()))


def _format_captured(result):
    click.echo(f"Saved: {result.data.get('content', '')}")


def _format_searched(result):
    results = result.data.get("results", [])
    if not results:
        click.echo("No results found.")
        return
    for r in results:
        tag = r.get("type", "?")
        src = r.get("source", "?")
        score = r.get("score", 0)
        click.echo(f"  [{tag}] [{src} {score:.3f}] {r['id']}: {r['content']}")


def _format_answered(result):
    click.echo(result.data.get("answer", ""))
    sources = result.data.get("sources", [])
    if sources:
        click.echo(f"\nSources: {', '.join(sources)}")


def _format_todo_created(result):
    click.echo(f"Todo: {result.data.get('content', '')}")


def _format_todo_listed(result):
    items_data = result.data.get("items", [])
    if not items_data:
        click.echo("No pending todos.")
        return
    for todo in items_data:
        click.echo(f"  [ ] {todo['id']}: {todo['content']}")


def _format_todo_completed(result):
    click.echo(f"Done: {result.data.get('content', '')}")


def _format_todo_not_found(result):
    click.echo("No matching todo found.")


def _format_app_command(result):
    click.echo(f"Command: {result.data.get('command', '')}")


def _format_default(result):
    click.echo(f"[{result.action}] OK")
