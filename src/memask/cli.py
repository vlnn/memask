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


def _get_daemon_client(ctx: click.Context):
    url = ctx.obj.get("daemon_url")
    if url is None:
        return None
    from memask.client import DaemonClient
    return DaemonClient(url)


@click.group()
@click.option(
    "--db",
    default=None,
    help="Database path (default: ~/.memask/memask.db)",
)
@click.option(
    "--url",
    default=None,
    help="Daemon URL (e.g. http://127.0.0.1:7394)",
)
@click.pass_context
def cli(ctx: click.Context, db: str | None, url: str | None) -> None:
    ctx.ensure_object(dict)
    ctx.obj["daemon_url"] = url
    if url is None:
        app = AppContext(db_path=db)
        ctx.obj["app"] = app
        ctx.call_on_close(app.shutdown)


@cli.command()
@click.argument("text", nargs=-1, required=True)
@click.pass_context
def input(ctx: click.Context, text: tuple[str, ...]) -> None:
    full_text = " ".join(text)

    client = _get_daemon_client(ctx)
    if client is not None:
        try:
            data = client.input(full_text)
            _format_result(data.get("action", "unknown"), data.get("data", {}))
            return
        except ConnectionError as exc:
            raise click.ClickException(f"daemon not reachable: {exc}")

    svc = _get_svc(ctx)
    result = dispatch(svc, full_text)
    _format_result(result.action, result.data)


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
    client = _get_daemon_client(ctx)
    if client is not None:
        try:
            data = client.items(type=item_type, status=status)
            items_list = data.get("items", [])
            if not items_list:
                click.echo("No items found.")
                return
            for item in items_list:
                prefix = f"[{item['type']}]"
                if item.get("status"):
                    prefix += f" ({item['status']})"
                click.echo(f"{prefix} {item['id']}: {item['content']}")
            return
        except ConnectionError as exc:
            raise click.ClickException(f"daemon not reachable: {exc}")

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
    client = _get_daemon_client(ctx)
    if client is not None:
        try:
            data = client.search(query)
            results = data.get("results", [])
            if not results:
                click.echo("No results found.")
                return
            for r in results:
                click.echo(f"[{r.get('type', '?')}] {r['id']}: {r['content']}")
            return
        except ConnectionError as exc:
            raise click.ClickException(f"daemon not reachable: {exc}")

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


@cli.command("serve")
@click.option("--host", default="127.0.0.1", help="Bind address")
@click.option("--port", default=7394, type=int, help="Port number")
@click.pass_context
def serve_cmd(ctx, host, port):
    """Start the memask daemon."""
    from memask.daemon import run_daemon
    db = ctx.parent.params.get("db") if ctx.parent else None
    run_daemon(db_path=db, host=host, port=port)


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
    from memask.rag.models import download_model
    from memask.rag.models import model_status as get_status

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


def _format_result(action, data):
    formatters = {
        "captured": lambda d: click.echo(f"Saved: {d.get('content', '')}"),
        "searched": _format_search_results,
        "answered": _format_answer,
        "todo_created": lambda d: click.echo(f"Todo: {d.get('content', '')}"),
        "todo_created_batch": lambda d: [
            click.echo(f"Todo: {i.get('content', '')}") for i in d.get("items", [])
        ],
        "todo_listed": _format_todo_list,
        "todo_completed": lambda d: click.echo(f"Done: {d.get('content', '')}"),
        "todo_not_found": lambda d: click.echo(
            d.get("message", "No matching todo found.")
        ),
        "todo_ambiguous": _format_todo_list,
        "app_command": lambda d: click.echo(f"Command: {d.get('command', '')}"),
        "listed": _format_item_listing,
        "help": _format_help,
        "status": _format_status,
    }
    formatter = formatters.get(action)
    if formatter:
        formatter(data)
    else:
        click.echo(f"[{action}] OK")


def _format_search_results(data):
    results = data.get("results", [])
    if not results:
        click.echo("No results found.")
        return
    for r in results:
        tag = r.get("type", "?")
        src = r.get("source", "?")
        score = r.get("score", 0)
        click.echo(f"  [{tag}] [{src} {score:.3f}] {r['id']}: {r['content']}")


def _format_answer(data):
    click.echo(data.get("answer", ""))
    sources = data.get("sources", [])
    if sources:
        click.echo(f"\nSources: {', '.join(sources)}")


def _format_todo_list(data):
    items_data = data.get("items", data.get("matches", []))
    if not items_data:
        click.echo("No pending todos.")
        return
    for todo in items_data:
        status = todo.get("status", " ")
        mark = "x" if status == "done" else " "
        click.echo(f"  [{mark}] {todo['id']}: {todo['content']}")


def _format_item_listing(data):
    items_data = data.get("items", [])
    label = data.get("date_range")
    if label:
        click.echo(f"({label})")
    if not items_data:
        click.echo("No items found.")
        return
    for item in items_data:
        prefix = f"[{item.get('type', '?')}]"
        if item.get("status"):
            prefix += f" ({item['status']})"
        click.echo(f"  {prefix} {item['id']}: {item['content']}")


def _format_help(data):
    for cmd in data.get("commands", []):
        click.echo(f"  {cmd['command']:30s} {cmd['description']}")


def _format_status(data):
    click.echo(f"LLM available: {data.get('llm', False)}")
    click.echo(f"Embedder: {data.get('embedder', 'none')}")
    click.echo(f"Vectors: {data.get('vectors', 0)}")
    click.echo(f"Items: {data.get('items', 0)}")


@cli.command("install")
def install_cmd():
    """Install memask daemon to start on login."""
    from memask.autostart import install

    result = install()
    if not result["installed"]:
        raise click.ClickException(result.get("error", "installation failed"))

    platform = result.get("platform", "unknown")
    path = result["path"]
    click.echo(f"Installed ({platform}): {path}")

    if platform == "launchd":
        click.echo("To start now:  launchctl load " + path)
        click.echo("To stop:       launchctl unload " + path)
    elif platform == "systemd":
        click.echo("To start now:  systemctl --user enable --now memask")
        click.echo("To stop:       systemctl --user disable --now memask")


@cli.command("uninstall")
def uninstall_cmd():
    """Remove memask daemon from login startup."""
    from memask.autostart import status, uninstall

    current = status()
    if not current["installed"]:
        click.echo("Autostart not installed.")
        return

    result = uninstall()
    if result.get("uninstalled"):
        click.echo(f"Removed: {result['path']}")
    else:
        raise click.ClickException(result.get("error", "uninstall failed"))
