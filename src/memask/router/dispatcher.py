import re
import sqlite3
from dataclasses import dataclass, field

from memask.models.item import Item
from memask.repository.items import create_item, list_items, update_item
from memask.router.intents import Intent
from memask.router.router import route
from memask.search.keyword import keyword_search


@dataclass(frozen=True)
class DispatchResult:
    action: str
    item: Item | None = None
    items: list[Item] = field(default_factory=list)
    command: str | None = None


def dispatch(conn: sqlite3.Connection, text: str) -> DispatchResult:
    routing = route(text)

    handlers = {
        Intent.CAPTURE: _handle_capture,
        Intent.SEARCH: _handle_search,
        Intent.TODO_CREATE: _handle_todo_create,
        Intent.TODO_LIST: _handle_todo_list,
        Intent.TODO_COMPLETE: _handle_todo_complete,
        Intent.APP_COMMAND: _handle_app_command,
    }

    handler = handlers.get(routing.intent, _handle_capture)
    return handler(conn, text, routing)


def _handle_capture(conn, text, routing):
    item = create_item(conn, text.strip())
    return DispatchResult(action="captured", item=item)


def _handle_search(conn, text, routing):
    query = routing.query_context.raw_query
    ctx = routing.query_context
    results = keyword_search(
        conn, query,
        type=ctx.type_filter,
        status=ctx.status_filter,
    )
    return DispatchResult(
        action="searched",
        items=[r.item for r in results],
    )


def _handle_todo_create(conn, text, routing):
    content = _extract_todo_content(text)
    item = create_item(conn, content, type="todo", status="pending")
    return DispatchResult(action="todo_created", item=item)


def _handle_todo_list(conn, text, routing):
    todos = list_items(conn, type="todo", status="pending")
    return DispatchResult(action="todo_listed", items=list(todos))


def _handle_todo_complete(conn, text, routing):
    search_text = _extract_complete_query(text)
    todos = list_items(conn, type="todo", status="pending")

    match = _find_todo(search_text, todos)
    if match is None:
        return DispatchResult(action="todo_not_found")

    updated = update_item(conn, match.id, status="done")
    return DispatchResult(action="todo_completed", item=updated)


def _handle_app_command(conn, text, routing):
    command = _extract_command_name(text)
    return DispatchResult(action="app_command", command=command)


def _extract_todo_content(text: str) -> str:
    stripped = text.strip()

    prefixed = re.match(r"^/todo\s+(?:add\s+)?(.+)$", stripped, re.I)
    if prefixed:
        return prefixed.group(1).strip()

    for pattern in [
        re.compile(r"^remind\s+me\s+(?:to|about)\s+(.+)$", re.I),
        re.compile(r"^remind\s+(?:to|about)\s+(.+)$", re.I),
        re.compile(r"^remind\s+(.+)$", re.I),
        re.compile(r"^todo[:\s]+(.+)$", re.I),
        re.compile(r"^add\s+todo\s+(.+)$", re.I),
        re.compile(r"^(?:i\s+need\s+to|don'?t\s+forget\s+to|remember\s+to|i\s+have\s+to|i\s+must|i\s+should)\s+(.+)$", re.I),
    ]:
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip()

    return stripped


def _extract_complete_query(text: str) -> str:
    match = re.match(r"^/todo\s+(?:done|complete)\s+(.+)$", text.strip(), re.I)
    if match:
        return match.group(1).strip()
    return text.strip()


def _find_todo(query: str, todos: list[Item]) -> Item | None:
    for todo in todos:
        if query.lower() == todo.id.lower():
            return todo

    query_lower = query.lower()
    for todo in todos:
        if query_lower in todo.content.lower():
            return todo
    return None


def _extract_command_name(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^[!/](\w+)", stripped)
    if match:
        return match.group(1).lower()
    return stripped.lower()
