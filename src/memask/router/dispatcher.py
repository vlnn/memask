import re
from dataclasses import dataclass, field
from typing import Any

from memask.context import ServiceContext
from memask.models.item import Item
from memask.repository.items import create_item, list_items, update_item
from memask.router.intents import Intent
from memask.router.router import route
from memask.search.hybrid import hybrid_search
from memask.search.keyword import keyword_search


@dataclass(frozen=True)
class DispatchResult:
    action: str
    data: dict[str, Any] = field(default_factory=dict)
    item: Item | None = None
    items: list[Item] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = {"action": self.action, "data": self.data}
        if self.item:
            result["item"] = {
                "id": self.item.id,
                "type": self.item.type,
                "content": self.item.content,
                "status": self.item.status,
            }
        return result


def dispatch(svc: ServiceContext, text: str) -> DispatchResult:
    routing = route(text, embedding_service=svc.embedder)

    handlers = {
        Intent.CAPTURE: _handle_capture,
        Intent.SEARCH: _handle_search,
        Intent.TODO_CREATE: _handle_todo_create,
        Intent.TODO_LIST: _handle_todo_list,
        Intent.TODO_COMPLETE: _handle_todo_complete,
        Intent.APP_COMMAND: _handle_app_command,
    }

    handler = handlers.get(routing.intent, _handle_capture)
    return handler(svc, text, routing)


def _handle_capture(svc, text, routing):
    item = create_item(svc.conn, text.strip())
    return DispatchResult(
        action="captured",
        data={"id": item.id, "content": item.content},
        item=item,
    )


def _handle_search(svc, text, routing):
    ctx = routing.query_context
    query = ctx.raw_query

    if svc.store is not None and svc.embedder is not None:
        results = hybrid_search(
            svc.conn,
            svc.store,
            svc.embedder,
            query,
            type=ctx.type_filter,
            status=ctx.status_filter,
            date_from=ctx.date_hints[0] if ctx.date_hints else None,
        )
    else:
        results = keyword_search(
            svc.conn,
            query,
            type=ctx.type_filter,
            status=ctx.status_filter,
        )

    items = [r.item for r in results]
    return DispatchResult(
        action="searched",
        data={
            "results": [
                {
                    "id": r.item.id,
                    "content": r.item.content,
                    "type": r.item.type,
                    "score": r.score,
                    "source": r.source,
                }
                for r in results
            ]
        },
        items=items,
    )


def _handle_todo_create(svc, text, routing):
    content = _extract_todo_content(text)
    item = create_item(svc.conn, content, type="todo", status="pending")
    return DispatchResult(
        action="todo_created",
        data={"id": item.id, "content": item.content},
        item=item,
    )


def _handle_todo_list(svc, text, routing):
    todos = list_items(svc.conn, type="todo", status="pending")
    return DispatchResult(
        action="todo_listed",
        data={
            "items": [
                {"id": t.id, "content": t.content, "status": t.status} for t in todos
            ]
        },
        items=list(todos),
    )


def _handle_todo_complete(svc, text, routing):
    search_text = _extract_complete_query(text)
    todos = list_items(svc.conn, type="todo", status="pending")

    match = _find_todo(search_text, todos)
    if match is None:
        return DispatchResult(action="todo_not_found")

    updated = update_item(svc.conn, match.id, status="done")
    return DispatchResult(
        action="todo_completed",
        data={"id": updated.id, "content": updated.content},
        item=updated,
    )


def _handle_app_command(svc, text, routing):
    command = _extract_command_name(text)
    return DispatchResult(
        action="app_command",
        data={"command": command},
    )


def _extract_todo_content(text: str) -> str:
    stripped = text.strip()

    prefixed = re.match(r"^/todo\s+(?:add\s+)?(.+)$", stripped, re.I)
    if prefixed:
        return prefixed.group(1).strip()

    for pattern in [
        re.compile(r"^remind\s+me\s+(?:to\s+|about\s+)(.+)$", re.I),
        re.compile(r"^remind\s+(?:to\s+|about\s+)(.+)$", re.I),
        re.compile(r"^remind\s+(.+)$", re.I),
        re.compile(r"^todo[:\s]+(.+)$", re.I),
        re.compile(r"^add\s+todo\s+(.+)$", re.I),
        re.compile(
            r"^(?:i\s+need\s+to|don'?t\s+forget\s+to|remember\s+to|i\s+have\s+to|i\s+must|i\s+should)\s+(.+)$",
            re.I,
        ),
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
