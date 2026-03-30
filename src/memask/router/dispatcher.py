import re
from dataclasses import dataclass, field
from typing import Any

from memask.context import ServiceContext
from memask.models.item import Item
from memask.rag.pipeline import answer_question
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
    results = _retrieve(svc, query, ctx)

    session_history = None
    if svc.session is not None:
        session_history = svc.session.history()

    answer = answer_question(
        query,
        results,
        llm=svc.llm,
        reranker=svc.reranker,
        session_history=session_history,
    )

    if answer.synthesized:
        if svc.session is not None:
            svc.session.add_exchange(query, answer.answer)

        return DispatchResult(
            action="answered",
            data={
                "answer": answer.answer,
                "sources": answer.sources,
                "results": _serialize_results(answer.raw_results),
            },
            items=[r.item for r in answer.raw_results],
        )

    return DispatchResult(
        action="searched",
        data={
            "results": _serialize_results(answer.raw_results or results),
        },
        items=[r.item for r in (answer.raw_results or results)],
    )


def _retrieve(svc, query, ctx):
    if svc.store is not None and svc.embedder is not None:
        return hybrid_search(
            svc.conn,
            svc.store,
            svc.embedder,
            query,
            type=ctx.type_filter,
            status=ctx.status_filter,
            date_from=ctx.date_hints[0] if ctx.date_hints else None,
        )
    return keyword_search(
        svc.conn,
        query,
        type=ctx.type_filter,
        status=ctx.status_filter,
    )


def _serialize_results(results):
    return [
        {
            "id": r.item.id,
            "content": r.item.content,
            "type": r.item.type,
            "score": r.score,
            "source": r.source,
        }
        for r in results
    ]


def _handle_todo_create(svc, text, routing):
    content = _extract_todo_content(text) or text.strip()
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
    if not search_text:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to complete."},
        )

    todos = list_items(svc.conn, type="todo", status="pending")
    matches = _find_todos(search_text, todos)

    if len(matches) == 0:
        return DispatchResult(action="todo_not_found")

    if len(matches) == 1:
        updated = update_item(svc.conn, matches[0].id, status="done")
        return DispatchResult(
            action="todo_completed",
            data={"id": updated.id, "content": updated.content},
            item=updated,
        )

    return DispatchResult(
        action="todo_ambiguous",
        data={
            "message": f"Multiple todos match '{search_text}':",
            "matches": [
                {"id": t.id, "content": t.content} for t in matches
            ],
        },
        items=list(matches),
    )


def _handle_app_command(svc, text, routing):
    command = _extract_command_name(text)

    command_handlers = {
        "list": lambda: _handle_list(svc, text),
        "notes": lambda: _handle_list_shortcut(svc, "note"),
        "todos": lambda: _handle_list_shortcut(svc, "todo"),
        "done": lambda: _handle_done_bare(svc),
        "undone": lambda: _handle_undone(svc, text),
        "help": lambda: _handle_help(),
        "status": lambda: _handle_status(svc),
    }

    handler = command_handlers.get(command)
    if handler:
        return handler()

    return DispatchResult(
        action="app_command",
        data={"command": command},
    )


def _handle_list(svc, text):
    type_filter = _extract_list_type(text)
    all_items = list_items(svc.conn, type=type_filter, limit=50)
    return DispatchResult(
        action="listed",
        data={
            "items": [
                {
                    "id": i.id,
                    "content": i.content,
                    "type": i.type,
                    "status": i.status,
                    "created_at": i.created_at,
                }
                for i in all_items
            ],
        },
        items=list(all_items),
    )


def _handle_list_shortcut(svc, type_filter):
    all_items = list_items(svc.conn, type=type_filter, limit=50)
    return DispatchResult(
        action="listed",
        data={
            "items": [
                {
                    "id": i.id,
                    "content": i.content,
                    "type": i.type,
                    "status": i.status,
                    "created_at": i.created_at,
                }
                for i in all_items
            ],
        },
        items=list(all_items),
    )


def _handle_done_bare(svc):
    todos = list_items(svc.conn, type="todo", status="done", limit=50)
    return DispatchResult(
        action="listed",
        data={
            "items": [
                {
                    "id": i.id,
                    "content": i.content,
                    "type": i.type,
                    "status": i.status,
                    "created_at": i.created_at,
                }
                for i in todos
            ],
        },
        items=list(todos),
    )


def _handle_undone(svc, text):
    search_text = _extract_undone_query(text)
    if not search_text:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to reopen."},
        )

    done_todos = list_items(svc.conn, type="todo", status="done", limit=100)
    matches = _find_todos(search_text, done_todos)

    if len(matches) == 0:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "No completed todo matches."},
        )

    if len(matches) == 1:
        updated = update_item(svc.conn, matches[0].id, status="pending")
        return DispatchResult(
            action="todo_reopened",
            data={"id": updated.id, "content": updated.content},
            item=updated,
        )

    return DispatchResult(
        action="todo_ambiguous",
        data={
            "message": f"Multiple completed todos match '{search_text}':",
            "matches": [
                {"id": t.id, "content": t.content} for t in matches
            ],
        },
        items=list(matches),
    )


def _handle_help():
    commands = [
        ("(any text)", "Capture a note"),
        ("remind me to ...", "Create a todo"),
        ("?query", "Search / ask a question"),
        ("/todo add ...", "Create a todo"),
        ("/todo list", "List pending todos"),
        ("/todo done ...", "Complete a todo"),
        ("/done ...", "Complete a todo (shortcut)"),
        ("/undone ...", "Reopen a completed todo"),
        ("/list", "List all recent items"),
        ("/notes", "List recent notes"),
        ("/todos", "List pending todos"),
        ("/done", "List completed todos"),
        ("!help", "Show this help"),
        ("!status", "Show daemon status"),
    ]
    return DispatchResult(
        action="help",
        data={
            "commands": [
                {"command": cmd, "description": desc}
                for cmd, desc in commands
            ],
        },
    )


def _handle_status(svc):
    from memask.repository.jobs import queue_status

    llm_available = svc.llm is not None and svc.llm.is_available()
    embedder_model = svc.embedder.model_name if svc.embedder else None
    index_count = svc.store.count() if svc.store else 0
    jobs = queue_status(svc.conn)
    item_count = len(list_items(svc.conn, limit=10000))

    return DispatchResult(
        action="status",
        data={
            "llm": llm_available,
            "embedder": embedder_model,
            "vectors": index_count,
            "items": item_count,
            "jobs": jobs,
        },
    )


def _extract_list_type(text: str) -> str | None:
    m = re.match(r"^/list\s+(\w+)", text.strip(), re.I)
    if not m:
        return None
    word = m.group(1).lower()
    type_map = {
        "notes": "note",
        "note": "note",
        "todos": "todo",
        "todo": "todo",
    }
    return type_map.get(word)


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

    return ""


def _extract_complete_query(text: str) -> str:
    for pattern in [
        re.compile(r"^/todo\s+(?:done|complete)\s+(.+)$", re.I),
        re.compile(r"^/done\s+(.+)$", re.I),
    ]:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return ""


def _extract_undone_query(text: str) -> str:
    m = re.match(r"^/undone\s+(.+)$", text.strip(), re.I)
    return m.group(1).strip() if m else ""


def _find_todos(query: str, todos: list[Item]) -> list[Item]:
    query_lower = query.lower()
    return [t for t in todos if query_lower in t.content.lower()]


def _find_todo(query: str, todos: list[Item]) -> Item | None:
    matches = _find_todos(query, todos)
    return matches[0] if len(matches) == 1 else None


def _extract_command_name(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^[!/](\w+)", stripped)
    if match:
        return match.group(1).lower()
    return stripped.lower()
