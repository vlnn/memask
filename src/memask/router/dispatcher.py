from datetime import UTC
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from memask.context import ServiceContext
from memask.models.item import Item
from memask.rag.pipeline import answer_question
from memask.repository.items import create_item, list_items, update_item, soft_delete_item
from memask.router.action_resolution import resolve_update
from memask.router.intents import Intent
from memask.router.query_refinement import needs_refinement, refine_search_query
from memask.router.router import route
from memask.search.hybrid import hybrid_search
from memask.search.keyword import keyword_search, SearchResult


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
        Intent.TODO_DELETE: _handle_todo_delete,
        Intent.TODO_UPDATE: _handle_todo_update,
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


def _to_utc_iso(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _date_filter(ctx):
    if ctx.date_range:
        return _to_utc_iso(ctx.date_range.start), _to_utc_iso(ctx.date_range.end)
    return None, None


def _effective_query(svc, query):
    if needs_refinement(query) and svc.llm and svc.llm.is_available():
        return refine_search_query(query, svc.llm)
    return query


def _handle_search(svc, text, routing):
    ctx = routing.query_context
    search_query = _effective_query(svc, ctx.raw_query)

    if not search_query:
        if ctx.date_range:
            return _handle_date_list(svc, ctx)
        return DispatchResult(action="searched", data={"results": []})

    if not search_query.strip() and ctx.date_range:
        return _handle_date_list(svc, ctx)

    results = _retrieve(svc, search_query, ctx)

    synthesis_query = _synthesis_question(text)

    session_history = None
    if svc.session is not None:
        session_history = svc.session.history()

    answer = answer_question(
        synthesis_query,
        results,
        llm=svc.llm,
        reranker=svc.reranker,
        session_history=session_history,
    )

    if answer.synthesized:
        if svc.session is not None:
            svc.session.add_exchange(synthesis_query, answer.answer)

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


def _handle_todo_create(svc, text, routing):
    content = _extract_todo_content(text) or text.strip()
    parts = _split_multi_todo(content)

    if len(parts) > 1:
        items = []
        for part in parts:
            item = create_item(svc.conn, part, type="todo", status="pending")
            items.append(item)
        return DispatchResult(
            action="todo_created_batch",
            data={
                "items": [
                    {"id": i.id, "content": i.content} for i in items
                ],
            },
            items=items,
        )

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

    if not search_text and svc.embedder is None:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to complete."},
        )

    todos = list_items(svc.conn, type="todo", status="pending")

    if search_text:
        matches = _find_todos(search_text, todos)
        if len(matches) == 0 and svc.embedder is not None:
            matches = _find_todos_semantic(search_text, todos, svc.embedder)
    elif svc.embedder is not None:
        matches = _find_todos_semantic(text.strip(), todos, svc.embedder)
    else:
        matches = []

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


def _handle_todo_delete(svc, text, routing):
    search_text = _extract_delete_query(text)
    if not search_text:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to delete."},
        )

    todos = list_items(svc.conn, type="todo", status="pending")
    matches = _find_todos(search_text, todos)

    if len(matches) == 0:
        return DispatchResult(action="todo_not_found")

    if len(matches) == 1:
        soft_delete_item(svc.conn, matches[0].id)
        return DispatchResult(
            action="todo_deleted",
            data={"id": matches[0].id, "content": matches[0].content},
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


def _handle_todo_update(svc, text, routing):
    search_text, new_content = _extract_update_parts(text)

    todos = list_items(svc.conn, type="todo", status="pending")
    if not todos:
        return DispatchResult(action="todo_not_found")

    matches = _find_todos(search_text, todos) if search_text else []

    if len(matches) == 1 and new_content:
        updated = update_item(svc.conn, matches[0].id, content=new_content)
        return DispatchResult(
            action="todo_updated",
            data={"id": updated.id, "content": updated.content},
            item=updated,
        )

    candidates = matches if matches else todos

    llm = svc.llm
    llm_available = llm is not None and llm.is_available()

    if llm_available:
        plan = resolve_update(text, candidates, llm)
        if plan:
            updated = update_item(svc.conn, plan.target_id, content=plan.new_content)
            return DispatchResult(
                action="todo_updated",
                data={"id": updated.id, "content": updated.content},
                item=updated,
            )

    if matches:
        return DispatchResult(
            action="todo_ambiguous",
            data={
                "message": f"Multiple todos match '{search_text}':",
                "matches": [{"id": t.id, "content": t.content} for t in matches],
            },
            items=list(matches),
        )

    return DispatchResult(action="todo_not_found")


def _handle_app_command(svc, text, routing):
    command = _extract_command_name(text)

    command_handlers = {
        "list": lambda: _handle_list(svc, text, routing),
        "notes": lambda: _handle_list_shortcut(svc, "note", routing),
        "todos": lambda: _handle_list_shortcut(svc, "todo", routing),
        "done": lambda: _handle_done_bare(svc),
        "undone": lambda: _handle_undone(svc, text),
        "help": lambda: _handle_help(),
        "status": lambda: _handle_status(svc),
    }

    handler = command_handlers.get(command)
    if handler:
        return handler()

    if _is_nl_help(text):
        return _handle_help()

    return DispatchResult(
        action="app_command",
        data={"command": command},
    )


def _handle_list(svc, text, routing):
    type_filter = _extract_list_type(text)
    date_from, date_to = _date_filter(routing.query_context)
    all_items = list_items(
        svc.conn, type=type_filter, date_from=date_from, date_to=date_to, limit=50,
    )
    data = {
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
    }
    if routing.query_context.date_range:
        data["date_range"] = routing.query_context.date_range.label
    return DispatchResult(action="listed", data=data, items=list(all_items))


def _handle_list_shortcut(svc, type_filter, routing):
    date_from, date_to = _date_filter(routing.query_context)
    all_items = list_items(
        svc.conn, type=type_filter, date_from=date_from, date_to=date_to, limit=50,
    )
    data = {
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
    }
    return DispatchResult(action="listed", data=data, items=list(all_items))


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
        ("/list -1w", "List items from last 7 days"),
        ("/notes", "List recent notes"),
        ("/todos", "List pending todos"),
        ("/done", "List completed todos"),
        ("-1d", "Show everything from yesterday"),
        ("today", "Show everything from today"),
        ("!help", "Show this help"),
        ("!status", "Show daemon status"),
        ("finished X", "Mark todo X as done"),
        ("remove X from my todos", "Delete todo X"),
        ("what's on my todo list?", "List pending todos"),
        ("change X to Y", "Update a todo"),
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


def _handle_date_list(svc, ctx):
    date_from, date_to = _date_filter(ctx)
    all_items = list_items(
        svc.conn,
        type=ctx.type_filter,
        status=ctx.status_filter,
        date_from=date_from,
        date_to=date_to,
        limit=50,
    )
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
            "date_range": ctx.date_range.label,
        },
        items=list(all_items),
    )


def _retrieve(svc, query, ctx):
    date_from, date_to = _date_filter(ctx)

    if svc.store is not None and svc.embedder is not None:
        return hybrid_search(
            svc.conn, svc.store, svc.embedder, query,
            type=ctx.type_filter, status=ctx.status_filter,
            date_from=date_from, date_to=date_to,
        )
    return keyword_search(
        svc.conn, query,
        type=ctx.type_filter, status=ctx.status_filter,
        date_from=date_from, date_to=date_to,
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


def _synthesis_question(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("?"):
        stripped = stripped[1:].strip()
    return stripped


def _extract_list_type(text: str) -> str | None:
    m = re.match(r"^/list\s+(\w+)", text.strip(), re.I)
    if not m:
        return None
    word = m.group(1).lower()
    type_map = {"notes": "note", "note": "note", "todos": "todo", "todo": "todo"}
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
        re.compile(r"^todos?[:\s]+(.+)$", re.I),
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
        re.compile(r"^finished\s+(.+)$", re.I),
        re.compile(r"^i\s+completed\s+(.+)$", re.I),
        re.compile(r"^i\s+already\s+(.+)$", re.I),
        re.compile(r"^mark\s+(.+?)\s+as\s+done$", re.I),
        re.compile(r"^(?:my\s+)?task\s+(?:about\s+)?(.+?)\s+is\s+(?:done|complete)\s*$", re.I),
        re.compile(r"^(.+?)\s+is\s+(?:done|complete)\s*$", re.I),
    ]:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return ""


def _extract_delete_query(text: str) -> str:
    for pattern in [
        re.compile(r"^remove\s+(.+?)\s+from\s+my\s+(?:todos?|tasks?|list)", re.I),
        re.compile(r"^delete\s+(?:the\s+)?(?:todo|task)\s+(?:about\s+)?(.+)$", re.I),
        re.compile(r"^delete\s+(?:my\s+)?(?:todo|task)\s+(?:about\s+)?(.+)$", re.I),
        re.compile(r"^cancel\s+the\s+(.+?)\s+(?:todo|task|reminder)$", re.I),
        re.compile(r"^remove\s+the\s+(.+?)\s+(?:task|todo)\s+from\s+my\s+list$", re.I),
        re.compile(r"^delete\s+(.+)$", re.I),
        re.compile(r"^remove\s+(.+)$", re.I),
    ]:
        match = pattern.match(text.strip())
        if match:
            return match.group(1).strip()
    return ""


UPDATE_PATTERNS = [
    re.compile(r"^(?:change|update)\s+(?:the\s+)?(.+?)\s+(?:todo\s+)?to\s+(.+)$", re.I),
    re.compile(r"^reschedule\s+(?:the\s+)?(.+?)\s+to\s+(.+)$", re.I),
    re.compile(r"^rename\s+(?:the\s+)?(.+?)\s+to\s+(.+)$", re.I),
    re.compile(r"^set\s+(?:the\s+)?(.+?)\s+to\s+(.+)$", re.I),
]


def _extract_update_parts(text: str) -> tuple[str | None, str | None]:
    stripped = text.strip()
    for pattern in UPDATE_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return stripped, None


def _extract_undone_query(text: str) -> str:
    m = re.match(r"^/undone\s+(.+)$", text.strip(), re.I)
    return m.group(1).strip() if m else ""


def _extract_command_name(text: str) -> str:
    stripped = text.strip()
    match = re.match(r"^[!/](\w+)", stripped)
    if match:
        return match.group(1).lower()
    return stripped.lower()


def _is_nl_help(text: str) -> bool:
    return bool(re.match(
        r"(?:what\s+can\s+you\s+do|help\s+me|how\s+does\s+this\s+work)",
        text.strip(),
        re.I,
    ))


def _find_todos(query: str, todos: list[Item]) -> list[Item]:
    query_lower = query.lower()
    return [t for t in todos if query_lower in t.content.lower()]


def _find_todo(query: str, todos: list[Item]) -> Item | None:
    matches = _find_todos(query, todos)
    return matches[0] if len(matches) == 1 else None


def _find_todos_semantic(
    query: str,
    todos: list[Item],
    embedder,
    threshold: float = 0.5,
) -> list[Item]:
    if not todos or embedder is None:
        return []

    query_vec = np.asarray(embedder.embed_one(query), dtype=np.float32)
    todo_texts = [t.content for t in todos]
    todo_vecs = np.asarray(embedder.embed_many(todo_texts), dtype=np.float32)

    scores = _cosine_similarities(query_vec, todo_vecs)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return [todos[best_idx]]
    return []


def _cosine_similarities(query_vec, candidate_vecs):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    norms = np.linalg.norm(candidate_vecs, axis=1, keepdims=True) + 1e-9
    normalized = candidate_vecs / norms
    return normalized @ query_norm


def _split_multi_todo(content: str) -> list[str]:
    normalized = re.sub(r",\s*and\s+", ", ", content)
    if "," not in normalized:
        return [content.strip()]
    parts = [p.strip() for p in normalized.split(",")]
    return [p for p in parts if p]
