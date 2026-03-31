from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from memask.context import ServiceContext
from memask.models.item import Item
from memask.rag.pipeline import answer_question
from memask.repository.items import (
    create_item,
    list_items,
    soft_delete_item,
    update_item,
)
from memask.router.action_resolution import resolve_update
from memask.router.instruction_handler import handle_instruction_command
from memask.router.intents import Intent
from memask.router.query_refinement import needs_refinement, refine_search_query
from memask.router.router import route
from memask.router.text_extraction import (
    extract_command_name,
    extract_complete_query,
    extract_delete_query,
    extract_list_type,
    extract_todo_content,
    extract_undone_query,
    extract_update_parts,
    is_nl_help,
    split_multi_todo,
    synthesis_question,
)
from memask.router.todo_matching import find_todos, find_todos_semantic
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
            result["item"] = self.item.to_summary()
            result["item"]["type"] = self.item.type
            result["item"]["status"] = self.item.status
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
        data=item.to_summary(),
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

    if not search_query or not search_query.strip():
        if ctx.date_range:
            return _build_item_listing(svc, ctx)
        return DispatchResult(action="searched", data={"results": []})

    results = _retrieve(svc, search_query, ctx)

    session_history = None
    if svc.session is not None:
        session_history = svc.session.history()

    answer = answer_question(
        synthesis_question(text),
        results,
        llm=svc.llm,
        reranker=svc.reranker,
        session_history=session_history,
        instructions=_load_instruction_texts(svc),
    )

    if answer.synthesized:
        if svc.session is not None:
            svc.session.add_exchange(synthesis_question(text), answer.answer)

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
    content = extract_todo_content(text) or text.strip()
    parts = split_multi_todo(content)

    if len(parts) > 1:
        items = [
            create_item(svc.conn, part, type="todo", status="pending")
            for part in parts
        ]
        return DispatchResult(
            action="todo_created_batch",
            data={"items": [i.to_summary() for i in items]},
            items=items,
        )

    item = create_item(svc.conn, content, type="todo", status="pending")
    return DispatchResult(
        action="todo_created",
        data=item.to_summary(),
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
    search_text = extract_complete_query(text)

    if not search_text and svc.embedder is None:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to complete."},
        )

    todos = list_items(svc.conn, type="todo", status="pending")

    if search_text:
        matches = find_todos(search_text, todos)
        if len(matches) == 0 and svc.embedder is not None:
            matches = find_todos_semantic(search_text, todos, svc.embedder)
    elif svc.embedder is not None:
        matches = find_todos_semantic(text.strip(), todos, svc.embedder)
    else:
        matches = []

    if len(matches) == 0:
        return DispatchResult(action="todo_not_found")

    if len(matches) == 1:
        updated = update_item(svc.conn, matches[0].id, status="done")
        return DispatchResult(
            action="todo_completed",
            data=updated.to_summary(),
            item=updated,
        )

    return _ambiguous_todo(search_text, matches)


def _handle_todo_delete(svc, text, routing):
    search_text = extract_delete_query(text)
    if not search_text:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to delete."},
        )

    todos = list_items(svc.conn, type="todo", status="pending")
    matches = find_todos(search_text, todos)

    if len(matches) == 0:
        return DispatchResult(action="todo_not_found")

    if len(matches) == 1:
        soft_delete_item(svc.conn, matches[0].id)
        return DispatchResult(
            action="todo_deleted",
            data=matches[0].to_summary(),
        )

    return _ambiguous_todo(search_text, matches)


def _handle_todo_update(svc, text, routing):
    search_text, new_content = extract_update_parts(text)

    todos = list_items(svc.conn, type="todo", status="pending")
    if not todos:
        return DispatchResult(action="todo_not_found")

    matches = find_todos(search_text, todos) if search_text else []

    if len(matches) == 1 and new_content:
        updated = update_item(svc.conn, matches[0].id, content=new_content)
        return DispatchResult(
            action="todo_updated",
            data=updated.to_summary(),
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
                data=updated.to_summary(),
                item=updated,
            )

    if matches:
        return _ambiguous_todo(search_text, matches)

    return DispatchResult(action="todo_not_found")


def _ambiguous_todo(search_text, matches):
    return DispatchResult(
        action="todo_ambiguous",
        data={
            "message": f"Multiple todos match '{search_text}':",
            "matches": [t.to_summary() for t in matches],
        },
        items=list(matches),
    )


def _handle_app_command(svc, text, routing):
    command = extract_command_name(text)

    command_handlers = {
        "list": lambda: _build_item_listing(
            svc, routing.query_context, type_override=extract_list_type(text),
        ),
        "notes": lambda: _build_item_listing(
            svc, routing.query_context, type_override="note",
        ),
        "todos": lambda: _build_item_listing(
            svc, routing.query_context, type_override="todo",
        ),
        "done": lambda: _handle_done_bare(svc),
        "undone": lambda: _handle_undone(svc, text),
        "help": lambda: _handle_help(),
        "status": lambda: _handle_status(svc),
        "instruction": lambda: _handle_instruction(svc, text),
    }

    handler = command_handlers.get(command)
    if handler:
        return handler()

    if is_nl_help(text):
        return _handle_help()

    return DispatchResult(
        action="app_command",
        data={"command": command},
    )


def _build_item_listing(svc, ctx, *, type_override=None):
    date_from, date_to = _date_filter(ctx)
    type_filter = type_override or ctx.type_filter
    all_items = list_items(
        svc.conn,
        type=type_filter,
        status=ctx.status_filter if not type_override else None,
        date_from=date_from,
        date_to=date_to,
        limit=50,
    )
    data = {"items": [i.to_list_entry() for i in all_items]}
    if ctx.date_range:
        data["date_range"] = ctx.date_range.label
    return DispatchResult(action="listed", data=data, items=list(all_items))


def _handle_done_bare(svc):
    todos = list_items(svc.conn, type="todo", status="done", limit=50)
    return DispatchResult(
        action="listed",
        data={"items": [i.to_list_entry() for i in todos]},
        items=list(todos),
    )


def _handle_undone(svc, text):
    search_text = extract_undone_query(text)
    if not search_text:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "Specify which todo to reopen."},
        )

    done_todos = list_items(svc.conn, type="todo", status="done", limit=100)
    matches = find_todos(search_text, done_todos)

    if len(matches) == 0:
        return DispatchResult(
            action="todo_not_found",
            data={"message": "No completed todo matches."},
        )

    if len(matches) == 1:
        updated = update_item(svc.conn, matches[0].id, status="pending")
        return DispatchResult(
            action="todo_reopened",
            data=updated.to_summary(),
            item=updated,
        )

    return _ambiguous_todo(search_text, matches)


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
        ("/instruction <text>", "Save a persistent instruction"),
        ("/instruction", "List active instructions"),
        ("/instruction clear", "Clear all instructions"),
        ("/instruction remove <id>", "Remove one instruction"),
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


def _handle_instruction(svc, text):
    result = handle_instruction_command(svc.conn, text)
    return DispatchResult(
        action=result["action"],
        data=result.get("data", {}),
    )


def _load_instruction_texts(svc) -> list[str]:
    from memask.repository.instructions import list_active_instructions
    active = list_active_instructions(svc.conn)
    return [i["content"] for i in active]


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
