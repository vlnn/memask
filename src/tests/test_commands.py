import pytest

from memask.context import ServiceContext
from memask.repository.items import create_item, update_item
from memask.router.dispatcher import dispatch
from memask.router.intents import Intent
from memask.router.rules import classify_by_rules


class TestBareTodoRouting:
    def test_bare_todo_routes_to_list(self):
        result = classify_by_rules("/todo")
        assert result.intent == Intent.TODO_LIST, (
            "bare /todo should route to TODO_LIST not TODO_CREATE"
        )

    def test_bare_todo_with_spaces(self):
        result = classify_by_rules("/todo  ")
        assert result.intent == Intent.TODO_LIST, (
            "/todo with trailing spaces should route to TODO_LIST"
        )

    def test_bare_todo_dispatches_to_list(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")
        result = dispatch(svc, "/todo")
        assert result.action == "todo_listed", "bare /todo should list todos"
        assert len(result.data["items"]) == 1, "should show pending todos"


class TestDoneShortcut:
    def test_done_routes_to_complete(self):
        result = classify_by_rules("/done buy milk")
        assert result.intent == Intent.TODO_COMPLETE, (
            "/done should route to TODO_COMPLETE"
        )

    def test_done_completes_todo(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")
        result = dispatch(svc, "/done buy milk")
        assert result.action == "todo_completed", "/done should complete the todo"

    def test_bare_done_lists_completed(self, conn):
        svc = ServiceContext(conn=conn)
        item = create_item(conn, "done task", type="todo", status="pending")
        update_item(conn, item.id, status="done")
        result = dispatch(svc, "/done")
        assert result.action == "listed", "bare /done should list completed todos"
        assert len(result.data["items"]) == 1, "should show completed todos"


class TestUndone:
    def test_undone_routes_to_app_command(self):
        result = classify_by_rules("/undone buy milk")
        assert result.intent == Intent.APP_COMMAND, (
            "/undone should route to APP_COMMAND"
        )

    def test_undone_reopens_todo(self, conn):
        svc = ServiceContext(conn=conn)
        item = create_item(conn, "buy milk", type="todo", status="pending")
        update_item(conn, item.id, status="done")
        result = dispatch(svc, "/undone buy milk")
        assert result.action == "todo_reopened", "/undone should reopen todo"
        assert result.data["content"] == "buy milk", "should return reopened content"

    def test_undone_not_found(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "/undone nonexistent")
        assert result.action == "todo_not_found", (
            "should report not found for unmatched undone"
        )


class TestMultiMatchTodoComplete:
    def test_single_match_completes(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")
        create_item(conn, "call dentist", type="todo", status="pending")
        result = dispatch(svc, "/done milk")
        assert result.action == "todo_completed", (
            "single match should complete directly"
        )

    def test_multiple_matches_returns_ambiguous(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk from store", type="todo", status="pending")
        create_item(conn, "buy milk from market", type="todo", status="pending")
        result = dispatch(svc, "/done buy milk")
        assert result.action == "todo_ambiguous", (
            "multiple matches should return ambiguous"
        )
        assert len(result.data["matches"]) == 2, "should list both matches"

    def test_no_match_returns_not_found(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy eggs", type="todo", status="pending")
        result = dispatch(svc, "/done milk")
        assert result.action == "todo_not_found", (
            "no match should return not found"
        )


class TestListCommand:
    @pytest.mark.parametrize("command", ["/list", "/List", "/LIST"])
    def test_list_returns_items(self, conn, command):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note")
        result = dispatch(svc, command)
        assert result.action == "listed", f"'{command}' should return listed"
        assert len(result.data["items"]) == 1, "should list items"

    @pytest.mark.parametrize("command,expected_type", [
        ("/list notes", "note"),
        ("/list todos", "todo"),
    ])
    def test_list_filters(self, conn, command, expected_type):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note", type="note")
        create_item(conn, "a todo", type="todo", status="pending")
        result = dispatch(svc, command)
        assert result.action == "listed", f"'{command}' should return listed"
        for item in result.data["items"]:
            assert item["type"] == expected_type, (
                f"'{command}' should only return {expected_type}"
            )


class TestNotesAndTodosShortcuts:
    def test_notes_lists_notes(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note", type="note")
        create_item(conn, "a todo", type="todo", status="pending")
        result = dispatch(svc, "/notes")
        assert result.action == "listed", "/notes should return listed"
        for item in result.data["items"]:
            assert item["type"] == "note", "/notes should only return notes"

    def test_todos_lists_todos(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note", type="note")
        create_item(conn, "a todo", type="todo", status="pending")
        result = dispatch(svc, "/todos")
        assert result.action == "listed", "/todos should return listed"
        for item in result.data["items"]:
            assert item["type"] == "todo", "/todos should only return todos"


class TestHelpCommand:
    @pytest.mark.parametrize("command", ["!help", "/help"])
    def test_help_returns_commands(self, conn, command):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, command)
        assert result.action == "help", f"'{command}' should return help"
        assert len(result.data["commands"]) > 0, "help should list commands"

    def test_help_includes_key_commands(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "!help")
        cmds = [c["command"] for c in result.data["commands"]]
        assert any("/done" in c for c in cmds), "help should mention /done"
        assert any("/undone" in c for c in cmds), "help should mention /undone"
        assert any("/list" in c for c in cmds), "help should mention /list"
        assert any("?query" in c for c in cmds), "help should mention ?query"


class TestStatusCommand:
    @pytest.mark.parametrize("command", ["!status", "/status"])
    def test_status_returns_info(self, conn, command):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, command)
        assert result.action == "status", f"'{command}' should return status"
        assert "items" in result.data, "status should include item count"
        assert "llm" in result.data, "status should include LLM info"

    def test_status_counts_items(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "note one")
        create_item(conn, "note two")
        result = dispatch(svc, "!status")
        assert result.data["items"] == 2, "status should count items"
