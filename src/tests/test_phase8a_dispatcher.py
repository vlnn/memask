import pytest

from memask.context import ServiceContext
from memask.repository.items import create_item, list_items
from memask.router.dispatcher import (
    DispatchResult,
    dispatch,
    _handle_todo_create,
    _handle_todo_delete,
    _split_multi_todo,
)
from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context


class TestTodoDeleteDispatch:
    def test_delete_single_match(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")
        create_item(conn, "call dentist", type="todo", status="pending")

        routing = RoutingResult(
            intent=Intent.TODO_DELETE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("remove buy milk from my todos"),
            raw_input="remove buy milk from my todos",
            source="rules",
        )
        result = _handle_todo_delete(svc, "remove buy milk from my todos", routing)
        assert result.action == "todo_deleted", (
            "single match should soft-delete the todo"
        )
        assert "milk" in result.data["content"], (
            "should return deleted todo content"
        )

    def test_delete_marks_item_as_deleted(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")

        routing = RoutingResult(
            intent=Intent.TODO_DELETE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("delete buy milk"),
            raw_input="delete buy milk",
            source="rules",
        )
        _handle_todo_delete(svc, "delete buy milk", routing)

        remaining = list_items(conn, type="todo", status="pending")
        assert len(remaining) == 0, (
            "deleted todo should not appear in pending list"
        )

    def test_delete_no_match(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy eggs", type="todo", status="pending")

        routing = RoutingResult(
            intent=Intent.TODO_DELETE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("remove milk"),
            raw_input="remove milk",
            source="rules",
        )
        result = _handle_todo_delete(svc, "remove milk", routing)
        assert result.action == "todo_not_found", (
            "no match should return todo_not_found"
        )

    def test_delete_ambiguous(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk from store", type="todo", status="pending")
        create_item(conn, "buy milk from market", type="todo", status="pending")

        routing = RoutingResult(
            intent=Intent.TODO_DELETE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("remove buy milk"),
            raw_input="remove buy milk",
            source="rules",
        )
        result = _handle_todo_delete(svc, "remove buy milk", routing)
        assert result.action == "todo_ambiguous", (
            "multiple matches should return todo_ambiguous"
        )
        assert len(result.data["matches"]) == 2, (
            "should list both matching todos"
        )

    def test_delete_empty_search_text(self, conn):
        svc = ServiceContext(conn=conn)
        routing = RoutingResult(
            intent=Intent.TODO_DELETE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("delete"),
            raw_input="delete",
            source="rules",
        )
        result = _handle_todo_delete(svc, "delete", routing)
        assert result.action == "todo_not_found", (
            "empty search text should return todo_not_found"
        )


class TestTodoDeleteEndToEnd:
    def test_nl_delete_dispatches_correctly(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy groceries", type="todo", status="pending")
        result = dispatch(svc, "remove buy groceries from my todos")
        assert result.action == "todo_deleted", (
            "NL delete should route through and soft-delete the matching todo"
        )


class TestNLTodoListEndToEnd:
    def test_nl_todo_list_dispatches(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")
        result = dispatch(svc, "what's on my todo list?")
        assert result.action == "todo_listed", (
            "'what's on my todo list?' should list pending todos"
        )
        assert len(result.data["items"]) == 1, (
            "should find the pending todo"
        )

    def test_show_my_todos_dispatches(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "call mom", type="todo", status="pending")
        result = dispatch(svc, "show my todos")
        assert result.action == "todo_listed", (
            "'show my todos' should list pending todos"
        )


class TestNLHelpEndToEnd:
    def test_what_can_you_do_returns_help(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "what can you do")
        assert result.action == "help", (
            "'what can you do' should return help"
        )
        assert "commands" in result.data, (
            "help should include commands list"
        )

    def test_help_me_returns_help(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "help me")
        assert result.action == "help", (
            "'help me' should return help"
        )


class TestNLCompletionEndToEnd:
    def test_finished_buying_groceries_completes(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buying groceries", type="todo", status="pending")
        result = dispatch(svc, "finished buying groceries")
        assert result.action == "todo_completed", (
            "'finished buying groceries' should complete matching todo"
        )

    def test_task_is_done_completes(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "jumping 12 times", type="todo", status="pending")
        result = dispatch(svc, "my task about jumping 12 times is complete")
        assert result.action == "todo_completed", (
            "'my task about jumping 12 times is complete' should complete it"
        )


class TestMultiTodoSplitting:
    def test_split_comma_separated(self):
        parts = _split_multi_todo("buy milk, jump 12 times, call mom")
        assert parts == ["buy milk", "jump 12 times", "call mom"], (
            "should split comma-separated items"
        )

    def test_split_comma_and(self):
        parts = _split_multi_todo("buy milk, and jump 12 times")
        assert parts == ["buy milk", "jump 12 times"], (
            "should split ', and ' separated items"
        )

    def test_no_split_for_single_item(self):
        parts = _split_multi_todo("buy milk")
        assert parts == ["buy milk"], (
            "single item should not be split"
        )

    def test_no_split_for_short_comma_phrase(self):
        parts = _split_multi_todo("buy milk, eggs")
        assert len(parts) == 2, (
            "comma-separated pair should split into 2 items"
        )

    def test_strips_whitespace(self):
        parts = _split_multi_todo("  buy milk ,  call mom  ")
        assert parts == ["buy milk", "call mom"], (
            "should strip whitespace from split items"
        )

    def test_filters_empty_parts(self):
        parts = _split_multi_todo("buy milk,,call mom")
        assert all(p for p in parts), (
            "should not include empty parts"
        )


class TestMultiTodoCreateEndToEnd:
    def test_multi_todo_creates_batch(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "todos: buy milk, jump 12 times, call mom")
        assert result.action == "todo_created_batch", (
            "comma-separated todos should create a batch"
        )
        assert len(result.data["items"]) == 3, (
            "should create 3 separate todos"
        )
        contents = [i["content"] for i in result.data["items"]]
        assert "buy milk" in contents, "should include 'buy milk'"
        assert "jump 12 times" in contents, "should include 'jump 12 times'"
        assert "call mom" in contents, "should include 'call mom'"

    def test_multi_todo_items_are_pending(self, conn):
        svc = ServiceContext(conn=conn)
        dispatch(svc, "todos: buy milk, call mom")
        todos = list_items(conn, type="todo", status="pending")
        assert len(todos) == 2, (
            "both todos should be pending in the database"
        )

    def test_single_todo_still_works(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "remind me to buy milk")
        assert result.action == "todo_created", (
            "single todo should still use todo_created action"
        )
