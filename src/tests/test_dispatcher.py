import pytest

from memask.router.dispatcher import dispatch


class TestDispatchCapture:
    def test_plain_text_creates_note(self, conn):
        result = dispatch(conn, "kubernetes needs more RAM")
        assert result.action == "captured", "plain text should create a note"
        assert result.item is not None, "should return the created item"
        assert result.item.type == "note", "should create a note type"
        assert result.item.content == "kubernetes needs more RAM", (
            "should store the original content"
        )


class TestDispatchTodoCreate:
    @pytest.mark.parametrize("text,expected_content", [
        ("/todo buy milk", "buy milk"),
        ("/todo add fix the faucet", "fix the faucet"),
        ("remind me to buy groceries", "buy groceries"),
        ("remind about reading", "reading"),
        ("remind about reading book", "reading book"),
        ("remind me about the meeting", "the meeting"),
        ("todo call the dentist", "call the dentist"),
        ("don't forget to send the email", "send the email"),
        ("i need to finish the report", "finish the report"),
        ("remember to water the plants", "water the plants"),
    ])
    def test_todo_patterns_create_todo(self, conn, text, expected_content):
        result = dispatch(conn, text)
        assert result.action == "todo_created", (
            f"'{text}' should create a todo"
        )
        assert result.item is not None, "should return the created item"
        assert result.item.type == "todo", "should create todo type"
        assert result.item.status == "pending", "new todo should be pending"
        assert result.item.content == expected_content, (
            f"should extract '{expected_content}' from '{text}'"
        )


class TestDispatchTodoList:
    def test_todo_list_returns_todos(self, conn):
        dispatch(conn, "/todo buy milk")
        dispatch(conn, "/todo fix the bug")
        result = dispatch(conn, "/todo list")
        assert result.action == "todo_listed", "should list todos"
        assert len(result.items) == 2, "should return both todos"

    def test_todo_list_empty(self, conn):
        result = dispatch(conn, "/todo list")
        assert result.action == "todo_listed", "should list todos even when empty"
        assert result.items == [], "should return empty list"


class TestDispatchTodoComplete:
    def test_todo_complete_marks_done(self, conn):
        create_result = dispatch(conn, "/todo buy milk")
        item_id = create_result.item.id
        result = dispatch(conn, f"/todo done {item_id}")
        assert result.action == "todo_completed", "should complete the todo"
        assert result.item.status == "done", "todo should be marked done"

    def test_todo_complete_by_keyword(self, conn):
        dispatch(conn, "/todo buy milk")
        result = dispatch(conn, "/todo done buy milk")
        assert result.action == "todo_completed", (
            "should find and complete todo by keyword match"
        )

    def test_todo_complete_not_found(self, conn):
        result = dispatch(conn, "/todo done nonexistent task")
        assert result.action == "todo_not_found", (
            "should report not found for missing todo"
        )


class TestDispatchSearch:
    def test_question_triggers_search(self, conn):
        dispatch(conn, "kubernetes cluster needs more RAM")
        result = dispatch(conn, "?kubernetes")
        assert result.action == "searched", "should trigger search"
        assert len(result.items) >= 1, "should find the matching item"

    def test_search_no_results(self, conn):
        result = dispatch(conn, "?nonexistent thing")
        assert result.action == "searched", "should still report searched"
        assert result.items == [], "should return empty results"


class TestDispatchAppCommand:
    @pytest.mark.parametrize("text", [
        "!help",
        "!status",
        "/help",
        "/status",
    ])
    def test_app_commands_return_command_result(self, conn, text):
        result = dispatch(conn, text)
        assert result.action == "app_command", (
            f"'{text}' should dispatch as app_command"
        )


class TestEndToEndFlow:
    def test_full_workflow(self, conn):
        dispatch(conn, "learned about lancedb for vector storage")
        dispatch(conn, "/todo review lancedb documentation")

        search_result = dispatch(conn, "?lancedb")
        assert search_result.action == "searched", "should search"
        assert len(search_result.items) >= 1, "should find lancedb note"

        todo_result = dispatch(conn, "/todo list")
        assert len(todo_result.items) == 1, "should have one todo"

        dispatch(conn, "/todo done review lancedb documentation")
        todo_result = dispatch(conn, "/todo list")
        assert len(todo_result.items) == 0, (
            "completed todo should not appear in pending list"
        )
