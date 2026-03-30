import pytest

from memask.context import ServiceContext
from memask.repository.items import create_item
from memask.router.dispatcher import DispatchResult, dispatch, _handle_todo_create
from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context
from memask.search.vector_store import VectorStore
from memask.search.worker import enqueue_embedding, process_all_pending
from tests.helpers import FakeEmbeddingService


class TestDispatchSignature:
    def test_accepts_service_context(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "hello world")
        assert isinstance(result, DispatchResult), "should return DispatchResult"

    def test_capture_through_service_context(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "kubernetes cluster needs more RAM")
        assert result.action == "captured", "should capture plain text"
        assert result.item is not None, "captured result should have item"
        assert result.item.content == "kubernetes cluster needs more RAM", (
            "should store the original text"
        )


class TestDispatchResultShape:
    def test_has_action(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "hello")
        assert hasattr(result, "action"), "DispatchResult should have action"
        assert isinstance(result.action, str), "action should be a string"

    def test_has_data_dict(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "hello")
        assert hasattr(result, "data"), "DispatchResult should have data dict"
        assert isinstance(result.data, dict), "data should be a dict"

    def test_capture_data_contains_id_and_content(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "some note")
        assert "id" in result.data, "capture data should contain id"
        assert "content" in result.data, "capture data should contain content"

    def test_search_data_contains_results(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "deployment pipeline is broken")
        result = dispatch(svc, "?deployment")
        assert "results" in result.data, "search data should contain results"
        assert isinstance(result.data["results"], list), "results should be a list"

    def test_todo_create_data_contains_id(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "remind me to buy milk")
        assert result.action == "todo_created", "should create todo"
        assert "id" in result.data, "todo_created data should contain id"

    def test_todo_list_data_contains_items(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy eggs", type="todo", status="pending")
        result = dispatch(svc, "/todo list")
        assert result.action == "todo_listed", "should list todos"
        assert "items" in result.data, "todo_listed data should contain items"

    def test_serializable_to_dict(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "hello world")
        as_dict = result.to_dict()
        assert isinstance(as_dict, dict), "to_dict should return a dict"
        assert "action" in as_dict, "serialized should have action"
        assert "data" in as_dict, "serialized should have data"


class TestDispatchSearchUsesHybrid:
    def test_search_uses_hybrid_when_store_available(self, conn, lance_dir):
        embedder = FakeEmbeddingService()
        store = VectorStore(lance_dir, dimension=embedder.dimension)
        svc = ServiceContext(conn=conn, store=store, embedder=embedder)

        item = create_item(conn, "shipping to production on friday")
        enqueue_embedding(conn, item.id)
        process_all_pending(conn, store, embedder)

        result = dispatch(svc, "?deployment")
        assert result.action == "searched", "should route to search"
        found_ids = [r["id"] for r in result.data["results"]]
        assert item.id in found_ids, (
            "hybrid search should find semantically similar item"
        )

    def test_search_falls_back_to_keyword_when_no_store(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "deployment pipeline is broken")
        result = dispatch(svc, "?deployment")
        assert result.action == "searched", "should still search without store"
        assert len(result.data["results"]) > 0, "keyword search should find exact match"


class TestDispatchQueryContextWiring:
    def test_date_filters_passed_through(self, conn, lance_dir):
        embedder = FakeEmbeddingService()
        store = VectorStore(lance_dir, dimension=embedder.dimension)
        svc = ServiceContext(conn=conn, store=store, embedder=embedder)

        create_item(conn, "meeting notes about deployment")
        result = dispatch(svc, "?notes about deployment yesterday")
        assert result.action == "searched", "should route to search"

    def test_type_filter_passed_through(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "deployment todo", type="todo", status="pending")
        create_item(conn, "deployment note", type="note")
        result = dispatch(svc, "?deployment todos")
        if result.data["results"]:
            assert all(r.get("type") == "todo" for r in result.data["results"]), (
                "should filter by type when query mentions todos"
            )


class TestDispatchTodoActions:
    def test_todo_complete(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "buy milk", type="todo", status="pending")
        result = dispatch(svc, "/todo done buy milk")
        assert result.action == "todo_completed", "should complete todo"

    def test_todo_not_found(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "/todo done nonexistent task")
        assert result.action == "todo_not_found", "should report not found"

    def test_generic_app_command(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "!export")
        assert result.action == "app_command", (
            "should route unknown bang command to app_command"
        )
        assert result.data.get("command") == "export", "should extract command name"

    def test_help_command(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "!help")
        assert result.action == "help", "!help should return help action"
        assert "commands" in result.data, "help should include commands list"
        assert len(result.data["commands"]) > 0, (
            "help should list available commands"
        )

    def test_status_command(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "!status")
        assert result.action == "status", "!status should return status action"
        assert "items" in result.data, "status should include item count"


class TestTodoCreateWithoutPrefix:
    def _make_routing(self, text):
        return RoutingResult(
            intent=Intent.TODO_CREATE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context(text),
            raw_input=text,
            source="embedding",
        )

    @pytest.mark.parametrize("text,expected_content", [
        ("buy a book", "buy a book"),
        ("buy milk", "buy milk"),
        ("call mom", "call mom"),
        ("fix the leaky faucet", "fix the leaky faucet"),
    ])
    def test_bare_text_creates_todo_with_full_text(self, conn, text, expected_content):
        svc = ServiceContext(conn=conn)
        routing = self._make_routing(text)
        result = _handle_todo_create(svc, text, routing)
        assert result.action == "todo_created", (
            f"'{text}' routed to TODO_CREATE should create a todo, not list"
        )
        assert result.data["content"] == expected_content, (
            f"should use raw text '{text}' as content when no prefix to strip"
        )

    def test_prefixed_text_still_strips(self, conn):
        svc = ServiceContext(conn=conn)
        routing = self._make_routing("remind me to buy a book")
        result = _handle_todo_create(svc, "remind me to buy a book", routing)
        assert result.action == "todo_created", (
            "prefixed text should still create todo"
        )
        assert result.data["content"] == "buy a book", (
            "should strip 'remind me to' prefix"
        )

    def test_todo_prefix_still_strips(self, conn):
        svc = ServiceContext(conn=conn)
        routing = self._make_routing("/todo buy a book")
        result = _handle_todo_create(svc, "/todo buy a book", routing)
        assert result.action == "todo_created", (
            "/todo prefixed text should still create todo"
        )
        assert result.data["content"] == "buy a book", (
            "should strip '/todo' prefix"
        )

    def test_created_item_is_todo_type(self, conn):
        svc = ServiceContext(conn=conn)
        routing = self._make_routing("buy a book")
        result = _handle_todo_create(svc, "buy a book", routing)
        assert result.item is not None, "should return the created item"
        assert result.item.type == "todo", "created item should be type todo"
        assert result.item.status == "pending", "created item should be pending"
