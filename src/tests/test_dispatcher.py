from memask.context import ServiceContext
from memask.repository.items import create_item
from memask.router.dispatcher import DispatchResult, dispatch
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

    def test_app_command(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "!help")
        assert result.action == "app_command", "should route to app command"
        assert result.data.get("command") == "help", "should extract command name"
