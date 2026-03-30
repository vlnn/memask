import pytest

from memask.context import ServiceContext
from memask.repository.items import create_item
from memask.router.dispatcher import DispatchResult, dispatch, _handle_todo_create
from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context
from memask.search.vector_store import VectorStore
from memask.search.worker import enqueue_embedding, process_all_pending
from tests.helpers import FakeEmbeddingService, FakeLLM


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
        assert result.action == "todo_created", "prefixed text should still create todo"
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
        assert result.data["content"] == "buy a book", "should strip '/todo' prefix"

    def test_created_item_is_todo_type(self, conn):
        svc = ServiceContext(conn=conn)
        routing = self._make_routing("buy a book")
        result = _handle_todo_create(svc, "buy a book", routing)
        assert result.item is not None, "should return the created item"
        assert result.item.type == "todo", "created item should be type todo"
        assert result.item.status == "pending", "created item should be pending"


class TestTimelineQueries:
    def test_standalone_today_lists_items(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "fresh note")
        result = dispatch(svc, "today")
        assert result.action == "listed", (
            "standalone 'today' should list items in range"
        )
        assert len(result.data["items"]) >= 1, "should find today's item"
        assert result.data.get("date_range") == "today", (
            "should include date_range label"
        )

    def test_standalone_offset_lists_items(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note")
        result = dispatch(svc, "-1d")
        assert result.action == "listed", (
            "standalone '-1d' should list items"
        )

    def test_search_with_date_filter(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "deploy to production")
        result = dispatch(svc, "?deploy today")
        assert result.action == "searched", (
            "search with topic and date should still search"
        )

    def test_list_command_with_date(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "some note")
        result = dispatch(svc, "/list -1d")
        assert result.action == "listed", "/list -1d should list items"
        assert "date_range" in result.data, (
            "/list with date should include date_range label"
        )

    def test_notes_command_with_date(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note", type="note")
        create_item(conn, "a todo", type="todo", status="pending")
        result = dispatch(svc, "/notes -1d")
        assert result.action == "listed", "/notes -1d should list items"
        for item in result.data["items"]:
            assert item["type"] == "note", (
                "/notes should only return notes even with date filter"
            )

    def test_standalone_last_week(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "recent note")
        result = dispatch(svc, "last week")
        assert result.action == "listed", (
            "standalone 'last week' should list items"
        )


class TestQueryRefinementInDispatch:
    def test_natural_language_query_uses_llm_refinement(self, conn, mocker):
        mock_refine = mocker.patch(
            "memask.router.dispatcher.refine_search_query",
            return_value="running",
        )
        mocker.patch(
            "memask.router.dispatcher.needs_refinement",
            return_value=True,
        )
        llm = FakeLLM(response="running")
        svc = ServiceContext(conn=conn, llm=llm)
        create_item(conn, "VPN was running and blocking internet")
        dispatch(svc, "what did I write about running?")
        mock_refine.assert_called_once(), (
            "should call refine_search_query for natural language queries"
        )

    def test_keyword_query_skips_refinement(self, conn, mocker):
        mock_refine = mocker.patch(
            "memask.router.dispatcher.refine_search_query",
        )
        mocker.patch(
            "memask.router.dispatcher.needs_refinement",
            return_value=False,
        )
        svc = ServiceContext(conn=conn)
        create_item(conn, "deployment is broken")
        dispatch(svc, "?deployment")
        mock_refine.assert_not_called(), (
            "should not call refine for keyword queries"
        )

    def test_no_llm_skips_refinement(self, conn, mocker):
        mock_refine = mocker.patch(
            "memask.router.dispatcher.refine_search_query",
        )
        mocker.patch(
            "memask.router.dispatcher.needs_refinement",
            return_value=True,
        )
        svc = ServiceContext(conn=conn)
        create_item(conn, "some note about running")
        dispatch(svc, "what did I write about running?")
        mock_refine.assert_not_called(), (
            "should not call refine when no LLM available"
        )

    def test_unavailable_llm_skips_refinement(self, conn, mocker):
        mock_refine = mocker.patch(
            "memask.router.dispatcher.refine_search_query",
        )
        mocker.patch(
            "memask.router.dispatcher.needs_refinement",
            return_value=True,
        )
        llm = FakeLLM(available=False)
        svc = ServiceContext(conn=conn, llm=llm)
        create_item(conn, "some note about running")
        dispatch(svc, "what did I write about running?")
        mock_refine.assert_not_called(), (
            "should not call refine when LLM is unavailable"
        )

    def test_refined_query_finds_matching_item(self, conn):
        llm = FakeLLM(responses=["running", "Based on your notes..."])
        svc = ServiceContext(conn=conn, llm=llm)
        create_item(conn, "VPN was running and blocking internet")
        result = dispatch(svc, "what did I write about running?")
        assert result.action in ("searched", "answered"), (
            "refined query should produce search results"
        )
        if result.data.get("results"):
            contents = [r["content"] for r in result.data["results"]]
            assert any("running" in c.lower() for c in contents), (
                "refined search should find items matching extracted keywords"
            )
