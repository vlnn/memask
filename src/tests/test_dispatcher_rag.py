import pytest

from memask.context import ServiceContext
from memask.models.item import Item
from memask.router.dispatcher import DispatchResult, dispatch
from memask.search.keyword import SearchResult
from tests.helpers import FakeLLM, FakeReranker


def _seed_item(conn, content, **kwargs):
    from memask.repository.items import create_item
    return create_item(conn, content, **kwargs)


class TestDispatchAnsweredWithLLM:
    def test_search_with_llm_returns_answered(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="deploy is friday"))
        _seed_item(conn, "deploy on friday")
        result = dispatch(svc, "?deployment")
        assert result.action == "answered", (
            "search with available LLM should return answered"
        )

    def test_answered_has_answer_text(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="the answer"))
        _seed_item(conn, "some note")
        result = dispatch(svc, "?note")
        assert result.data["answer"] == "the answer", (
            "answered result should contain LLM response"
        )

    def test_answered_has_sources(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="ok"))
        item = _seed_item(conn, "deployment notes")
        result = dispatch(svc, "?deployment")
        assert item.id in result.data["sources"], (
            "answered result should list source item ids"
        )

    def test_answered_has_results(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="ok"))
        _seed_item(conn, "deploy info")
        result = dispatch(svc, "?deploy")
        assert len(result.data["results"]) > 0, (
            "answered result should include raw search results"
        )

    def test_answered_results_have_expected_keys(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="ok"))
        _seed_item(conn, "deployment details")
        result = dispatch(svc, "?deployment")
        r = result.data["results"][0]
        assert "id" in r, "result should have id"
        assert "content" in r, "result should have content"
        assert "score" in r, "result should have score"
        assert "source" in r, "result should have source"


class TestDispatchFallbackWithoutLLM:
    def test_no_llm_returns_searched(self, conn):
        svc = ServiceContext(conn=conn)
        _seed_item(conn, "deployment notes")
        result = dispatch(svc, "?deployment")
        assert result.action == "searched", (
            "search without LLM should return searched"
        )

    def test_unavailable_llm_returns_searched(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(available=False))
        _seed_item(conn, "deployment notes")
        result = dispatch(svc, "?deployment")
        assert result.action == "searched", (
            "search with unavailable LLM should return searched"
        )

    def test_fallback_has_results(self, conn):
        svc = ServiceContext(conn=conn)
        _seed_item(conn, "deploy info")
        result = dispatch(svc, "?deploy")
        assert len(result.data["results"]) > 0, (
            "fallback should still return search results"
        )

    def test_fallback_has_no_answer_key(self, conn):
        svc = ServiceContext(conn=conn)
        _seed_item(conn, "something")
        result = dispatch(svc, "?something")
        assert "answer" not in result.data, (
            "fallback should not have an answer key"
        )


class TestDispatchWithReranker:
    def test_reranker_used_when_present(self, conn):
        reranker = FakeReranker()
        svc = ServiceContext(
            conn=conn, llm=FakeLLM(response="ok"), reranker=reranker,
        )
        _seed_item(conn, "deployment pipeline")
        _seed_item(conn, "cats and dogs")
        result = dispatch(svc, "?deployment")
        assert result.action == "answered", (
            "should answer with reranker and LLM"
        )
        first_result = result.data["results"][0]
        assert "deployment" in first_result["content"].lower(), (
            "reranker should promote deployment result"
        )

    def test_reranker_without_llm_still_reranks(self, conn):
        reranker = FakeReranker()
        svc = ServiceContext(conn=conn, reranker=reranker)
        _seed_item(conn, "cats and dogs")
        _seed_item(conn, "deployment pipeline")
        result = dispatch(svc, "?deployment")
        assert result.action == "searched", (
            "no LLM means searched even with reranker"
        )


class TestDispatchNonSearchUnchanged:
    def test_capture_still_works(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="ignored"))
        result = dispatch(svc, "just a plain note")
        assert result.action == "captured", (
            "capture should be unaffected by LLM presence"
        )

    def test_todo_create_still_works(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="ignored"))
        result = dispatch(svc, "remind me to buy milk")
        assert result.action == "todo_created", (
            "todo creation should be unaffected by LLM presence"
        )

    def test_todo_list_still_works(self, conn):
        svc = ServiceContext(conn=conn, llm=FakeLLM(response="ignored"))
        result = dispatch(svc, "/todo list")
        assert result.action == "todo_listed", (
            "todo listing should be unaffected by LLM presence"
        )
