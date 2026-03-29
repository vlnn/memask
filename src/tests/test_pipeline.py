import pytest

from memask.models.item import Item
from memask.rag.pipeline import answer_question, AnswerResult
from memask.search.keyword import SearchResult
from tests.helpers import FakeLLM, FakeReranker


def _item(id="item-1", content="some content", **kwargs):
    defaults = dict(
        id=id, type="note", content=content, title=None,
        status=None, priority=None, due_date=None, category=None,
        source="manual", tags=None, created_at="2025-06-15T10:00:00",
        updated_at="2025-06-15T10:00:00", deleted_at=None, metadata=None,
    )
    defaults.update(kwargs)
    return Item(**defaults)


def _result(id="item-1", content="some content", score=0.5):
    return SearchResult(item=_item(id=id, content=content), score=score)


class TestAnswerResultShape:
    def test_has_required_fields(self):
        result = AnswerResult(
            answer="test", sources=["a"], raw_results=[], synthesized=True,
        )
        assert result.answer == "test", "should store answer"
        assert result.sources == ["a"], "should store sources"
        assert result.synthesized is True, "should store synthesized flag"

    def test_is_frozen(self):
        result = AnswerResult(
            answer="test", sources=[], raw_results=[], synthesized=True,
        )
        with pytest.raises(AttributeError):
            result.answer = "modified"


class TestAnswerQuestionWithLLM:
    def test_returns_synthesized_answer(self):
        results = [_result(id="a", content="deploy on friday")]
        llm = FakeLLM(response="Deployment is planned for friday. [1]")
        result = answer_question("when is deployment?", results, llm=llm)
        assert result.synthesized is True, "should be marked as synthesized"
        assert "friday" in result.answer.lower(), "answer should contain LLM output"

    def test_sources_from_context(self):
        results = [
            _result(id="src-1", content="alpha"),
            _result(id="src-2", content="beta"),
        ]
        llm = FakeLLM(response="Combined info from sources.")
        result = answer_question("query", results, llm=llm)
        assert "src-1" in result.sources, "should include first source"
        assert "src-2" in result.sources, "should include second source"

    def test_llm_receives_context_and_query(self):
        results = [_result(content="secret deployment plan")]
        llm = FakeLLM(response="ok")
        answer_question("deployment?", results, llm=llm)
        assert len(llm.call_log) == 1, "should call LLM once"
        prompt = llm.call_log[0]["prompt"]
        assert "secret deployment plan" in prompt, "prompt should include context"
        assert "deployment?" in prompt, "prompt should include query"

    def test_llm_receives_system_message(self):
        results = [_result(content="note")]
        llm = FakeLLM(response="ok")
        answer_question("query", results, llm=llm)
        system = llm.call_log[0]["system"]
        assert system is not None, "should pass system message to LLM"
        assert len(system) > 0, "system message should not be empty"

    def test_preserves_raw_results(self):
        results = [
            _result(id="a", content="one"),
            _result(id="b", content="two"),
        ]
        llm = FakeLLM(response="answer")
        result = answer_question("query", results, llm=llm)
        assert len(result.raw_results) == 2, "should preserve raw results"


class TestAnswerQuestionWithReranker:
    def test_reranker_reorders_before_context(self):
        results = [
            _result(id="irrelevant", content="cats and dogs", score=0.9),
            _result(id="relevant", content="deployment pipeline", score=0.1),
        ]
        llm = FakeLLM(response="ok")
        reranker = FakeReranker()
        result = answer_question(
            "deployment", results, llm=llm, reranker=reranker,
        )
        assert result.sources[0] == "relevant", (
            "reranker should promote relevant result to first source"
        )

    def test_works_without_reranker(self):
        results = [_result(content="note")]
        llm = FakeLLM(response="ok")
        result = answer_question("query", results, llm=llm, reranker=None)
        assert result.synthesized is True, "should still synthesize without reranker"


class TestAnswerQuestionFallback:
    def test_falls_back_when_no_llm(self):
        results = [_result(id="a", content="deploy on friday", score=0.9)]
        result = answer_question("deployment", results, llm=None)
        assert result.synthesized is False, "should not be synthesized without LLM"

    def test_falls_back_when_llm_unavailable(self):
        results = [_result(id="a", content="deploy on friday")]
        llm = FakeLLM(available=False)
        result = answer_question("deployment", results, llm=llm)
        assert result.synthesized is False, (
            "should not be synthesized when LLM unavailable"
        )

    def test_fallback_returns_search_results(self):
        results = [
            _result(id="a", content="alpha", score=0.9),
            _result(id="b", content="beta", score=0.5),
        ]
        result = answer_question("query", results, llm=None)
        assert len(result.raw_results) == 2, "fallback should include raw results"
        assert result.sources == ["a", "b"], "fallback sources should list item ids"

    def test_fallback_answer_is_empty(self):
        results = [_result(content="note")]
        result = answer_question("query", results, llm=None)
        assert result.answer == "", "fallback answer should be empty string"

    def test_fallback_with_reranker_still_reranks(self):
        results = [
            _result(id="bad", content="cats", score=0.9),
            _result(id="good", content="deployment plan", score=0.1),
        ]
        reranker = FakeReranker()
        result = answer_question(
            "deployment", results, llm=None, reranker=reranker,
        )
        assert result.sources[0] == "good", (
            "fallback should still rerank even without LLM"
        )


class TestAnswerQuestionEmptyResults:
    def test_empty_results_with_llm(self):
        llm = FakeLLM(response="I don't have any notes about that.")
        result = answer_question("anything", [], llm=llm)
        assert result.synthesized is True, "should still call LLM with empty results"
        assert result.sources == [], "no sources from empty results"

    def test_empty_results_without_llm(self):
        result = answer_question("anything", [], llm=None)
        assert result.synthesized is False, "should fallback with empty results"
        assert result.sources == [], "no sources from empty results"


class TestAnswerQuestionSessionHistory:
    def test_session_history_passed_to_prompt(self):
        results = [_result(content="note about deploy")]
        llm = FakeLLM(response="ok")
        history = [
            {"role": "user", "content": "tell me about deploy"},
            {"role": "assistant", "content": "deploy is friday"},
        ]
        answer_question("what time?", results, llm=llm, session_history=history)
        prompt = llm.call_log[0]["prompt"]
        assert "tell me about deploy" in prompt, (
            "session history should appear in prompt"
        )

    def test_no_session_history(self):
        results = [_result(content="note")]
        llm = FakeLLM(response="ok")
        answer_question("query", results, llm=llm)
        assert len(llm.call_log) == 1, "should work without session history"


class TestAnswerQuestionTopN:
    @pytest.mark.parametrize("top_n", [1, 3, 5])
    def test_limits_context_sources(self, top_n):
        results = [_result(id=f"item-{i}", content=f"note {i}") for i in range(10)]
        llm = FakeLLM(response="ok")
        result = answer_question("query", results, llm=llm, top_n=top_n)
        assert len(result.sources) == top_n, f"should limit to top_n={top_n} sources"
