import json

import numpy as np
import pytest

from memask.context import ServiceContext
from memask.models.item import Item
from memask.repository.items import create_item, get_item, list_items, update_item
from memask.router.dispatcher import (
    DispatchResult,
    dispatch,
)
from memask.router.todo_matching import find_todos, find_todos_semantic
from memask.router.text_extraction import extract_update_parts
from tests.helpers import FakeEmbeddingService, FakeLLM


class TestExtractUpdateParts:
    @pytest.mark.parametrize("text,search,new_content", [
        ("change the meeting todo to next week", "meeting", "next week"),
        ("update buy milk to buy oat milk", "buy milk", "buy oat milk"),
        ("change dentist to friday", "dentist", "friday"),
    ])
    def test_extracts_search_and_new_content(self, text, search, new_content):
        s, n = extract_update_parts(text)
        assert s is not None, f"should extract search text from '{text}'"
        assert n is not None, f"should extract new content from '{text}'"

    @pytest.mark.parametrize("text", [
        "change the meeting todo to next week",
        "update buy milk to buy oat milk",
    ])
    def test_search_part_is_nonempty(self, text):
        s, _ = extract_update_parts(text)
        assert s and len(s.strip()) > 0, (
            f"search part should be nonempty for '{text}'"
        )

    def test_fallback_returns_full_text(self):
        s, n = extract_update_parts("something unparseable here")
        assert s is not None, "should return fallback search text"
        assert n is None, "should return None for new_content on fallback"


class TestHandleTodoUpdate:
    def test_single_match_with_extractable_content_applies_directly(self, conn):
        item = create_item(conn, "meeting at 3pm", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "change the meeting todo to next week")
        assert result.action == "todo_updated", (
            f"single match with extractable content should apply directly, got {result.action}"
        )
        assert result.data["content"] == "next week", (
            "should contain the extracted new content"
        )

    def test_no_extractable_content_uses_llm(self, conn):
        item = create_item(conn, "meeting at 3pm", type="todo", status="pending")
        create_item(conn, "meeting notes", type="todo", status="pending")
        response = json.dumps({"target_id": item.id, "new_content": "meeting at 5pm"})
        llm = FakeLLM(response=response)
        svc = ServiceContext(conn=conn, llm=llm)

        result = dispatch(svc, "change the meeting todo to 5pm")
        assert result.action == "todo_updated", (
            "multi-match update should resolve via LLM"
        )
        assert len(llm.call_log) > 0, "should have called LLM for ambiguous case"

    def test_updated_item_persisted(self, conn):
        item = create_item(conn, "meeting at 3pm", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        dispatch(svc, "change the meeting todo to next week")
        refreshed = get_item(conn, item.id)
        assert refreshed.content == "next week", (
            "updated content should be persisted in DB"
        )

    def test_no_match_returns_not_found(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "change nonexistent todo to something")
        assert result.action == "todo_not_found", (
            "should return not_found when no todos match"
        )

    def test_without_llm_single_match_with_new_content_applies(self, conn):
        item = create_item(conn, "meeting at 3pm", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "change the meeting todo to next week")
        assert result.action == "todo_updated", (
            "single match with extractable new_content should apply directly without LLM"
        )
        assert result.data["content"] == "next week", (
            "should use the extracted new_content"
        )

    def test_multiple_matches_without_llm_returns_ambiguous(self, conn):
        create_item(conn, "meeting with alice", type="todo", status="pending")
        create_item(conn, "meeting with bob", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "change the meeting todo to next week")
        assert result.action == "todo_ambiguous", (
            "multiple matches without LLM should return ambiguous"
        )
        assert "matches" in result.data, "ambiguous should include matches"
        assert len(result.data["matches"]) == 2, "should list both candidates"

    def test_multiple_matches_with_llm_resolves(self, conn):
        item_a = create_item(conn, "meeting with alice", type="todo", status="pending")
        item_b = create_item(conn, "meeting with bob", type="todo", status="pending")
        response = json.dumps({"target_id": item_b.id, "new_content": "meeting with bob next week"})
        llm = FakeLLM(response=response)
        svc = ServiceContext(conn=conn, llm=llm)

        result = dispatch(svc, "change the bob meeting to next week")
        assert result.action == "todo_updated", (
            "LLM should resolve ambiguous update"
        )
        assert result.data["id"] == item_b.id, "should update the correct item"

    def test_llm_fails_returns_ambiguous(self, conn):
        create_item(conn, "meeting at 3pm", type="todo", status="pending")
        create_item(conn, "meeting tomorrow", type="todo", status="pending")

        class BrokenLLM:
            def generate(self, prompt, *, system=None):
                raise RuntimeError("model crashed")
            def is_available(self):
                return True

        svc = ServiceContext(conn=conn, llm=BrokenLLM())
        result = dispatch(svc, "change the meeting todo to next week")
        assert result.action == "todo_ambiguous", (
            "broken LLM should degrade to ambiguous with candidate list"
        )

    def test_llm_unavailable_single_match_still_applies(self, conn):
        item = create_item(conn, "meeting at 3pm", type="todo", status="pending")
        llm = FakeLLM(available=False)
        svc = ServiceContext(conn=conn, llm=llm)

        result = dispatch(svc, "change the meeting todo to next week")
        assert result.action == "todo_updated", (
            "single match with extractable content applies even with unavailable LLM"
        )

    def test_uses_all_pending_when_no_substring_match(self, conn):
        item = create_item(conn, "appointment on friday", type="todo", status="pending")
        response = json.dumps({"target_id": item.id, "new_content": "appointment on monday"})
        llm = FakeLLM(response=response)
        svc = ServiceContext(conn=conn, llm=llm)

        result = dispatch(svc, "change my friday thing to monday")
        assert result.action == "todo_updated", (
            "should resolve via LLM when no substring match but LLM available"
        )
        assert result.data["content"] == "appointment on monday", (
            "should apply LLM-determined new content"
        )


class TestFindTodosSemantic:
    def test_finds_semantically_similar(self):
        class SemanticEmbedder:
            def __init__(self):
                self._vectors = {
                    "buy milk": np.array([1.0, 0.0, 0.0]),
                    "call dentist": np.array([0.0, 1.0, 0.0]),
                    "I purchased the milk": np.array([0.95, 0.05, 0.0]),
                }

            @property
            def dimension(self):
                return 3

            @property
            def model_name(self):
                return "test"

            def embed_one(self, text):
                if text in self._vectors:
                    return self._vectors[text]
                return np.array([0.0, 0.0, 1.0])

            def embed_many(self, texts):
                return np.array([self.embed_one(t) for t in texts])

        embedder = SemanticEmbedder()
        todos = [
            _make_item(id="a", content="buy milk"),
            _make_item(id="b", content="call dentist"),
        ]

        matches = find_todos_semantic("I purchased the milk", todos, embedder, threshold=0.5)
        assert len(matches) == 1, "should find one semantic match"
        assert matches[0].id == "a", "should match 'buy milk'"

    def test_returns_empty_below_threshold(self):
        embedder = FakeEmbeddingService()
        todos = [_make_item(id="a", content="buy milk")]

        matches = find_todos_semantic("completely unrelated xyz", todos, embedder, threshold=0.99)
        assert len(matches) == 0, "should return empty when below threshold"

    def test_returns_empty_with_no_todos(self):
        embedder = FakeEmbeddingService()
        matches = find_todos_semantic("anything", [], embedder)
        assert matches == [], "should return empty for empty todo list"


class TestEmbeddingBasedCompletion:
    def test_purchased_milk_completes_buy_milk(self, conn, mocker):
        class MilkEmbedder:
            @property
            def dimension(self):
                return 3

            @property
            def model_name(self):
                return "test"

            def embed_one(self, text):
                if "milk" in text.lower() or "purchased" in text.lower():
                    return np.array([0.9, 0.1, 0.0])
                return np.array([0.0, 0.0, 1.0])

            def embed_many(self, texts):
                return np.array([self.embed_one(t) for t in texts])

        from memask.router.intents import Intent, Confidence, QueryContext, RoutingResult

        mocker.patch(
            "memask.router.dispatcher.route",
            return_value=RoutingResult(
                intent=Intent.TODO_COMPLETE,
                confidence=Confidence.MEDIUM,
                query_context=QueryContext(raw_query="I purchased the milk"),
                raw_input="I purchased the milk",
                source="embedding",
            ),
        )

        item = create_item(conn, "buy milk", type="todo", status="pending")
        svc = ServiceContext(conn=conn, embedder=MilkEmbedder())

        result = dispatch(svc, "I purchased the milk")
        assert result.action == "todo_completed", (
            f"'I purchased the milk' should complete 'buy milk' via embedding, got {result.action}"
        )
        assert result.data["id"] == item.id, "should complete the correct item"

    def test_embedding_completion_falls_back_to_substring(self, conn):
        create_item(conn, "buying groceries", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "finished buying groceries")
        assert result.action == "todo_completed", (
            "substring match should still work without embedder"
        )

    def test_no_match_at_all(self, conn, mocker):
        from memask.router.intents import Intent, Confidence, QueryContext, RoutingResult

        mocker.patch(
            "memask.router.dispatcher.route",
            return_value=RoutingResult(
                intent=Intent.TODO_COMPLETE,
                confidence=Confidence.MEDIUM,
                query_context=QueryContext(raw_query="I purchased the milk"),
                raw_input="I purchased the milk",
                source="embedding",
            ),
        )

        create_item(conn, "buy milk", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "I purchased the milk")
        assert result.action == "todo_not_found", (
            "without embedder and no substring match, should report not found"
        )


class TestDegradationPaths:
    def test_update_without_llm_single_candidate_with_content(self, conn):
        create_item(conn, "meeting at 3pm", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "change the meeting to next week")
        assert result.action == "todo_updated", (
            "single match with extractable new_content applies directly without LLM"
        )

    def test_update_without_llm_multi_match_degrades(self, conn):
        create_item(conn, "meeting at 3pm", type="todo", status="pending")
        create_item(conn, "meeting notes review", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "change the meeting to next week")
        assert result.action == "todo_ambiguous", (
            "multi-match update without LLM should return ambiguous candidates"
        )

    def test_embedding_completion_without_embedder(self, conn, mocker):
        from memask.router.intents import Intent, Confidence, QueryContext, RoutingResult

        mocker.patch(
            "memask.router.dispatcher.route",
            return_value=RoutingResult(
                intent=Intent.TODO_COMPLETE,
                confidence=Confidence.MEDIUM,
                query_context=QueryContext(raw_query="I purchased the milk"),
                raw_input="I purchased the milk",
                source="embedding",
            ),
        )

        create_item(conn, "buy milk", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "I purchased the milk")
        assert result.action == "todo_not_found", (
            "without embedder, no substring match should yield not_found"
        )

    def test_existing_substring_completion_still_works(self, conn):
        create_item(conn, "buy milk", type="todo", status="pending")
        svc = ServiceContext(conn=conn)

        result = dispatch(svc, "finished buy milk")
        assert result.action == "todo_completed", (
            "substring-based completion must still work"
        )


def _make_item(id="item-1", content="buy milk"):
    return Item(
        id=id,
        type="todo",
        content=content,
        title=None,
        status="pending",
        priority=None,
        due_date=None,
        category=None,
        source="manual",
        tags=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        deleted_at=None,
        metadata=None,
    )
