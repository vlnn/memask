import pytest

from memask.models.item import Item
from memask.rag.context import assemble_context, ContextBlock
from memask.search.keyword import SearchResult


def _item(
    id="item-1",
    content="some content",
    title=None,
    type="note",
    created_at="2025-06-15T10:00:00",
    **kwargs,
):
    defaults = dict(
        id=id,
        type=type,
        content=content,
        title=title,
        status=None,
        priority=None,
        due_date=None,
        category=None,
        source="manual",
        tags=None,
        created_at=created_at,
        updated_at=created_at,
        deleted_at=None,
        metadata=None,
    )
    defaults.update(kwargs)
    return Item(**defaults)


def _result(id="item-1", content="some content", score=0.9, **kwargs):
    return SearchResult(item=_item(id=id, content=content, **kwargs), score=score)


class TestAssembleContextBasic:
    def test_empty_results_returns_empty(self):
        ctx = assemble_context([], "any query")
        assert ctx.text == "", "empty results should produce empty context"
        assert ctx.sources == [], "empty results should produce no sources"

    def test_single_result_includes_content(self):
        results = [_result(content="deploy on friday")]
        ctx = assemble_context(results, "deployment")
        assert "deploy on friday" in ctx.text, "context should contain the item content"

    def test_single_result_includes_item_id(self):
        results = [_result(id="abc-123", content="some note")]
        ctx = assemble_context(results, "query")
        assert "abc-123" in ctx.text, "context should reference the item id"

    def test_multiple_results_all_included(self):
        results = [
            _result(id="a", content="first note"),
            _result(id="b", content="second note"),
            _result(id="c", content="third note"),
        ]
        ctx = assemble_context(results, "query")
        assert "first note" in ctx.text, "first result should be in context"
        assert "second note" in ctx.text, "second result should be in context"
        assert "third note" in ctx.text, "third result should be in context"

    def test_sources_track_item_ids(self):
        results = [
            _result(id="x1", content="alpha"),
            _result(id="x2", content="beta"),
        ]
        ctx = assemble_context(results, "query")
        assert ctx.sources == ["x1", "x2"], "sources should list item ids in order"


class TestAssembleContextMetadata:
    def test_includes_title_when_present(self):
        results = [_result(content="details here", title="Project Plan")]
        ctx = assemble_context(results, "query")
        assert "Project Plan" in ctx.text, "context should include item title"

    def test_includes_type(self):
        results = [_result(content="buy milk", type="todo")]
        ctx = assemble_context(results, "query")
        assert "todo" in ctx.text, "context should include item type"

    def test_includes_created_at(self):
        results = [_result(content="meeting notes", created_at="2025-06-15T10:00:00")]
        ctx = assemble_context(results, "query")
        assert "2025-06-15" in ctx.text, "context should include creation date"


class TestAssembleContextTruncation:
    def test_respects_max_tokens_estimate(self):
        long_content = "word " * 5000
        results = [_result(content=long_content)]
        ctx = assemble_context(results, "query", max_chars=500)
        assert len(ctx.text) <= 600, "context should be truncated near max_chars"

    def test_truncation_prefers_higher_scored_results(self):
        results = [
            _result(id="high", content="important " * 200, score=0.95),
            _result(id="low", content="less relevant " * 200, score=0.3),
        ]
        ctx = assemble_context(results, "query", max_chars=500)
        assert "important" in ctx.text, "higher scored result should survive truncation"

    def test_truncated_item_content_is_marked(self):
        long_content = "word " * 5000
        results = [_result(content=long_content)]
        ctx = assemble_context(results, "query", max_chars=500)
        assert "[truncated]" in ctx.text, "truncated content should be marked"


class TestAssembleContextTopN:
    def test_default_top_n_limits_results(self):
        results = [_result(id=f"item-{i}", content=f"note {i}") for i in range(20)]
        ctx = assemble_context(results, "query")
        assert len(ctx.sources) <= 10, "should limit to default top_n"

    @pytest.mark.parametrize("top_n", [1, 3, 5])
    def test_custom_top_n(self, top_n):
        results = [_result(id=f"item-{i}", content=f"note {i}") for i in range(20)]
        ctx = assemble_context(results, "query", top_n=top_n)
        assert len(ctx.sources) == top_n, f"should limit to top_n={top_n}"


class TestContextBlock:
    def test_dataclass_fields(self):
        block = ContextBlock(text="hello", sources=["a", "b"])
        assert block.text == "hello", "should store text"
        assert block.sources == ["a", "b"], "should store sources"

    def test_is_frozen(self):
        block = ContextBlock(text="hello", sources=[])
        with pytest.raises(AttributeError):
            block.text = "modified"
