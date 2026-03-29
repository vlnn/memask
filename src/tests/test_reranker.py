import pytest

from memask.models.item import Item
from memask.rag.reranker import rerank, CrossEncoderReranker
from memask.search.keyword import SearchResult
from tests.helpers import FakeReranker


def _item(id="item-1", content="some content", **kwargs):
    defaults = dict(
        id=id, type="note", content=content, title=None,
        status=None, priority=None, due_date=None, category=None,
        source="manual", tags=None, created_at="2025-06-15T10:00:00",
        updated_at="2025-06-15T10:00:00", deleted_at=None, metadata=None,
    )
    defaults.update(kwargs)
    return Item(**defaults)


def _result(id="item-1", content="some content", score=0.5, source="hybrid"):
    return SearchResult(item=_item(id=id, content=content), score=score, source=source)


class TestRerankPassthrough:
    def test_none_reranker_returns_unchanged(self):
        results = [_result(id="a", score=0.9), _result(id="b", score=0.1)]
        reranked = rerank("query", results, reranker=None)
        assert [r.item.id for r in reranked] == ["a", "b"], (
            "None reranker should pass through unchanged"
        )

    def test_none_reranker_preserves_scores(self):
        results = [_result(id="a", score=0.9)]
        reranked = rerank("query", results, reranker=None)
        assert reranked[0].score == 0.9, "scores should be preserved with None reranker"

    def test_empty_results(self):
        reranked = rerank("query", [], reranker=None)
        assert reranked == [], "empty input should return empty output"


class TestRerankWithFake:
    def test_reorders_by_relevance(self):
        results = [
            _result(id="irrelevant", content="cats and dogs", score=0.9),
            _result(id="relevant", content="deployment pipeline", score=0.1),
        ]
        reranker = FakeReranker()
        reranked = rerank("deployment", results, reranker=reranker)
        assert reranked[0].item.id == "relevant", (
            "reranker should promote content matching the query"
        )

    def test_replaces_scores_with_reranker_scores(self):
        results = [_result(id="a", content="deployment notes", score=0.1)]
        reranker = FakeReranker()
        reranked = rerank("deployment", results, reranker=reranker)
        assert reranked[0].score != 0.1, "reranker should replace original scores"

    def test_preserves_all_items(self):
        results = [
            _result(id="a", content="alpha"),
            _result(id="b", content="beta"),
            _result(id="c", content="gamma"),
        ]
        reranker = FakeReranker()
        reranked = rerank("query", results, reranker=reranker)
        ids = {r.item.id for r in reranked}
        assert ids == {"a", "b", "c"}, "reranker should not drop any items"

    def test_marks_source_as_reranked(self):
        results = [_result(id="a", content="hello", source="hybrid")]
        reranker = FakeReranker()
        reranked = rerank("hello", results, reranker=reranker)
        assert reranked[0].source == "reranked", (
            "reranked results should be marked with source=reranked"
        )

    def test_sorted_descending_by_score(self):
        results = [
            _result(id="a", content="x"),
            _result(id="b", content="y"),
            _result(id="c", content="z"),
        ]
        reranker = FakeReranker()
        reranked = rerank("query", results, reranker=reranker)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True), (
            "results should be sorted by score descending"
        )


class TestRerankTopN:
    def test_limits_output_to_top_n(self):
        results = [_result(id=f"item-{i}", content=f"note {i}") for i in range(10)]
        reranker = FakeReranker()
        reranked = rerank("query", results, reranker=reranker, top_n=3)
        assert len(reranked) == 3, "should limit to top_n results"

    def test_top_n_keeps_highest_scored(self):
        results = [
            _result(id="match", content="deployment details"),
            _result(id="no-match-1", content="unrelated a"),
            _result(id="no-match-2", content="unrelated b"),
        ]
        reranker = FakeReranker()
        reranked = rerank("deployment", results, reranker=reranker, top_n=1)
        assert reranked[0].item.id == "match", (
            "top_n=1 should keep the highest scored result"
        )

    @pytest.mark.parametrize("top_n", [1, 5, 20])
    def test_top_n_with_fewer_results(self, top_n):
        results = [_result(id="only", content="single")]
        reranker = FakeReranker()
        reranked = rerank("query", results, reranker=reranker, top_n=top_n)
        assert len(reranked) == 1, "should not exceed available results"


class TestFakeReranker:
    def test_scores_based_on_word_overlap(self):
        reranker = FakeReranker()
        pairs = [("deploy to prod", "how to deploy")]
        scores = reranker.score_pairs(pairs)
        assert len(scores) == 1, "should return one score per pair"
        assert scores[0] > 0.0, "overlapping words should produce positive score"

    def test_no_overlap_scores_low(self):
        reranker = FakeReranker()
        pairs = [("cats are cute", "deploy to prod")]
        scores = reranker.score_pairs(pairs)
        assert scores[0] < 0.5, "no word overlap should score low"

    def test_full_overlap_scores_high(self):
        reranker = FakeReranker()
        pairs = [("deploy notes", "deploy notes")]
        scores = reranker.score_pairs(pairs)
        assert scores[0] >= 0.9, "identical text should score very high"

    def test_multiple_pairs(self):
        reranker = FakeReranker()
        pairs = [("a b c", "a b"), ("x y z", "a b")]
        scores = reranker.score_pairs(pairs)
        assert len(scores) == 2, "should return score for each pair"
        assert scores[0] > scores[1], "higher overlap should score higher"


class TestCrossEncoderRerankerInterface:
    def test_has_score_pairs_method(self):
        assert hasattr(CrossEncoderReranker, "score_pairs"), (
            "CrossEncoderReranker should have score_pairs method"
        )
