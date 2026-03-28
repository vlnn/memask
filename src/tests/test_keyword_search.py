import pytest

from memask.repository.items import create_item
from memask.search.keyword import SearchResult, keyword_search, _build_fts_query


class TestBuildFtsQuery:
    @pytest.mark.parametrize("input_text,expected", [
        ("hello world", '"hello" "world"'),
        ("  spaced  ", '"spaced"'),
        ("single", '"single"'),
        ('with"quote', '"with""quote"'),
    ])
    def test_tokenizes_and_quotes(self, input_text, expected):
        assert _build_fts_query(input_text) == expected, "should tokenize and quote each word"


class TestKeywordSearch:
    def test_finds_exact_match(self, conn):
        create_item(conn, "deploy to production")
        results = keyword_search(conn, "deploy")
        assert len(results) == 1, "should find the item with exact word match"
        assert results[0].item.content == "deploy to production", "should return the matching item"
        assert results[0].source == "keyword", "should tag source as keyword"

    def test_finds_partial_content_match(self, conn):
        create_item(conn, "kubernetes cluster setup guide")
        results = keyword_search(conn, "kubernetes")
        assert len(results) == 1, "should match on single keyword"

    def test_matches_title(self, conn):
        create_item(conn, "some content", title="deployment checklist")
        results = keyword_search(conn, "checklist")
        assert len(results) == 1, "should search in title field"

    def test_matches_tags(self, conn):
        create_item(conn, "some content", tags="devops,infrastructure")
        results = keyword_search(conn, "devops")
        assert len(results) == 1, "should search in tags field"

    def test_returns_empty_for_no_match(self, conn):
        create_item(conn, "apples and oranges")
        results = keyword_search(conn, "kubernetes")
        assert results == [], "should return empty list when nothing matches"

    def test_returns_empty_for_blank_query(self, conn):
        create_item(conn, "some content")
        results = keyword_search(conn, "   ")
        assert results == [], "should return empty for blank query"

    def test_excludes_soft_deleted(self, conn):
        item = create_item(conn, "deleted note about deploy")
        conn.execute(
            "UPDATE items SET deleted_at = '2025-01-01' WHERE id = ?", (item.id,)
        )
        conn.commit()
        results = keyword_search(conn, "deploy")
        assert results == [], "should not return soft-deleted items"

    def test_filters_by_type(self, conn):
        create_item(conn, "deploy note", type="note")
        create_item(conn, "deploy todo", type="todo")
        results = keyword_search(conn, "deploy", type="todo")
        assert len(results) == 1, "should filter by type"
        assert results[0].item.type == "todo", "should return only matching type"

    def test_filters_by_category(self, conn):
        create_item(conn, "deploy work", category="work")
        create_item(conn, "deploy personal", category="personal")
        results = keyword_search(conn, "deploy", category="work")
        assert len(results) == 1, "should filter by category"

    def test_filters_by_status(self, conn):
        create_item(conn, "deploy pending", type="todo", status="pending")
        create_item(conn, "deploy done", type="todo", status="done")
        results = keyword_search(conn, "deploy", status="pending")
        assert len(results) == 1, "should filter by status"

    def test_respects_limit(self, conn):
        for i in range(10):
            create_item(conn, f"deploy version {i}")
        results = keyword_search(conn, "deploy", limit=3)
        assert len(results) == 3, "should respect limit parameter"

    def test_results_have_positive_scores(self, conn):
        create_item(conn, "deploy to production")
        results = keyword_search(conn, "deploy")
        assert results[0].score > 0, "BM25 score should be positive"

    def test_multi_word_query(self, conn):
        create_item(conn, "deploy to production environment")
        create_item(conn, "only about production")
        results = keyword_search(conn, "deploy production")
        assert len(results) >= 1, "should match multi-word queries"

    def test_special_characters_in_query(self, conn):
        create_item(conn, "c++ programming guide")
        results = keyword_search(conn, "c++")
        assert isinstance(results, list), "should handle special characters without crashing"
