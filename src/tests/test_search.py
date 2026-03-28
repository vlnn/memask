import pytest

from memask.repository.items import create_item, soft_delete_item
from memask.repository.search import SearchResult, search_items


@pytest.fixture
def populated_db(conn):
    create_item(
        conn, "deploy the app to production", title="deployment notes", category="devops",
    )
    create_item(
        conn, "buy groceries and cook dinner", type="todo", status="pending", category="personal",
    )
    create_item(conn, "review pull request for authentication module", category="work")
    create_item(
        conn, "python packaging with setuptools and wheels", title="packaging guide", type="guide",
    )
    create_item(conn, "meeting notes from sprint retrospective", category="work")
    return conn


class TestExactMatch:
    def test_finds_exact_word(self, populated_db):
        results = search_items(populated_db, "deploy")
        assert len(results) >= 1, "should find item containing 'deploy'"
        assert any(
            "deploy" in r.item.content for r in results
        ), "result should contain the search term"

    def test_finds_by_title(self, populated_db):
        results = search_items(populated_db, "deployment")
        assert len(results) >= 1, "should find item by title match"
        assert any(
            r.item.title == "deployment notes" for r in results
        ), "should match against title field"

    def test_finds_multiple_matches(self, populated_db):
        results = search_items(populated_db, "notes")
        assert len(results) >= 2, "should find multiple items containing 'notes'"

    @pytest.mark.parametrize("query,expected_fragment", [
        ("groceries", "buy groceries"),
        ("authentication", "authentication module"),
        ("retrospective", "sprint retrospective"),
    ])
    def test_finds_specific_terms(self, populated_db, query, expected_fragment):
        results = search_items(populated_db, query)
        assert len(results) >= 1, f"should find item for '{query}'"
        assert any(
            expected_fragment in r.item.content for r in results
        ), f"result should contain '{expected_fragment}'"


class TestMultiWordQuery:
    def test_matches_multiple_terms(self, populated_db):
        results = search_items(populated_db, "pull request")
        assert len(results) >= 1, "should find item with both terms"
        assert any(
            "pull request" in r.item.content for r in results
        ), "should match multi-word query"

    def test_matches_terms_in_different_fields(self, populated_db):
        results = search_items(populated_db, "deployment notes")
        assert len(results) >= 1, "should match terms across title and content"


class TestFilters:
    def test_filters_by_type(self, populated_db):
        results = search_items(populated_db, "notes", type="guide")
        contents = [r.item.content for r in results]
        assert all(
            "packaging" in c for c in contents
        ), "type filter should restrict to guide items only"

    def test_filters_by_status(self, populated_db):
        results = search_items(populated_db, "buy", status="pending")
        assert len(results) >= 1, "should find pending items"
        assert all(
            r.item.status == "pending" for r in results
        ), "all results should have pending status"

    def test_filters_by_category(self, populated_db):
        results = search_items(populated_db, "notes", category="work")
        assert len(results) >= 1, "should find work items"
        assert all(
            r.item.category == "work" for r in results
        ), "all results should be in work category"

    def test_filters_by_date_range(self, conn):
        create_item(conn, "early note")
        early = search_items(conn, "early")
        assert len(early) >= 1, "should find item before date filter"

        ts = early[0].item.created_at
        results = search_items(conn, "early", date_from=ts, date_to=ts)
        assert len(results) >= 1, "should find item within date range"

    def test_date_from_excludes_older(self, conn):
        create_item(conn, "old item")
        results = search_items(conn, "old", date_from="2099-01-01T00:00:00")
        assert len(results) == 0, "should exclude items before date_from"

    def test_date_to_excludes_newer(self, conn):
        create_item(conn, "new item")
        results = search_items(conn, "new", date_to="2000-01-01T00:00:00")
        assert len(results) == 0, "should exclude items after date_to"

    def test_combined_filters(self, populated_db):
        results = search_items(populated_db, "notes", type="note", category="work")
        assert len(results) >= 1, "should apply type and category filters together"
        for r in results:
            assert r.item.type == "note", "type filter should apply"
            assert r.item.category == "work", "category filter should apply"


class TestDeletedItems:
    def test_excludes_deleted_by_default(self, conn):
        item = create_item(conn, "soon to be deleted")
        soft_delete_item(conn, item.id)
        results = search_items(conn, "deleted")
        assert len(results) == 0, "should exclude soft-deleted items by default"

    def test_includes_deleted_when_asked(self, conn):
        item = create_item(conn, "soft deleted searchable")
        soft_delete_item(conn, item.id)
        results = search_items(conn, "searchable", include_deleted=True)
        assert len(results) >= 1, "should include deleted items when requested"


class TestEmptyResults:
    def test_no_matches(self, populated_db):
        results = search_items(populated_db, "xyznonexistent")
        assert results == [], "should return empty list for no matches"

    def test_empty_query(self, populated_db):
        results = search_items(populated_db, "")
        assert results == [], "should return empty for blank query"

    def test_whitespace_only_query(self, populated_db):
        results = search_items(populated_db, "   ")
        assert results == [], "should return empty for whitespace-only query"


class TestSpecialCharacters:
    @pytest.mark.parametrize("content,query", [
        ("error: connection refused", "connection"),
        ("config.yaml settings", "config"),
        ("user@example.com contact", "user"),
        ("price is $100", "price"),
        ("100% complete", "complete"),
        ("C++ programming", "programming"),
        ("key=value pairs", "key"),
    ])
    def test_handles_special_chars_in_content(self, conn, content, query):
        create_item(conn, content)
        results = search_items(conn, query)
        assert len(results) >= 1, f"should find item with special chars: '{content}'"

    @pytest.mark.parametrize("query", [
        "hello*world",
        "test()",
        '"quoted"',
        "back\\slash",
        "semi;colon",
        "angle<bracket>",
    ])
    def test_handles_special_chars_in_query(self, conn, query):
        create_item(conn, "some test content with hello world")
        results = search_items(conn, query)
        assert isinstance(results, list), f"should not crash on query: '{query}'"


class TestResultStructure:
    def test_returns_search_results(self, populated_db):
        results = search_items(populated_db, "deploy")
        assert len(results) >= 1, "should return results"
        assert isinstance(results[0], SearchResult), "should return SearchResult instances"

    def test_result_has_item_and_rank(self, populated_db):
        results = search_items(populated_db, "deploy")
        result = results[0]
        assert result.item is not None, "result should have item"
        assert isinstance(result.rank, float), "result should have numeric rank"

    def test_results_are_ranked(self, conn):
        create_item(conn, "python is great")
        create_item(conn, "python python python everywhere", title="python guide")
        results = search_items(conn, "python")
        assert len(results) >= 2, "should return multiple results"
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks), "results should be sorted by rank"


class TestLimit:
    def test_respects_limit(self, conn):
        for i in range(10):
            create_item(conn, f"searchable item number {i}")
        results = search_items(conn, "searchable", limit=3)
        assert len(results) == 3, "should return at most limit results"

    def test_returns_fewer_than_limit_when_not_enough(self, conn):
        create_item(conn, "only one searchable")
        results = search_items(conn, "searchable", limit=10)
        assert len(results) == 1, "should return all matches even if fewer than limit"


class TestFtsSyncWithItems:
    def test_new_items_are_searchable(self, conn):
        create_item(conn, "freshly created searchable content")
        results = search_items(conn, "freshly")
        assert len(results) == 1, "newly created items should be immediately searchable"

    def test_updated_items_reflect_changes(self, conn):
        from memask.repository.items import update_item

        item = create_item(conn, "original text")
        update_item(conn, item.id, content="modified text unique")
        old_results = search_items(conn, "original")
        new_results = search_items(conn, "modified")
        assert len(old_results) == 0, "old content should not be findable"
        assert len(new_results) == 1, "new content should be findable"

    def test_deleted_items_still_in_fts_but_filtered(self, conn):
        item = create_item(conn, "deletable content")
        soft_delete_item(conn, item.id)
        default_results = search_items(conn, "deletable")
        assert len(default_results) == 0, "deleted items hidden by default"
        all_results = search_items(conn, "deletable", include_deleted=True)
        assert len(all_results) == 1, "deleted items findable with include_deleted"
