import pytest

from memask.repository.items import create_item
from memask.suggest import suggest, score_command, COMMANDS


class TestScoreCommand:
    @pytest.mark.parametrize("query,command,expected_positive", [
        ("/t", "/todo", True),
        ("/todo", "/todo", True),
        ("/li", "/list", True),
        ("/d", "/done", True),
        ("xyz", "/todo", False),
    ])
    def test_prefix_match_scores(self, query, command, expected_positive):
        score = score_command(query, command)
        if expected_positive:
            assert score > 0, f"'{query}' should match '{command}'"
        else:
            assert score == 0.0, f"'{query}' should not match '{command}'"

    def test_exact_match_scores_highest(self):
        exact = score_command("/todo", "/todo")
        prefix = score_command("/to", "/todo")
        assert exact > prefix, "exact match should score higher than prefix match"

    def test_substring_match_scores_lower_than_prefix(self):
        prefix = score_command("/to", "/todo")
        substring = score_command("odo", "/todo")
        assert prefix > substring, "prefix match should score higher than substring"

    def test_case_insensitive(self):
        score = score_command("/TODO", "/todo")
        assert score > 0, "matching should be case-insensitive"


class TestSuggestCommandsOnly:
    def test_slash_prefix_returns_commands(self, conn):
        results = suggest(conn, "/")
        kinds = {r["kind"] for r in results}
        assert "command" in kinds, "/ prefix should return command suggestions"

    def test_slash_todo_matches_todo_commands(self, conn):
        results = suggest(conn, "/to")
        texts = [r["text"] for r in results]
        assert any("/todo" in t for t in texts), "/to should suggest /todo"

    def test_all_commands_present(self, conn):
        results = suggest(conn, "/", limit=20)
        suggested_commands = {r["text"] for r in results if r["kind"] == "command"}
        for cmd in COMMANDS:
            assert cmd["text"] in suggested_commands, (
                f"/ should suggest all commands including {cmd['text']}"
            )


class TestSuggestItemsOnly:
    def test_finds_items_by_content(self, conn):
        create_item(conn, "deploy to production")
        results = suggest(conn, "deploy")
        items = [r for r in results if r["kind"] == "item"]
        assert len(items) >= 1, "should find items matching query"

    def test_item_text_is_content(self, conn):
        create_item(conn, "buy groceries")
        results = suggest(conn, "buy")
        items = [r for r in results if r["kind"] == "item"]
        assert items[0]["text"] == "buy groceries", "item text should be the content"

    def test_excludes_deleted_items(self, conn):
        item = create_item(conn, "deleted deploy note")
        conn.execute(
            "UPDATE items SET deleted_at = '2025-01-01' WHERE id = ?",
            (item.id,),
        )
        conn.commit()
        results = suggest(conn, "deploy")
        items = [r for r in results if r["kind"] == "item"]
        assert len(items) == 0, "should not suggest soft-deleted items"

    def test_empty_query_returns_recent_items(self, conn):
        create_item(conn, "first note")
        create_item(conn, "second note")
        results = suggest(conn, "")
        items = [r for r in results if r["kind"] == "item"]
        assert len(items) >= 2, "empty query should return recent items"


class TestSuggestMixed:
    def test_respects_limit(self, conn):
        for i in range(15):
            create_item(conn, f"note number {i}")
        results = suggest(conn, "note", limit=10)
        assert len(results) <= 10, "should respect the limit parameter"

    def test_commands_and_items_mixed_by_score(self, conn):
        create_item(conn, "todo list for deployment")
        results = suggest(conn, "todo")
        kinds = {r["kind"] for r in results}
        assert "command" in kinds, "should include command matches"
        assert "item" in kinds, "should include item matches"

    def test_sorted_by_score_descending(self, conn):
        create_item(conn, "deploy to staging")
        create_item(conn, "deployment checklist")
        results = suggest(conn, "deploy")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            "results should be sorted by score descending"
        )

    def test_includes_item_type_and_id(self, conn):
        create_item(conn, "meeting notes", type="note")
        results = suggest(conn, "meeting")
        items = [r for r in results if r["kind"] == "item"]
        assert "id" in items[0], "item suggestions should include id"
        assert "item_type" in items[0], "item suggestions should include item_type"


class TestSuggestEdgeCases:
    def test_whitespace_only_returns_recent(self, conn):
        create_item(conn, "something")
        results = suggest(conn, "   ")
        assert len(results) >= 1, "whitespace query should return recent items"

    def test_special_characters_dont_crash(self, conn):
        results = suggest(conn, "it's a %test\"")
        assert isinstance(results, list), "should handle special characters gracefully"
