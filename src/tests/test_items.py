import pytest

from memask.repository.items import (
    create_item,
    get_item,
    list_items,
    soft_delete_item,
    update_item,
)


class TestCreateItem:
    def test_creates_note_with_defaults(self, conn):
        item = create_item(conn, "hello world")
        assert item.content == "hello world", "should store the content"
        assert item.type == "note", "should default to note type"
        assert item.source == "manual", "should default to manual source"
        assert item.deleted_at is None, "should not be deleted"

    def test_creates_todo_with_status(self, conn):
        item = create_item(conn, "buy milk", type="todo", status="pending")
        assert item.type == "todo", "should set the type"
        assert item.status == "pending", "should set the status"

    def test_generates_unique_ids(self, conn):
        a = create_item(conn, "first")
        b = create_item(conn, "second")
        assert a.id != b.id, "each item should get a unique ULID"

    def test_sets_timestamps(self, conn):
        item = create_item(conn, "timestamped")
        assert item.created_at is not None, "should set created_at"
        assert item.updated_at is not None, "should set updated_at"

    @pytest.mark.parametrize("item_type", ["note", "todo", "url", "decision", "guide"])
    def test_accepts_valid_types(self, conn, item_type):
        item = create_item(conn, "test", type=item_type)
        assert item.type == item_type, f"should accept {item_type} as valid type"

    def test_rejects_invalid_type(self, conn):
        with pytest.raises(ValueError, match="Invalid type"):
            create_item(conn, "bad", type="bogus")

    def test_stores_optional_fields(self, conn):
        item = create_item(
            conn,
            "deploy plan",
            type="note",
            title="Deployment",
            category="work",
            tags="ops,deploy",
            priority=1,
        )
        assert item.title == "Deployment", "should store title"
        assert item.category == "work", "should store category"
        assert item.tags == "ops,deploy", "should store tags"
        assert item.priority == 1, "should store priority"

    def test_ignores_unknown_fields(self, conn):
        item = create_item(conn, "safe", bogus_field="ignored")
        assert item.content == "safe", "should create item ignoring unknown fields"


class TestGetItem:
    def test_returns_existing_item(self, conn):
        created = create_item(conn, "findable")
        found = get_item(conn, created.id)
        assert found is not None, "should find existing item"
        assert found.id == created.id, "should return the same item"

    def test_returns_none_for_missing(self, conn):
        assert get_item(conn, "nonexistent") is None, "should return None for missing id"


class TestUpdateItem:
    def test_updates_content(self, conn):
        item = create_item(conn, "original")
        updated = update_item(conn, item.id, content="changed")
        assert updated is not None, "updated result is never empty"
        assert updated.content == "changed", "should update content"

    def test_updates_multiple_fields(self, conn):
        item = create_item(conn, "multi", type="todo")
        updated = update_item(conn, item.id, status="done", priority=5)
        assert updated is not None, "updated result is never empty"
        assert updated.status == "done", "should update status"
        assert updated.priority == 5, "should update priority"

    def test_advances_updated_at(self, conn):
        item = create_item(conn, "timestamped")
        updated = update_item(conn, item.id, content="new")
        assert updated is not None, "updated result is never empty"
        assert updated.updated_at >= item.updated_at, "should advance updated_at"

    def test_returns_none_for_missing(self, conn):
        assert update_item(conn, "gone", content="x") is None, "should return None for missing id"

    def test_no_op_without_valid_fields(self, conn):
        item = create_item(conn, "unchanged")
        result = update_item(conn, item.id, bogus="ignored")
        assert result is not None, "result is never empty"
        assert result.content == "unchanged", "should not change when no valid fields given"

    def test_skips_deleted_item(self, conn):
        item = create_item(conn, "will delete")
        soft_delete_item(conn, item.id)
        result = update_item(conn, item.id, content="too late")
        assert result is not None, "result is never empty"
        assert result.content == "will delete", "should not update a soft-deleted item"


class TestSoftDelete:
    def test_marks_as_deleted(self, conn):
        item = create_item(conn, "ephemeral")
        assert soft_delete_item(conn, item.id) is True, "should return True on success"
        deleted = get_item(conn, item.id)
        assert deleted.deleted_at is not None, "should set deleted_at"

    def test_returns_false_for_missing(self, conn):
        assert soft_delete_item(conn, "nope") is False, "should return False for missing id"

    def test_double_delete_returns_false(self, conn):
        item = create_item(conn, "once")
        soft_delete_item(conn, item.id)
        assert soft_delete_item(conn, item.id) is False, "should return False on second delete"


class TestListItems:
    def test_returns_all_non_deleted(self, conn):
        create_item(conn, "a")
        create_item(conn, "b")
        deleted = create_item(conn, "c")
        soft_delete_item(conn, deleted.id)

        result = list_items(conn)
        assert len(result) == 2, "should return only non-deleted items"

    def test_includes_deleted_when_asked(self, conn):
        create_item(conn, "alive")
        dead = create_item(conn, "dead")
        soft_delete_item(conn, dead.id)

        result = list_items(conn, include_deleted=True)
        assert len(result) == 2, "should return all items including deleted"

    def test_filters_by_type(self, conn):
        create_item(conn, "note1", type="note")
        create_item(conn, "todo1", type="todo")
        create_item(conn, "todo2", type="todo")

        result = list_items(conn, type="todo")
        assert len(result) == 2, "should return only matching type"
        assert all(i.type == "todo" for i in result), "all results should be todos"

    def test_filters_by_status(self, conn):
        create_item(conn, "pending", type="todo", status="pending")
        create_item(conn, "done", type="todo", status="done")

        result = list_items(conn, status="pending")
        assert len(result) == 1, "should filter by status"
        assert result[0].status == "pending", "should return pending item"

    def test_filters_by_category(self, conn):
        create_item(conn, "work stuff", category="work")
        create_item(conn, "personal stuff", category="personal")

        result = list_items(conn, category="work")
        assert len(result) == 1, "should filter by category"

    def test_combined_filters(self, conn):
        create_item(conn, "work todo", type="todo", status="pending", category="work")
        create_item(conn, "work note", type="note", category="work")
        create_item(conn, "personal todo", type="todo", status="pending", category="personal")

        result = list_items(conn, type="todo", status="pending", category="work")
        assert len(result) == 1, "should apply all filters together"
        assert result[0].content == "work todo", "should return the matching item"

    def test_empty_result(self, conn):
        result = list_items(conn, type="guide")
        assert result == [], "should return empty list when nothing matches"

    def test_ordered_by_created_at_desc(self, conn):
        a = create_item(conn, "first")
        b = create_item(conn, "second")

        result = list_items(conn)
        assert result[0].id == b.id, "newest item should come first"
        assert result[1].id == a.id, "oldest item should come last"


class TestListItemsDateFilter:
    def test_date_from_excludes_older(self, conn):
        create_item(conn, "old item")
        result = list_items(conn, date_from="2099-01-01T00:00:00")
        assert len(result) == 0, "should exclude items before date_from"

    def test_date_to_excludes_newer(self, conn):
        create_item(conn, "new item")
        result = list_items(conn, date_to="2000-01-01T00:00:00")
        assert len(result) == 0, "should exclude items after date_to"

    def test_date_range_includes_matching(self, conn):
        item = create_item(conn, "today's note")
        ts = item.created_at
        result = list_items(conn, date_from=ts, date_to=ts)
        assert len(result) == 1, "should include item within date range"
        assert result[0].id == item.id, "should return the matching item"

    def test_date_filters_combine_with_type(self, conn):
        create_item(conn, "a note", type="note")
        create_item(conn, "a todo", type="todo", status="pending")
        result = list_items(conn, type="note", date_from="2000-01-01T00:00:00")
        assert all(i.type == "note" for i in result), (
            "date filter should combine with type filter"
        )

    def test_none_date_filters_ignored(self, conn):
        create_item(conn, "always visible")
        result = list_items(conn, date_from=None, date_to=None)
        assert len(result) == 1, "None date filters should be ignored"
