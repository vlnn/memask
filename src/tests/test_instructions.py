import pytest

from memask.repository.instructions import (
    create_instruction,
    deactivate_instruction,
    deactivate_all_instructions,
    list_active_instructions,
)


class TestCreateInstruction:
    def test_stores_instruction(self, conn):
        instruction = create_instruction(conn, "show todos with emojis")
        assert instruction["content"] == "show todos with emojis", (
            "create_instruction should store content"
        )
        assert instruction["active"] == 1, (
            "new instruction should be active"
        )
        assert instruction["id"] is not None, (
            "new instruction should have an id"
        )

    def test_stores_timestamps(self, conn):
        instruction = create_instruction(conn, "be brief")
        assert instruction["created_at"] is not None, (
            "new instruction should have created_at"
        )
        assert instruction["updated_at"] is not None, (
            "new instruction should have updated_at"
        )

    def test_multiple_instructions(self, conn):
        create_instruction(conn, "first")
        create_instruction(conn, "second")
        active = list_active_instructions(conn)
        assert len(active) == 2, (
            "should store multiple instructions"
        )


class TestListActiveInstructions:
    def test_empty_list(self, conn):
        active = list_active_instructions(conn)
        assert active == [], (
            "should return empty list when no instructions exist"
        )

    def test_returns_only_active(self, conn):
        inst = create_instruction(conn, "will deactivate")
        create_instruction(conn, "stays active")
        deactivate_instruction(conn, inst["id"])
        active = list_active_instructions(conn)
        assert len(active) == 1, (
            "should only return active instructions"
        )
        assert active[0]["content"] == "stays active", (
            "should return the non-deactivated instruction"
        )

    def test_ordered_by_created_at(self, conn):
        create_instruction(conn, "first")
        create_instruction(conn, "second")
        active = list_active_instructions(conn)
        contents = [i["content"] for i in active]
        assert contents == ["first", "second"], (
            "should return instructions in creation order"
        )


class TestDeactivateInstruction:
    def test_deactivates_by_id(self, conn):
        inst = create_instruction(conn, "to remove")
        result = deactivate_instruction(conn, inst["id"])
        assert result is True, (
            "deactivate should return True on success"
        )
        active = list_active_instructions(conn)
        assert len(active) == 0, (
            "deactivated instruction should not appear in active list"
        )

    def test_returns_false_for_missing_id(self, conn):
        result = deactivate_instruction(conn, "nonexistent-id")
        assert result is False, (
            "deactivate should return False for unknown id"
        )

    def test_does_not_affect_other_instructions(self, conn):
        inst1 = create_instruction(conn, "first")
        create_instruction(conn, "second")
        deactivate_instruction(conn, inst1["id"])
        active = list_active_instructions(conn)
        assert len(active) == 1, (
            "deactivating one should not affect others"
        )


class TestDeactivateAllInstructions:
    def test_deactivates_all(self, conn):
        create_instruction(conn, "one")
        create_instruction(conn, "two")
        create_instruction(conn, "three")
        count = deactivate_all_instructions(conn)
        assert count == 3, (
            "should return number of deactivated instructions"
        )
        active = list_active_instructions(conn)
        assert len(active) == 0, (
            "no instructions should be active after clear"
        )

    def test_returns_zero_when_empty(self, conn):
        count = deactivate_all_instructions(conn)
        assert count == 0, (
            "should return 0 when no active instructions"
        )
