import pytest

from memask.repository.instructions import (
    create_instruction,
    list_active_instructions,
)
from memask.router.instruction_handler import handle_instruction_command


class TestInstructionSave:
    def test_saves_instruction(self, conn):
        result = handle_instruction_command(conn, "/instruction show todos with emojis")
        assert result["action"] == "instruction_saved", (
            "should return instruction_saved action"
        )
        active = list_active_instructions(conn)
        assert len(active) == 1, (
            "should persist the instruction"
        )
        assert active[0]["content"] == "show todos with emojis", (
            "should store the text after /instruction"
        )

    def test_strips_whitespace(self, conn):
        handle_instruction_command(conn, "/instruction   be brief   ")
        active = list_active_instructions(conn)
        assert active[0]["content"] == "be brief", (
            "should strip leading/trailing whitespace from content"
        )


class TestInstructionList:
    def test_bare_command_lists(self, conn):
        create_instruction(conn, "rule one")
        create_instruction(conn, "rule two")
        result = handle_instruction_command(conn, "/instruction")
        assert result["action"] == "instruction_listed", (
            "bare /instruction should list instructions"
        )
        assert len(result["data"]["instructions"]) == 2, (
            "should return all active instructions"
        )

    def test_empty_list(self, conn):
        result = handle_instruction_command(conn, "/instruction")
        assert result["action"] == "instruction_listed", (
            "should return listed even when empty"
        )
        assert result["data"]["instructions"] == [], (
            "should return empty list"
        )


class TestInstructionClear:
    def test_clears_all(self, conn):
        create_instruction(conn, "one")
        create_instruction(conn, "two")
        result = handle_instruction_command(conn, "/instruction clear")
        assert result["action"] == "instruction_cleared", (
            "should return instruction_cleared action"
        )
        assert result["data"]["count"] == 2, (
            "should report how many were cleared"
        )
        assert list_active_instructions(conn) == [], (
            "should have no active instructions after clear"
        )


class TestInstructionRemove:
    def test_removes_by_id(self, conn):
        inst = create_instruction(conn, "remove me")
        result = handle_instruction_command(
            conn, f"/instruction remove {inst['id']}"
        )
        assert result["action"] == "instruction_removed", (
            "should return instruction_removed action"
        )
        assert list_active_instructions(conn) == [], (
            "should deactivate the instruction"
        )

    def test_remove_unknown_id(self, conn):
        result = handle_instruction_command(conn, "/instruction remove fake-id")
        assert result["action"] == "instruction_not_found", (
            "should return not_found for unknown id"
        )
