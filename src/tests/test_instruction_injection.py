import pytest

from memask.rag.prompts import build_system_message


class TestInstructionInjection:
    def test_no_instructions_produces_base_message(self):
        message = build_system_message()
        assert "User preferences" not in message, (
            "should not include preferences block when no instructions"
        )

    def test_no_instructions_with_empty_list(self):
        message = build_system_message(instructions=[])
        assert "User preferences" not in message, (
            "should not include preferences block for empty list"
        )

    def test_single_instruction_injected(self):
        message = build_system_message(
            instructions=["show todos with emojis"]
        )
        assert "User preferences" in message, (
            "should include preferences heading"
        )
        assert "show todos with emojis" in message, (
            "should include the instruction text"
        )

    def test_multiple_instructions_injected(self):
        message = build_system_message(
            instructions=["be brief", "use bullet points"]
        )
        assert "be brief" in message, (
            "should include first instruction"
        )
        assert "use bullet points" in message, (
            "should include second instruction"
        )

    def test_instructions_appear_before_base_content(self):
        message = build_system_message(
            instructions=["be brief"]
        )
        prefs_pos = message.index("User preferences")
        assert prefs_pos < len(message) // 2, (
            "instructions should appear near the top of the system message"
        )
