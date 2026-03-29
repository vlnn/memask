import pytest

from memask.rag.context import ContextBlock
from memask.rag.prompts import build_answer_prompt, build_system_message, SYSTEM_MESSAGE


class TestSystemMessage:
    def test_is_nonempty_string(self):
        assert len(SYSTEM_MESSAGE) > 0, "system message should not be empty"

    def test_mentions_sources(self):
        assert "source" in SYSTEM_MESSAGE.lower(), (
            "system message should instruct about source references"
        )

    def test_build_system_message_returns_default(self):
        assert build_system_message() == SYSTEM_MESSAGE, (
            "build_system_message should return the default system message"
        )


class TestBuildAnswerPrompt:
    def test_includes_query(self):
        ctx = ContextBlock(text="some context", sources=["a"])
        prompt = build_answer_prompt("what about deployment?", ctx)
        assert "what about deployment?" in prompt, (
            "prompt should include the user query"
        )

    def test_includes_context(self):
        ctx = ContextBlock(text="deploy on friday at 3pm", sources=["a"])
        prompt = build_answer_prompt("deployment", ctx)
        assert "deploy on friday at 3pm" in prompt, (
            "prompt should include the context text"
        )

    def test_empty_context(self):
        ctx = ContextBlock(text="", sources=[])
        prompt = build_answer_prompt("anything", ctx)
        assert "anything" in prompt, "prompt should still include query"

    def test_context_and_query_are_separated(self):
        ctx = ContextBlock(text="context block", sources=["a"])
        prompt = build_answer_prompt("query text", ctx)
        ctx_pos = prompt.index("context block")
        query_pos = prompt.index("query text")
        assert ctx_pos != query_pos, "context and query should be at different positions"


class TestBuildAnswerPromptWithSession:
    def test_no_session_history(self):
        ctx = ContextBlock(text="ctx", sources=["a"])
        prompt = build_answer_prompt("query", ctx, session_history=[])
        assert "query" in prompt, "should work with empty session history"

    def test_includes_session_history(self):
        ctx = ContextBlock(text="ctx", sources=["a"])
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        prompt = build_answer_prompt("follow up", ctx, session_history=history)
        assert "previous question" in prompt, (
            "prompt should include prior user message"
        )
        assert "previous answer" in prompt, (
            "prompt should include prior assistant message"
        )

    def test_session_history_before_current_query(self):
        ctx = ContextBlock(text="ctx", sources=["a"])
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "response"},
        ]
        prompt = build_answer_prompt("current", ctx, session_history=history)
        earlier_pos = prompt.index("earlier")
        current_pos = prompt.index("current")
        assert earlier_pos < current_pos, (
            "session history should appear before current query"
        )

    @pytest.mark.parametrize("history_len", [1, 3, 5])
    def test_variable_history_lengths(self, history_len):
        ctx = ContextBlock(text="ctx", sources=["a"])
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"}
            for i in range(history_len)
        ]
        prompt = build_answer_prompt("query", ctx, session_history=history)
        for i in range(history_len):
            assert f"msg-{i}" in prompt, f"message {i} should be in prompt"


class TestPromptStructure:
    def test_prompt_has_clear_sections(self):
        ctx = ContextBlock(
            text="[1] (note, 2025-06-15) id: abc\nsome note",
            sources=["abc"],
        )
        prompt = build_answer_prompt("my question", ctx)
        assert "Context" in prompt or "context" in prompt or "CONTEXT" in prompt, (
            "prompt should have a labeled context section"
        )
        assert "Question" in prompt or "question" in prompt or "QUESTION" in prompt, (
            "prompt should have a labeled question section"
        )
