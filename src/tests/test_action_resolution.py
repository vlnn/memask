import json

import pytest

from memask.router.action_resolution import (
    ActionPlan,
    RESOLVE_UPDATE_SYSTEM,
    RESOLVE_UPDATE_PROMPT,
    resolve_update,
    _format_candidates,
    _parse_action_plan,
)
from memask.models.item import Item
from tests.helpers import FakeLLM


def _make_item(id="item-1", content="buy milk"):
    return Item(
        id=id,
        type="todo",
        content=content,
        title=None,
        status="pending",
        priority=None,
        due_date=None,
        category=None,
        source="manual",
        tags=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
        deleted_at=None,
        metadata=None,
    )


class TestActionPlan:
    def test_is_frozen_dataclass(self):
        plan = ActionPlan(target_id="abc", new_content="new text")
        assert plan.target_id == "abc", "should store target_id"
        assert plan.new_content == "new text", "should store new_content"

    def test_immutable(self):
        plan = ActionPlan(target_id="abc", new_content="text")
        with pytest.raises(AttributeError):
            plan.target_id = "changed"


class TestFormatCandidates:
    def test_single_candidate(self):
        items = [_make_item(id="x1", content="buy milk")]
        result = _format_candidates(items)
        assert 'id=x1' in result, "should include item id"
        assert 'buy milk' in result, "should include item content"

    def test_multiple_candidates(self):
        items = [
            _make_item(id="x1", content="buy milk"),
            _make_item(id="x2", content="call dentist"),
        ]
        result = _format_candidates(items)
        assert 'id=x1' in result, "should include first item"
        assert 'id=x2' in result, "should include second item"


class TestParseActionPlan:
    def test_valid_json(self):
        candidates = [_make_item(id="abc")]
        raw = '{"target_id": "abc", "new_content": "buy oat milk"}'
        plan = _parse_action_plan(raw, candidates)
        assert plan is not None, "should parse valid JSON"
        assert plan.target_id == "abc", "should extract target_id"
        assert plan.new_content == "buy oat milk", "should extract new_content"

    def test_strips_markdown_fences(self):
        candidates = [_make_item(id="abc")]
        raw = '```json\n{"target_id": "abc", "new_content": "updated"}\n```'
        plan = _parse_action_plan(raw, candidates)
        assert plan is not None, "should handle markdown-wrapped JSON"

    def test_null_target_returns_none(self):
        candidates = [_make_item(id="abc")]
        raw = '{"target_id": null, "new_content": null}'
        plan = _parse_action_plan(raw, candidates)
        assert plan is None, "null target should return None"

    def test_invalid_json_returns_none(self):
        candidates = [_make_item(id="abc")]
        plan = _parse_action_plan("not json at all", candidates)
        assert plan is None, "garbage input should return None"

    def test_unknown_id_returns_none(self):
        candidates = [_make_item(id="abc")]
        raw = '{"target_id": "unknown-id", "new_content": "text"}'
        plan = _parse_action_plan(raw, candidates)
        assert plan is None, "unknown target_id should return None"

    def test_empty_new_content_returns_none(self):
        candidates = [_make_item(id="abc")]
        raw = '{"target_id": "abc", "new_content": ""}'
        plan = _parse_action_plan(raw, candidates)
        assert plan is None, "empty new_content should return None"


class TestResolveUpdate:
    def test_returns_plan_from_llm(self):
        item = _make_item(id="todo-1", content="meeting at 3pm")
        response = json.dumps({"target_id": "todo-1", "new_content": "meeting next week"})
        llm = FakeLLM(response=response)

        plan = resolve_update("change the meeting to next week", [item], llm)
        assert plan is not None, "should return a plan when LLM responds"
        assert plan.target_id == "todo-1", "should pick the correct target"
        assert plan.new_content == "meeting next week", "should extract new content"

    def test_passes_text_in_prompt(self):
        item = _make_item(id="todo-1")
        response = json.dumps({"target_id": "todo-1", "new_content": "x"})
        llm = FakeLLM(response=response)

        resolve_update("change milk to oat milk", [item], llm)
        assert "change milk to oat milk" in llm.call_log[0]["prompt"], (
            "should include user text in prompt"
        )

    def test_passes_system_message(self):
        item = _make_item(id="todo-1")
        response = json.dumps({"target_id": "todo-1", "new_content": "x"})
        llm = FakeLLM(response=response)

        resolve_update("change something", [item], llm)
        assert llm.call_log[0]["system"] == RESOLVE_UPDATE_SYSTEM, (
            "should use the resolve update system prompt"
        )

    def test_includes_candidates_in_prompt(self):
        items = [
            _make_item(id="a", content="buy milk"),
            _make_item(id="b", content="call dentist"),
        ]
        response = json.dumps({"target_id": "a", "new_content": "x"})
        llm = FakeLLM(response=response)

        resolve_update("update milk", items, llm)
        prompt = llm.call_log[0]["prompt"]
        assert "buy milk" in prompt, "should include first candidate content"
        assert "call dentist" in prompt, "should include second candidate content"

    def test_returns_none_on_empty_candidates(self):
        llm = FakeLLM(response="anything")
        plan = resolve_update("update something", [], llm)
        assert plan is None, "should return None with no candidates"

    def test_returns_none_on_llm_exception(self):
        class BrokenLLM:
            def generate(self, prompt, *, system=None):
                raise RuntimeError("model exploded")
            def is_available(self):
                return True

        item = _make_item(id="todo-1")
        plan = resolve_update("change something", [item], BrokenLLM())
        assert plan is None, "should return None on LLM failure"

    @pytest.mark.parametrize("bad_response", [
        "I'm not sure which one you mean",
        '{"target_id": null}',
        "",
        "```\nerror\n```",
    ])
    def test_returns_none_on_unparseable_response(self, bad_response):
        item = _make_item(id="todo-1")
        llm = FakeLLM(response=bad_response)
        plan = resolve_update("change something", [item], llm)
        assert plan is None, (
            f"should return None for unparseable response: {bad_response!r}"
        )

    def test_picks_correct_target_among_multiple(self):
        items = [
            _make_item(id="a", content="buy milk"),
            _make_item(id="b", content="meeting at 3pm"),
            _make_item(id="c", content="call dentist"),
        ]
        response = json.dumps({"target_id": "b", "new_content": "meeting next week"})
        llm = FakeLLM(response=response)

        plan = resolve_update("change the meeting to next week", items, llm)
        assert plan.target_id == "b", "should select the meeting todo"
        assert plan.new_content == "meeting next week", "should set new content"
