import pytest

from memask.router.intents import Confidence, Intent
from memask.router.rules import classify_by_rules


class TestNLUpdateRouting:
    @pytest.mark.parametrize("text", [
        "change the meeting todo to next week",
        "update buy milk to buy oat milk",
        "reschedule the dentist to friday",
        "rename the deployment task to staging deploy",
        "change meeting to next thursday",
        "update the grocery todo to buy organic milk",
        "set the deadline to monday",
    ])
    def test_nl_update_routes_to_todo_update(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_UPDATE, (
            f"'{text}' should route to TODO_UPDATE, got {result.intent.value}"
        )
        assert result.confidence == Confidence.MEDIUM, (
            f"'{text}' should have MEDIUM confidence for NL update"
        )

    @pytest.mark.parametrize("text", [
        "change is the only constant",
        "update me on what happened",
        "set a reminder to buy milk",
    ])
    def test_non_update_phrases_dont_match(self, text):
        result = classify_by_rules(text)
        assert result.intent != Intent.TODO_UPDATE, (
            f"'{text}' should NOT route to TODO_UPDATE, got {result.intent.value}"
        )


class TestUpdateDoesntBreakExisting:
    @pytest.mark.parametrize("text,expected_intent", [
        ("/todo buy milk", Intent.TODO_CREATE),
        ("/done buy milk", Intent.TODO_COMPLETE),
        ("?what about deployment", Intent.SEARCH),
        ("!help", Intent.APP_COMMAND),
        ("remind me to buy milk", Intent.TODO_CREATE),
        ("the meeting went well", Intent.CAPTURE),
        ("finished buying groceries", Intent.TODO_COMPLETE),
        ("remove milk from my todos", Intent.TODO_DELETE),
        ("what's on my todo list?", Intent.TODO_LIST),
    ])
    def test_existing_rules_unbroken(self, text, expected_intent):
        result = classify_by_rules(text)
        assert result.intent == expected_intent, (
            f"'{text}' should still route to {expected_intent.value}, got {result.intent.value}"
        )
