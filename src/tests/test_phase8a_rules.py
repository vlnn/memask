import pytest

from memask.router.intents import Confidence, Intent
from memask.router.rules import classify_by_rules


class TestNLCompletionRouting:
    @pytest.mark.parametrize("text", [
        "my task about jumping 12 times is complete",
        "finished buying groceries",
        "I already bought the milk",
        "the deployment task is done",
        "mark laundry as done",
        "I completed the code review",
        "buying groceries is done",
    ])
    def test_nl_completion_routes_to_todo_complete(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_COMPLETE, (
            f"'{text}' should route to TODO_COMPLETE, got {result.intent.value}"
        )
        assert result.confidence == Confidence.MEDIUM, (
            f"'{text}' should have MEDIUM confidence for NL completion"
        )

    @pytest.mark.parametrize("text", [
        "/done buy milk",
        "/todo done laundry",
    ])
    def test_prefix_completion_still_routes_high_confidence(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_COMPLETE, (
            f"'{text}' should still route to TODO_COMPLETE"
        )
        assert result.confidence == Confidence.HIGH, (
            f"'{text}' with prefix should remain HIGH confidence"
        )


class TestNLDeletionRouting:
    @pytest.mark.parametrize("text", [
        "remove buy milk from my todos",
        "delete the todo about groceries",
        "cancel the dentist todo",
        "remove the deployment task from my list",
        "delete my todo about the meeting",
    ])
    def test_nl_deletion_routes_to_todo_delete(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_DELETE, (
            f"'{text}' should route to TODO_DELETE, got {result.intent.value}"
        )
        assert result.confidence == Confidence.MEDIUM, (
            f"'{text}' should have MEDIUM confidence for NL deletion"
        )


class TestNLTodoListRouting:
    @pytest.mark.parametrize("text", [
        "what's on my todo list?",
        "show my todos",
        "any pending tasks?",
        "what do I need to do?",
        "what are my pending todos?",
        "show me my tasks",
    ])
    def test_nl_listing_routes_to_todo_list(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_LIST, (
            f"'{text}' should route to TODO_LIST, got {result.intent.value}"
        )
        assert result.confidence == Confidence.MEDIUM, (
            f"'{text}' should have MEDIUM confidence for NL listing"
        )

    @pytest.mark.parametrize("text", [
        "/todo list",
        "/todo ls",
    ])
    def test_prefix_listing_still_high_confidence(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_LIST, (
            f"'{text}' should still route to TODO_LIST"
        )
        assert result.confidence == Confidence.HIGH, (
            f"'{text}' with prefix should remain HIGH confidence"
        )


class TestNLHelpRouting:
    @pytest.mark.parametrize("text", [
        "what can you do",
        "help me",
        "how does this work",
        "what can you do?",
    ])
    def test_nl_help_routes_to_app_command(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.APP_COMMAND, (
            f"'{text}' should route to APP_COMMAND, got {result.intent.value}"
        )
        assert result.confidence == Confidence.MEDIUM, (
            f"'{text}' should have MEDIUM confidence for NL help"
        )


class TestNewIntentsExist:
    def test_todo_delete_intent_exists(self):
        assert hasattr(Intent, "TODO_DELETE"), "Intent enum should have TODO_DELETE"
        assert Intent.TODO_DELETE.value == "todo_delete", (
            "TODO_DELETE should have value 'todo_delete'"
        )

    def test_todo_update_intent_exists(self):
        assert hasattr(Intent, "TODO_UPDATE"), "Intent enum should have TODO_UPDATE"
        assert Intent.TODO_UPDATE.value == "todo_update", (
            "TODO_UPDATE should have value 'todo_update'"
        )


class TestExistingRulesUnbroken:
    @pytest.mark.parametrize("text,expected_intent", [
        ("/todo buy milk", Intent.TODO_CREATE),
        ("/done buy milk", Intent.TODO_COMPLETE),
        ("?what about deployment", Intent.SEARCH),
        ("!help", Intent.APP_COMMAND),
        ("remind me to buy milk", Intent.TODO_CREATE),
        ("the meeting went well", Intent.CAPTURE),
    ])
    def test_existing_rules_still_work(self, text, expected_intent):
        result = classify_by_rules(text)
        assert result.intent == expected_intent, (
            f"'{text}' should still route to {expected_intent.value}"
        )
