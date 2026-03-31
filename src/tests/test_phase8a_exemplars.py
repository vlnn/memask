import pytest

from memask.router.embedding_classifier import INTENT_EXEMPLARS
from memask.router.intents import Intent


class TestNewExemplarsPresent:
    def test_todo_delete_exemplars_exist(self):
        assert Intent.TODO_DELETE in INTENT_EXEMPLARS, (
            "INTENT_EXEMPLARS should have TODO_DELETE exemplars"
        )
        assert len(INTENT_EXEMPLARS[Intent.TODO_DELETE]) >= 3, (
            "TODO_DELETE should have at least 3 exemplars"
        )

    def test_todo_update_exemplars_exist(self):
        assert Intent.TODO_UPDATE in INTENT_EXEMPLARS, (
            "INTENT_EXEMPLARS should have TODO_UPDATE exemplars"
        )
        assert len(INTENT_EXEMPLARS[Intent.TODO_UPDATE]) >= 3, (
            "TODO_UPDATE should have at least 3 exemplars"
        )

    def test_todo_delete_exemplars_are_deletion_phrases(self):
        texts = " ".join(INTENT_EXEMPLARS[Intent.TODO_DELETE]).lower()
        assert any(w in texts for w in ["remove", "delete", "cancel"]), (
            "TODO_DELETE exemplars should contain deletion language"
        )

    def test_todo_update_exemplars_are_update_phrases(self):
        texts = " ".join(INTENT_EXEMPLARS[Intent.TODO_UPDATE]).lower()
        assert any(w in texts for w in ["change", "update", "reschedule"]), (
            "TODO_UPDATE exemplars should contain update language"
        )


class TestExistingExemplarsUnbroken:
    @pytest.mark.parametrize("intent", [
        Intent.CAPTURE,
        Intent.SEARCH,
        Intent.TODO_CREATE,
        Intent.TODO_LIST,
        Intent.TODO_COMPLETE,
    ])
    def test_existing_exemplars_still_present(self, intent):
        assert intent in INTENT_EXEMPLARS, (
            f"INTENT_EXEMPLARS should still have {intent.value} exemplars"
        )
        assert len(INTENT_EXEMPLARS[intent]) >= 3, (
            f"{intent.value} should still have at least 3 exemplars"
        )
