import pytest

from memask.rag.session import Session


class TestSessionBasic:
    def test_empty_history(self):
        session = Session()
        assert session.history() == [], "new session should have empty history"

    def test_add_exchange(self):
        session = Session()
        session.add_exchange("what is deployment?", "deploy on friday")
        history = session.history()
        assert len(history) == 2, "one exchange should produce two entries"
        assert history[0] == {"role": "user", "content": "what is deployment?"}, (
            "first entry should be the user message"
        )
        assert history[1] == {"role": "assistant", "content": "deploy on friday"}, (
            "second entry should be the assistant message"
        )

    def test_multiple_exchanges(self):
        session = Session()
        session.add_exchange("q1", "a1")
        session.add_exchange("q2", "a2")
        history = session.history()
        assert len(history) == 4, "two exchanges should produce four entries"
        assert history[0]["content"] == "q1", "first query should be oldest"
        assert history[2]["content"] == "q2", "second query should be newer"

    def test_history_order(self):
        session = Session()
        session.add_exchange("first", "r1")
        session.add_exchange("second", "r2")
        session.add_exchange("third", "r3")
        contents = [e["content"] for e in session.history()]
        assert contents == ["first", "r1", "second", "r2", "third", "r3"], (
            "history should be in chronological order"
        )


class TestSessionMaxExchanges:
    def test_default_max_exchanges(self):
        session = Session()
        assert session.max_exchanges == 10, "default max exchanges should be 10"

    def test_custom_max_exchanges(self):
        session = Session(max_exchanges=3)
        assert session.max_exchanges == 3, "should accept custom max"

    def test_evicts_oldest_when_full(self):
        session = Session(max_exchanges=2)
        session.add_exchange("q1", "a1")
        session.add_exchange("q2", "a2")
        session.add_exchange("q3", "a3")
        history = session.history()
        assert len(history) == 4, "should keep only max_exchanges pairs"
        contents = [e["content"] for e in history]
        assert "q1" not in contents, "oldest exchange should be evicted"
        assert "q2" in contents, "second exchange should remain"
        assert "q3" in contents, "newest exchange should remain"

    def test_evicts_full_pairs(self):
        session = Session(max_exchanges=1)
        session.add_exchange("old-q", "old-a")
        session.add_exchange("new-q", "new-a")
        history = session.history()
        assert len(history) == 2, "should keep exactly one exchange"
        assert history[0]["content"] == "new-q", "only newest query should remain"
        assert history[1]["content"] == "new-a", "only newest answer should remain"

    @pytest.mark.parametrize("max_ex", [1, 3, 5])
    def test_never_exceeds_max(self, max_ex):
        session = Session(max_exchanges=max_ex)
        for i in range(max_ex * 3):
            session.add_exchange(f"q{i}", f"a{i}")
        history = session.history()
        assert len(history) == max_ex * 2, (
            f"should have exactly {max_ex} exchanges ({max_ex * 2} entries)"
        )


class TestSessionClear:
    def test_clear_empties_history(self):
        session = Session()
        session.add_exchange("q", "a")
        session.clear()
        assert session.history() == [], "clear should empty history"

    def test_clear_allows_new_exchanges(self):
        session = Session()
        session.add_exchange("old", "old-a")
        session.clear()
        session.add_exchange("new", "new-a")
        assert len(session.history()) == 2, (
            "should accept new exchanges after clear"
        )


class TestSessionHistoryIsCopy:
    def test_history_returns_copy(self):
        session = Session()
        session.add_exchange("q", "a")
        h1 = session.history()
        h1.append({"role": "user", "content": "injected"})
        h2 = session.history()
        assert len(h2) == 2, "mutating returned history should not affect session"

    def test_entries_are_copies(self):
        session = Session()
        session.add_exchange("q", "a")
        h = session.history()
        h[0]["content"] = "tampered"
        assert session.history()[0]["content"] == "q", (
            "mutating returned entries should not affect session"
        )


class TestSessionLen:
    def test_len_empty(self):
        session = Session()
        assert len(session) == 0, "empty session should have length 0"

    def test_len_counts_exchanges(self):
        session = Session()
        session.add_exchange("q1", "a1")
        session.add_exchange("q2", "a2")
        assert len(session) == 2, "len should count exchanges not entries"
