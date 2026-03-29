import threading

import pytest
from click.testing import CliRunner

from memask.app import AppContext
from memask.cli import cli
from memask.server import create_app
from tests.helpers import FakeEmbeddingService, FakeLLM, FakeReranker


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def daemon(tmp_path):
    ctx = AppContext(
        db_path=str(tmp_path / "test.db"),
        embedder_factory=lambda: FakeEmbeddingService(),
        llm_factory=lambda: FakeLLM(),
        reranker_factory=lambda: FakeReranker(),
    )
    app = create_app(ctx)
    app.config["TESTING"] = True

    from werkzeug.serving import make_server
    srv = make_server("127.0.0.1", 0, app)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    ctx.shutdown()


class TestInputViaDaemon:
    def test_capture(self, runner, daemon):
        result = runner.invoke(cli, ["--url", daemon, "input", "hello daemon"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Saved" in result.output, "should confirm capture"

    def test_todo_create(self, runner, daemon):
        result = runner.invoke(cli, ["--url", daemon, "input", "remind me to buy milk"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Todo" in result.output, "should confirm todo creation"

    def test_search(self, runner, daemon):
        runner.invoke(cli, ["--url", daemon, "input", "deployment pipeline broken"])
        result = runner.invoke(cli, ["--url", daemon, "input", "?deployment"])
        assert result.exit_code == 0, f"should succeed: {result.output}"


class TestSearchViaDaemon:
    def test_finds_results(self, runner, daemon):
        runner.invoke(cli, ["--url", daemon, "input", "deployment pipeline broken"])
        result = runner.invoke(cli, ["--url", daemon, "search", "deployment"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "deployment" in result.output, "should find matching item"

    def test_no_results(self, runner, daemon):
        result = runner.invoke(cli, ["--url", daemon, "search", "nonexistent"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "No results" in result.output, "should show no results"


class TestListViaDaemon:
    def test_lists_items(self, runner, daemon):
        runner.invoke(cli, ["--url", daemon, "input", "note one"])
        runner.invoke(cli, ["--url", daemon, "input", "note two"])
        result = runner.invoke(cli, ["--url", daemon, "list"])
        assert result.exit_code == 0, f"should succeed: {result.output}"

    def test_empty_list(self, runner, daemon):
        result = runner.invoke(cli, ["--url", daemon, "list"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "No items" in result.output, "should show empty message"


class TestFallbackToDirect:
    def test_input_works_without_daemon(self, runner, tmp_path):
        db_path = str(tmp_path / "test.db")
        result = runner.invoke(cli, ["--db", db_path, "input", "direct mode note"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Saved" in result.output, "direct mode should still work"

    def test_unreachable_daemon_shows_error(self, runner):
        result = runner.invoke(cli, ["--url", "http://127.0.0.1:1", "input", "test"])
        assert result.exit_code != 0, "should fail when daemon unreachable"
        assert "daemon" in result.output.lower() or "error" in result.output.lower(), (
            "should mention daemon or error"
        )
