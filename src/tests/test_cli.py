import pytest
from click.testing import CliRunner

from memask.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


class TestCreateCommand:
    def test_creates_note(self, runner, db_path):
        result = runner.invoke(cli, ["--db", db_path, "create", "hello world"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "hello world" in result.output, "should echo the content"

    def test_creates_todo(self, runner, db_path):
        result = runner.invoke(
            cli,
            [
                "--db",
                db_path,
                "create",
                "buy milk",
                "--type",
                "todo",
                "--status",
                "pending",
            ],
        )
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "buy milk" in result.output, "should echo todo content"


class TestListCommand:
    def test_lists_items(self, runner, db_path):
        runner.invoke(cli, ["--db", db_path, "create", "note one"])
        runner.invoke(cli, ["--db", db_path, "create", "note two"])
        result = runner.invoke(cli, ["--db", db_path, "list"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "note one" in result.output, "should list first item"
        assert "note two" in result.output, "should list second item"

    def test_filters_by_type(self, runner, db_path):
        runner.invoke(cli, ["--db", db_path, "create", "a note"])
        runner.invoke(cli, ["--db", db_path, "create", "a task", "--type", "todo"])
        result = runner.invoke(cli, ["--db", db_path, "list", "--type", "todo"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "a task" in result.output, "should show todo"
        assert "a note" not in result.output, "should not show note"

    def test_empty_list(self, runner, db_path):
        result = runner.invoke(cli, ["--db", db_path, "list"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "No items" in result.output, "should show empty message"


class TestSearchCommand:
    def test_keyword_search(self, runner, db_path):
        runner.invoke(cli, ["--db", db_path, "create", "deployment pipeline broken"])
        result = runner.invoke(
            cli, ["--db", db_path, "search", "deployment", "--keyword-only"]
        )
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "deployment" in result.output, "should find matching item"

    def test_no_results(self, runner, db_path):
        result = runner.invoke(
            cli, ["--db", db_path, "search", "nonexistent", "--keyword-only"]
        )
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "No results" in result.output, "should show no results message"


class TestInputCommand:
    def test_capture(self, runner, db_path):
        result = runner.invoke(cli, ["--db", db_path, "input", "kubernetes needs RAM"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Saved" in result.output, "should confirm capture"

    def test_todo_create(self, runner, db_path):
        result = runner.invoke(cli, ["--db", db_path, "input", "remind me to buy milk"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Todo" in result.output, "should confirm todo creation"

    def test_todo_list(self, runner, db_path):
        runner.invoke(cli, ["--db", db_path, "input", "remind me to buy milk"])
        result = runner.invoke(cli, ["--db", db_path, "input", "/todo list"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "buy milk" in result.output, "should list the todo"

    def test_search(self, runner, db_path):
        runner.invoke(cli, ["--db", db_path, "create", "deployment pipeline broken"])
        result = runner.invoke(cli, ["--db", db_path, "input", "?deployment"])
        assert result.exit_code == 0, f"should succeed: {result.output}"


class TestJobStatusCommand:
    def test_no_jobs(self, runner, db_path):
        result = runner.invoke(cli, ["--db", db_path, "job-status"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "No jobs" in result.output, "should show no jobs"


class TestReindexCommand:
    def test_reindex_no_embedder(self, runner, db_path):
        result = runner.invoke(cli, ["--db", db_path, "reindex"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "No embedding" in result.output, "should report no embedder"
