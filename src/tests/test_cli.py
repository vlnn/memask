import pytest
from click.testing import CliRunner

from memask.cli import cli


@pytest.fixture
def run(tmp_path):
    db_path = str(tmp_path / "test.db")
    cli_runner = CliRunner()

    def invoke(*args):
        return cli_runner.invoke(cli, ["--db", db_path, *args])

    return invoke


class TestCreateCommand:
    def test_creates_note(self, run):
        result = run("create", "hello world")
        assert result.exit_code == 0, f"create note should succeed: {result.output}"
        assert "Created note:" in result.output, "should confirm note creation"

    def test_creates_todo(self, run):
        result = run("create", "buy milk", "--type", "todo")
        assert result.exit_code == 0, f"create todo should succeed: {result.output}"
        assert "Created todo:" in result.output, "should confirm todo creation"

    def test_with_all_options(self, run):
        result = run(
            "create", "deploy v2",
            "--type", "todo",
            "--title", "Deploy",
            "--status", "pending",
            "--priority", "1",
            "--due-date", "2025-12-01",
            "--category", "work",
            "--tags", "ops,deploy",
        )
        assert result.exit_code == 0, f"create with all options should succeed: {result.output}"


class TestListCommand:
    def test_lists_items(self, run):
        run("create", "first")
        run("create", "second")

        result = run("list")
        assert "first" in result.output, "should show first item"
        assert "second" in result.output, "should show second item"

    def test_filters_by_type(self, run):
        run("create", "a note")
        run("create", "a todo", "--type", "todo")

        result = run("list", "--type", "todo")
        assert "a todo" in result.output, "should show matching todo"
        assert "a note" not in result.output, "should not show non-matching note"

    def test_empty_result(self, run):
        result = run("list")
        assert "No items found" in result.output, "should show empty message"


class TestJobStatusCommand:
    def test_empty_queue(self, run):
        result = run("job-status")
        assert "No jobs" in result.output, "should report empty queue"
