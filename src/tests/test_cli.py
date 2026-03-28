import pytest
from click.testing import CliRunner

from memask.cli import cli


@pytest.fixture
def runner(tmp_path):
    db_path = str(tmp_path / "test.db")
    return CliRunner(), ["--db", db_path]


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






class TestCliInput:
    def test_plain_text_captures_note(self, runner):
        r, opts = runner
        result = r.invoke(cli, [*opts, "input", "kubernetes needs RAM"])
        assert result.exit_code == 0, result.output
        assert "Saved:" in result.output, "should confirm note saved"

    def test_todo_prefix_creates_todo(self, runner):
        r, opts = runner
        result = r.invoke(cli, [*opts, "input", "/todo", "buy", "milk"])
        assert result.exit_code == 0, result.output
        assert "Todo:" in result.output, "should confirm todo created"

    def test_todo_list_shows_todos(self, runner):
        r, opts = runner
        r.invoke(cli, [*opts, "input", "/todo", "buy", "milk"])
        result = r.invoke(cli, [*opts, "input", "/todo", "list"])
        assert result.exit_code == 0, result.output
        assert "buy milk" in result.output, "should list the created todo"

    def test_todo_complete(self, runner):
        r, opts = runner
        r.invoke(cli, [*opts, "input", "/todo", "buy", "milk"])
        result = r.invoke(cli, [*opts, "input", "/todo", "done", "buy", "milk"])
        assert result.exit_code == 0, result.output
        assert "Done:" in result.output, "should confirm completion"

    def test_search_finds_note(self, runner):
        r, opts = runner
        r.invoke(cli, [*opts, "input", "deployment pipeline is broken"])
        result = r.invoke(cli, [*opts, "input", "?deployment"])
        assert result.exit_code == 0, result.output
        assert "deployment" in result.output.lower(), "should find the note"

    def test_remind_creates_todo(self, runner):
        r, opts = runner
        result = r.invoke(cli, [*opts, "input", "remind", "me", "to", "buy", "groceries"])
        assert result.exit_code == 0, result.output
        assert "Todo:" in result.output, "remind pattern should create todo"

    @pytest.mark.parametrize("text,expected_fragment", [
        (["the", "meeting", "went", "well"], "Saved:"),
        (["/todo", "fix", "the", "faucet"], "Todo:"),
        (["?what", "is", "python"], ""),
        (["!help"], "Command:"),
    ])
    def test_routing_via_cli(self, runner, text, expected_fragment):
        r, opts = runner
        result = r.invoke(cli, [*opts, "input", *text])
        assert result.exit_code == 0, result.output
        if expected_fragment:
            assert expected_fragment in result.output, (
                f"'{' '.join(text)}' should produce output containing '{expected_fragment}'"
            )


class TestCliList:
    def test_list_items(self, runner):
        r, opts = runner
        r.invoke(cli, [*opts, "input", "a plain note"])
        result = r.invoke(cli, [*opts, "list"])
        assert result.exit_code == 0, result.output
        assert "a plain note" in result.output, "should list items"

    def test_list_by_type(self, runner):
        r, opts = runner
        r.invoke(cli, [*opts, "input", "a plain note"])
        r.invoke(cli, [*opts, "input", "/todo", "buy", "milk"])
        result = r.invoke(cli, [*opts, "list", "--type", "todo"])
        assert "buy milk" in result.output, "should filter by type"
        assert "a plain note" not in result.output, "should exclude other types"
