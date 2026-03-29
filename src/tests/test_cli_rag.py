import pytest
from click.testing import CliRunner

from memask.cli import cli
from memask.rag.models import DEFAULT_MODEL_NAME


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


class TestModelStatusCommand:
    def test_shows_not_downloaded(self, runner, mocker, tmp_path):
        mocker.patch("memask.rag.models.model_path", return_value=tmp_path / "models" / DEFAULT_MODEL_NAME)
        result = runner.invoke(cli, ["model", "status"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "not downloaded" in result.output, "should show not downloaded"

    def test_shows_model_name(self, runner, mocker, tmp_path):
        mocker.patch("memask.rag.models.model_path", return_value=tmp_path / "models" / DEFAULT_MODEL_NAME)
        result = runner.invoke(cli, ["model", "status"])
        assert DEFAULT_MODEL_NAME in result.output, "should show model filename"

    def test_shows_download_hint(self, runner, mocker, tmp_path):
        mocker.patch("memask.rag.models.model_path", return_value=tmp_path / "models" / DEFAULT_MODEL_NAME)
        result = runner.invoke(cli, ["model", "status"])
        assert "memask model download" in result.output, (
            "should suggest download command"
        )

    def test_shows_ready_when_downloaded(self, runner, mocker, tmp_path):
        mdir = tmp_path / "models"
        mdir.mkdir()
        (mdir / DEFAULT_MODEL_NAME).write_bytes(b"x" * 200_000)
        mocker.patch("memask.rag.models.model_path", return_value=mdir / DEFAULT_MODEL_NAME)
        result = runner.invoke(cli, ["model", "status"])
        assert "ready" in result.output, "should show ready when model exists"


class TestModelPathCommand:
    def test_prints_path(self, runner, mocker, tmp_path):
        mocker.patch("memask.rag.models.model_path", return_value=tmp_path / "models" / DEFAULT_MODEL_NAME)
        result = runner.invoke(cli, ["model", "path"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert DEFAULT_MODEL_NAME in result.output, "should include model filename"
        assert result.output.strip().endswith(".gguf"), (
            "path should end with .gguf"
        )


class TestModelDownloadCommand:
    def test_already_downloaded(self, runner, tmp_path):
        models_dir = tmp_path / ".memask" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / DEFAULT_MODEL_NAME).write_bytes(b"x" * 1000)

        result = runner.invoke(
            cli, ["model", "status"],
            env={"HOME": str(tmp_path)},
        )
        assert result.exit_code == 0, f"should succeed: {result.output}"


class TestInputAnsweredFormat:
    def test_answered_shows_answer_text(self, runner, db_path, mocker):
        from memask.router.dispatcher import DispatchResult

        mocker.patch(
            "memask.cli.dispatch",
            return_value=DispatchResult(
                action="answered",
                data={
                    "answer": "Deploy is planned for friday. [1]",
                    "sources": ["abc-123"],
                    "results": [],
                },
            ),
        )
        result = runner.invoke(cli, ["--db", db_path, "input", "?deployment"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Deploy is planned for friday" in result.output, (
            "should display the synthesized answer"
        )

    def test_answered_shows_sources(self, runner, db_path, mocker):
        from memask.router.dispatcher import DispatchResult

        mocker.patch(
            "memask.cli.dispatch",
            return_value=DispatchResult(
                action="answered",
                data={
                    "answer": "Some answer",
                    "sources": ["src-1", "src-2"],
                    "results": [],
                },
            ),
        )
        result = runner.invoke(cli, ["--db", db_path, "input", "?query"])
        assert "src-1" in result.output, "should show first source"
        assert "src-2" in result.output, "should show second source"

    def test_answered_no_sources(self, runner, db_path, mocker):
        from memask.router.dispatcher import DispatchResult

        mocker.patch(
            "memask.cli.dispatch",
            return_value=DispatchResult(
                action="answered",
                data={
                    "answer": "No relevant notes found.",
                    "sources": [],
                    "results": [],
                },
            ),
        )
        result = runner.invoke(cli, ["--db", db_path, "input", "?anything"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Sources" not in result.output, (
            "should not show Sources line when empty"
        )
