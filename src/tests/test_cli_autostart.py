import pytest
from click.testing import CliRunner

from memask.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestInstallCommand:
    def test_succeeds_when_executable_found(self, runner, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)
        mocker.patch("memask.autostart.find_executable", return_value="/usr/local/bin/memask")

        result = runner.invoke(cli, ["install"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Installed" in result.output, "should confirm installation"

    def test_fails_when_executable_not_found(self, runner, mocker):
        mocker.patch("memask.autostart.find_executable", return_value=None)
        result = runner.invoke(cli, ["install"])
        assert result.exit_code != 0, "should fail when executable not found"

    def test_shows_path(self, runner, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)
        mocker.patch("memask.autostart.find_executable", return_value="/usr/local/bin/memask")

        result = runner.invoke(cli, ["install"])
        assert str(dest) in result.output, "should show installed path"


class TestUninstallCommand:
    def test_succeeds(self, runner, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        dest.parent.mkdir(parents=True)
        dest.write_text("plist")
        mocker.patch("memask.autostart.plist_path", return_value=dest)

        result = runner.invoke(cli, ["uninstall"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "Removed" in result.output, "should confirm removal"

    def test_succeeds_when_not_installed(self, runner, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)

        result = runner.invoke(cli, ["uninstall"])
        assert result.exit_code == 0, f"should succeed: {result.output}"
        assert "not installed" in result.output.lower() or "Removed" in result.output, (
            "should handle not-installed gracefully"
        )
