import sys

import pytest

from memask.autostart import (
    generate_launchd_plist,
    generate_systemd_unit,
    plist_path,
    systemd_unit_path,
    install,
    uninstall,
    status,
)


class TestGenerateLaunchdPlist:
    def test_contains_label(self):
        plist = generate_launchd_plist("/usr/local/bin/memask")
        assert "com.memask.daemon" in plist, "plist should contain the service label"

    def test_contains_executable(self):
        plist = generate_launchd_plist("/usr/local/bin/memask")
        assert "/usr/local/bin/memask" in plist, "plist should contain the executable path"

    def test_contains_serve_argument(self):
        plist = generate_launchd_plist("/usr/local/bin/memask")
        assert "serve" in plist, "plist should contain the serve subcommand"

    def test_runs_at_load(self):
        plist = generate_launchd_plist("/usr/local/bin/memask")
        assert "RunAtLoad" in plist, "plist should run at load"

    def test_keep_alive(self):
        plist = generate_launchd_plist("/usr/local/bin/memask")
        assert "KeepAlive" in plist, "plist should keep the daemon alive"

    def test_is_valid_xml(self):
        import xml.etree.ElementTree as ET

        plist = generate_launchd_plist("/usr/local/bin/memask")
        ET.fromstring(plist), "plist should be valid XML"

    def test_stdout_log_path(self):
        plist = generate_launchd_plist("/usr/local/bin/memask")
        assert "memask" in plist, "plist should reference memask in log path"

    @pytest.mark.parametrize("exe_path", [
        "/usr/local/bin/memask",
        "/home/user/.local/bin/memask",
        "/opt/memask/bin/memask",
    ])
    def test_uses_provided_executable(self, exe_path):
        plist = generate_launchd_plist(exe_path)
        assert exe_path in plist, "plist should use the provided executable path"


class TestGenerateSystemdUnit:
    def test_contains_description(self):
        unit = generate_systemd_unit("/usr/local/bin/memask")
        assert "Memory" in unit, "unit should contain a description"

    def test_contains_exec_start(self):
        unit = generate_systemd_unit("/usr/local/bin/memask")
        assert "ExecStart=" in unit, "unit should contain ExecStart"

    def test_contains_executable(self):
        unit = generate_systemd_unit("/usr/local/bin/memask")
        assert "/usr/local/bin/memask" in unit, "unit should contain the executable"

    def test_contains_serve_argument(self):
        unit = generate_systemd_unit("/usr/local/bin/memask")
        assert "serve" in unit, "unit should contain the serve subcommand"

    def test_restart_on_failure(self):
        unit = generate_systemd_unit("/usr/local/bin/memask")
        assert "Restart=" in unit, "unit should configure restart policy"

    def test_wanted_by_default_target(self):
        unit = generate_systemd_unit("/usr/local/bin/memask")
        assert "default.target" in unit, "unit should be wanted by default.target"

    @pytest.mark.parametrize("exe_path", [
        "/usr/local/bin/memask",
        "/home/user/.local/bin/memask",
    ])
    def test_uses_provided_executable(self, exe_path):
        unit = generate_systemd_unit(exe_path)
        assert exe_path in unit, "unit should use the provided executable path"


class TestPlistPath:
    def test_returns_path_in_launch_agents(self):
        path = plist_path()
        assert "LaunchAgents" in str(path), "plist should be in LaunchAgents"
        assert str(path).endswith(".plist"), "path should end with .plist"


class TestSystemdUnitPath:
    def test_returns_path_in_systemd_user(self):
        path = systemd_unit_path()
        assert "systemd" in str(path), "unit path should be in systemd directory"
        assert str(path).endswith(".service"), "path should end with .service"


class TestInstall:
    def test_creates_plist_on_darwin(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)
        mocker.patch("memask.autostart.find_executable", return_value="/usr/local/bin/memask")

        result = install()
        assert dest.exists(), "install should create the plist file"
        assert "com.memask.daemon" in dest.read_text(), (
            "plist should contain the service label"
        )
        assert result["installed"] is True, "install should report success"
        assert result["path"] == str(dest), "install should report the file path"

    def test_creates_unit_on_linux(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "linux"

        dest = tmp_path / "systemd" / "user" / "memask.service"
        mocker.patch("memask.autostart.systemd_unit_path", return_value=dest)
        mocker.patch("memask.autostart.find_executable", return_value="/usr/local/bin/memask")

        result = install()
        assert dest.exists(), "install should create the unit file"
        assert "ExecStart=" in dest.read_text(), (
            "unit should contain ExecStart directive"
        )
        assert result["installed"] is True, "install should report success"

    def test_idempotent(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)
        mocker.patch("memask.autostart.find_executable", return_value="/usr/local/bin/memask")

        install()
        first_content = dest.read_text()
        result = install()
        assert dest.read_text() == first_content, (
            "reinstalling should produce the same content"
        )
        assert result["installed"] is True, "reinstall should still report success"

    def test_fails_when_executable_not_found(self, mocker):
        mocker.patch("memask.autostart.find_executable", return_value=None)
        result = install()
        assert result["installed"] is False, "should fail when executable not found"
        assert "error" in result, "should include error message"


class TestUninstall:
    def test_removes_plist(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        dest.parent.mkdir(parents=True)
        dest.write_text("plist content")
        mocker.patch("memask.autostart.plist_path", return_value=dest)

        result = uninstall()
        assert not dest.exists(), "uninstall should remove the plist"
        assert result["uninstalled"] is True, "should report success"

    def test_removes_unit(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "linux"

        dest = tmp_path / "systemd" / "user" / "memask.service"
        dest.parent.mkdir(parents=True)
        dest.write_text("unit content")
        mocker.patch("memask.autostart.systemd_unit_path", return_value=dest)

        result = uninstall()
        assert not dest.exists(), "uninstall should remove the unit"
        assert result["uninstalled"] is True, "should report success"

    def test_idempotent(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)

        result = uninstall()
        assert result["uninstalled"] is True, (
            "uninstalling when not installed should still succeed"
        )


class TestStatus:
    def test_not_installed(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        mocker.patch("memask.autostart.plist_path", return_value=dest)

        result = status()
        assert result["installed"] is False, "should report not installed"

    def test_installed(self, tmp_path, mocker):
        mocker.patch("memask.autostart.sys")
        import memask.autostart as mod
        mod.sys.platform = "darwin"

        dest = tmp_path / "LaunchAgents" / "com.memask.daemon.plist"
        dest.parent.mkdir(parents=True)
        dest.write_text("plist content")
        mocker.patch("memask.autostart.plist_path", return_value=dest)

        result = status()
        assert result["installed"] is True, "should report installed"
        assert result["path"] == str(dest), "should report the file path"
