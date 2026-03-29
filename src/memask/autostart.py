from __future__ import annotations

import shutil
import sys
from pathlib import Path
from textwrap import dedent

LABEL = "com.memask.daemon"
SERVICE_NAME = "memask"


def find_executable() -> str | None:
    return shutil.which("memask")


def generate_launchd_plist(executable: str) -> str:
    log_dir = Path.home() / ".memask" / "logs"
    return dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{LABEL}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{executable}</string>
                <string>serve</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{log_dir / "daemon.log"}</string>
            <key>StandardErrorPath</key>
            <string>{log_dir / "daemon.err"}</string>
        </dict>
        </plist>
    """)


def generate_systemd_unit(executable: str) -> str:
    return dedent(f"""\
        [Unit]
        Description=Memory daemon (memask)
        After=network.target

        [Service]
        Type=simple
        ExecStart={executable} serve
        Restart=on-failure
        RestartSec=5

        [Install]
        WantedBy=default.target
    """)


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def systemd_unit_path() -> Path:
    config_dir = Path.home() / ".config" / "systemd" / "user"
    return config_dir / f"{SERVICE_NAME}.service"


def install() -> dict:
    executable = find_executable()
    if executable is None:
        return {"installed": False, "error": "memask executable not found in PATH"}

    if sys.platform == "darwin":
        return _install_launchd(executable)
    elif sys.platform == "linux":
        return _install_systemd(executable)
    else:
        return {"installed": False, "error": f"unsupported platform: {sys.platform}"}


def uninstall() -> dict:
    if sys.platform == "darwin":
        return _uninstall_file(plist_path())
    elif sys.platform == "linux":
        return _uninstall_file(systemd_unit_path())
    else:
        return {"uninstalled": False, "error": f"unsupported platform: {sys.platform}"}


def status() -> dict:
    if sys.platform == "darwin":
        path = plist_path()
    elif sys.platform == "linux":
        path = systemd_unit_path()
    else:
        return {"installed": False, "platform": sys.platform}

    if path.exists():
        return {"installed": True, "path": str(path)}
    return {"installed": False}


def _install_launchd(executable: str) -> dict:
    dest = plist_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path.home() / ".memask" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    content = generate_launchd_plist(executable)
    dest.write_text(content)
    return {"installed": True, "path": str(dest), "platform": "launchd"}


def _install_systemd(executable: str) -> dict:
    dest = systemd_unit_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    content = generate_systemd_unit(executable)
    dest.write_text(content)
    return {"installed": True, "path": str(dest), "platform": "systemd"}


def _uninstall_file(path: Path) -> dict:
    if path.exists():
        path.unlink()
    return {"uninstalled": True, "path": str(path)}
