import plistlib
from pathlib import Path

import pytest

from veronica.ui import login_item


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_launchctl(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(login_item.subprocess, "run", fake_run)
    return calls


def test_is_enabled_false_when_no_plist(fake_home):
    assert login_item.is_enabled() is False


def test_enable_writes_plist_and_calls_launchctl(fake_home, fake_launchctl):
    app_path = fake_home / "Applications" / "Veronica.app"
    login_item.enable(app_path)

    path = login_item.plist_path()
    assert path.is_file()
    assert login_item.is_enabled() is True

    with open(path, "rb") as f:
        data = plistlib.load(f)
    assert data["Label"] == "io.manik.veronica"
    assert data["ProgramArguments"] == [str(app_path / "Contents" / "MacOS" / "Veronica")]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is False
    expected_log = str(fake_home / ".veronica" / "logs" / "launchd.log")
    assert data["StandardOutPath"] == expected_log
    assert data["StandardErrorPath"] == expected_log

    assert any(args[0] == "launchctl" and args[1] == "bootstrap" for args in fake_launchctl)


def test_disable_removes_plist_and_calls_launchctl(fake_home, fake_launchctl):
    app_path = fake_home / "Applications" / "Veronica.app"
    login_item.enable(app_path)
    assert login_item.is_enabled() is True

    login_item.disable()
    assert login_item.is_enabled() is False
    assert any(args[0] == "launchctl" and args[1] == "bootout" for args in fake_launchctl)


def test_disable_is_noop_when_not_enabled(fake_home, fake_launchctl):
    login_item.disable()
    assert fake_launchctl == []


def test_enable_ignores_launchctl_failure(fake_home, monkeypatch):
    def raising_run(*a, **k):
        raise OSError("no launchctl")

    monkeypatch.setattr(login_item.subprocess, "run", raising_run)
    app_path = fake_home / "Applications" / "Veronica.app"
    login_item.enable(app_path)  # must not raise
    assert login_item.is_enabled() is True


def test_is_running_from_bundle():
    assert login_item.is_running_from_bundle("/Applications/Veronica.app/Contents/MacOS/Veronica") is True
    assert login_item.is_running_from_bundle("/usr/bin/python3") is False
    assert login_item.is_running_from_bundle("veronica/__main__.py") is False


def test_bundle_app_path():
    assert login_item.bundle_app_path("/Applications/Veronica.app/Contents/MacOS/Veronica") == Path(
        "/Applications/Veronica.app"
    )
    assert login_item.bundle_app_path("/usr/bin/python3") is None
