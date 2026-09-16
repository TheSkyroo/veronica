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


def test_bundle_app_path():
    assert login_item.bundle_app_path("/Applications/Veronica.app/Contents/MacOS/Veronica", env={}) == Path(
        "/Applications/Veronica.app"
    )
    assert login_item.bundle_app_path("/usr/bin/python3", env={}) is None
    assert login_item.bundle_app_path("veronica/__main__.py", env={}) is None


def test_bundle_app_path_prefers_app_bundle_env():
    env = {"VERONICA_APP_BUNDLE": "/Apps/Veronica.app"}
    # the launcher runs `python -m veronica`, so argv0 is __main__.py: env must win
    assert login_item.bundle_app_path("veronica/__main__.py", env=env, exists=lambda p: True) == Path(
        "/Apps/Veronica.app"
    )


def test_bundle_app_path_env_must_exist_and_end_with_app():
    env = {"VERONICA_APP_BUNDLE": "/Apps/Veronica.app"}
    assert login_item.bundle_app_path("veronica/__main__.py", env=env, exists=lambda p: False) is None
    env = {"VERONICA_APP_BUNDLE": "/Apps/Veronica"}
    assert login_item.bundle_app_path("veronica/__main__.py", env=env, exists=lambda p: True) is None
    assert login_item.bundle_app_path("veronica/__main__.py", env={"VERONICA_APP_BUNDLE": ""}, exists=lambda p: True) is None


def test_bundle_app_path_derives_from_bundle_build():
    env = {"VERONICA_BUNDLE_BUILD": "/Apps/Veronica.app/Contents/Resources/build.json"}
    assert login_item.bundle_app_path("veronica/__main__.py", env=env, exists=lambda p: True) == Path(
        "/Apps/Veronica.app"
    )
    # a build.json that isn't inside a .app is ignored
    env = {"VERONICA_BUNDLE_BUILD": "/tmp/x/y/build.json"}
    assert login_item.bundle_app_path("veronica/__main__.py", env=env, exists=lambda p: True) is None
    env = {"VERONICA_BUNDLE_BUILD": "/Apps/Veronica.app/Contents/Resources/build.json"}
    assert login_item.bundle_app_path("veronica/__main__.py", env=env, exists=lambda p: False) is None


def test_bundle_app_path_falls_back_to_argv_when_env_empty():
    assert login_item.bundle_app_path(
        "/Applications/Veronica.app/Contents/MacOS/Veronica", env={}, exists=lambda p: False
    ) == Path("/Applications/Veronica.app")
    assert login_item.bundle_app_path("veronica/__main__.py", env={}) is None
