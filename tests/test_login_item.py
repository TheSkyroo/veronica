"""Start at Login via HKCU\\...\\Run, with an in-memory fake winreg."""
from pathlib import Path

import pytest

from veronica.ui import login_item


class FakeKey:
    def __init__(self, reg, path):
        self.reg, self.path = reg, path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_READ, KEY_SET_VALUE = 1, 2
    REG_SZ = 1

    def __init__(self):
        self.keys: dict[str, dict] = {}
        self.calls = []

    def OpenKey(self, root, path, reserved=0, access=0):  # noqa: N802 — winreg's names
        self.calls.append(("open", root, path, access))
        if path not in self.keys:
            raise FileNotFoundError(path)
        return FakeKey(self, path)

    def CreateKeyEx(self, root, path, reserved=0, access=0):  # noqa: N802
        self.calls.append(("create", root, path, access))
        self.keys.setdefault(path, {})
        return FakeKey(self, path)

    def QueryValueEx(self, key, name):  # noqa: N802
        values = self.keys[key.path]
        if name not in values:
            raise FileNotFoundError(name)
        return values[name]

    def SetValueEx(self, key, name, reserved, kind, value):  # noqa: N802
        self.keys[key.path][name] = (value, kind)

    def DeleteValue(self, key, name):  # noqa: N802
        if name not in self.keys[key.path]:
            raise FileNotFoundError(name)
        del self.keys[key.path][name]


@pytest.fixture
def reg():
    return FakeWinreg()


EXE = Path(r"C:\Users\Me\Veronica App\dist\Veronica\Veronica.exe")


def test_is_enabled_false_when_no_run_value(reg):
    assert login_item.is_enabled(reg) is False
    reg.keys[login_item.RUN_KEY] = {}             # key exists, value doesn't
    assert login_item.is_enabled(reg) is False


def test_enable_writes_quoted_exe_path_under_hkcu_run(reg):
    login_item.enable(EXE, reg)
    assert reg.keys[login_item.RUN_KEY]["Veronica"] == (f'"{EXE}"', reg.REG_SZ)
    assert ("create", "HKCU", r"Software\Microsoft\Windows\CurrentVersion\Run", reg.KEY_SET_VALUE) in reg.calls
    assert login_item.is_enabled(reg) is True
    assert login_item.registered_command(reg) == f'"{EXE}"'


def test_enable_leaves_other_run_values_alone(reg):
    reg.keys[login_item.RUN_KEY] = {"OneDrive": ("onedrive.exe", 1)}
    login_item.enable(EXE, reg)
    login_item.disable(reg)
    assert reg.keys[login_item.RUN_KEY] == {"OneDrive": ("onedrive.exe", 1)}


def test_disable_removes_the_value(reg):
    login_item.enable(EXE, reg)
    login_item.disable(reg)
    assert login_item.is_enabled(reg) is False


def test_disable_is_noop_when_not_enabled(reg):
    login_item.disable(reg)                       # no key at all
    reg.keys[login_item.RUN_KEY] = {}
    login_item.disable(reg)                       # key, no value


def test_is_enabled_false_off_windows(monkeypatch):
    def no_winreg():
        raise ImportError("winreg")

    monkeypatch.setattr(login_item, "_winreg", no_winreg)
    assert login_item.is_enabled() is False


def test_command_for_quotes_the_path():
    assert login_item.command_for(EXE) == f'"{EXE}"'


def test_app_exe_path_only_when_frozen():
    assert login_item.app_exe_path(frozen=False, executable=str(EXE), exists=lambda p: True) is None
    assert login_item.app_exe_path(frozen=True, executable=str(EXE), exists=lambda p: True) == EXE


def test_app_exe_path_must_be_an_existing_exe():
    assert login_item.app_exe_path(frozen=True, executable=str(EXE), exists=lambda p: False) is None
    assert login_item.app_exe_path(frozen=True, executable="/usr/bin/python3", exists=lambda p: True) is None


def test_app_exe_path_dev_run_is_none():
    # the test process is a plain interpreter, never a frozen exe
    assert login_item.app_exe_path() is None
