r"""Manage "Start at Login" for the built Veronica.exe.

Windows starts everything listed under
HKCU\Software\Microsoft\Windows\CurrentVersion\Run when the user signs in, so
enabling writes a "Veronica" value there holding the exe's quoted path, and
disabling deletes it. The value is the source of truth for `is_enabled()`.
(Settings > Apps > Startup lists it too, and can switch it off without
removing it; we don't second-guess that.)

It only makes sense when running from the PyInstaller-built exe
(`app_exe_path()`): a dev run (`uv run python -m veronica`) has no stable
executable to point at. `winreg` is stdlib but Windows-only, so it's imported
lazily, and every function takes an injectable `reg` for tests.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "Veronica"


def _winreg():
    import winreg  # Windows only

    return winreg


def app_exe_path(
    frozen: bool | None = None,
    executable: str | None = None,
    exists: Callable[[Path], bool] = Path.exists,
) -> Path | None:
    """The built Veronica.exe when running from it (a PyInstaller "frozen"
    process, whose sys.executable is the exe itself), else None. This is also
    the "are we running from the built app" check that Start at Login and
    relaunch need. `frozen`/`executable`/`exists` are injectable for tests."""
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    if not frozen:
        return None
    exe = Path(executable if executable is not None else sys.executable)
    if exe.suffix.lower() != ".exe" or not exists(exe):
        return None
    return exe


def command_for(exe_path: Path | str) -> str:
    """The Run value: the exe's path, quoted (it may contain spaces)."""
    return f'"{exe_path}"'


def registered_command(reg=None) -> str | None:
    """What the Run key currently starts for us, or None."""
    reg = reg or _winreg()
    try:
        with reg.OpenKey(reg.HKEY_CURRENT_USER, RUN_KEY, 0, reg.KEY_READ) as key:
            value, _kind = reg.QueryValueEx(key, VALUE_NAME)
    except OSError:   # FileNotFoundError: no such value (or key)
        return None
    return str(value) if value else None


def is_enabled(reg=None) -> bool:
    try:
        return registered_command(reg) is not None
    except ImportError:   # not on Windows
        return False


def enable(exe_path: Path | str, reg=None) -> None:
    """Start `exe_path` (the built Veronica.exe) at every sign-in."""
    reg = reg or _winreg()
    with reg.CreateKeyEx(reg.HKEY_CURRENT_USER, RUN_KEY, 0, reg.KEY_SET_VALUE) as key:
        reg.SetValueEx(key, VALUE_NAME, 0, reg.REG_SZ, command_for(Path(exe_path)))


def disable(reg=None) -> None:
    reg = reg or _winreg()
    try:
        with reg.OpenKey(reg.HKEY_CURRENT_USER, RUN_KEY, 0, reg.KEY_SET_VALUE) as key:
            reg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        pass   # already off
