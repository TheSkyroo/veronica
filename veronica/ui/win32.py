"""Small Win32 helpers for the HUD/Settings windows, via ctypes (no pywin32).

Everything here is lazy: `ctypes.windll` only exists on Windows, so user32 is
loaded on first use and this module still imports — and the window classes'
unit tests (which pass fakes for all of it) still run — on Linux.

Coordinates are physical pixels with the origin at the primary monitor's
top-left corner, y growing downwards (pywebview makes the process DPI aware
before it creates the first window, so EnumDisplayMonitors/GetWindowRect/
SetWindowPos all agree on that space).
"""
from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes
from dataclasses import dataclass
from functools import cache

log = logging.getLogger("veronica.ui.win32")

GWL_EXSTYLE = -20
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
MONITORINFOF_PRIMARY = 1


@dataclass(frozen=True)
class Rect:
    """A screen rectangle: top-left (x, y) plus width/height, in pixels."""

    x: float
    y: float
    w: float
    h: float


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


@cache
def _user32():
    u = ctypes.WinDLL("user32", use_last_error=True)   # AttributeError off Windows
    u.GetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
    u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    u.SetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
    u.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    u.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, wintypes.UINT)
    u.SetWindowPos.restype = wintypes.BOOL
    u.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    u.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    u.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    u.SetForegroundWindow.argtypes = (wintypes.HWND,)
    u.IsWindowVisible.argtypes = (wintypes.HWND,)
    u.GetMonitorInfoW.argtypes = (wintypes.HMONITOR, ctypes.POINTER(_MONITORINFO))
    u.PostMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    return u


def _rect(r: wintypes.RECT) -> Rect:
    return Rect(r.left, r.top, r.right - r.left, r.bottom - r.top)


def work_areas() -> list[Rect]:
    """The work area (screen minus taskbar) of every attached monitor, the
    primary one first. [] if it can't be read."""
    user32 = _user32()
    found: list[tuple[bool, Rect]] = []
    proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                   ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def _cb(hmon, _hdc, _rc, _data):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(_MONITORINFO)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            found.append((bool(info.dwFlags & MONITORINFOF_PRIMARY), _rect(info.rcWork)))
        return True

    user32.EnumDisplayMonitors(None, None, proc_type(_cb), 0)
    found.sort(key=lambda item: not item[0])   # stable: primary first, then OS order
    return [r for _, r in found]


def hwnd_of(window, timeout: float = 20.0) -> int | None:
    """The native HWND behind a pywebview window (its WinForms form), once
    the GUI loop has created it; None if it never shows up."""
    shown = getattr(getattr(window, "events", None), "shown", None)
    if shown is not None and not shown.wait(timeout):
        return None
    native = getattr(window, "native", None)
    handle = getattr(native, "Handle", None)
    if handle is None:
        return None
    try:
        return int(handle.ToInt64())
    except Exception:
        return int(handle)


def make_floating(hwnd: int) -> None:
    """Never take focus, stay out of the taskbar/Alt-Tab, stay on top."""
    user32 = _user32()
    ex = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    ex = (ex | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST) & ~WS_EX_APPWINDOW
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED)


def show_no_activate(hwnd: int) -> None:
    user32 = _user32()
    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


def hide(hwnd: int) -> None:
    _user32().ShowWindow(hwnd, SW_HIDE)


def window_rect(hwnd: int) -> Rect:
    r = wintypes.RECT()
    if not _user32().GetWindowRect(hwnd, ctypes.byref(r)):
        raise OSError(ctypes.get_last_error(), "GetWindowRect failed")
    return _rect(r)


def set_window_rect(hwnd: int, x: float, y: float, w: float | None = None, h: float | None = None) -> None:
    """Move (and, with w/h, resize) without activating; keeps it topmost."""
    flags = SWP_NOACTIVATE | (SWP_NOSIZE if w is None or h is None else 0)
    _user32().SetWindowPos(hwnd, HWND_TOPMOST, int(round(x)), int(round(y)),
                           int(round(w or 0)), int(round(h or 0)), flags)


def dpi_scale(hwnd: int | None = None) -> float:
    """Physical pixels per CSS pixel for `hwnd`'s monitor (1.0 = 96 dpi)."""
    try:
        user32 = _user32()
        dpi = user32.GetDpiForWindow(hwnd) if hwnd else user32.GetDpiForSystem()
        return dpi / 96.0 if dpi else 1.0
    except Exception:
        return 1.0


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    _user32().GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def bring_to_front(hwnd: int) -> None:
    """Best effort: Windows may refuse a background process the foreground
    (it flashes the taskbar button instead)."""
    try:
        _user32().SetForegroundWindow(hwnd)
    except Exception:
        log.debug("SetForegroundWindow failed", exc_info=True)


def post_message(hwnd: int, msg: int, wparam: int = 0, lparam: int = 0) -> bool:
    return bool(_user32().PostMessageW(hwnd, msg, wparam, lparam))


def open_target(target) -> None:
    """Open a file/folder with its default app, or an ms-settings:/https: URL
    (the Windows `open`)."""
    os.startfile(str(target))   # type: ignore[attr-defined]  # Windows only
