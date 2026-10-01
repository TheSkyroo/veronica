"""Low-level Windows primitives for computer use: synthetic mouse/keyboard
input via `SendInput`, the per-monitor DPI awareness every coordinate
here relies on, a password-field check via UI Automation, and a
foreground-window lookup.

Coordinates here are **physical screen pixels** in the virtual-screen
space: (0, 0) is the primary monitor's top-left corner and the space runs
on across every attached monitor — one left of or above the primary has
*negative* coordinates, and nothing here clamps them. That only holds once
the process is per-monitor DPI aware (`ensure_dpi_awareness`, done the
first time the Win32 backend is touched); an unaware process would see
DPI-virtualised (scaled) coordinates instead. The `computer` tools convert
screenshot pixels before calling in.

Every Win32 call goes through `_win32()` (and UI Automation through
`_uia()`) so tests can swap in recording fakes — nothing in this module
touches the real OS under pytest, and nothing Windows-only is imported at
module level.
"""
import contextlib
import functools
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Shown when the foreground window belongs to an elevated (administrator)
# process and Veronica isn't elevated: Windows' UIPI silently drops
# synthetic input sent to it, so acting would "succeed" and do nothing.
PERMISSION_HINT = (
    "That window is running as administrator, so Windows won't let me control it. "
    "Please do this one yourself, or switch to another window."
)

DRAG_STEPS = 8
DRAG_DURATION_S = 0.2
TYPE_CHUNK_UTF16 = 20
TYPE_CHUNK_GAP_S = 0.01
CLICK_GAP_S = 0.02

# --- SendInput constants -------------------------------------------------------

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x1000
MOUSEEVENTF_VIRTUALDESK = 0x4000
MOUSEEVENTF_ABSOLUTE = 0x8000
# Every move: an absolute position normalised over the whole virtual
# desktop (all monitors), not just the primary one.
_ABS = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

WHEEL_DELTA = 120
# How many pixels of the tools' "scroll by N pixels" one wheel notch is
# worth (a notch scrolls ~3 lines by default). Deltas are sent in whole
# notches: plenty of Win32 apps ignore a partial (< 120) delta.
WHEEL_PX_PER_NOTCH = 100

# Window classes / executables of Windows' permission and security
# prompts — where "Allow"-style clicks are refused and every action needs
# a confirm. Lowercase executable names (what `Front.bundle_id` holds).
# The UAC consent prompt (consent.exe) normally renders on the secure
# desktop, where no foreground window is visible to Veronica at all (and
# no input from it can arrive); it's listed for the case where secure
# desktop prompting is turned off.
SYSTEM_DIALOG_APPS = frozenset({
    "consent.exe",                    # UAC elevation prompt
    "credentialuibroker.exe",         # "Windows Security" credential prompt
    "logonui.exe",                    # sign-in / lock screen UI
    "lockapp.exe",                    # lock screen
    "smartscreen.exe",                # "Windows protected your PC"
    "sechealthui.exe",                # Windows Security app
    "securityhealthhost.exe",
    "useraccountcontrolsettings.exe", # UAC slider
    "systemsettings.exe",             # Settings: every page is one click from Privacy & security
})
SYSTEM_DIALOG_CLASSES = frozenset({
    "credential dialog xaml host",                              # credential prompt
    "$$$secure uap dummy window class for interim dialog",      # UAC dimmed backdrop
})
# Window titles that mean a security prompt whatever process shows it
# (e.g. explorer's "Open File - Security Warning" Run button).
SYSTEM_DIALOG_TITLES = frozenset({
    "windows security",
    "user account control",
    "open file - security warning",
    "windows protected your pc",
    "windows defender smartscreen",
    "microsoft defender smartscreen",
})

# Button labels Veronica must never click while a system dialog is up.
DISALLOWED_DIALOG_TARGETS = frozenset({
    "allow", "always allow", "allow access", "ok", "yes", "continue", "install", "trust",
    "run", "run anyway", "more info", "open settings", "open system settings", "turn off",
})

# Win32 virtual-key codes. Punctuation keys are the US-layout OEM codes
# (on another layout the same physical key may carry another symbol);
# `type_text` is layout-independent and is the way to enter literal text.
KEYCODES: dict[str, int] = {
    **{chr(c): c - 32 for c in range(ord("a"), ord("z") + 1)},   # 'a' → 0x41 ('A')
    **{str(d): 0x30 + d for d in range(10)},
    **{f"f{n}": 0x6F + n for n in range(1, 25)},                 # f1 = 0x70 … f24 = 0x87
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "space": 0x20, "spacebar": 0x20,
    "backspace": 0x08, "bksp": 0x08, "delete": 0x2E, "forwarddelete": 0x2E, "insert": 0x2D,
    "ins": 0x2D, "esc": 0x1B,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "pgdown": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "arrowleft": 0x25, "arrowup": 0x26, "arrowright": 0x27, "arrowdown": 0x28,
    "printscreen": 0x2C, "prtsc": 0x2C, "capslock": 0x14, "numlock": 0x90, "scrolllock": 0x91,
    "pause": 0x13, "apps": 0x5D, "menu": 0x5D, "contextmenu": 0x5D,
    "volumemute": 0xAD, "volumedown": 0xAE, "volumeup": 0xAF,
    "nexttrack": 0xB0, "prevtrack": 0xB1, "stop": 0xB2, "playpause": 0xB3,
    "semicolon": 0xBA, "equal": 0xBB, "comma": 0xBC, "minus": 0xBD, "period": 0xBE,
    "slash": 0xBF, "grave": 0xC0, "leftbracket": 0xDB, "backslash": 0xDC,
    "rightbracket": 0xDD, "quote": 0xDE,
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD, ".": 0xBE, "/": 0xBF, "`": 0xC0,
    "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}

VK_RETURN = 0x0D
VK_TAB = 0x09

# Modifier keys (the left-hand ones).
MODIFIERS: dict[str, int] = {"ctrl": 0xA2, "alt": 0xA4, "shift": 0xA0, "win": 0x5B}

# Keys that need KEYEVENTF_EXTENDEDKEY, or Windows reads them as their
# numpad twins (Home → numpad 7 with NumLock on, ...).
EXTENDED_KEYS = frozenset({
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,   # pgup/pgdn/end/home/arrows
    0x2C, 0x2D, 0x2E,                                 # printscreen, insert, delete
    0x5B, 0x5C, 0x5D,                                 # win keys, apps
    0x90,                                             # numlock
    0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,         # media / volume
})

_BUTTONS = ("left", "right", "middle")
_BUTTON_FLAGS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


@dataclass(frozen=True)
class Front:
    """The foreground app and window at the time of the lookup.

    `bundle_id` keeps its historical name but on Windows holds the app's
    identity: its lowercase executable file name ("chrome.exe",
    "windowsterminal.exe"). `window_class` is the Win32 window class."""
    app: str
    bundle_id: str
    window_title: str
    pid: int
    window_class: str = ""


@dataclass(frozen=True)
class MouseInput:
    """One MOUSEINPUT: `dx`/`dy` already normalised to 0..65535 over the
    virtual desktop (ignored unless `flags` has MOUSEEVENTF_ABSOLUTE);
    `x`/`y` are the screen pixels they stand for (for logs and tests)."""
    flags: int
    dx: int = 0
    dy: int = 0
    data: int = 0
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class KeyInput:
    """One KEYBDINPUT: a virtual key, or with KEYEVENTF_UNICODE a UTF-16
    code unit in `scan` (and `vk` 0)."""
    vk: int = 0
    flags: int = 0
    scan: int = 0


# --- Win32 backend (ctypes) -------------------------------------------------------

class _CtypesWin32:
    """The real Win32 calls, via ctypes. Built lazily by `_win32()` — on a
    non-Windows interpreter the constructor raises."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self.ctypes, self.wt = ctypes, wintypes
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        self._dwmapi = None
        ulong_ptr = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ulong_ptr)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                        ("szDevice", wintypes.WCHAR * 32)]

        self.INPUT, self.MONITORINFOEXW = INPUT, MONITORINFOEXW
        self.WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self.MONITORENUMPROC = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
        )
        u, k = self.user32, self.kernel32
        u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        u.SendInput.restype = wintypes.UINT
        u.GetForegroundWindow.restype = wintypes.HWND
        for fn in ("GetWindowTextLengthW", "IsWindowVisible", "IsIconic"):
            getattr(u, fn).argtypes = [wintypes.HWND]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.EnumWindows.argtypes = [self.WNDENUMPROC, wintypes.LPARAM]
        u.EnumChildWindows.argtypes = [wintypes.HWND, self.WNDENUMPROC, wintypes.LPARAM]
        u.EnumDisplayMonitors.argtypes = [wintypes.HDC, ctypes.c_void_p, self.MONITORENUMPROC, wintypes.LPARAM]
        u.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
        u.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        u.MapVirtualKeyW.restype = wintypes.UINT
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
        ]
        self.advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        self.advapi32.GetTokenInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ]

    # -- DPI -----------------------------------------------------------------

    def set_dpi_awareness(self) -> str:
        """Per-monitor v2 awareness, falling back to per-monitor (8.1+
        API), then system awareness. "already-set" when something set the
        awareness before us (mss, a GUI toolkit, the app manifest) — the
        call fails with ERROR_ACCESS_DENIED then."""
        ctypes = self.ctypes
        fn = getattr(self.user32, "SetProcessDpiAwarenessContext", None)
        if fn is not None:
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = self.wt.BOOL
            if fn(ctypes.c_void_p(-4)):      # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
                return "per-monitor-v2"
            if ctypes.get_last_error() == 5:  # ERROR_ACCESS_DENIED: already set
                return "already-set"
        try:
            hr = ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)   # PROCESS_PER_MONITOR_DPI_AWARE
            if hr == 0:
                return "per-monitor"
            if hr & 0xFFFFFFFF == 0x80070005:                       # E_ACCESSDENIED
                return "already-set"
        except (OSError, AttributeError):
            pass
        return "system" if self.user32.SetProcessDPIAware() else "unaware"

    # -- input ---------------------------------------------------------------

    def virtual_screen(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the virtual desktop (every monitor)."""
        m = self.user32.GetSystemMetrics
        return (m(76), m(77), m(78), m(79))   # SM_[XY]VIRTUALSCREEN, SM_C[XY]VIRTUALSCREEN

    def send_input(self, events) -> int:
        items = []
        for e in events:
            inp = self.INPUT()
            if isinstance(e, MouseInput):
                inp.type = INPUT_MOUSE
                inp.u.mi.dx, inp.u.mi.dy = int(e.dx), int(e.dy)
                inp.u.mi.mouseData = int(e.data) & 0xFFFFFFFF   # wheel deltas are signed
                inp.u.mi.dwFlags = e.flags
            else:
                inp.type = INPUT_KEYBOARD
                inp.u.ki.wVk = e.vk
                inp.u.ki.dwFlags = e.flags
                # A real scan code alongside the vk: some apps (games,
                # remote desktop clients) read only the scan code.
                inp.u.ki.wScan = e.scan if e.flags & KEYEVENTF_UNICODE else self.user32.MapVirtualKeyW(e.vk, 0)
            items.append(inp)
        if not items:
            return 0
        arr = (self.INPUT * len(items))(*items)
        return int(self.user32.SendInput(len(items), arr, self.ctypes.sizeof(self.INPUT)))

    def set_cursor_pos(self, x: int, y: int) -> None:
        if not self.user32.SetCursorPos(int(x), int(y)):
            raise OSError(f"SetCursorPos({x}, {y}) failed")

    def clipboard_sequence(self) -> int:
        return int(self.user32.GetClipboardSequenceNumber())

    # -- windows -------------------------------------------------------------

    def foreground_window(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    def window_title(self, hwnd: int) -> str:
        n = self.user32.GetWindowTextLengthW(hwnd)
        buf = self.ctypes.create_unicode_buffer(max(n, 0) + 1)
        self.user32.GetWindowTextW(hwnd, buf, len(buf))
        return buf.value

    def window_class(self, hwnd: int) -> str:
        buf = self.ctypes.create_unicode_buffer(256)
        self.user32.GetClassNameW(hwnd, buf, len(buf))
        return buf.value

    def window_pid(self, hwnd: int) -> int:
        pid = self.wt.DWORD(0)
        self.user32.GetWindowThreadProcessId(hwnd, self.ctypes.byref(pid))
        return int(pid.value)

    def _dwm(self):
        if self._dwmapi is None:
            self._dwmapi = self.ctypes.WinDLL("dwmapi")
            self._dwmapi.DwmGetWindowAttribute.argtypes = [
                self.wt.HWND, self.wt.DWORD, self.ctypes.c_void_p, self.wt.DWORD,
            ]
        return self._dwmapi

    def window_rect(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """(left, top, right, bottom) of the window as drawn: DWM's
        extended frame bounds (no invisible resize border / drop shadow),
        else GetWindowRect."""
        ctypes, rect = self.ctypes, self.wt.RECT()
        with contextlib.suppress(OSError):
            if self._dwm().DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) == 0:
                return (rect.left, rect.top, rect.right, rect.bottom)   # DWMWA_EXTENDED_FRAME_BOUNDS
        if self.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return (rect.left, rect.top, rect.right, rect.bottom)
        return None

    def window_visible(self, hwnd: int) -> bool:
        """Visible, not minimised and not cloaked (a UWP app suspended on
        another virtual desktop is "visible" but cloaked)."""
        if not self.user32.IsWindowVisible(hwnd) or self.user32.IsIconic(hwnd):
            return False
        cloaked = self.wt.DWORD(0)
        with contextlib.suppress(OSError):
            self._dwm().DwmGetWindowAttribute(hwnd, 14, self.ctypes.byref(cloaked), 4)   # DWMWA_CLOAKED
        return not cloaked.value

    def top_level_windows(self) -> list[int]:
        """Every top-level window, front to back (EnumWindows' z-order)."""
        out: list[int] = []

        def cb(hwnd, _lparam):
            out.append(int(hwnd or 0))
            return True

        self.user32.EnumWindows(self.WNDENUMPROC(cb), 0)
        return out

    def uwp_child_pid(self, hwnd: int) -> int | None:
        """For an ApplicationFrameHost frame: the pid of the hosted app's
        CoreWindow (the child owned by another process), or None."""
        own = self.window_pid(hwnd)
        found: list[int] = []

        def cb(child, _lparam):
            pid = self.window_pid(child)
            if pid and pid != own:
                found.append(pid)
                return False
            return True

        self.user32.EnumChildWindows(hwnd, self.WNDENUMPROC(cb), 0)
        return found[0] if found else None

    # -- processes -------------------------------------------------------------

    def process_path(self, pid: int) -> str:
        h = self.kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            size = self.wt.DWORD(32768)
            buf = self.ctypes.create_unicode_buffer(size.value)
            if self.kernel32.QueryFullProcessImageNameW(h, 0, buf, self.ctypes.byref(size)):
                return buf.value
            return ""
        finally:
            self.kernel32.CloseHandle(h)

    def app_name(self, path: str) -> str:
        """The executable's FileDescription ("Google Chrome"), or ""."""
        return _file_description(path)

    def _token_elevated(self, process) -> bool | None:
        ctypes, wt = self.ctypes, self.wt
        token = wt.HANDLE()
        if not self.advapi32.OpenProcessToken(process, 0x0008, ctypes.byref(token)):   # TOKEN_QUERY
            # Denied for a process at a higher integrity level than ours.
            return True if ctypes.get_last_error() == 5 else None
        try:
            elevated, size = wt.DWORD(0), wt.DWORD(0)
            if not self.advapi32.GetTokenInformation(token, 20, ctypes.byref(elevated), 4, ctypes.byref(size)):
                return None                                                          # TokenElevation
            return bool(elevated.value)
        finally:
            self.kernel32.CloseHandle(token)

    def self_elevated(self) -> bool:
        return bool(self._token_elevated(self.kernel32.GetCurrentProcess()))

    def process_elevated(self, pid: int) -> bool | None:
        h = self.kernel32.OpenProcess(0x1000, False, pid)
        if not h:
            return True if self.ctypes.get_last_error() == 5 else None
        try:
            return self._token_elevated(h)
        finally:
            self.kernel32.CloseHandle(h)

    # -- monitors ----------------------------------------------------------------

    def monitors(self) -> list[dict]:
        """[{"handle", "rect": (l, t, r, b), "primary", "name"}] in
        EnumDisplayMonitors order."""
        ctypes = self.ctypes
        out: list[dict] = []

        def cb(hmon, _hdc, _rect, _lparam):
            info = self.MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(info)
            if self.user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcMonitor
                out.append({
                    "handle": int(hmon or 0), "rect": (r.left, r.top, r.right, r.bottom),
                    "primary": bool(info.dwFlags & 1),            # MONITORINFOF_PRIMARY
                    "name": info.szDevice,
                })
            return True

        self.user32.EnumDisplayMonitors(None, None, self.MONITORENUMPROC(cb), 0)
        return out


@functools.lru_cache(maxsize=64)
def _file_description(path: str) -> str:
    if not path:
        return ""
    try:
        import ctypes
        import struct
        from ctypes import wintypes

        ver = ctypes.WinDLL("version")
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return ""
        buf = ctypes.create_string_buffer(size)
        if not ver.GetFileVersionInfoW(path, 0, size, buf):
            return ""
        ptr, length = ctypes.c_void_p(), wintypes.UINT(0)
        if not ver.VerQueryValueW(buf, "\\VarFileInfo\\Translation", ctypes.byref(ptr), ctypes.byref(length)) \
                or length.value < 4:
            return ""
        lang, codepage = struct.unpack("<HH", ctypes.string_at(ptr.value, 4))
        key = f"\\StringFileInfo\\{lang:04x}{codepage:04x}\\FileDescription"
        if not ver.VerQueryValueW(buf, key, ctypes.byref(ptr), ctypes.byref(length)) or not length.value:
            return ""
        return ctypes.wstring_at(ptr.value, length.value).rstrip("\0").strip()
    except Exception:  # noqa: BLE001 — a missing description is not an error
        return ""


# --- framework seams ---------------------------------------------------------

_backend = None


def _win32():
    """The Win32 backend (built once; becoming DPI aware first thing)."""
    global _backend
    if _backend is None:
        _backend = _CtypesWin32()
        ensure_dpi_awareness()
    return _backend


def _uia():
    import uiautomation
    return uiautomation


_sleep = time.sleep
_dpi_state: str | None = None


def ensure_dpi_awareness() -> str:
    """Make the process per-monitor DPI aware (v2), once; idempotent.
    Returns how it went ("per-monitor-v2", "per-monitor", "system",
    "already-set", or "unavailable" when there is no Win32 here). Call it
    at startup before any window is created; it also runs the first time
    the Win32 backend is used, so coordinates are never DPI-virtualised."""
    global _dpi_state
    if _dpi_state is None:
        try:
            _dpi_state = _win32().set_dpi_awareness()
        except Exception as e:  # noqa: BLE001 — never fatal
            log.debug("DPI awareness unavailable: %s", e)
            _dpi_state = "unavailable"
        log.debug("DPI awareness: %s", _dpi_state)
    return _dpi_state


def post(*events) -> None:
    """Send `events` as one SendInput batch (so nothing interleaves with
    them). Raises OSError if Windows inserted fewer than asked — blocked
    by another thread, or (silently, with no error) by UIPI when the
    target is elevated; see `input_blocked`."""
    if not events:
        return
    sent = _win32().send_input(list(events))
    if sent != len(events):
        raise OSError(f"SendInput inserted {sent} of {len(events)} events")


def _release(events, what: str) -> None:
    """Send key-ups / button-ups from a `finally`: a failure here is
    logged rather than raised so it never masks the original error."""
    try:
        post(*events)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to release %s: %s", what, e)


# --- mouse -------------------------------------------------------------------

def _absolute(x: float, y: float) -> tuple[int, int]:
    """Screen pixel → SendInput's 0..65535 over the virtual desktop.
    Aims at the pixel's centre, so it lands on that pixel whether Windows
    scales back by width/65536 or by (width-1)/65535."""
    vx, vy, vw, vh = _win32().virtual_screen()
    vw, vh = max(int(vw), 1), max(int(vh), 1)

    def norm(p: float, origin: int, size: int) -> int:
        n = int((round(p) - origin + 0.5) * 65536 / size)
        return min(max(n, 0), 65535)

    return norm(x, vx, vw), norm(y, vy, vh)


def _mouse_event(flags: int, x: float, y: float) -> MouseInput:
    dx, dy = _absolute(x, y)
    return MouseInput(flags=_ABS | flags, dx=dx, dy=dy, x=float(x), y=float(y))


def _button_event(flags: int, x: float, y: float) -> MouseInput:
    """A button event at wherever the pointer is (`x`/`y` only say where
    that should be): no position of its own, so the exact pixel
    `move` snapped to is the one pressed."""
    return MouseInput(flags=flags, x=float(x), y=float(y))


def _button_flags(button: str) -> tuple[int, int]:
    try:
        return _BUTTON_FLAGS[button]
    except KeyError:
        raise ValueError(f"unknown mouse button {button!r} (use one of {', '.join(_BUTTONS)})") from None


def move(x: float, y: float) -> None:
    """Move the pointer to (x, y) screen pixels: a real (absolute,
    virtual-desktop) mouse move — so apps see input and hover — then
    SetCursorPos onto the exact pixel, which the 0..65535 normalisation
    can miss by one."""
    post(_mouse_event(0, x, y))
    _win32().set_cursor_pos(round(x), round(y))


def click(x: float, y: float, button: str = "left", double: bool = False) -> None:
    """Move to (x, y) then press and release `button`. `double` sends a
    second down/up pair right away — well inside the double-click time
    and on the same pixel, so Windows reports a double-click. The
    button-up is always attempted, so no button is left held."""
    down, up = _button_flags(button)
    move(x, y)

    def pair() -> None:
        try:
            post(_button_event(down, x, y))
        finally:
            _release([_button_event(up, x, y)], f"{button} button")

    pair()
    if double:
        _sleep(CLICK_GAP_S)
        pair()


def drag(x1: float, y1: float, x2: float, y2: float) -> None:
    """Left-drag from (x1, y1) to (x2, y2): press, `DRAG_STEPS`
    interpolated moves spread over `DRAG_DURATION_S`, release — at the
    end point even if a move in between failed."""
    move(x1, y1)
    step_s = DRAG_DURATION_S / DRAG_STEPS
    try:
        post(_button_event(MOUSEEVENTF_LEFTDOWN, x1, y1))
        for i in range(1, DRAG_STEPS + 1):
            t = i / DRAG_STEPS
            _sleep(step_s)
            move(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    finally:
        try:
            move(x2, y2)
        except Exception as e:  # noqa: BLE001 — release regardless
            log.warning("failed to reach the drag end point: %s", e)
        _release([_button_event(MOUSEEVENTF_LEFTUP, x2, y2)], "left button (drag)")


def _notches(px: float) -> int:
    """`px` pixels as a wheel delta in whole notches (at least one)."""
    if not px:
        return 0
    n = max(1, round(abs(px) / WHEEL_PX_PER_NOTCH))
    return WHEEL_DELTA * n * (1 if px > 0 else -1)


def scroll(x: float, y: float, dx: float = 0, dy: float = 0) -> None:
    """Move to (x, y) then turn the wheel by (dx, dy) pixels (rounded to
    whole notches of WHEEL_DELTA). The wheel's own sign convention:
    positive dy rotates it forward — scrolls *up*, toward the top of the
    document — and positive dx tilts it right (scrolls right)."""
    move(x, y)
    events = []
    if dy:
        events.append(MouseInput(flags=MOUSEEVENTF_WHEEL, data=_notches(dy), x=float(x), y=float(y)))
    if dx:
        events.append(MouseInput(flags=MOUSEEVENTF_HWHEEL, data=_notches(dx), x=float(x), y=float(y)))
    if events:
        post(*events)


# --- keyboard ----------------------------------------------------------------

def _utf16_units(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _chunks(text: str, limit: int = TYPE_CHUNK_UTF16) -> list[str]:
    """Split `text` into pieces of at most `limit` UTF-16 code units,
    never splitting a surrogate pair (astral characters count as 2)."""
    out: list[str] = []
    cur, cur_units = [], 0
    for ch in text:
        units = 2 if ord(ch) > 0xFFFF else 1
        if cur and cur_units + units > limit:
            out.append("".join(cur))
            cur, cur_units = [], 0
        cur.append(ch)
        cur_units += units
    if cur:
        out.append("".join(cur))
    return out


def _vk_press(vk: int) -> list[KeyInput]:
    ext = KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_KEYS else 0
    return [KeyInput(vk=vk, flags=ext), KeyInput(vk=vk, flags=ext | KEYEVENTF_KEYUP)]


def _text_inputs(chunk: str) -> list[KeyInput]:
    """Key events typing `chunk`: each UTF-16 code unit as a
    KEYEVENTF_UNICODE down/up (layout-independent; a surrogate pair is two
    units, which Windows reassembles), except line breaks and tabs, sent
    as real Enter/Tab presses — a unicode "\\n" is not Enter to most apps.
    "\\r\\n" counts as one line break."""
    out: list[KeyInput] = []
    prev = ""
    for ch in chunk:
        if ch == "\n" and prev == "\r":
            prev = ch
            continue
        prev = ch
        if ch in "\r\n":
            out += _vk_press(VK_RETURN)
        elif ch == "\t":
            out += _vk_press(VK_TAB)
        else:
            raw = ch.encode("utf-16-le")
            for i in range(0, len(raw), 2):
                unit = int.from_bytes(raw[i:i + 2], "little")
                out += [KeyInput(scan=unit, flags=KEYEVENTF_UNICODE),
                        KeyInput(scan=unit, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)]
    return out


def type_text(text: str) -> None:
    """Type `text` as Unicode key events (layout-independent), in chunks
    of `TYPE_CHUNK_UTF16` code units — one SendInput batch each — with a
    short gap between them."""
    chunks = _chunks(text.replace("\r\n", "\n"))
    for i, chunk in enumerate(chunks):
        post(*_text_inputs(chunk))
        if i < len(chunks) - 1:
            _sleep(TYPE_CHUNK_GAP_S)


# "cmd"/"command" (the macOS habit, and what a model trained on Mac
# shortcuts reaches for) means Ctrl here: cmd+c is copy, cmd+s save.
_MODIFIER_CANON = {
    "ctrl": "ctrl", "control": "ctrl", "ctl": "ctrl", "cmd": "ctrl", "command": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt",
    "shift": "shift",
    "win": "win", "windows": "win", "super": "win", "meta": "win", "start": "win",
}
_MODIFIER_ORDER = ("ctrl", "alt", "shift", "win")
_KEY_CANON = {"escape": "esc", "del": "delete", "forwarddelete": "delete"}


class DangerousCombo(ValueError):
    """Raised by `key()` for combos that close apps, lock or sign out of
    the PC, or open its security / power screens — never sent, whatever
    the caller asked."""


# Canonical form (see `normalize_combo`).
DANGEROUS_COMBOS = frozenset({
    "alt+f4",                # close the app (on the desktop: the shutdown dialog)
    "ctrl+q",                # quit app (Firefox, Signal, ...)
    "ctrl+shift+q",          # quit Chrome / Edge
    "win+l",                 # lock the PC
    "ctrl+alt+delete",       # security screen (sign out, task manager)
    "ctrl+alt+end",          # the same through Remote Desktop
    "ctrl+shift+esc",        # Task Manager (end any process)
    "win+x",                 # power-user menu (shut down, sign out)
})


def normalize_combo(combo: str) -> str:
    """Canonical form of a combo: lowercase, modifier aliases folded
    (control→ctrl, cmd/command→ctrl, option→alt, windows/super→win),
    modifiers deduplicated and ordered ctrl, alt, shift, win, then the key
    (escape→esc, del→delete). A lone modifier ("win") is that key itself.
    Raises ValueError when the shape isn't (modifiers)+one key; the key
    name itself is *not* validated here (see `key()`)."""
    parts = [p.strip().lower() for p in combo.split("+")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"can't parse key combo {combo!r}")
    *mods, name = parts
    canon = set()
    for m in mods:
        if m not in _MODIFIER_CANON:
            raise ValueError(f"unknown modifier {m!r} in {combo!r}")
        canon.add(_MODIFIER_CANON[m])
    if name in _MODIFIER_CANON:
        if mods:
            raise ValueError(f"key combo {combo!r} has modifiers but no key to press")
        return _MODIFIER_CANON[name]
    ordered = [m for m in _MODIFIER_ORDER if m in canon]
    return "+".join([*ordered, _KEY_CANON.get(name, name)])


def _parse_combo(combo: str) -> tuple[int, list[int]]:
    """`"ctrl+shift+s"` → (vk, [modifier vks in press order]). Raises
    DangerousCombo for the denylist and ValueError for anything that isn't
    (modifiers)+one key from `KEYCODES`."""
    norm = normalize_combo(combo)
    if norm in DANGEROUS_COMBOS:
        raise DangerousCombo(f"refusing to press {norm} (closes apps, locks or signs out)")
    *mods, name = norm.split("+")
    if not mods and name in MODIFIERS:
        return MODIFIERS[name], []
    if name not in KEYCODES:
        raise ValueError(f"unknown key {name!r} in {combo!r}")
    return KEYCODES[name], [MODIFIERS[m] for m in mods]


def key(combo: str) -> None:
    """Press and release a key combo such as "enter", "ctrl+s" or
    "ctrl+shift+tab": modifiers down, key down, then key up and modifiers
    up in reverse. The ups are always attempted even if the downs fail, so
    no modifier is left stuck. (A modifier physically held at the time —
    e.g. a push-to-talk key — does combine with the synthetic ones.)"""
    vk, mods = _parse_combo(combo)
    downs: list[KeyInput] = []
    ups: list[KeyInput] = []
    for code in [*mods, vk]:
        d, u = _vk_press(code)
        downs.append(d)
        ups.insert(0, u)
    try:
        post(*downs)
    finally:
        _release(ups, f"key {combo!r}")


# --- permissions ----------------------------------------------------------------

def accessibility_trusted(prompt: bool = False) -> bool:
    """Kept for callers from the macOS days: Windows needs no grant for
    synthetic input or UI Automation, so this is always True (`prompt` is
    ignored). What can block input on Windows is elevation — see
    `input_blocked`."""
    return True


def input_blocked() -> bool:
    """True when the foreground window belongs to an elevated process and
    Veronica isn't elevated: UIPI then drops synthetic input without any
    error. False when unknown (Windows enforces it anyway; this only buys
    an honest message)."""
    try:
        w = _win32()
        hwnd = w.foreground_window()
        if not hwnd or w.self_elevated():
            return False
        return bool(w.process_elevated(w.window_pid(hwnd)))
    except Exception as e:  # noqa: BLE001 — never raises
        log.debug("elevation check failed: %s", e)
        return False


def focused_is_secure() -> bool:
    """True when the focused UI element is a password field (UI
    Automation's IsPassword — set for Win32 ES_PASSWORD edits, browser
    <input type=password>, UWP PasswordBox). False on any error."""
    try:
        uia = _uia()
        init = getattr(uia, "UIAutomationInitializerInThread", None)
        with init() if init is not None else contextlib.nullcontext():
            control = uia.GetFocusedControl()
            if control is None:
                return False
            return bool(control.Element.CurrentIsPassword)
    except Exception as e:  # noqa: BLE001
        log.debug("focused element check failed: %s", e)
        return False


# --- frontmost ---------------------------------------------------------------

UWP_HOST = "applicationframehost.exe"


def app_for_pid(pid: int, win32=None) -> tuple[str, str]:
    """(display name, lowercase exe file name) for `pid`; the display name
    is the executable's FileDescription, else its file name without .exe.
    Empty strings when the process can't be queried."""
    w = win32 if win32 is not None else _win32()
    path = w.process_path(pid) if pid else ""
    exe = os.path.basename(path.replace("\\", "/")).lower() if path else ""
    name = w.app_name(path) if path else ""
    if not name and exe:
        name = os.path.splitext(os.path.basename(path.replace("\\", "/")))[0]
    return name, exe


def frontmost() -> Front:
    """The foreground window: its app (name, exe, pid), title and class.
    A UWP app's frame belongs to ApplicationFrameHost, so its hosted app's
    process is reported instead. Empty strings when there's nothing to
    report — including while UAC's secure desktop is up."""
    try:
        w = _win32()
        hwnd = w.foreground_window()
    except Exception as e:  # noqa: BLE001
        log.debug("foreground window lookup failed: %s", e)
        hwnd = 0
    if not hwnd:
        return Front(app="", bundle_id="", window_title="", pid=0)
    title = cls = ""
    pid = 0
    name = exe = ""
    try:
        title = str(w.window_title(hwnd) or "")
        cls = str(w.window_class(hwnd) or "")
        pid = int(w.window_pid(hwnd) or 0)
        name, exe = app_for_pid(pid, w)
        if exe == UWP_HOST:
            child = w.uwp_child_pid(hwnd)
            if child:
                pid = int(child)
                name, exe = app_for_pid(pid, w)
    except Exception as e:  # noqa: BLE001
        log.debug("foreground window details failed: %s", e)
    return Front(app=name, bundle_id=exe, window_title=title, pid=pid, window_class=cls)


def is_system_dialog(front: Front) -> bool:
    """True when `front` is a Windows permission/security prompt (UAC,
    credential prompt, SmartScreen, Windows Security, Settings) — where
    "Allow"-style clicks are refused and every action needs a confirm."""
    return (
        front.bundle_id.lower() in SYSTEM_DIALOG_APPS
        or front.window_class.lower() in SYSTEM_DIALOG_CLASSES
        or " ".join(front.window_title.lower().split()) in SYSTEM_DIALOG_TITLES
    )
