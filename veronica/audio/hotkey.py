"""Global push-to-talk hotkey monitor: a Win32 low-level keyboard hook
(SetWindowsHookExW(WH_KEYBOARD_LL)) watching one key's down/up (default:
Right Ctrl). The hook needs no permission on Windows; if it can't be
installed (user32 unavailable, SetWindowsHookExW fails) `available` is
False and neither callback is ever invoked.

Low-level hooks see left/right-specific virtual-key codes (VK_RCONTROL,
VK_RMENU, ...), so holding the *other* Ctrl never masks our key's release.
Windows silently unhooks a low-level hook whose callback is too slow
(LowLevelHooksTimeout), so the callback only marshals onto the asyncio
loop and returns at once.
"""
import asyncio
import contextlib
import ctypes
import logging
import threading
from collections.abc import Callable

log = logging.getLogger("veronica.audio.hotkey")

# Windows virtual-key codes for the keys push-to-talk can sensibly use:
# modifiers that type nothing on their own. Right Alt is AltGr on many
# keyboard layouts and a lone Alt tap focuses the foreground app's menu
# bar, so Right Ctrl is the default.
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4      # left Alt
VK_RMENU = 0xA5      # right Alt (AltGr)
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_APPS = 0x5D       # context-menu key
VK_SCROLL = 0x91
VK_PAUSE = 0x13
VK_F13 = 0x7C

RIGHT_CTRL_KEYCODE = VK_RCONTROL

# Names accepted in place of a VK code (settings may store either).
KEY_NAMES = {
    "right_ctrl": VK_RCONTROL,
    "right_control": VK_RCONTROL,
    "left_ctrl": VK_LCONTROL,
    "left_control": VK_LCONTROL,
    "right_alt": VK_RMENU,
    "left_alt": VK_LMENU,
    "right_shift": VK_RSHIFT,
    "left_shift": VK_LSHIFT,
    "right_win": VK_RWIN,
    "left_win": VK_LWIN,
    "menu": VK_APPS,
    "apps": VK_APPS,
    "scroll_lock": VK_SCROLL,
    "pause": VK_PAUSE,
    "f13": VK_F13,
}

WH_KEYBOARD_LL = 13
HC_ACTION = 0
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
LLKHF_INJECTED = 0x10

_DOWN = (WM_KEYDOWN, WM_SYSKEYDOWN)
_UP = (WM_KEYUP, WM_SYSKEYUP)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", ctypes.c_uint32),
        ("scanCode", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.c_size_t),
        ("lParam", ctypes.c_ssize_t),
        ("time", ctypes.c_uint32),
        ("pt_x", ctypes.c_long),
        ("pt_y", ctypes.c_long),
    ]


def resolve_keycode(key: int | str) -> int:
    """A VK code from an int (returned as is) or a KEY_NAMES name; unknown
    names fall back to Right Ctrl (logged)."""
    if isinstance(key, int):
        return key
    vk = KEY_NAMES.get(str(key).strip().lower())
    if vk is None:
        log.warning("hotkey monitor: unknown key %r, using Right Ctrl", key)
        return RIGHT_CTRL_KEYCODE
    return vk


def _hookproc_type():
    # LRESULT CALLBACK LowLevelKeyboardProc(int, WPARAM, LPARAM); WINFUNCTYPE
    # (stdcall) only exists on Windows — CFUNCTYPE is identical on x64.
    functype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)
    return functype(ctypes.c_ssize_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t)


class _Win32:
    """The handful of user32/kernel32 calls the monitor makes, with argtypes
    set so 64-bit handles and LPARAMs aren't truncated."""

    def __init__(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.HOOKPROC = _hookproc_type()
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, self.HOOKPROC, ctypes.c_void_p, ctypes.c_uint32]
        user32.CallNextHookEx.restype = ctypes.c_ssize_t
        user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_ssize_t]
        user32.UnhookWindowsHookEx.restype = ctypes.c_int
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        user32.GetMessageW.restype = ctypes.c_int
        user32.GetMessageW.argtypes = [ctypes.POINTER(MSG), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        user32.PostThreadMessageW.restype = ctypes.c_int
        user32.PostThreadMessageW.argtypes = [ctypes.c_uint32, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetCurrentThreadId.restype = ctypes.c_uint32
        self.SetWindowsHookExW = user32.SetWindowsHookExW
        self.CallNextHookEx = user32.CallNextHookEx
        self.UnhookWindowsHookEx = user32.UnhookWindowsHookEx
        self.GetMessageW = user32.GetMessageW
        self.PostThreadMessageW = user32.PostThreadMessageW
        self.GetModuleHandleW = kernel32.GetModuleHandleW
        self.GetCurrentThreadId = kernel32.GetCurrentThreadId


def _import_win32() -> _Win32:
    return _Win32()


class HotkeyMonitor:
    """Watches one key's down/up globally (i.e. even when Veronica isn't
    the focused app) and calls `on_press`/`on_release` when it goes
    down/up. The hook lives on its own daemon thread running a Win32
    message loop (a low-level hook is called on the installing thread, and
    only while it pumps messages); callbacks are marshalled onto the
    asyncio loop that was running when `start()` was called, via
    `call_soon_threadsafe`. Auto-repeat key-downs and synthesized
    (injected) events — e.g. our own computer-control tools typing — are
    ignored.
    """

    _import_win32 = staticmethod(_import_win32)  # swapped in tests

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        keycode: int | str = RIGHT_CTRL_KEYCODE,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._keycode = resolve_keycode(keycode)
        self.available = True
        self._pressed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._win32: _Win32 | None = None
        self._hook = None
        self._proc = None        # the ctypes callback: must outlive the hook
        self._thread_id: int | None = None

    # -- lifecycle --------------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start monitoring on a daemon thread. `loop` is the asyncio loop
        callbacks should be marshalled onto (needed whenever start() is
        called from a thread other than the one running that loop, e.g.
        the tray UI's main thread starting a monitor for a background
        orchestrator loop) — defaults to the calling thread's running loop,
        or none at all (callbacks then run directly on the hook thread) if
        there isn't one. Blocks briefly (bounded) for the hook to be
        installed so `available` reflects reality by the time this
        returns."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        self._loop = loop
        ready = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, args=(ready,), name="hotkey-hook", daemon=True)
        self._thread.start()
        ready.wait(timeout=2)

    def stop(self) -> None:
        win32, tid = self._win32, self._thread_id
        if win32 is not None and tid is not None:
            with contextlib.suppress(Exception):
                win32.PostThreadMessageW(tid, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=1)

    # -- worker thread ------------------------------------------------------------
    def _thread_main(self, ready: threading.Event) -> None:
        ok = self._install_hook()
        ready.set()
        if not ok:
            return
        try:
            self._message_loop()
        finally:
            with contextlib.suppress(Exception):
                self._win32.UnhookWindowsHookEx(self._hook)
            self._hook = None
            # The key-up may never come now: don't leave push-to-talk held.
            if self._pressed:
                self._dispatch(False)

    def _install_hook(self) -> bool:
        try:
            win32 = self._import_win32()
        except Exception:
            log.warning("hotkey monitor: user32 unavailable", exc_info=True)
            self.available = False
            return False
        self._win32 = win32
        self._proc = win32.HOOKPROC(self._hook_proc)
        hook = None
        with contextlib.suppress(Exception):
            hook = win32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, win32.GetModuleHandleW(None), 0)
        if not hook:
            log.warning("hotkey monitor: could not install the keyboard hook (error %s)",
                        ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else "?")
            self.available = False
            return False
        self._hook = hook
        self._thread_id = win32.GetCurrentThreadId()
        self.available = True
        return True

    def _message_loop(self) -> None:
        # GetMessageW returns 0 on WM_QUIT (stop()) and -1 on error; the
        # hook is called from inside it, on this thread.
        msg = MSG()
        while self._win32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            pass

    # -- event handling -------------------------------------------------------
    def _hook_proc(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code == HC_ACTION and l_param:
            with contextlib.suppress(Exception):
                kb = KBDLLHOOKSTRUCT.from_address(l_param)
                self._on_key(int(w_param), int(kb.vkCode), int(kb.flags))
        try:
            return self._win32.CallNextHookEx(None, n_code, w_param, l_param)
        except Exception:
            return 0

    def _on_key(self, message: int, vk: int, flags: int) -> None:
        if vk != self._keycode or flags & LLKHF_INJECTED:
            return
        if message in _DOWN:
            self._dispatch(True)
        elif message in _UP:
            self._dispatch(False)

    def _dispatch(self, is_down: bool) -> None:
        if is_down == self._pressed:
            return
        self._pressed = is_down
        cb = self._on_press if is_down else self._on_release
        loop = self._loop
        if loop is None:
            cb()
            return
        try:
            loop.call_soon_threadsafe(cb)
        except RuntimeError:
            pass  # loop closed/closing
