"""Global push-to-talk hotkey monitor: a Win32 low-level keyboard hook
(SetWindowsHookExW(WH_KEYBOARD_LL)) watching one hotkey's hold and release.

The hotkey is either a combo — modifiers plus one key, "win+space" by
default — or a single key ("right_ctrl"). A combo's key is swallowed while
it belongs to us, so Win+Space doesn't also switch the keyboard layout, and
a masking key is injected while Win/Alt are held so letting go of them
doesn't open the Start menu (or focus a menu bar). A single key is only
watched, never swallowed. Releasing either the key or a required modifier
ends the hold.

The hook needs no permission on Windows; if it can't be installed
(user32 unavailable, SetWindowsHookExW fails) `available` is False and
neither callback is ever invoked. Windows silently unhooks a low-level
hook whose callback is too slow (LowLevelHooksTimeout), so the callback
only marshals onto the asyncio loop (and asks its own thread to inject the
mask) and returns at once.
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
VK_SPACE = 0x20
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12       # Alt, side-neutral
# Unassigned VK that AutoHotkey also uses as its "menu mask": a key event
# between Win/Alt down and up stops Windows treating the release as a lone
# tap (Start menu / menu bar).
VK_MASK = 0xE8

RIGHT_CTRL_KEYCODE = VK_RCONTROL
DEFAULT_HOTKEY = "win+space"

# Modifier name -> every VK that counts as holding it.
MODIFIERS: dict[str, frozenset[int]] = {
    "win": frozenset({VK_LWIN, VK_RWIN}),
    "ctrl": frozenset({VK_LCONTROL, VK_RCONTROL, VK_CONTROL}),
    "alt": frozenset({VK_LMENU, VK_RMENU, VK_MENU}),
    "shift": frozenset({VK_LSHIFT, VK_RSHIFT, VK_SHIFT}),
}
_MODIFIER_ALIASES = {"windows": "win", "super": "win", "meta": "win", "start": "win",
                     "control": "ctrl", "option": "alt"}

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
    "space": VK_SPACE,
    "spacebar": VK_SPACE,
}
KEY_NAMES.update({f"f{n}": 0x70 + n - 1 for n in range(1, 25)})
KEY_NAMES.update({c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz0123456789"})

WH_KEYBOARD_LL = 13
HC_ACTION = 0
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_QUIT = 0x0012
WM_APP_MASK = 0x8000 + 1   # WM_APP + 1: "inject the menu mask now"
LLKHF_INJECTED = 0x10
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002

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


def parse_hotkey(spec: int | str) -> tuple[frozenset[str], int]:
    """(required modifiers, trigger VK) for "win+space", "ctrl+alt+p",
    "right_ctrl" or a bare VK code. Anything unparseable (an unknown key,
    a combo of modifiers only, "fn" — which Windows never sees) falls back
    to DEFAULT_HOTKEY, logged."""
    if isinstance(spec, int):
        return frozenset(), spec
    parts = [p.strip().lower() for p in str(spec).replace(" ", "").split("+") if p.strip()]
    mods = {_MODIFIER_ALIASES.get(p, p) for p in parts[:-1]}
    key = parts[-1] if parts else ""
    if (parts and mods <= MODIFIERS.keys() and key in KEY_NAMES
            and not (mods and _MODIFIER_ALIASES.get(key, key) in MODIFIERS)):
        return frozenset(mods), KEY_NAMES[key]
    if spec != DEFAULT_HOTKEY:
        log.warning("hotkey monitor: can't use hotkey %r, using %s", spec, DEFAULT_HOTKEY)
    return parse_hotkey(DEFAULT_HOTKEY)


def describe_hotkey(spec: int | str) -> str:
    """"Win+Space", "Right Ctrl": the hotkey as the UI should name it."""
    mods, vk = parse_hotkey(spec)
    names = {v: k for k, v in KEY_NAMES.items() if not k.endswith("bar") and k not in ("apps", "right_control", "left_control")}
    key = names.get(vk, f"VK {vk:#x}").replace("_", " ").title()
    order = [m for m in ("ctrl", "alt", "shift", "win") if m in mods]
    return "+".join([m.title() for m in order] + [key])


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_uint16), ("wScan", ctypes.c_uint16), ("dwFlags", ctypes.c_uint32),
                ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t)]


class _MOUSEINPUT(ctypes.Structure):   # only here so INPUT has the size Windows expects
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_uint32),
                ("dwFlags", ctypes.c_uint32), ("time", ctypes.c_uint32), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_uint32), ("u", _U)]


def _mask_inputs() -> "ctypes.Array[_INPUT]":
    """VK_MASK down + up, as one SendInput batch."""
    arr = (_INPUT * 2)()
    for i, flags in enumerate((0, KEYEVENTF_KEYUP)):
        arr[i].type = INPUT_KEYBOARD
        arr[i].ki = _KEYBDINPUT(wVk=VK_MASK, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return arr


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
        user32.SendInput.restype = ctypes.c_uint
        user32.SendInput.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
        self.SendInput = user32.SendInput
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
    """Watches one hotkey's hold/release globally (i.e. even when Veronica isn't
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
        hotkey: int | str = DEFAULT_HOTKEY,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._mods, self._keycode = parse_hotkey(hotkey)
        self.description = describe_hotkey(hotkey)
        self.available = True
        self._pressed = False
        self._held: set[int] = set()     # modifier VKs physically down right now
        self._swallowing = False         # the trigger's down was ours: eat its up too
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
            if msg.message == WM_APP_MASK:
                self._send_mask()

    def _send_mask(self) -> None:
        """Inject VK_MASK (flagged as injected, so our own hook skips it)."""
        with contextlib.suppress(Exception):
            inputs = _mask_inputs()
            self._win32.SendInput(len(inputs), ctypes.byref(inputs), ctypes.sizeof(_INPUT))

    def _request_mask(self) -> None:
        # Not from inside the hook callback itself: post to our own thread,
        # which injects it on its next turn of the message loop.
        if self._win32 is not None and self._thread_id is not None:
            with contextlib.suppress(Exception):
                self._win32.PostThreadMessageW(self._thread_id, WM_APP_MASK, 0, 0)

    # -- event handling -------------------------------------------------------
    def _hook_proc(self, n_code: int, w_param: int, l_param: int) -> int:
        swallow = False
        if n_code == HC_ACTION and l_param:
            with contextlib.suppress(Exception):
                kb = KBDLLHOOKSTRUCT.from_address(l_param)
                swallow = self._on_key(int(w_param), int(kb.vkCode), int(kb.flags))
        if swallow:
            return 1     # non-zero without chaining: Windows drops the event
        try:
            return self._win32.CallNextHookEx(None, n_code, w_param, l_param)
        except Exception:
            return 0

    def _on_key(self, message: int, vk: int, flags: int) -> bool:
        """Track the key; True when this event must be swallowed."""
        if flags & LLKHF_INJECTED:
            return False
        is_down, is_up = message in _DOWN, message in _UP
        if any(vk in MODIFIERS[m] for m in MODIFIERS) and vk != self._keycode:
            if is_down:
                self._held.add(vk)
            elif is_up:
                self._held.discard(vk)
                if self._pressed and not self._mods_held():
                    self._dispatch(False)     # let go of Win first: the hold ends
            return False
        if vk != self._keycode:
            return False
        if not self._mods:                    # a single key: watch, never swallow
            if is_down:
                self._dispatch(True)
            elif is_up:
                self._dispatch(False)
            return False
        if is_down:
            if self._swallowing:
                return True                   # auto-repeat of a hold we own
            if self._mods_held():
                self._swallowing = True
                if self._mods & {"win", "alt"}:
                    self._request_mask()
                self._dispatch(True)
                return True
            return False
        if is_up and self._swallowing:
            self._swallowing = False
            self._dispatch(False)
            return True
        return False

    def _mods_held(self) -> bool:
        return all(self._held & MODIFIERS[m] for m in self._mods)

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
