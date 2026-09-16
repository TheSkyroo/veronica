"""Global push-to-talk hotkey monitor: a Quartz CGEventTap watching
kCGEventFlagsChanged for one modifier keycode (default: Right Option, 61).
Requires Accessibility (Input Monitoring) permission — if the tap can't be
created (permission not granted), `available` is False and neither callback
is ever invoked; callers should offer a menu item pointing at the
Accessibility privacy pane.
"""
import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable

log = logging.getLogger("veronica.audio.hotkey")

RIGHT_OPTION_KEYCODE = 61

# Device-specific modifier bits (IOKit NX_DEVICE*KEYMASK) carried in
# CGEventFlags, keyed by the modifier's keycode. These distinguish the left
# and right instances of a modifier, which the generic
# kCGEventFlagMaskAlternate & co. do not: with left-Option held down, a
# right-Option release still leaves the generic Alternate bit set, and a
# monitor reading only that bit would think the key was still down.
DEVICE_FLAG_MASKS = {
    54: 0x10,    # right command  (NX_DEVICERCMDKEYMASK)
    55: 0x08,    # left command   (NX_DEVICELCMDKEYMASK)
    56: 0x02,    # left shift     (NX_DEVICELSHIFTKEYMASK)
    58: 0x20,    # left option    (NX_DEVICELALTKEYMASK)
    59: 0x01,    # left control   (NX_DEVICELCTLKEYMASK)
    60: 0x04,    # right shift    (NX_DEVICERSHIFTKEYMASK)
    61: 0x40,    # right option   (NX_DEVICERALTKEYMASK)
    62: 0x2000,  # right control  (NX_DEVICERCTLKEYMASK)
}

# CGEventType values the OS posts *to the tap's callback* when it has
# disabled the tap: after the callback was too slow for too long (Timeout)
# or because of user input (UserInput, e.g. a secure-input field). A
# disabled tap never fires again unless re-enabled, which would silently
# kill push-to-talk for the rest of the session.
TAP_DISABLED_BY_TIMEOUT = 0xFFFFFFFE
TAP_DISABLED_BY_USER_INPUT = 0xFFFFFFFF


def _import_quartz():
    import Quartz
    return Quartz


class HotkeyMonitor:
    """Watches one modifier key's flags-changed events globally (i.e. even
    when Veronica isn't the focused app) and calls `on_press`/`on_release`
    when it goes down/up. The CGEventTap runs on its own CFRunLoop in a
    daemon thread; callbacks are marshalled onto the asyncio loop that was
    running when `start()` was called, via `call_soon_threadsafe`.
    """

    _import_quartz = staticmethod(_import_quartz)  # swapped in tests

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        keycode: int = RIGHT_OPTION_KEYCODE,
    ) -> None:
        self._on_press = on_press
        self._on_release = on_release
        self._keycode = keycode
        self._device_mask = DEVICE_FLAG_MASKS.get(keycode)
        self.available = True
        self._pressed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._run_loop = None
        self._quartz = None
        self._tap = None
        self._reenable_logged = False
        self.reenable_count = 0

    # -- lifecycle --------------------------------------------------------------
    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start monitoring on a daemon thread. `loop` is the asyncio loop
        callbacks should be marshalled onto (needed whenever start() is
        called from a thread other than the one running that loop, e.g.
        the menu bar's AppKit main thread starting a monitor for a
        background orchestrator loop) — defaults to the calling thread's
        running loop, or none at all (callbacks then run directly on the
        tap thread) if there isn't one. Blocks briefly (bounded) for the
        tap to be set up so `available` reflects reality by the time this
        returns."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        self._loop = loop
        ready = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, args=(ready,), daemon=True)
        self._thread.start()
        ready.wait(timeout=2)

    def stop(self) -> None:
        quartz, run_loop = self._quartz, self._run_loop
        if quartz is not None and run_loop is not None:
            with contextlib.suppress(Exception):
                quartz.CFRunLoopStop(run_loop)
        if self._thread is not None:
            self._thread.join(timeout=1)

    # -- worker thread ------------------------------------------------------------
    def _thread_main(self, ready: threading.Event) -> None:
        ok = self._setup_tap()
        ready.set()
        if not ok:
            return
        self._quartz.CFRunLoopRun()

    def _setup_tap(self) -> bool:
        try:
            quartz = self._import_quartz()
        except Exception:
            log.warning("hotkey monitor: Quartz unavailable")
            self.available = False
            return False
        self._quartz = quartz
        tap = quartz.CGEventTapCreate(
            quartz.kCGSessionEventTap,
            quartz.kCGHeadInsertEventTap,
            quartz.kCGEventTapOptionListenOnly,
            quartz.CGEventMaskBit(quartz.kCGEventFlagsChanged),
            self._callback,
            None,
        )
        if tap is None:
            log.warning(
                "hotkey monitor: could not create event tap — grant Accessibility "
                "(Input Monitoring) permission to Veronica in System Settings"
            )
            self.available = False
            return False
        self._tap = tap
        source = quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._run_loop = quartz.CFRunLoopGetCurrent()
        quartz.CFRunLoopAddSource(self._run_loop, source, quartz.kCFRunLoopCommonModes)
        quartz.CGEventTapEnable(tap, True)
        self.available = True
        return True

    # -- event handling -------------------------------------------------------
    def _callback(self, proxy, event_type, event, refcon):
        quartz = self._quartz
        if event_type in (TAP_DISABLED_BY_TIMEOUT, TAP_DISABLED_BY_USER_INPUT):
            self._reenable_tap(event_type)
            return event
        if event_type == quartz.kCGEventFlagsChanged:
            keycode = quartz.CGEventGetIntegerValueField(event, quartz.kCGKeyboardEventKeycode)
            if keycode == self._keycode:
                flags = quartz.CGEventGetFlags(event)
                mask = self._device_mask if self._device_mask is not None else quartz.kCGEventFlagMaskAlternate
                self._dispatch(bool(flags & mask))
        return event

    def _reenable_tap(self, event_type) -> None:
        """The OS disabled our tap (timeout or user input); turn it back on
        so push-to-talk keeps working. Logged once per monitor so a flaky
        tap doesn't spam the log."""
        if self._tap is None or self._quartz is None:
            return
        self.reenable_count += 1
        if not self._reenable_logged:
            why = "timeout" if event_type == TAP_DISABLED_BY_TIMEOUT else "user input"
            log.warning("hotkey monitor: event tap disabled by %s; re-enabling", why)
            self._reenable_logged = True
        with contextlib.suppress(Exception):
            self._quartz.CGEventTapEnable(self._tap, True)

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
