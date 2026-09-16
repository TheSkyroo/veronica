"""Default-input-device tracking for PortAudio.

macOS switches the default input device on its own (AirPods connect, a USB
mic is plugged in), but PortAudio snapshots the device table at init and a
stream opened afterwards still lands on the *old* default. `InputWatch`
polls CoreAudio for the current default input id; on a change the mic
reader closes its stream, `refresh_portaudio()` re-initialises PortAudio
(after closing any persistent output stream via `before`), and the next
open follows the new device.
"""

import contextlib
import ctypes
import logging
import threading
import time
from collections.abc import Callable

import sounddevice as sd

log = logging.getLogger("veronica.audio")

_COREAUDIO = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"
_SYSTEM_OBJECT = 1                                       # kAudioObjectSystemObject
_DEFAULT_INPUT = int.from_bytes(b"dIn ", "big")          # kAudioHardwarePropertyDefaultInputDevice
_SCOPE_GLOBAL = int.from_bytes(b"glob", "big")           # kAudioObjectPropertyScopeGlobal
_ELEMENT_MAIN = 0

# Hook the orchestrator/Recorder sets so a PortAudio re-init never runs
# while a capture stream is open (Recorder registers `lambda: self._capturing`).
busy: Callable[[], bool] = lambda: False

# Held while PortAudio is re-initialised and while any stream is being
# opened, so a re-init can't land between an open starting and the stream
# being live (Pa_Terminate under a live stream is undefined behaviour).
refresh_lock = threading.RLock()


class _PropertyAddress(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32), ("mScope", ctypes.c_uint32), ("mElement", ctypes.c_uint32)]


_getter_cache: list = []


def _coreaudio_getter():
    """AudioObjectGetPropertyData from CoreAudio, or None if unavailable."""
    if _getter_cache:
        return _getter_cache[0]
    try:
        lib = ctypes.cdll.LoadLibrary(_COREAUDIO)
        fn = lib.AudioObjectGetPropertyData
        fn.restype = ctypes.c_int32
        fn.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(_PropertyAddress), ctypes.c_uint32,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
        ]
    except Exception:
        fn = None
    _getter_cache.append(fn)
    return fn


def default_input_id(get: Callable | None = None) -> int | None:
    """CoreAudio's current default input device id, or None on any failure.
    `get` stands in for AudioObjectGetPropertyData (same call signature) in
    tests."""
    try:
        fn = get if get is not None else _coreaudio_getter()
        if fn is None:
            return None
        addr = _PropertyAddress(_DEFAULT_INPUT, _SCOPE_GLOBAL, _ELEMENT_MAIN)
        data = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(data))
        status = fn(_SYSTEM_OBJECT, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(data))
        if status != 0:
            return None
        return int(data.value)
    except Exception:
        return None


def refresh_portaudio(before: Callable[[], None] | None = None) -> None:
    """Re-initialise PortAudio so it re-reads the device list. `before` runs
    first (used to close the persistent Player output stream — PortAudio
    must not be terminated under an open stream). Every step is attempted;
    errors are logged, never raised."""
    with refresh_lock:
        if before is not None:
            try:
                before()
            except Exception:
                log.warning("pre-refresh hook failed", exc_info=True)
        try:
            sd._terminate()
        except Exception:
            log.warning("PortAudio terminate failed", exc_info=True)
        try:
            sd._initialize()
        except Exception:
            log.warning("PortAudio initialize failed", exc_info=True)


class InputWatch:
    """Polls the default input device id at most every `poll_s` seconds.
    `check(now)` is cheap enough to call per mic frame: it returns True (and
    calls `on_change(old, new)`) only when the id differs from the last
    observation; the very first observation never counts as a change."""

    def __init__(
        self,
        poll_s: float = 2.0,
        get_id: Callable[[], int | None] | None = None,
        on_change: Callable[[int | None, int | None], None] | None = None,
    ) -> None:
        self.poll_s = poll_s
        self._get_id = get_id
        self._on_change = on_change
        self.last: int | None = None
        self.previous: int | None = None   # id before the most recent change
        self._seen = False
        self._next_poll: float | None = None

    def _poll(self) -> int | None:
        get = self._get_id if self._get_id is not None else default_input_id
        with contextlib.suppress(Exception):
            return get()
        return None

    def check(self, now: float | None = None) -> bool:
        if now is None:
            now = time.monotonic()
        if self._next_poll is not None and now < self._next_poll:
            return False
        self._next_poll = now + self.poll_s
        current = self._poll()
        if not self._seen:
            self._seen = True
            self.last = current
            return False
        if current == self.last:
            return False
        self.previous, self.last = self.last, current
        if self._on_change is not None:
            self._on_change(self.previous, current)
        return True
