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
_COREFOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
_SYSTEM_OBJECT = 1                                       # kAudioObjectSystemObject
_DEFAULT_INPUT = int.from_bytes(b"dIn ", "big")          # kAudioHardwarePropertyDefaultInputDevice
_OBJECT_NAME = int.from_bytes(b"lnam", "big")            # kAudioObjectPropertyName (CFStringRef)
_CF_UTF8 = 0x08000100                                    # kCFStringEncodingUTF8
_SCOPE_GLOBAL = int.from_bytes(b"glob", "big")           # kAudioObjectPropertyScopeGlobal
_ELEMENT_MAIN = 0

# -- shared state -----------------------------------------------------------
# Kept at module level (not per InputWatch) because wake readers come and go:
# WhisperWake/WakeWord close their frame generator every time wait() returns,
# so a per-reader baseline would miss any change that happens between readers
# (e.g. AirPods connecting during a capture/turn).
last_input_id: int | None = None      # most recent poll
initialised_for: int | None = None    # default input id PortAudio was last (re)initialised for
pending: bool = False                 # last_input_id != initialised_for: a refresh is owed
generation: int = 0                   # bumped by every refresh_portaudio(); streams opened
                                      # under an older generation are dead (Pa_Terminate closes them)
_baselined = False
# Callbacks run at the end of every successful refresh_portaudio() (the
# input-volume guard re-checks the level right after a device switch).
_change_subscribers: list[Callable[[], None]] = []

# Held while PortAudio is re-initialised and while any stream is being
# opened or read, so a re-init can't land between an open starting and the
# stream being live (Pa_Terminate under a live stream is undefined behaviour).
refresh_lock = threading.RLock()


def _never_busy() -> bool:
    return False


_busy_fn: Callable[[], bool] = _never_busy


def register_busy(fn: Callable[[], bool]) -> None:
    """Install the hook that says whether a capture stream is open (Recorder
    registers `lambda: self._capturing`); a refresh is deferred while it's True."""
    global _busy_fn
    _busy_fn = fn


def reset_busy() -> None:
    global _busy_fn
    _busy_fn = _never_busy


def busy() -> bool:
    return bool(_busy_fn())


def reset() -> None:
    """Forget all shared state (tests)."""
    global last_input_id, initialised_for, pending, generation, _baselined
    last_input_id = initialised_for = None
    pending = False
    generation = 0
    _baselined = False
    _change_subscribers.clear()
    reset_busy()


def subscribe_change(fn: Callable[[], None]) -> None:
    """Run `fn` after every successful PortAudio refresh (i.e. right after the
    mic has followed a default-input switch). Callbacks run on a short
    daemon thread, never on the refreshing thread: the real callers hold
    `refresh_lock` around the whole refresh, and a subscriber may shell out
    (the input-volume guard runs osascript). Errors are logged, not raised."""
    _change_subscribers.append(fn)


def _run_subscribers() -> None:
    for fn in list(_change_subscribers):
        try:
            fn()
        except Exception:
            log.warning("device change callback failed", exc_info=True)


def observe(current: int | None) -> bool:
    """Record a polled default input id. The first observation is the
    baseline PortAudio was initialised for. Returns True when the id differs
    from the previous observation; `pending` says whether a refresh is owed
    (it clears itself if the device switches back). A None poll after a
    valid baseline means "couldn't read it" (a transient CoreAudio hiccup),
    not "no device": the last id is kept and nothing changes."""
    global last_input_id, initialised_for, pending, _baselined
    if not _baselined:
        _baselined = True
        last_input_id = initialised_for = current
        return False
    if current is None and last_input_id is not None:
        return False
    changed = current != last_input_id
    last_input_id = current
    pending = current != initialised_for
    return changed


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


_cf_to_str_cache: list = []


def _cfstring_to_str():
    """A CFStringRef -> str converter (releases the ref), or None if
    CoreFoundation is unavailable."""
    if _cf_to_str_cache:
        return _cf_to_str_cache[0]
    try:
        cf = ctypes.cdll.LoadLibrary(_COREFOUNDATION)
        get_c = cf.CFStringGetCString
        get_c.restype = ctypes.c_bool
        get_c.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        release = cf.CFRelease
        release.restype = None
        release.argtypes = [ctypes.c_void_p]

        def to_str(ref: int) -> str | None:
            try:
                buf = ctypes.create_string_buffer(256)
                if not get_c(ref, buf, 256, _CF_UTF8):
                    return None
                return buf.value.decode("utf-8", "replace")
            finally:
                release(ref)
    except Exception:
        to_str = None
    _cf_to_str_cache.append(to_str)
    return to_str


def default_input_name(
    get: Callable | None = None,
    device_id: int | None = None,
    to_str: Callable[[int], str | None] | None = None,
) -> str | None:
    """The current default input device's name ("MacBook Air Microphone",
    "AirPods Pro"), or None on any failure. CoreAudio's kAudioObjectPropertyName
    is a CFStringRef (owned by the caller), read with CFStringGetCString and
    released. `get`/`device_id`/`to_str` are test injection points."""
    try:
        dev = device_id if device_id is not None else default_input_id(get)
        if dev is None:
            return None
        fn = get if get is not None else _coreaudio_getter()
        conv = to_str if to_str is not None else _cfstring_to_str()
        if fn is None or conv is None:
            return None
        addr = _PropertyAddress(_OBJECT_NAME, _SCOPE_GLOBAL, _ELEMENT_MAIN)
        ref = ctypes.c_void_p(0)
        size = ctypes.c_uint32(ctypes.sizeof(ref))
        status = fn(dev, ctypes.byref(addr), 0, None, ctypes.byref(size), ctypes.byref(ref))
        if status != 0 or not ref.value:
            return None
        return conv(ref.value) or None
    except Exception:
        return None


def refresh_portaudio(before: Callable[[], None] | None = None) -> None:
    """Re-initialise PortAudio so it re-reads the device list. `before` runs
    first (used to close the persistent Player output stream — PortAudio
    must not be terminated under an open stream). `before`/terminate
    failures are logged and skipped; an initialize failure is logged at
    ERROR and re-raised — the caller has no working audio to fall back to."""
    global initialised_for, pending, generation
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
        # Whether or not init succeeds, every stream opened before is dead
        # (Pa_Terminate ran) and the device table is whatever it is now.
        generation += 1
        initialised_for = last_input_id
        pending = False
        try:
            sd._initialize()
        except Exception:
            # Without PortAudio there is no audio at all: surface it (the
            # mic reader dies, wake.wait() raises, the orchestrator backs
            # off and retries) rather than carrying on silently.
            log.error("PortAudio initialize failed", exc_info=True)
            raise
    # Off this thread: mic.refresh_if_pending / play._ensure_stream call us
    # inside their own `with refresh_lock:` (RLock), so anything run inline
    # here would still hold the lock and stall every other waiter for as
    # long as the subscriber takes (osascript: up to 2x5 s worst case).
    if _change_subscribers:
        threading.Thread(target=_run_subscribers, name="audio-change-subscribers", daemon=True).start()


def _snapshot_baseline(get: Callable[[], int | None] = default_input_id) -> None:
    """Take the baseline at import time — right after `import sounddevice`
    initialised PortAudio — so a default-input change during warmup (model
    loads take seconds; AirPods connect meanwhile) is already a change by the
    time the first mic reader's InputWatch polls, instead of becoming the
    baseline. Guarded: never raises, and only baselines if nothing has yet."""
    if _baselined:
        return
    try:
        observe(get())
    except Exception:
        log.debug("could not snapshot default input device at import", exc_info=True)


_snapshot_baseline()


class InputWatch:
    """Polls the default input device id at most every `poll_s` seconds and
    feeds `observe()`. `check(now)` is cheap enough to call per mic frame: it
    returns True (and calls `on_change(old, new)`) when the id differs from
    the previous observation; the very first observation (module-wide, not
    per instance) never counts as a change. Whether a refresh is owed is
    `devices.pending`."""

    def __init__(
        self,
        poll_s: float = 2.0,
        get_id: Callable[[], int | None] | None = None,
        on_change: Callable[[int | None, int | None], None] | None = None,
    ) -> None:
        self.poll_s = poll_s
        self._get_id = get_id
        self._on_change = on_change
        self._next_poll: float | None = None

    @property
    def last(self) -> int | None:
        return last_input_id

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
        old = last_input_id
        if not observe(self._poll()):
            return False
        if self._on_change is not None:
            self._on_change(old, last_input_id)
        return True
