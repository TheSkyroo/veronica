"""Default-input-device tracking for PortAudio.

Windows switches the default input device on its own (a Bluetooth headset
connects, a USB mic is plugged in), but PortAudio snapshots the device
table at init and a stream opened afterwards (WASAPI/MME alike) still lands
on the *old* default. `InputWatch` polls Core Audio (MMDevice API) for the
current default capture endpoint id; on a change the mic
reader closes its stream, `refresh_portaudio()` re-initialises PortAudio
(after closing any persistent output stream via `before`), and the next
open follows the new device.
"""

import contextlib
import logging
import threading
import time
from collections.abc import Callable

import sounddevice as sd

log = logging.getLogger("veronica.audio")

# -- shared state -----------------------------------------------------------
# Kept at module level (not per InputWatch) because wake readers come and go:
# WhisperWake/WakeWord close their frame generator every time wait() returns,
# so a per-reader baseline would miss any change that happens between readers
# (e.g. a headset connecting during a capture/turn).
last_input_id: str | None = None      # most recent poll (endpoint id string)
initialised_for: str | None = None    # default input id PortAudio was last (re)initialised for
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
    (the input-volume guard makes COM calls). Errors are logged, not raised."""
    _change_subscribers.append(fn)


def _run_subscribers() -> None:
    for fn in list(_change_subscribers):
        try:
            fn()
        except Exception:
            log.warning("device change callback failed", exc_info=True)


def observe(current: str | None) -> bool:
    """Record a polled default input id. The first observation is the
    baseline PortAudio was initialised for. Returns True when the id differs
    from the previous observation; `pending` says whether a refresh is owed
    (it clears itself if the device switches back). A None poll after a
    valid baseline means "couldn't read it" (a transient Core Audio hiccup),
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


# PKEY_Device_FriendlyName ("Microphone (Realtek(R) Audio)") in an
# endpoint's property store: {a45c254e-df1c-4efd-8020-67d146a850e0}, 14.
_FRIENDLY_NAME_FMTID = "{A45C254E-DF1C-4EFD-8020-67D146A850E0}"
_FRIENDLY_NAME_PID = 14
_STGM_READ = 0

_com_ready = threading.local()


def com_init() -> None:
    """Make sure COM is initialised on the calling thread (once per thread).
    Core Audio is COM: every thread that touches it needs CoInitialize, and
    the mic reader, the input-volume guard and the subscriber threads are
    all workers. Never uninitialised: the threads are few and the interface
    pointers comtypes hands out are released lazily (__del__), possibly
    after a CoUninitialize would already have run. Already-initialised (in
    either apartment model) is fine; any failure is left for the actual
    COM call to report."""
    if getattr(_com_ready, "done", False):
        return
    _com_ready.done = True
    try:
        import comtypes
        comtypes.CoInitialize()
    except Exception:
        log.debug("CoInitialize failed", exc_info=True)


def default_capture_device():
    """The default capture endpoint (eCapture, eConsole) as a pycaw/comtypes
    IMMDevice. Raises on any failure (no capture device, pycaw missing)."""
    com_init()
    import comtypes
    from pycaw.constants import CLSID_MMDeviceEnumerator, EDataFlow, ERole
    from pycaw.pycaw import IMMDeviceEnumerator

    enumerator = comtypes.CoCreateInstance(
        CLSID_MMDeviceEnumerator, IMMDeviceEnumerator, comtypes.CLSCTX_INPROC_SERVER,
    )
    return enumerator.GetDefaultAudioEndpoint(EDataFlow.eCapture.value, ERole.eConsole.value)


def default_input_id(device: Callable | None = None) -> str | None:
    """The default capture endpoint's id string (e.g.
    "{0.0.1.00000000}.{8d3c...}"), or None on any failure (no input device,
    COM/pycaw unavailable). `device` stands in for `default_capture_device`
    (returns an IMMDevice-like object) in tests."""
    try:
        dev = (device or default_capture_device)()
        if dev is None:
            return None
        return str(dev.GetId()) or None
    except Exception:
        return None


def _friendly_name(dev) -> str | None:
    """PKEY_Device_FriendlyName from an IMMDevice's property store."""
    store = dev.OpenPropertyStore(_STGM_READ)
    for i in range(store.GetCount()):
        key = store.GetAt(i)
        if str(key.fmtid).upper() != _FRIENDLY_NAME_FMTID or int(key.pid) != _FRIENDLY_NAME_PID:
            continue
        value = store.GetValue(key)
        try:
            name = value.GetValue()
        finally:
            with contextlib.suppress(Exception):
                value.clear()
        return str(name) if name else None
    return None


def default_input_name(device: Callable | None = None) -> str | None:
    """The current default input device's friendly name ("Microphone
    (Realtek(R) Audio)", "Headset (WH-1000XM4 Hands-Free)"), or None on any
    failure. `device` stands in for `default_capture_device` in tests."""
    try:
        dev = (device or default_capture_device)()
        if dev is None:
            return None
        return _friendly_name(dev) or None
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
    # long as the subscriber takes (a COM call into the audio service can hang).
    if _change_subscribers:
        threading.Thread(target=_run_subscribers, name="audio-change-subscribers", daemon=True).start()


def _snapshot_baseline(get: Callable[[], str | None] = default_input_id) -> None:
    """Take the baseline at import time — right after `import sounddevice`
    initialised PortAudio — so a default-input change during warmup (model
    loads take seconds; a headset connects meanwhile) is already a change by the
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
    """Polls the default input endpoint id at most every `poll_s` seconds and
    feeds `observe()`. `check(now)` is cheap enough to call per mic frame: it
    returns True (and calls `on_change(old, new)`) when the id differs from
    the previous observation; the very first observation (module-wide, not
    per instance) never counts as a change. Whether a refresh is owed is
    `devices.pending`."""

    def __init__(
        self,
        poll_s: float = 2.0,
        get_id: Callable[[], str | None] | None = None,
        on_change: Callable[[str | None, str | None], None] | None = None,
    ) -> None:
        self.poll_s = poll_s
        self._get_id = get_id
        self._on_change = on_change
        self._next_poll: float | None = None

    @property
    def last(self) -> str | None:
        return last_input_id

    def _poll(self) -> str | None:
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
