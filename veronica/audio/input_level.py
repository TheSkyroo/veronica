"""Input-volume floor guard.

macOS keeps lowering the system input volume behind our back: call apps
with auto-gain (Zoom, Meet, FaceTime) and default-device switches leave it
at 27–33 %, and far-field wake detection dies at that level. `InputLevelGuard`
reads the input volume via osascript and raises it back to a configurable
floor (never lowers it). It runs periodically from the orchestrator loop
and once, forced, right after every device switch.
"""

import asyncio
import logging
import subprocess
import threading
import time
from collections.abc import Callable

from veronica.audio import devices

log = logging.getLogger("veronica.audio")

_TIMEOUT_S = 5


def get_input_volume(run: Callable = subprocess.run) -> int | None:
    """The Mac's input volume (0-100) via osascript; None on any failure
    (osascript missing, timeout, non-zero exit, non-integer output)."""
    try:
        proc = run(
            ["osascript", "-e", "input volume of (get volume settings)"],
            capture_output=True, text=True, timeout=_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return None
        return int(str(proc.stdout).strip())
    except Exception:
        return None


def set_input_volume(level: int, run: Callable = subprocess.run) -> bool:
    """Set the Mac's input volume to `level` (clamped to 0-100). False on failure."""
    level = max(0, min(100, int(level)))
    try:
        proc = run(
            ["osascript", "-e", f"set volume input volume {level}"],
            capture_output=True, text=True, timeout=_TIMEOUT_S,
        )
        return proc.returncode == 0
    except Exception:
        return False


class InputLevelGuard:
    """Raise the input volume to `floor()` whenever it is found below it.

    `check()` is throttled to one read per `interval_s` unless `force` is
    set (the device-switch hook forces one), and is a no-op while
    `floor() <= 0`. Safe to call from the mic reader thread: it only runs
    subprocesses, under a lock so two callers can't race the same fix.
    """

    def __init__(
        self,
        floor: Callable[[], int],
        *,
        get: Callable[[], int | None] = get_input_volume,
        set: Callable[[int], bool] = set_input_volume,
        device_name: Callable[[], str | None] = devices.default_input_name,
        on_corrected: Callable[[int, int, str], None] | None = None,
        interval_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._floor = floor
        self._get = get
        self._set = set
        self._device_name = device_name
        self._on_corrected = on_corrected
        self.interval_s = interval_s
        self._clock = clock
        self._lock = threading.Lock()
        self._next: float | None = None

    def check(self, force: bool = False) -> int | None:
        """Read the input volume and raise it to the floor if it is lower.
        Returns the level it was raised to, or None (nothing to do, throttled,
        disabled, or the read/set failed)."""
        with self._lock:
            floor = int(self._floor() or 0)
            if floor <= 0:
                return None
            now = self._clock()
            if not force and self._next is not None and now < self._next:
                return None
            self._next = now + self.interval_s
            vol = self._get()
            if vol is None or vol >= floor:
                return None
            if not self._set(floor):
                log.warning("could not raise input volume %d → %d", vol, floor)
                return None
            name = self._name()
            log.info("input volume %d → %d (%s)", vol, floor, name)
            if self._on_corrected is not None:
                try:
                    self._on_corrected(vol, floor, name)
                except Exception:
                    log.exception("input volume on_corrected failed")
            return floor

    def _name(self) -> str:
        try:
            return self._device_name() or "unknown input"
        except Exception:
            return "unknown input"


async def run_periodic(guard: InputLevelGuard, stop: asyncio.Event) -> None:
    """One forced check now (startup), then a throttled check every
    `guard.interval_s` until `stop` is set. Checks run on a thread (they
    shell out to osascript); a failing check is logged and the loop goes on."""
    force = True
    while not stop.is_set():
        try:
            await asyncio.to_thread(guard.check, force)
        except Exception:
            log.warning("input volume check failed", exc_info=True)
        force = False
        try:
            await asyncio.wait_for(stop.wait(), timeout=guard.interval_s)
        except TimeoutError:
            pass
