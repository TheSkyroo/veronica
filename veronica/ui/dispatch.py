"""The UI thread: one daemon worker that every HUD/Settings/tray mutation is
marshalled onto, plus the periodic timers the tray app runs (the 0.25 s
refresh, the 30 Hz HUD drain, the hourly update check).

pywebview's `webview.start()` owns the process's main thread and runs the
WinForms/WebView2 GUI loop on its own STA thread. Its window methods are
thread-safe, but they *block* until the GUI thread has run them
(`run_js`/`evaluate_js` even wait for the script's result), so producers —
the orchestrator's asyncio loop, pywebview's js_api call threads, pystray's
menu thread — never call them directly: they post here with `on_ui_thread`,
the calls run in order on this one thread, and nobody but it ever waits on
the GUI. It plays the part AppKit's main thread + `AppHelper.callAfter`
played on macOS: code running here may touch window/tray state freely.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable

log = logging.getLogger("veronica.ui.dispatch")


class Timer:
    """Handle for a repeating `Dispatcher.every` callback."""

    def __init__(self, interval: float, fn: Callable[[], None]) -> None:
        self.interval = interval
        self.fn = fn
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class Dispatcher:
    def __init__(self, name: str = "veronica-ui", clock: Callable[[], float] = time.monotonic) -> None:
        self._name = name
        self._clock = clock
        self._queue: queue.Queue = queue.Queue()
        self._timers: list[tuple[float, int, Timer]] = []
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stopped = False

    # -- public ------------------------------------------------------------------
    def is_ui_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def call(self, fn: Callable[[], None]) -> None:
        """Run `fn` on the UI thread, after everything already posted."""
        if self._stopped:
            return
        self._ensure_started()
        self._queue.put(fn)

    def every(self, interval: float, fn: Callable[[], None]) -> Timer:
        """Run `fn` on the UI thread every `interval` seconds (first run one
        interval from now) until the returned Timer is cancelled."""
        timer = Timer(interval, fn)
        with self._lock:
            heapq.heappush(self._timers, (self._clock() + interval, next(self._seq), timer))
        self._ensure_started()
        self._queue.put(None)   # wake the loop so it recomputes its timeout
        return timer

    def stop(self) -> None:
        self._stopped = True
        self._queue.put(None)

    # -- loop ---------------------------------------------------------------------
    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()

    def _next_timeout(self) -> float | None:
        with self._lock:
            if not self._timers:
                return None
            return max(0.0, self._timers[0][0] - self._clock())

    def _run_due_timers(self) -> None:
        while True:
            with self._lock:
                if not self._timers or self._timers[0][0] > self._clock():
                    return
                due, _, timer = heapq.heappop(self._timers)
                if timer.cancelled:
                    continue
                # Fixed-rate, but never trying to "catch up" a backlog after
                # a stall (a slow GUI round-trip): reschedule from now.
                heapq.heappush(self._timers, (max(due + timer.interval, self._clock()), next(self._seq), timer))
            self._invoke(timer.fn)

    @staticmethod
    def _invoke(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("UI callback failed")

    def _run(self) -> None:
        while not self._stopped:
            try:
                fn = self._queue.get(timeout=self._next_timeout())
            except queue.Empty:
                fn = None
            if fn is not None and not self._stopped:
                self._invoke(fn)
            if not self._stopped:
                self._run_due_timers()


_default = Dispatcher()


def on_ui_thread(fn: Callable[[], None]) -> None:
    """Run `fn` on the UI thread: inline when already there (keeps ordering
    with whatever the caller is in the middle of), else posted."""
    if _default.is_ui_thread():
        fn()
    else:
        _default.call(fn)


def every(interval: float, fn: Callable[[], None]) -> Timer:
    return _default.every(interval, fn)


def stop() -> None:
    _default.stop()
