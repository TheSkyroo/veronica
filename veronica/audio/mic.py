"""Shared mic frame source for the wake-word engines."""

import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator

import sounddevice as sd

from veronica.audio import devices
from veronica.config import Settings

log = logging.getLogger("veronica.audio")


def mic_frames(
    settings: Settings,
    chunk: int,
    log_prefix: str,
    *,
    watch: devices.InputWatch | None = None,
    before_refresh: Callable[[], None] | None = None,
    on_backlog: Callable[[Callable[[], int]], None] | None = None,
) -> Iterator[bytes]:
    """Mic frames of `chunk` samples, read on a dedicated thread into a queue.

    The wake loop transcribes a window every hop and tiny.en can take
    longer than one hop under CPU load; reading the device inline would
    let PortAudio's ring buffer overflow during that stall and silently
    drop audio — chopping the wake word in half. The reader thread keeps
    draining the device no matter how long a transcription takes, so the
    loop only ever falls behind, never loses frames. Closing this generator
    stops the thread and the stream.

    `watch` (an InputWatch) is checked at start and per frame; when the
    default input device differs from the one PortAudio was initialised for
    (`devices.pending`) the reader closes its stream, re-initialises
    PortAudio (`before_refresh` runs first, to close the Player's output
    stream) and reopens on the new device — deferred while `devices.busy()`
    says a capture stream is open. That state lives in `veronica.audio.
    devices`, so a change seen while no reader is running, or deferred when
    one stopped, is applied by the next reader before its first open. `on_backlog` receives a callable returning the
    number of queued frames when the reader starts, and one returning 0
    when it stops.
    """
    q: queue.Queue[bytes | None] = queue.Queue()
    done = threading.Event()

    def open_stream():
        with devices.refresh_lock:
            stream = sd.RawInputStream(samplerate=settings.sample_rate, channels=1, dtype="int16", blocksize=chunk)
            stream.__enter__()
            return stream

    def reader() -> None:
        stream = None
        opened_gen = -1
        deferred_logged = False

        def open_fresh():
            nonlocal stream, opened_gen
            stream = open_stream()
            opened_gen = devices.generation

        def close_current():
            nonlocal stream
            if stream is None:
                return
            if devices.generation == opened_gen:
                stream.__exit__(None, None, None)
            # else: PortAudio was re-initialised under it (Pa_Terminate
            # closes every open stream), so the handle is already dead —
            # just drop it rather than closing a dangling pointer.
            stream = None

        def refresh_if_pending() -> None:
            """Under the lock: if a refresh is owed and no capture is open,
            close our stream, refresh PortAudio and reopen; otherwise (busy)
            keep the current stream and leave `pending` set for a later
            check. Logs the deferral once per pending change."""
            nonlocal deferred_logged
            if not devices.pending:
                deferred_logged = False
                return
            with devices.refresh_lock:
                if devices.busy():
                    if not deferred_logged:
                        log.debug("deferring device refresh: capture in flight")
                        deferred_logged = True
                    return
                log.info("input device changed (%s -> %s); reopening mic",
                         devices.initialised_for, devices.last_input_id)
                close_current()
                devices.refresh_portaudio(before=before_refresh)
                open_fresh()
                deferred_logged = False

        try:
            if watch is not None:
                # A change that happened while no reader was alive (or one
                # deferred when the last reader stopped) is still pending:
                # refresh before the first open so it lands on the new device.
                watch.check(time.monotonic())
                refresh_if_pending()
            if stream is None:
                open_fresh()
            while not done.is_set():
                # Reading under the lock means a refresh from another thread
                # (Player's open-retry path) can only run between reads, and
                # the generation check below then reopens on the new device.
                with devices.refresh_lock:
                    if devices.generation != opened_gen:
                        log.info("%s mic stream invalidated by PortAudio refresh; reopening", log_prefix)
                        close_current()
                        open_fresh()
                    data, overflowed = stream.read(chunk)
                if overflowed:
                    log.warning("%s mic overflow (reader thread stalled)", log_prefix)
                q.put(bytes(data))
                if watch is None:
                    continue
                watch.check(time.monotonic())
                refresh_if_pending()
        except Exception:
            log.exception("%s mic reader died", log_prefix)
        finally:
            close_current()
            q.put(None)

    t = threading.Thread(target=reader, name=f"{log_prefix}-mic", daemon=True)
    t.start()
    if on_backlog is not None:
        on_backlog(q.qsize)
    try:
        while True:
            frame = q.get()
            if frame is None:
                return
            yield frame
    finally:
        done.set()
        if on_backlog is not None:
            on_backlog(lambda: 0)
