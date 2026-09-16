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

    `watch` (an InputWatch) is checked per frame; when the default input
    device changes the reader closes its stream, re-initialises PortAudio
    (`before_refresh` runs first, to close the Player's output stream) and
    reopens on the new device — deferred while `devices.busy()` says a
    capture stream is open. `on_backlog` receives a callable returning the
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
        pending: tuple[int | None, int | None] | None = None
        try:
            stream = open_stream()
            while not done.is_set():
                data, overflowed = stream.read(chunk)
                if overflowed:
                    log.warning("%s mic overflow (reader thread stalled)", log_prefix)
                q.put(bytes(data))
                if watch is None:
                    continue
                if watch.check(time.monotonic()):
                    pending = (watch.previous, watch.last)
                if pending is None:
                    continue
                with devices.refresh_lock:
                    # busy() is checked under the lock: a Recorder capture
                    # that arms after this check can't open its stream until
                    # the refresh below has finished.
                    if devices.busy():
                        log.debug("deferring device refresh: capture in flight")
                        continue
                    log.info("input device changed (%s -> %s); reopening mic", *pending)
                    pending = None
                    stream.__exit__(None, None, None)
                    stream = None
                    devices.refresh_portaudio(before=before_refresh)
                    stream = open_stream()
        except Exception:
            log.exception("%s mic reader died", log_prefix)
        finally:
            if stream is not None:
                stream.__exit__(None, None, None)
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
