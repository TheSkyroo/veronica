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

# Mic overflow warnings are summarised at most this often (see the read loop).
OVERFLOW_LOG_EVERY_S = 60


def base_latency(chunk: int, sample_rate: int) -> float:
    """Suggested input latency for a `chunk`-frame blocking stream.

    PortAudio's CoreAudio blocking read buffers input in a ring sized
    `pow2ceil(max(2 * latency * rate, 3 * device IO buffer))`
    (pa_mac_core_utilities.c computeRingBufferSize) — it ignores our
    blocksize, yet its callback writes a whole block at once. sounddevice's
    default 'high' latency is the device's, snapshotted when PortAudio was
    initialised; for a Bluetooth hands-free mic seen right at launch that was
    ~30 ms with a small IO buffer, so the ring was 1024 frames under a
    1280-frame block: every callback overflowed and dropped 256 frames (20%),
    ~10 overflows a second for days. Asking for at least one chunk's worth
    of latency makes the ring at least two chunks whatever the device says;
    a device whose own high latency is larger keeps it."""
    floor = chunk / sample_rate
    try:
        high = float(sd.query_devices(kind="input")["default_high_input_latency"])
    except Exception:
        high = 0.0
    return max(floor, high)


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
    stops the thread and the stream. If the reader dies (the device can't be
    opened, or reopened after a refresh) the generator raises that exception
    to its consumer.

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
    # Frames, then either None (clean stop) or the exception that killed the
    # reader (open/refresh failure) — re-raised in the consumer so the wake
    # engine's wait() fails and the orchestrator's retry backoff applies,
    # instead of the generator just ending and the loop reopening the mic
    # in a tight spin.
    q: queue.Queue[bytes | None | BaseException] = queue.Queue()
    done = threading.Event()

    def open_stream():
        with devices.refresh_lock:
            stream = sd.RawInputStream(samplerate=settings.sample_rate, channels=1, dtype="int16",
                                       blocksize=chunk, latency=base_latency(chunk, settings.sample_rate))
            stream.__enter__()
            return stream

    def reader() -> None:
        stream = None
        opened_gen = -1
        deferred_logged = False
        failure: BaseException | None = None

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

        overflow_count, overflow_logged_at = 0, -1e9
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
                    # One line per minute at most: a device that overflows on
                    # every chunk (seen with AirPods' hands-free profile) wrote
                    # ~15 lines a second and rotated every other log line away
                    # within hours, destroying the evidence of what went wrong.
                    overflow_count += 1
                    now_s = time.monotonic()
                    if now_s - overflow_logged_at >= OVERFLOW_LOG_EVERY_S:
                        log.warning("%s mic overflow (reader thread stalled) x%d in the last %ds — input: %s",
                                    log_prefix, overflow_count, OVERFLOW_LOG_EVERY_S,
                                    getattr(devices, "last_input_name", None) or devices.last_input_id)
                        overflow_count = 0
                        overflow_logged_at = now_s
                q.put(bytes(data))
                if watch is None:
                    continue
                watch.check(time.monotonic())
                refresh_if_pending()
        except Exception as e:
            log.exception("%s mic reader died", log_prefix)
            failure = e
        finally:
            with devices.refresh_lock:
                close_current()
            q.put(failure)

    t = threading.Thread(target=reader, name=f"{log_prefix}-mic", daemon=True)
    t.start()
    if on_backlog is not None:
        on_backlog(q.qsize)
    try:
        while True:
            frame = q.get()
            if frame is None:
                return
            if isinstance(frame, BaseException):
                raise frame
            yield frame
    finally:
        done.set()
        if on_backlog is not None:
            on_backlog(lambda: 0)
