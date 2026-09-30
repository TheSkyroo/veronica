"""Shared mic frame source for the wake-word engines."""

import contextlib
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
# No audio at all for this long means the stream is dead, not quiet (a live
# input delivers frames even when muted): PortAudio's CoreAudio input callback
# stops the unit for good when AudioUnitRender fails (-10863 "cannot do in
# current context", -50 — seen in launchd.log when AirPods leave or the Mac
# sleeps), and a blocking read() on it then waits forever.
STALL_S = 2.0
# Overflowing on at least half the reads for this long is a broken stream
# configuration, not a CPU hiccup: reopen with a bigger buffer.
OVERFLOW_HEAL_S = 3.0
# ...doubling the latency each time, up to this; past it we stop reopening.
MAX_LATENCY_S = 1.0
# Frames the queue holds before dropping the oldest (~60 s of 80 ms chunks):
# a consumer that stops pulling must never block the reader or grow memory.
QUEUE_MAX = 750


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


def _input_label() -> str:
    """The default input's name for log lines ("AirPods - Find My"), else its id."""
    with contextlib.suppress(Exception):
        name = devices.default_input_name()
        if name:
            return name
    return f"device id {devices.last_input_id}"


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

    The reader never blocks inside PortAudio: it polls `read_available` and
    reads only what is buffered, so it can tell a stream that died (no audio
    for STALL_S: reopen, then re-initialise PortAudio if the fresh stream is
    dead too) from one that is merely slow, and never sits on
    `devices.refresh_lock` waiting for audio. A stream that overflows on
    most reads for OVERFLOW_HEAL_S is reopened with twice the latency (up to
    MAX_LATENCY_S). The queue is bounded (QUEUE_MAX, oldest dropped), so a
    consumer that stops pulling can't block or balloon the reader.

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
    q: queue.Queue[bytes | None | BaseException] = queue.Queue(maxsize=QUEUE_MAX)
    done = threading.Event()
    chunk_bytes = chunk * 2          # int16 mono
    poll_s = chunk / settings.sample_rate / 4

    def put(item) -> bool:
        """Queue `item` without ever blocking: when full, drop the oldest
        frame to make room. Returns True if a frame was dropped. Only this
        thread puts, so the end-of-stream item (None / the exception) always
        lands, after every frame."""
        dropped = False
        while True:
            try:
                q.put_nowait(item)
                return dropped
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
                    dropped = True

    def open_stream(latency: float):
        with devices.refresh_lock:
            stream = sd.RawInputStream(samplerate=settings.sample_rate, channels=1, dtype="int16",
                                       blocksize=chunk, latency=latency)
            stream.__enter__()
            return stream

    def reader() -> None:
        stream = None
        opened_gen = -1
        deferred_logged = False
        failure: BaseException | None = None
        # Under the lock: Pa_GetDeviceInfo racing a Pa_Terminate on the
        # Player's thread would read a freed device table.
        with devices.refresh_lock:
            latency = base_latency(chunk, settings.sample_rate)
        last_audio = time.monotonic()      # last time a read returned frames
        stalls = 0                         # consecutive stall reopens with no audio between
        win_start, win_reads, win_over = last_audio, 0, 0
        heal_exhausted = False
        pending = bytearray()              # partial chunk carried between reads

        def open_fresh():
            nonlocal stream, opened_gen, last_audio, win_start, win_reads, win_over
            stream = open_stream(latency)
            opened_gen = devices.generation
            # A fresh stream gets a fresh stall clock and overflow window.
            last_audio = win_start = time.monotonic()
            win_reads = win_over = 0
            pending.clear()

        def close_current():
            nonlocal stream
            if stream is None:
                return
            if devices.generation == opened_gen:
                # A stream PortAudio stopped on its own may complain on the
                # way out; that must not kill the reader that's replacing it.
                try:
                    stream.__exit__(None, None, None)
                except Exception:
                    log.debug("%s mic stream close failed", log_prefix, exc_info=True)
            # else: PortAudio was re-initialised under it (Pa_Terminate
            # closes every open stream), so the handle is already dead —
            # just drop it rather than closing a dangling pointer.
            stream = None

        def read_some():
            """(data, overflowed) for whatever is buffered, or (None, False)
            when nothing is. Never blocks: a read() for more than is buffered
            would wait in PortAudio's ReadStream, forever if the unit
            stopped. Streams without `read_available` (test fakes) get a
            plain read of one chunk."""
            avail = getattr(stream, "read_available", None)
            if avail is None:
                return stream.read(chunk)
            if avail <= 0:
                return None, False
            return stream.read(avail)

        def heal_stall() -> None:
            """No audio for STALL_S: reopen; if the last reopen already got
            nothing, the device table itself is stale (the device came back
            under a new id), so re-initialise PortAudio first — unless a
            capture's stream is open (Pa_Terminate would pull it away)."""
            nonlocal stalls, latency
            stalls += 1
            reinit = stalls > 1 and not devices.busy()
            log.warning("%s mic: no audio from %s for %.1fs (stream active=%s); %s", log_prefix, _input_label(),
                        STALL_S, getattr(stream, "active", "?"),
                        "re-initialising PortAudio and reopening" if reinit else "reopening the stream")
            with devices.refresh_lock:
                close_current()
                if reinit:
                    devices.refresh_portaudio(before=before_refresh)
                    latency = base_latency(chunk, settings.sample_rate)
                open_fresh()

        def heal_overflow(now: float) -> None:
            """Called once per OVERFLOW_HEAL_S window: if most reads in it
            overflowed, reopen with twice the latency (a bigger PortAudio
            ring), once per window, until MAX_LATENCY_S."""
            nonlocal win_start, win_reads, win_over, latency, heal_exhausted
            reads, over = win_reads, win_over
            win_start, win_reads, win_over = now, 0, 0
            if heal_exhausted or reads < 3 or over * 2 < reads:
                return
            bigger = min(latency * 2, MAX_LATENCY_S)
            if bigger <= latency:
                log.warning("%s mic: %s still overflowing on %d/%d reads at latency %.2fs; not reopening again",
                            log_prefix, _input_label(), over, reads, latency)
                heal_exhausted = True
                return
            log.warning("%s mic: %s overflowing on %d/%d reads for %.0fs; reopening with latency %.2fs (was %.2fs)",
                        log_prefix, _input_label(), over, reads, OVERFLOW_HEAL_S, bigger, latency)
            latency = bigger
            with devices.refresh_lock:
                close_current()
                open_fresh()

        def refresh_if_pending() -> None:
            """Under the lock: if a refresh is owed and no capture is open,
            close our stream, refresh PortAudio and reopen; otherwise (busy)
            keep the current stream and leave `pending` set for a later
            check. Logs the deferral once per pending change."""
            nonlocal deferred_logged, latency, heal_exhausted
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
                # A new device: start again from its own latency, not one
                # grown to heal the old device's overflows.
                latency = base_latency(chunk, settings.sample_rate)
                heal_exhausted = False
                open_fresh()
                deferred_logged = False

        overflow_count, overflow_logged_at, drop_logged_at = 0, -1e9, -1e9
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
                # read_some() never blocks, so the lock is only ever held for
                # a copy out of PortAudio's ring, never while audio is awaited.
                with devices.refresh_lock:
                    if devices.generation != opened_gen:
                        log.info("%s mic stream invalidated by PortAudio refresh; reopening", log_prefix)
                        close_current()
                        open_fresh()
                    data, overflowed = read_some()
                now_s = time.monotonic()
                if data is None:
                    if now_s - last_audio >= STALL_S:
                        heal_stall()
                    else:
                        done.wait(poll_s)
                else:
                    last_audio = now_s
                    stalls = 0
                    win_reads += 1
                    if overflowed:
                        win_over += 1
                        # One line per minute at most: a device that overflows
                        # on every chunk (seen with Bluetooth hands-free mics)
                        # wrote ~10 lines a second and rotated every other log
                        # line away within hours, destroying the evidence.
                        overflow_count += 1
                        if now_s - overflow_logged_at >= OVERFLOW_LOG_EVERY_S:
                            log.warning("%s mic overflow x%d in the last %ds (latency %.2fs) — input: %s",
                                        log_prefix, overflow_count, OVERFLOW_LOG_EVERY_S, latency, _input_label())
                            overflow_count = 0
                            overflow_logged_at = now_s
                    if now_s - win_start >= OVERFLOW_HEAL_S:
                        heal_overflow(now_s)
                    pending.extend(data)
                    while len(pending) >= chunk_bytes:
                        if put(bytes(pending[:chunk_bytes])) and now_s - drop_logged_at >= OVERFLOW_LOG_EVERY_S:
                            log.warning("%s mic consumer not draining: queue full at %d frames, dropping the oldest",
                                        log_prefix, QUEUE_MAX)
                            drop_logged_at = now_s
                        del pending[:chunk_bytes]
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
            put(failure)

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
