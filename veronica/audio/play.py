import asyncio
import contextlib
import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from veronica.audio import devices

log = logging.getLogger("veronica.audio.play")
_PortAudioError = sd.PortAudioError  # bound here so tests can swap `sd` for a fake

FADE_MS = 3
DEFAULT_TIMEOUT_MARGIN_S = 2.0

# Players whose persistent output stream must be closed before PortAudio is
# re-initialised (see veronica.audio.devices.refresh_portaudio).
_registry: list["Player"] = []


def register_for_refresh(player: "Player") -> None:
    if player not in _registry:
        _registry.append(player)


def close_registered_streams() -> None:
    """`before` hook for devices.refresh_portaudio: closes every registered
    Player's output stream (playback resumes on the next play(), on the
    new default device)."""
    for player in list(_registry):
        player.close_stream()


class Player:
    """Plays float32 mono audio through one persistent output stream.

    Sentences are streamed back-to-back into the same `sd.OutputStream`
    (opened lazily on first `play()`) instead of each getting its own
    `sd.play()/sd.wait()` call — that per-call approach re-opens the audio
    device for every sentence, which is what produced an audible click at
    each boundary. A short linear fade-in/out on every enqueued chunk kills
    the sample-discontinuity click that would otherwise happen at its own
    edges.
    """

    def __init__(self, sample_rate: int = 24000, blocksize: int = 1024) -> None:
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self._stopped = False
        self._stream = None
        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._drained = threading.Event()
        self._drained.set()  # nothing queued yet
        # Bound on how long play() waits for a chunk to drain, beyond the
        # chunk's own playback duration — a stalled/dead device must not hang
        # a caller forever. Exposed as an attribute so tests can shrink it.
        self._timeout_margin_s = DEFAULT_TIMEOUT_MARGIN_S
        # Set by close_stream() (device refresh) so an interrupted play()
        # logs the expected interruption at INFO rather than WARNING.
        self._refreshed = False

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def reset(self) -> None:
        """Clear a previous stop() so new playback is accepted."""
        self._stopped = False

    # -- stream lifecycle -------------------------------------------------
    def _open_stream(self):
        return sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            latency="high",
            callback=self._cb,
            finished_callback=self._on_finished,
        )

    def _ensure_stream(self):
        if self._stream is not None:
            return self._stream
        # Under devices.refresh_lock so the mic reader's device refresh can't
        # terminate PortAudio between this open and the stream starting.
        with devices.refresh_lock:
            try:
                stream = self._open_stream()
            except _PortAudioError as e:
                # PortAudio's device table goes stale when the default output
                # device changes mid-session (headphones plugged in, AirPods
                # connected): opening a new stream then fails with an internal
                # error until PortAudio is re-initialised. One retry, routed
                # through devices.refresh_portaudio so the other registered
                # Players close first and the generation bump tells the wake
                # mic reader (which only reads under refresh_lock, so it is
                # between reads right now) to reopen its stream. A Recorder
                # capture's blocking read is *not* under the lock, and
                # Pa_Terminate under it would be a use-after-free (the
                # capture raises, it doesn't return None), so while a capture
                # is in flight we don't re-initialise at all: this play()
                # fails and the mic reader's watch refreshes once the
                # capture is done.
                if devices.busy():
                    log.warning("output stream open failed (%s); not re-initialising PortAudio: capture in flight", e)
                    raise
                log.warning("output stream open failed (%s); re-initialising PortAudio", e)
                devices.refresh_portaudio(before=close_registered_streams)
                stream = self._open_stream()
            try:
                stream.start()
            except Exception:
                with contextlib.suppress(Exception):
                    stream.close()
                raise
            with self._lock:
                self._stream = stream
                self._refreshed = False   # a refresh only explains the stream it closed
            return stream

    def _on_finished(self) -> None:
        # The stream itself ended (device error/disconnect, or an explicit
        # stop/close) — nobody is going to drain the queue for us any more,
        # so unblock any play() waiting on it and mark the stream dead so
        # the next play() opens a fresh one.
        with self._lock:
            self._stream = None
        self._drained.set()

    @staticmethod
    def _close_stream_obj(stream) -> None:
        with contextlib.suppress(Exception):
            stream.stop()
        with contextlib.suppress(Exception):
            stream.close()

    def close_stream(self) -> None:
        """Close the persistent output stream without stopping playback for
        good: `_stopped` is untouched, so the next play() reopens a fresh
        stream (on whatever the default output device is by then). A play()
        in flight sees the stream go inactive and returns early."""
        with self._lock:
            stream, self._stream = self._stream, None
            self._refreshed = stream is not None
        if stream is not None:
            self._close_stream_obj(stream)

    def close(self) -> None:
        self.close_stream()
        self._refreshed = False

    def _cb(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0] if outdata.ndim > 1 else outdata
        with self._lock:
            n = 0
            while n < frames and self._queue:
                chunk = self._queue[0]
                take = min(len(chunk), frames - n)
                out[n:n + take] = chunk[:take]
                n += take
                if take >= len(chunk):
                    self._queue.popleft()
                else:
                    self._queue[0] = chunk[take:]
            if n < frames:
                out[n:] = 0.0
            if not self._queue:
                self._drained.set()

    @staticmethod
    def _with_fades(samples: np.ndarray, fade_n: int) -> np.ndarray:
        samples = np.array(samples, dtype=np.float32, copy=True)
        n = min(fade_n, len(samples) // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            samples[:n] *= ramp
            samples[-n:] *= ramp[::-1]
        return samples

    # -- playback -----------------------------------------------------------
    async def play(self, samples: np.ndarray) -> None:
        if self._stopped:
            return
        fade_n = max(1, self.sample_rate * FADE_MS // 1000)
        faded = self._with_fades(samples, fade_n)
        stream = self._ensure_stream()
        with self._lock:
            self._drained.clear()
            self._queue.append(faded)
        timeout = len(faded) / self.sample_rate + self._timeout_margin_s
        drained = await asyncio.to_thread(self._drained.wait, timeout)
        active = getattr(stream, "active", True)
        if not drained or not active:
            with self._lock:
                refreshed, self._refreshed = self._refreshed, False
            if refreshed and not active:
                log.info("playback interrupted by an audio device refresh; reopening the output stream")
            else:
                log.warning(
                    "playback %s; closing and reopening the output stream",
                    "timed out" if not drained else "stream went inactive",
                )
            with self._lock:
                if self._stream is stream:
                    self._stream = None
                self._queue.clear()
            # _on_finished may already have detached this stream from
            # self._stream, so close_registered_streams() can't close it for
            # a refresh: close it under refresh_lock so a refresh can't
            # terminate PortAudio while it is still live.
            with devices.refresh_lock:
                self._close_stream_obj(stream)
            return

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            self._queue.clear()
        self._drained.set()
