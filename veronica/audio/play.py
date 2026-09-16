import asyncio
import contextlib
import logging
import threading
from collections import deque

import numpy as np
import sounddevice as sd

log = logging.getLogger("veronica.audio.play")

FADE_MS = 3
DEFAULT_TIMEOUT_MARGIN_S = 2.0


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

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def reset(self) -> None:
        """Clear a previous stop() so new playback is accepted."""
        self._stopped = False

    # -- stream lifecycle -------------------------------------------------
    def _ensure_stream(self):
        if self._stream is not None:
            return self._stream
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            latency="high",
            callback=self._cb,
            finished_callback=self._on_finished,
        )
        try:
            stream.start()
        except Exception:
            with contextlib.suppress(Exception):
                stream.close()
            raise
        self._stream = stream
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

    def close(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            self._close_stream_obj(stream)

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
            log.warning(
                "playback %s; closing and reopening the output stream",
                "timed out" if not drained else "stream went inactive",
            )
            with self._lock:
                if self._stream is stream:
                    self._stream = None
                self._queue.clear()
            self._close_stream_obj(stream)
            return

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            self._queue.clear()
        self._drained.set()
