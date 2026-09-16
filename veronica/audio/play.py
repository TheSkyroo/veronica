import asyncio
import threading
from collections import deque

import numpy as np
import sounddevice as sd

FADE_MS = 3


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
        )
        try:
            stream.start()
        except Exception:
            self._stream = None
            raise
        self._stream = stream
        return stream

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
        self._ensure_stream()
        with self._lock:
            self._drained.clear()
            self._queue.append(faded)
        await asyncio.to_thread(self._drained.wait)

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            self._queue.clear()
        self._drained.set()
