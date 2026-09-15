import asyncio
import threading
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
import webrtcvad

from veronica.config import Settings


class Recorder:
    """Captures one utterance, endpointed by webrtcvad silence."""

    _vad_cls = webrtcvad.Vad  # swapped in tests

    def __init__(self, settings: Settings, frames: Callable[[], Iterator[bytes]] | None = None) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames
        self._vad = self._vad_cls(settings.vad_aggressiveness)
        self._stop = threading.Event()
        self._capturing = False

    def _mic_frames(self) -> Iterator[bytes]:
        n = self.s.sample_rate * self.s.frame_ms // 1000
        with sd.RawInputStream(samplerate=self.s.sample_rate, channels=1, dtype="int16", blocksize=n) as stream:
            while True:
                data, _ = stream.read(n)
                yield bytes(data)

    def stop(self) -> None:
        """Request that the in-flight capture() stop early, returning None.
        A no-op unless a capture is actually running (checked thread-side) —
        otherwise a stop() that arrives just after an unrelated capture()
        already returned on its own would linger and cut short the *next*
        capture(). There's an unavoidable, acceptably tiny window right
        around the last frame where this check can still race the capture
        thread finishing on its own; callers should treat stop() as best-
        effort, not a guarantee."""
        if self._capturing:
            self._stop.set()

    async def capture(self, max_s: int | None = None) -> np.ndarray | None:
        """Capture one utterance, waiting for speech onset and endpointed by silence.

        Args:
            max_s: seconds to wait for speech to begin; None = wait forever.
                   Utterance length is always capped by settings.max_utterance_s.

        Returns:
            int16 mono PCM array or None if speech shorter than min_speech_ms / no speech before max_s.
        """
        return await asyncio.to_thread(self._capture, max_s)

    def _capture(self, max_s: int | None) -> np.ndarray | None:
        fm = self.s.frame_ms
        silence_frames_needed = self.s.vad_silence_ms // fm
        min_speech_frames = self.s.min_speech_ms // fm
        max_frames = self.s.max_utterance_s * 1000 // fm
        wait_frames = (max_s * 1000 // fm) if max_s else None

        buf: list[bytes] = []
        speech_frames = 0
        silence_run = 0
        started = False
        waited = 0

        self._capturing = True
        try:
            for frame in self._frames():
                if self._stop.is_set():
                    self._stop.clear()
                    return None
                is_speech = self._vad.is_speech(frame, self.s.sample_rate)
                if not started:
                    waited += 1
                    if is_speech:
                        started = True
                    elif wait_frames is not None and waited >= wait_frames:
                        return None
                    else:
                        continue
                buf.append(frame)
                if is_speech:
                    speech_frames += 1
                    silence_run = 0
                else:
                    silence_run += 1
                if silence_run >= silence_frames_needed or len(buf) >= max_frames:
                    break
        finally:
            self._capturing = False

        if speech_frames < min_speech_frames:
            return None
        return np.frombuffer(b"".join(buf), dtype=np.int16)
