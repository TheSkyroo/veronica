import asyncio
import logging
import threading
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
import webrtcvad

from veronica.config import Settings
from veronica.ui.events import rms

log = logging.getLogger("veronica.audio")


class Recorder:
    """Captures one utterance, endpointed by webrtcvad silence."""

    _vad_cls = webrtcvad.Vad  # swapped in tests

    def __init__(
        self,
        settings: Settings,
        frames: Callable[[], Iterator[bytes]] | None = None,
        on_level: Callable[[float], None] | None = None,
        on_audio: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames
        self._vad = self._vad_cls(settings.vad_aggressiveness)
        self._stop = threading.Event()
        self._capturing = False
        self._on_level = on_level
        self._level_error_logged = False
        # Public, reassignable: Orchestrator wires this up after construction
        # when partial live transcription is enabled.
        self.on_audio = on_audio
        self._audio_error_logged = False

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

    async def capture(self, max_s: int | None = None, preroll: np.ndarray | None = None) -> np.ndarray | None:
        """Capture one utterance, waiting for speech onset and endpointed by silence.

        Args:
            max_s: seconds to wait for speech to begin; None = wait forever.
                   Utterance length is always capped by settings.max_utterance_s.
            preroll: audio captured just before this call started (e.g. the
                     wake engine's tail buffer) — replayed through the VAD
                     ahead of live frames so a command spoken in the same
                     breath as the wake word isn't lost.

        Returns:
            int16 mono PCM array or None if speech shorter than min_speech_ms / no speech before max_s.
        """
        # Set on the event-loop thread, before handing off to the worker, so
        # a stop() issued in the (tiny) window between a caller flipping its
        # own "capturing" bookkeeping (e.g. Orchestrator._confirm_capturing)
        # and the worker thread actually starting is not a no-op.
        self._capturing = True
        return await asyncio.to_thread(self._capture, max_s, preroll)

    def _frame_bytes(self) -> int:
        return self.s.sample_rate * self.s.frame_ms // 1000

    def _preroll_frames(self, preroll: np.ndarray | None) -> Iterator[bytes]:
        if preroll is None or preroll.size == 0:
            return
        n = self._frame_bytes()
        usable = preroll.size - (preroll.size % n)
        for i in range(0, usable, n):
            yield preroll[i:i + n].tobytes()

    def has_speech(self, pcm: np.ndarray | None) -> bool:
        """True if any 30 ms frame of `pcm` is VAD speech. Used to decide
        whether the wake chime should still play given wake-word pre-roll."""
        if pcm is None or pcm.size == 0:
            return False
        n = self._frame_bytes()
        usable = pcm.size - (pcm.size % n)
        for i in range(0, usable, n):
            if self._vad.is_speech(pcm[i:i + n].tobytes(), self.s.sample_rate):
                return True
        return False

    def _capture(self, max_s: int | None, preroll: np.ndarray | None = None) -> np.ndarray | None:
        fm = self.s.frame_ms
        silence_frames_needed = self.s.vad_silence_ms // fm
        min_speech_frames = self.s.min_speech_ms // fm
        max_frames = self.s.max_utterance_s * 1000 // fm
        wait_frames = (max_s * 1000 // fm) if max_s else None
        hop_frames = max(1, int(self.s.partial_hop_s * 1000 / fm))

        buf: list[bytes] = []
        speech_frames = 0
        silence_run = 0
        started = False
        waited = 0
        frames_since_partial = 0

        def _all_frames():
            yield from self._preroll_frames(preroll)
            yield from self._frames()

        try:
            for frame in _all_frames():
                if self._stop.is_set():
                    self._stop.clear()
                    return None
                is_speech = self._vad.is_speech(frame, self.s.sample_rate)
                if self._on_level is not None:
                    try:
                        self._on_level(rms(np.frombuffer(frame, dtype=np.int16)))
                    except Exception:
                        if not self._level_error_logged:
                            log.exception("on_level callback failed")
                            self._level_error_logged = True
                if not started:
                    waited += 1
                    if is_speech:
                        started = True
                    elif wait_frames is not None and waited >= wait_frames:
                        return None
                    else:
                        continue
                buf.append(frame)
                if self.on_audio is not None:
                    frames_since_partial += 1
                    if frames_since_partial >= hop_frames:
                        frames_since_partial = 0
                        try:
                            self.on_audio(np.frombuffer(b"".join(buf), dtype=np.int16).copy())
                        except Exception:
                            if not self._audio_error_logged:
                                log.exception("on_audio callback failed")
                                self._audio_error_logged = True
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
