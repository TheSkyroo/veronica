import asyncio
import difflib
import logging
import queue
import string
import threading
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from veronica.config import Settings
from veronica.ui.events import rms

CHUNK = 1280  # 80 ms @ 16 kHz, same cadence as WakeWord

log = logging.getLogger("veronica.audio")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return text.lower().translate(_PUNCT_TABLE).strip()


def _matches(text: str, phrases: list[str]) -> bool:
    norm = _normalize(text)
    if not norm:
        return False
    for phrase in phrases:
        if _normalize(phrase) in norm:
            return True
    words = norm.split()
    for word in words:
        # Length-gated so short unrelated words (e.g. "verona", ratio ~0.86)
        # don't slip past the ratio bar meant for near-misses like
        # "veronika"/"veronicah" that are close to "veronica"'s own length.
        if abs(len(word) - len("veronica")) > 1:
            continue
        ratio = difflib.SequenceMatcher(None, word, "veronica").ratio()
        if ratio >= 0.8:
            return True
    return False


class WhisperWake:
    """Blocks until the configured wake phrase is heard via faster-whisper.

    Same public interface as WakeWord: __init__(settings, frames=None),
    async wait(threshold=None) -> bool, one-shot consume-on-use stop().
    """

    _model_cls = WhisperModel  # swapped in tests
    _warned_threshold = False

    def __init__(self, settings: Settings, frames: Callable[[], Iterator[bytes]] | None = None) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames
        self._model = self._model_cls(settings.wake_whisper_model, device="cpu", compute_type="int8")
        self._stop = threading.Event()
        self._window_samples = int(settings.wake_window_s * settings.sample_rate)
        self._hop_samples = int(settings.wake_hop_s * settings.sample_rate)
        self._buf = np.zeros(0, dtype=np.int16)
        self.preroll = np.zeros(0, dtype=np.int16)
        # Frames still queued by the mic reader thread; _wait skips a hop's
        # transcription while there's more than a hop of backlog so it
        # catches up to real time instead of analysing ever-older audio.
        self._backlog: Callable[[], int] = lambda: 0

    def _mic_frames(self) -> Iterator[bytes]:
        """Mic frames, read on a dedicated thread into a queue.

        The wake loop transcribes a window every hop and tiny.en can take
        longer than one hop under CPU load; reading the device inline would
        let PortAudio's ring buffer overflow during that stall and silently
        drop audio — chopping the wake word in half. The reader thread keeps
        draining the device no matter how long a transcription takes, so
        the loop only ever falls behind, never loses frames. Closing this
        generator (or _wait returning) stops the thread and the stream."""
        q: queue.Queue[bytes | None] = queue.Queue()
        done = threading.Event()

        def reader() -> None:
            try:
                with sd.RawInputStream(
                    samplerate=self.s.sample_rate, channels=1, dtype="int16", blocksize=CHUNK
                ) as stream:
                    while not done.is_set():
                        data, overflowed = stream.read(CHUNK)
                        if overflowed:
                            log.warning("wake mic overflow (reader thread stalled)")
                        q.put(bytes(data))
            except Exception:
                log.exception("wake mic reader died")
            finally:
                q.put(None)

        t = threading.Thread(target=reader, name="wake-mic", daemon=True)
        t.start()
        self._backlog = q.qsize
        try:
            while True:
                frame = q.get()
                if frame is None:
                    return
                yield frame
        finally:
            done.set()
            self._backlog = lambda: 0

    def stop(self) -> None:
        """Request that the in-flight (or next) wait() stop. Thread-safe, one-shot: a
        pending stop is consumed by the next wait() even if issued before it starts."""
        self._stop.set()

    async def wait(self, threshold: float | None = None, suppress: Callable[[], str] | None = None) -> bool:
        """Block until the wake phrase is detected (True) or stop() is called (False).
        Only one wait() should be in flight per WhisperWake instance at a time.
        `suppress`, if given, is called on every phrase match; if the text it
        returns also matches a wake phrase (i.e. Veronica is currently saying
        something like "I'm Veronica"), the match is treated as self-triggered
        and dropped rather than returned."""
        if threshold is not None and not WhisperWake._warned_threshold:
            log.debug("WhisperWake.wait: threshold=%s ignored (phrase match used instead)", threshold)
            WhisperWake._warned_threshold = True
        return await asyncio.to_thread(self._wait, suppress)

    def _transcribe(self, window: np.ndarray) -> list:
        audio = window.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        return list(segments)

    @staticmethod
    def _segments_text(segments: list) -> str:
        return " ".join(s.text.strip() for s in segments).strip()

    @staticmethod
    def _last_wake_word_end(segments: list, window_dur_s: float) -> float:
        """End time (seconds into the transcribed window) of the last word
        that itself fuzzy-matches "veronica". Falls back to window end - 0.3s
        (a rough guess at the wake word's length) when word timestamps aren't
        available or nothing at the word level matched (e.g. a substring-only
        phrase match like "hey veronica" mis-segmented by the model)."""
        last_end = None
        for seg in segments:
            for w in getattr(seg, "words", None) or []:
                word = _normalize(getattr(w, "word", ""))
                if not word or abs(len(word) - len("veronica")) > 1:
                    continue
                if difflib.SequenceMatcher(None, word, "veronica").ratio() >= 0.8:
                    last_end = w.end
        if last_end is None:
            return max(0.0, window_dur_s - 0.3)
        return last_end

    def take_preroll(self) -> np.ndarray:
        """Return (and clear) the audio captured just after the last matched
        wake word, so a command spoken in the same breath isn't lost."""
        p = self.preroll
        self.preroll = np.zeros(0, dtype=np.int16)
        return p

    def _wait(self, suppress: Callable[[], str] | None = None) -> bool:
        self._buf = np.zeros(0, dtype=np.int16)
        since_hop = 0
        for frame in self._frames():
            if self._stop.is_set():
                self._stop.clear()
                return False
            chunk = np.frombuffer(frame, dtype=np.int16)
            self._buf = np.concatenate([self._buf, chunk])[-self._window_samples:]
            since_hop += chunk.size
            if since_hop < self._hop_samples:
                continue
            since_hop = 0
            if self._backlog() * CHUNK > self._hop_samples:
                continue  # behind real time: catch up before transcribing again
            if rms(self._buf) < self.s.wake_min_rms:
                continue
            segments = self._transcribe(self._buf)
            text = self._segments_text(segments)
            if _matches(text, self.s.wake_phrases):
                window_dur_s = self._buf.size / self.s.sample_rate
                end_s = self._last_wake_word_end(segments, window_dur_s)
                self.preroll = self._buf[int(end_s * self.s.sample_rate):].copy()
                self._buf = np.zeros(0, dtype=np.int16)
                if suppress is not None and _matches(suppress(), self.s.wake_phrases):
                    log.debug("wake match suppressed (own speech)")
                    self.preroll = np.zeros(0, dtype=np.int16)
                    continue
                return True
        return False
