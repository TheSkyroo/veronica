import asyncio
import difflib
import logging
import string
import threading
from collections.abc import Callable, Iterator

import numpy as np
from faster_whisper import WhisperModel

from veronica.audio.devices import InputWatch
from veronica.audio.mic import mic_frames
from veronica.audio.play import close_registered_streams
from veronica.config import Settings
from veronica.ui.events import rms

CHUNK = 1280  # 80 ms @ 16 kHz, same cadence as WakeWord

log = logging.getLogger("veronica.audio")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return text.lower().translate(_PUNCT_TABLE).strip()


def _matches(text: str, phrases: list[str], *, strict: bool = False) -> bool:
    """`strict` (used while barging in on a turn that is already running)
    drops the fuzzy near-miss rule: mid-turn the mic is full of the user's
    own request and Veronica's speech, and a word that merely *sounds*
    like the name would throw the answer away and start listening again.
    At idle the fuzzy bar is what makes the wake word forgiving."""
    norm = _normalize(text)
    if not norm:
        return False
    for phrase in phrases:
        if _normalize(phrase) in norm:
            return True
    if strict:
        return False
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
        """Mic frames via the shared reader-thread source (see
        veronica.audio.mic.mic_frames); `_backlog` tracks its queue depth so
        _wait can skip hops while behind real time."""
        return mic_frames(self.s, CHUNK, "wake", watch=InputWatch(), before_refresh=close_registered_streams,
                          on_backlog=lambda f: setattr(self, "_backlog", f))

    def stop(self) -> None:
        """Request that the in-flight (or next) wait() stop. Thread-safe, one-shot: a
        pending stop is consumed by the next wait() even if issued before it starts."""
        self._stop.set()

    async def wait(self, threshold: float | None = None, suppress: Callable[[], str] | None = None,
                   strict: bool | None = None) -> bool:
        """Block until the wake phrase is detected (True) or stop() is called (False).
        Only one wait() should be in flight per WhisperWake instance at a time.
        `suppress`, if given, is called on every phrase match; if the text it
        returns also matches a wake phrase (i.e. Veronica is currently saying
        something like "I'm Veronica"), the match is treated as self-triggered
        and dropped rather than returned."""
        if threshold is not None and not WhisperWake._warned_threshold:
            log.debug("WhisperWake.wait: threshold=%s ignored (phrase match used instead)", threshold)
            WhisperWake._warned_threshold = True
        # A threshold is only ever passed by the barge listener (the idle
        # wake call takes no arguments), and mid-turn audio must match the
        # name exactly — see _matches.
        if strict is None:
            strict = threshold is not None
        return await asyncio.to_thread(self._wait, suppress, strict)

    def _transcribe(self, window: np.ndarray) -> list:
        audio = window.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            beam_size=1,
            language="en",
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            # No initial_prompt: priming the decoder with the wake phrases
            # makes tiny.en spell them out of room noise ("Hi. Veronika. Hi.
            # Veronika. ..." from a hop of −45 dB hiss), which both wakes her
            # when nobody spoke and costs ~4 s a hop instead of 0.1 s, so the
            # loop falls behind the mic and skips the hop the user really did
            # say her name in. wake_window_s is what carries slow speech.
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

    def _wait(self, suppress: Callable[[], str] | None = None, strict: bool = False) -> bool:
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
            level = rms(self._buf)
            below = level < self.s.wake_min_rms
            log.debug("wake hop rms=%.4f gate=%.4f%s", level, self.s.wake_min_rms, " (below gate)" if below else "")
            if below:
                continue
            segments = self._transcribe(self._buf)
            text = self._segments_text(segments)
            if _matches(text, self.s.wake_phrases, strict=strict):
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
