import asyncio
import logging
import threading
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
import webrtcvad

from veronica.audio import devices
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
        self._finish = threading.Event()
        self._capturing = False
        self._hold = False
        self._armed = False
        self._on_level = on_level
        self._level_error_logged = False
        # Public, reassignable: Orchestrator wires this up after construction
        # when partial live transcription is enabled.
        self.on_audio = on_audio
        self._audio_error_logged = False
        # A PortAudio re-init (default input device changed) must wait until
        # this capture's RawInputStream is closed; each capture opens a fresh
        # stream, so it lands on the new device by itself afterwards.
        devices.busy = lambda: self._capturing

    def _mic_frames(self) -> Iterator[bytes]:
        n = self.s.sample_rate * self.s.frame_ms // 1000
        # Opened under refresh_lock (and with _capturing already True, see
        # arm()) so a default-input-device refresh can't terminate PortAudio
        # underneath this stream; each capture opens fresh, so after a
        # refresh it lands on the new device by itself.
        with devices.refresh_lock:
            stream = sd.RawInputStream(samplerate=self.s.sample_rate, channels=1, dtype="int16", blocksize=n)
            stream.__enter__()
        try:
            while True:
                data, _ = stream.read(n)
                yield bytes(data)
        finally:
            stream.__exit__(None, None, None)

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

    def finish(self) -> None:
        """Request that the in-flight hold-mode capture() (`hold=True`, e.g.
        push-to-talk) end now, gracefully returning whatever has been
        recorded so far — unlike stop(), which discards it. A no-op unless
        a *hold-mode* capture is actually running: a finish() aimed at a
        normal capture (or one that lands between captures) is dropped
        rather than left lingering to cut short the next hold capture —
        and capture() clears both flags on entry anyway, as a belt-and-
        braces guard against the same race."""
        if self._capturing and self._hold:
            self._finish.set()

    async def capture(
        self,
        max_s: int | None = None,
        preroll: np.ndarray | None = None,
        partial: bool = False,
        skip_ms: int = 0,
        hold: bool = False,
    ) -> np.ndarray | None:
        """Capture one utterance, waiting for speech onset and endpointed by silence.

        Args:
            max_s: seconds to wait for speech to begin; None = wait forever.
                   Utterance length is always capped by settings.max_utterance_s.
            preroll: audio captured just before this call started (e.g. the
                     wake engine's tail buffer) — replayed through the VAD
                     ahead of live frames so a command spoken in the same
                     breath as the wake word isn't lost.
            partial: whether `on_audio` may be invoked during this capture.
                     False for captures whose transcript must not be treated
                     as a live partial (e.g. confirm()'s yes/no capture) —
                     otherwise a stray partial could overwrite the HUD's
                     "You" row with the wrong turn's text.
            skip_ms: discard this many milliseconds of *live* mic frames
                     before the VAD even looks at them (preroll is
                     unaffected). Used by the follow-up capture to drop the
                     tail/echo of Veronica's own just-spoken audio, which
                     would otherwise get endpointed as a false speech onset.
            hold: "hold to talk" mode (push-to-talk): the VAD silence
                  endpoint is ignored — recording continues until finish()
                  is called, at which point whatever's been captured so far
                  is returned (even if shorter than min_speech_ms). The
                  whole capture — waiting for onset *and* recording — is
                  hard-capped at max_s of live audio (max_utterance_s if
                  max_s is None), so a key that never comes back up (or a
                  release that got lost) can't leave the mic open forever.

        Returns:
            int16 mono PCM array, or None if nothing was captured (no speech
            before max_s, or — outside hold mode — total speech shorter
            than min_speech_ms).
        """
        # capture() is a coroutine: this body only runs on the task's first
        # step, one loop iteration after ensure_future(). Callers that
        # schedule a capture and may receive a stop()/finish() in that same
        # iteration must arm() synchronously first (Orchestrator._capture
        # does); arm() here is then an idempotent no-op.
        if not self._armed:
            self.arm(hold)
        self._armed = False
        return await asyncio.to_thread(self._capture, max_s, preroll, partial, skip_ms, hold)

    def arm(self, hold: bool = False) -> None:
        """Synchronously mark a capture as in flight *before* its coroutine
        gets its first step, so a stop()/finish() that lands in the gap
        between scheduling capture() and the worker thread actually
        starting is honored rather than silently dropped (a dropped
        finish() on a push-to-talk quick tap would leave the mic open for
        the whole ptt_max_s). Also discards any stale stop()/finish() left
        over from a previous capture (one that raced its natural end) so it
        can't cut this capture short. capture() arms itself unless the
        caller already did (so an explicit arm() is consumed by exactly one
        capture())."""
        self._stop.clear()
        self._finish.clear()
        self._hold = hold
        self._capturing = True
        self._armed = True

    def disarm(self) -> None:
        """Undo arm() when the capture it was meant for never got scheduled."""
        self._capturing = False
        self._armed = False
        self._hold = False

    def _frame_bytes(self) -> int:
        return self.s.sample_rate * self.s.frame_ms // 1000

    def _preroll_frames(self, preroll: np.ndarray | None) -> Iterator[bytes]:
        if preroll is None or preroll.size == 0:
            return
        n = self._frame_bytes()
        usable = preroll.size - (preroll.size % n)
        for i in range(0, usable, n):
            yield preroll[i:i + n].tobytes()

    # A pre-roll that only caught the wake word's own tail (a couple of VAD
    # frames right at the boundary) shouldn't count as "the user is already
    # talking" — that's just noise, not enough to skip the wake chime.
    _HAS_SPEECH_MIN_FRAMES = 5  # 150 ms at the default 30 ms frame size

    def has_speech(self, pcm: np.ndarray | None) -> bool:
        """True if `pcm` contains at least _HAS_SPEECH_MIN_FRAMES of VAD
        speech. Used to decide whether the wake chime should still play
        given wake-word pre-roll. Uses its own Vad instance (webrtcvad.Vad
        isn't documented thread-safe) since this can be called from the
        event-loop thread while a previous capture's worker thread is still
        winding down its own use of self._vad."""
        if pcm is None or pcm.size == 0:
            return False
        vad = self._vad_cls(self.s.vad_aggressiveness)
        n = self._frame_bytes()
        usable = pcm.size - (pcm.size % n)
        count = 0
        for i in range(0, usable, n):
            if vad.is_speech(pcm[i:i + n].tobytes(), self.s.sample_rate):
                count += 1
                if count >= self._HAS_SPEECH_MIN_FRAMES:
                    return True
        return False

    def _capture(
        self,
        max_s: int | None,
        preroll: np.ndarray | None = None,
        partial: bool = False,
        skip_ms: int = 0,
        hold: bool = False,
    ) -> np.ndarray | None:
        fm = self.s.frame_ms
        silence_frames_needed = self.s.vad_silence_ms // fm
        min_speech_frames = self.s.min_speech_ms // fm
        max_frames = self.s.max_utterance_s * 1000 // fm
        wait_frames = (max_s * 1000 // fm) if max_s else None
        # Hold mode: hard cap on *total* live frames from capture start
        # (onset wait + recording), so a lost key-up can't hold the mic open.
        hold_cap_frames = (wait_frames if wait_frames is not None else max_frames) if hold else None
        extra_frames = int(self.s.capture_extra_s * 1000 // fm)
        hop_frames = max(1, int(self.s.partial_hop_s * 1000 / fm))
        skip_frames = skip_ms // fm

        n = self._frame_bytes()
        preroll_frame_total = 0
        if preroll is not None and preroll.size > 0:
            preroll_frame_total = (preroll.size - preroll.size % n) // n

        buf: list[bytes] = []
        speech_frames = 0
        silence_run = 0
        started = False
        onset_from_preroll = False
        waited = 0
        elapsed = 0
        frames_since_partial = 0
        frame_idx = -1

        def _all_frames():
            yield from self._preroll_frames(preroll)
            live = self._frames()
            for _ in range(skip_frames):
                # Discarded before the VAD ever sees them — not "silence",
                # just not consulted at all.
                if next(live, None) is None:
                    return
            yield from live

        try:
            for frame in _all_frames():
                frame_idx += 1
                if frame_idx >= preroll_frame_total:
                    # Hard cap on live-frame time spent waiting for an onset:
                    # repeated false onsets (e.g. a bursty noise source) each
                    # get their own wait budget via the reset below, which
                    # can otherwise inflate the *real* wall-clock wait far
                    # past max_s. elapsed counts every live frame no matter
                    # what reset state we're in, so this always fires — but
                    # only while no utterance is in progress (not started):
                    # once real speech has started, it must be allowed to
                    # finish, bounded only by max_frames below, not cut short
                    # by this onset-wait cap.
                    elapsed += 1
                    if hold_cap_frames is not None and elapsed > hold_cap_frames:
                        break
                    if not started and wait_frames is not None and elapsed >= wait_frames + extra_frames:
                        return None
                if self._stop.is_set():
                    self._stop.clear()
                    return None
                if hold and self._finish.is_set():
                    self._finish.clear()
                    break
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
                        onset_from_preroll = frame_idx < preroll_frame_total
                    elif wait_frames is not None and waited >= wait_frames:
                        return None
                    else:
                        continue
                buf.append(frame)
                if is_speech:
                    speech_frames += 1
                    silence_run = 0
                    # Only speech frames count toward the partial-transcript
                    # hop — trailing silence shouldn't trigger a re-transcribe.
                    if partial and self.on_audio is not None:
                        frames_since_partial += 1
                        if frames_since_partial >= hop_frames:
                            frames_since_partial = 0
                            try:
                                self.on_audio(np.frombuffer(b"".join(buf), dtype=np.int16).copy())
                            except Exception:
                                if not self._audio_error_logged:
                                    log.exception("on_audio callback failed")
                                    self._audio_error_logged = True
                else:
                    silence_run += 1
                if (not hold and silence_run >= silence_frames_needed) or len(buf) >= max_frames:
                    has_wait_budget_left = wait_frames is not None and waited < wait_frames
                    if not hold and speech_frames < min_speech_frames and (onset_from_preroll or has_wait_budget_left):
                        # Too little real speech to count as an utterance —
                        # either it was just the tail of the wake word caught
                        # in the pre-roll, or a brief false onset (e.g. the
                        # echo/tail of Veronica's own voice during a
                        # follow-up capture). Either way, keep waiting for a
                        # genuine onset instead of giving up, as long as
                        # there's still wait budget left (max_s is None means
                        # no budget at all: fall through to the old
                        # behavior and return None below).
                        started = False
                        onset_from_preroll = False
                        buf = []
                        speech_frames = 0
                        silence_run = 0
                        frames_since_partial = 0
                        continue
                    break
        finally:
            self._capturing = False

        if not hold and speech_frames < min_speech_frames:
            return None
        if not buf:
            return None
        return np.frombuffer(b"".join(buf), dtype=np.int16)
