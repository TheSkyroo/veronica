"""Speaker verification: "only listen to my voice".

A voice profile is the mean CAM++ embedding of a few enrolment phrases,
kept in ~/.veronica/voice_profile.json (0600). With a profile and
Settings.speaker_verification on, the orchestrator scores each captured
utterance against it (cosine similarity of unit embeddings) and ignores one
under Settings.speaker_threshold as if nothing had been said. Every score is
logged ("speaker <where>: score=...") so the threshold can be tuned from the
log; the last few are also kept for the Settings window. Choices and numbers:
docs/superpowers/specs/2026-10-01-veronica-voice-isolation-design.md."""
import datetime as dt
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from veronica.audio import models
from veronica.audio.fbank import fbank
from veronica.config import Settings

log = logging.getLogger("veronica.audio")

PROFILE_FILE = "voice_profile.json"
PROFILE_VERSION = 1
# Enrolment clips of the same voice should agree at least this well with
# each other; under it one of them was probably someone (or something) else.
ENROL_MIN_AGREEMENT = 0.3
# Shorter than this there's too little voice to embed; such a capture is
# scored anyway (a short "yes" still has to pass), enrolment rejects it.
ENROL_MIN_S = 1.0
RECENT_MAX = 8
# A capture carries the VAD's trailing silence (vad_silence_ms) and some lead
# in; only 30 ms frames within 20 dB of the loud part, and above a floor that
# idle hiss and digital silence never reach, count as voice.
_FRAME = 480
_VOICED_DB = 20
_VOICED_MIN_RMS = 0.0005
_VOICED_MIN_FRAMES = 10
# Short answers ("yes", "haan") embed less reliably, so they score lower
# against the profile: the threshold scales down linearly from full at
# FULL_S of voiced audio to SHORT_FLOOR of it at none (see the spec's table)
# — but a confirm answer never gets under CONFIRM_FLOOR of it.
FULL_S = 2.0
SHORT_FLOOR = 0.6
CONFIRM_FLOOR = 0.8
# A model that failed to load is retried (in the background) at most this often.
RETRY_S = 600.0


def _voiced_mask(pcm16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = pcm16.size // _FRAME
    frames = pcm16[: n * _FRAME].reshape(n, _FRAME)
    if n == 0:
        return frames, np.zeros(0, dtype=bool)
    level = np.sqrt(np.mean((frames.astype(np.float32) / 32768.0) ** 2, axis=1))
    keep = (level >= np.percentile(level, 95) * 10 ** (-_VOICED_DB / 20)) & (level >= _VOICED_MIN_RMS)
    return frames, keep


def voiced(pcm16: np.ndarray) -> np.ndarray:
    """What gets embedded: the voiced frames of `pcm16`, joined; all of it
    when fewer than ten qualify (too little to stand on its own)."""
    frames, keep = _voiced_mask(pcm16)
    if keep.sum() < _VOICED_MIN_FRAMES:
        return pcm16
    return frames[keep].ravel()


def voiced_s(pcm16: np.ndarray) -> float:
    """Seconds of voice in `pcm16`: qualifying frames only, 0 under ten.
    What enrolment's length check and the length-scaled threshold use —
    never the whole capture, or silence would count as speech."""
    _frames, keep = _voiced_mask(pcm16)
    count = int(keep.sum())
    return 0.0 if count < _VOICED_MIN_FRAMES else count * _FRAME / 16000


def profile_path(settings: Settings) -> Path:
    return settings.home / PROFILE_FILE


def _new_session(path: str):
    import onnxruntime as ort  # heavy import; only once verification is in use

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


class SpeakerModel:
    """CAM++ (3D-Speaker, VoxCeleb): int16 16 kHz mono -> unit 512-d embedding.
    Features as the model was trained on: 80 Kaldi fbanks (Povey window) of
    the [-1, 1] waveform, mean-normalised over the utterance."""

    _session_factory = staticmethod(_new_session)  # swapped in tests

    def __init__(self, path: Path) -> None:
        self._sess = self._session_factory(str(path))
        self._input = self._sess.get_inputs()[0].name

    def embed(self, pcm16: np.ndarray) -> np.ndarray:
        feats = fbank(pcm16.astype(np.float32) / 32768.0, window="povey")
        if feats.shape[0] < 10:
            # under ~0.1 s: pad so the network has something to pool over
            feats = np.concatenate([feats, np.zeros((10 - feats.shape[0], feats.shape[1]), np.float32)])
        feats = feats - feats.mean(axis=0, keepdims=True)
        emb = np.asarray(self._sess.run(None, {self._input: feats[None]})[0][0], dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)


@dataclass(frozen=True)
class VoiceProfile:
    embedding: np.ndarray
    model: str
    created: str
    clips: int

    def to_json(self) -> dict:
        return {"version": PROFILE_VERSION, "model": self.model, "created": self.created,
                "clips": self.clips, "embedding": [round(float(v), 6) for v in self.embedding]}

    def save(self, path: Path) -> None:
        """Atomic and private: written to a 0600 temp file, then renamed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self.to_json(), f)
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "VoiceProfile | None":
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            emb = np.asarray(d["embedding"], dtype=np.float32)
            if d.get("version") != PROFILE_VERSION or emb.ndim != 1 or emb.size == 0:
                raise ValueError("unexpected profile layout")
            if d.get("model") != models.CAMPPLUS.name:
                raise ValueError(f"profile made with {d.get('model')!r}")
        except Exception as e:
            log.warning("ignoring voice profile %s (%s); enrol again", path, e)
            return None
        return cls(emb / (np.linalg.norm(emb) + 1e-9), d["model"], str(d.get("created", "")), int(d.get("clips", 0)))


class SpeakerGate:
    """Owns the voice profile and the embedding model.

    check() is the one question the orchestrator asks: is this capture the
    enrolled voice? It never waits on the network or on a model load: until
    the model is loaded (startup and enrolment load it, in the background or
    up front) it answers "yes", as it does with no profile or verification
    off — a missing model must not make her deaf, and the confirm gate never
    depends on it to say no. A load failure is remembered (`failed`, shown in
    Settings) and retried in the background at most every RETRY_S; a single
    scoring error is only logged."""

    _model_cls = SpeakerModel  # swapped in tests

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.path = profile_path(settings)
        self.profile = VoiceProfile.load(self.path)
        self._model: SpeakerModel | None = None
        self._lock = threading.Lock()
        self._failed_at: float | None = None      # monotonic time of the last failed load
        self._loader: threading.Thread | None = None
        self._not_ready_logged = False
        self._error_logged_at = -1e9
        # The last few checks, newest first, for the Settings window (read
        # from the AppKit thread while checks append from worker threads).
        self.recent: deque[dict] = deque(maxlen=RECENT_MAX)
        self._recent_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.profile is not None and self.s.speaker_verification

    @property
    def failed(self) -> bool:
        """The model couldn't be loaded: every voice is being accepted."""
        return self._failed_at is not None and self._model is None

    def model_ready(self) -> bool:
        return self._model is not None or (self.s.models_dir / models.CAMPPLUS.name).exists()

    def prepare(self) -> bool:
        """Fetch (if needed) and load the model; False if that failed. May
        block on a download: startup, enrolment and the background loader
        only, never a turn's check."""
        with self._lock:
            if self._model is not None:
                return True
            try:
                path = models.ensure(models.CAMPPLUS, self.s.models_dir)
                self._model = self._model_cls(path)
            except Exception:
                log.exception("speaker model unavailable; accepting every voice")
                self._failed_at = time.monotonic()
                return False
            self._failed_at = None
            self._not_ready_logged = False
            return True

    def _load_in_background(self) -> None:
        loader = self._loader
        if loader is not None and loader.is_alive():
            return
        if self._failed_at is not None and time.monotonic() - self._failed_at < RETRY_S:
            return
        self._loader = threading.Thread(target=self.prepare, name="veronica-speaker-model", daemon=True)
        self._loader.start()

    def embed(self, pcm16: np.ndarray) -> np.ndarray:
        """Enrolment's embedding: loads the model first if it has to."""
        if self._model is None and not self.prepare():
            raise RuntimeError("speaker model unavailable")
        return self._model.embed(voiced(pcm16))

    def threshold_for(self, pcm16: np.ndarray, where: str = "request") -> float:
        scale = SHORT_FLOOR + (1 - SHORT_FLOOR) * min(1.0, voiced_s(pcm16) / FULL_S)
        if where == "confirm":
            scale = max(scale, CONFIRM_FLOOR)
        return float(self.s.speaker_threshold) * scale

    def check(self, pcm16: np.ndarray, where: str) -> tuple[bool, float | None]:
        """(accepted, score). score is None when no check ran."""
        profile, model = self.profile, self._model
        if not self.active or profile is None:
            return True, None
        if model is None:
            if not self._not_ready_logged:
                self._not_ready_logged = True
                log.warning("speaker %s: model not loaded%s; accepting", where,
                            " (it failed, see above)" if self.failed else " yet")
            self._load_in_background()
            return True, None
        t0 = time.monotonic()
        try:
            score = float(profile.embedding @ model.embed(voiced(pcm16)))
        except Exception:
            if t0 - self._error_logged_at > 60:
                self._error_logged_at = t0
                log.exception("speaker %s: check failed; accepting this one", where)
            return True, None
        threshold = self.threshold_for(pcm16, where)
        ok = score >= threshold
        log.info("speaker %s: score=%.3f threshold=%.2f -> %s (%.1f s voiced of %.1f s, %d ms)", where, score,
                 threshold, "accepted" if ok else "ignored", voiced_s(pcm16),
                 pcm16.size / self.s.sample_rate, (time.monotonic() - t0) * 1000)
        with self._recent_lock:
            self.recent.appendleft({"where": where, "score": round(score, 3), "accepted": ok,
                                    "at": dt.datetime.now().strftime("%H:%M:%S")})
        return ok, score

    def check_wake(self, window: np.ndarray) -> bool:
        """The wake engine's hook: only consulted when the user asked for
        the wake word itself to be theirs (speaker_verification_wake)."""
        if not self.s.speaker_verification_wake:
            return True
        return self.check(window, "wake")[0]

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two recordings' voices (enrolment uses it to
        catch a take that is really her own voice coming back)."""
        return float(self.embed(a) @ self.embed(b))
    def enrol(self, clips: list[np.ndarray]) -> tuple[VoiceProfile | None, float]:
        """Build and save a profile from `clips` (raw int16). Returns
        (profile, agreement): agreement is the lowest similarity between any
        clip and the mean of the others; under ENROL_MIN_AGREEMENT nothing
        is saved and the profile is None."""
        embs = [self.embed(c) for c in clips]
        agreement = 1.0
        for i, e in enumerate(embs):
            rest = np.mean([o for j, o in enumerate(embs) if j != i], axis=0) if len(embs) > 1 else e
            agreement = min(agreement, float(e @ (rest / (np.linalg.norm(rest) + 1e-9))))
        log.info("voice enrolment: %d clips, agreement %.3f", len(clips), agreement)
        if agreement < ENROL_MIN_AGREEMENT:
            return None, agreement
        mean = np.mean(embs, axis=0)
        profile = VoiceProfile(mean / (np.linalg.norm(mean) + 1e-9), models.CAMPPLUS.name,
                               dt.datetime.now().isoformat(timespec="seconds"), len(clips))
        profile.save(self.path)
        self.profile = profile
        with self._recent_lock:
            self.recent.clear()
        return profile, agreement

    def forget(self) -> bool:
        """Delete the profile. True if there was one."""
        had = self.profile is not None or self.path.exists()
        self.path.unlink(missing_ok=True)
        self.profile = None
        with self._recent_lock:
            self.recent.clear()
        if had:
            log.info("voice profile forgotten")
        return had

    def status(self) -> dict:
        p = self.profile
        with self._recent_lock:
            recent = list(self.recent)
        return {"enrolled": p is not None, "created": p.created if p is not None else "",
                "active": self.active, "failed": self.active and self.failed, "recent": recent}
