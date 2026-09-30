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
# in; only 30 ms frames within 20 dB of the loud part are embedded.
_FRAME = 480
_VOICED_DB = 20
# Short answers ("yes", "haan") embed less reliably, so they score lower
# against the profile: the threshold scales down linearly from full at
# FULL_S of voiced audio to SHORT_FLOOR of it at none (see the spec's table).
FULL_S = 2.0
SHORT_FLOOR = 0.6


def voiced(pcm16: np.ndarray) -> np.ndarray:
    """The loud-enough 30 ms frames of `pcm16`, joined; all of it when fewer
    than ten frames qualify."""
    n = pcm16.size // _FRAME
    if n == 0:
        return pcm16
    frames = pcm16[: n * _FRAME].reshape(n, _FRAME)
    level = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    keep = level >= np.percentile(level, 95) * 10 ** (-_VOICED_DB / 20)
    if keep.sum() < 10:
        return pcm16
    return frames[keep].ravel()


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
    """Owns the voice profile and the (lazily loaded) embedding model.

    check() is the one question the orchestrator asks: is this capture the
    enrolled voice? It fails open — no profile, verification off, or a
    model that won't load all mean "yes" (logged), because a broken model
    must not make her deaf; the confirm gate never depends on it to say no."""

    _model_cls = SpeakerModel  # swapped in tests

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.path = profile_path(settings)
        self.profile = VoiceProfile.load(self.path)
        self._model: SpeakerModel | None = None
        self._lock = threading.Lock()
        self._load_failed = False
        # The last few checks, newest first, for the Settings window (read
        # from the AppKit thread while checks append from worker threads).
        self.recent: deque[dict] = deque(maxlen=RECENT_MAX)
        self._recent_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.profile is not None and self.s.speaker_verification

    def model_ready(self) -> bool:
        return self._model is not None or (self.s.models_dir / models.CAMPPLUS.name).exists()

    def _get_model(self) -> SpeakerModel:
        with self._lock:
            if self._model is None:
                path = models.ensure(models.CAMPPLUS, self.s.models_dir)
                self._model = self._model_cls(path)
            return self._model

    def prepare(self) -> bool:
        """Fetch (if needed) and load the model; False if that failed."""
        try:
            self._get_model()
            self._load_failed = False
            return True
        except Exception:
            log.exception("speaker model unavailable")
            return False

    def embed(self, pcm16: np.ndarray) -> np.ndarray:
        return self._get_model().embed(voiced(pcm16))

    def score(self, pcm16: np.ndarray) -> float:
        profile = self.profile
        if profile is None:
            raise RuntimeError("no voice profile")
        return float(profile.embedding @ self.embed(pcm16))

    def threshold_for(self, pcm16: np.ndarray) -> float:
        speech_s = voiced(pcm16).size / self.s.sample_rate
        scale = SHORT_FLOOR + (1 - SHORT_FLOOR) * min(1.0, speech_s / FULL_S)
        return float(self.s.speaker_threshold) * scale

    def check(self, pcm16: np.ndarray, where: str) -> tuple[bool, float | None]:
        """(accepted, score). score is None when no check ran."""
        if not self.active or self._load_failed:
            return True, None
        t0 = time.monotonic()
        try:
            score = self.score(pcm16)
        except Exception:
            if not self._load_failed:
                log.exception("speaker check failed; accepting every voice until restart")
                self._load_failed = True
            return True, None
        threshold = self.threshold_for(pcm16)
        ok = score >= threshold
        log.info("speaker %s: score=%.3f threshold=%.2f -> %s (%.1f s voiced of %.1f s, %d ms)", where, score,
                 threshold, "accepted" if ok else "ignored", voiced(pcm16).size / self.s.sample_rate,
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
                "active": self.active, "recent": recent}
