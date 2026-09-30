"""Streaming noise suppression (GTCRN) for the mic path.

GTCRN is a 48k-parameter speech-enhancement network that runs one 16 ms STFT
frame at a time (512-point sqrt-Hann window, 256-sample hop) and carries its
own recurrent state, so it can clean the mic stream frame by frame, before
the recorder's VAD and speech level floor see it. That is all it is used
for: whisper and the speaker check get the raw audio, and the wake check
doesn't use it (both measured worse on suppressed audio). On an M5 it costs
~3% of one core while a capture is open. Why this model and not RNNoise / a
spectral gate / WebRTC NS: see
docs/superpowers/specs/2026-10-01-veronica-voice-isolation-design.md."""
import functools
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from veronica.audio import models
from veronica.config import Settings

log = logging.getLogger("veronica.audio")

N_FFT = 512
HOP = 256
# Output lags input by this many samples: the analysis window's overlap
# (N_FFT - HOP) plus one hop of output buffered so any chunk size can be
# answered with a chunk of the same size.
LAG = N_FFT
_CACHE_SHAPES = {"conv_cache": (2, 1, 16, 16, 33), "tra_cache": (2, 3, 1, 1, 16), "inter_cache": (2, 1, 33, 16)}
# GTCRN's suppression depends on the input level: pink noise at a real mic's
# room level (rms 0.001-0.01) lost only ~10 dB, the same noise at 0.03-0.1
# lost ~30 dB. So each frame is scaled towards AGC_TARGET before the network
# and back after it — an envelope that rises fast (a voice starting must not
# be overdriven) and falls slowly (the gap between words must not pump).
AGC_TARGET = 0.05
AGC_MAX_GAIN = 32.0
AGC_ATTACK = 0.5      # per 16 ms hop
AGC_RELEASE = 0.01    # per hop: ~1.6 s to settle
# A 60 Hz high-pass first, as speech front ends do: GTCRN passes sub-audio
# rumble (desk thumps, a mic's DC drift) straight through, and at room
# level that rumble was most of what was left of the noise — enough to hold
# the speech energy gate open.
_HPF = butter(2, 60, btype="highpass", fs=16000, output="sos")
# periodic sqrt-Hann: analysis x synthesis = Hann, which sums to 1 at a half-window hop
_WINDOW = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


def _new_session(path: str):
    import onnxruntime as ort  # heavy import; only once a denoiser is wanted

    opts = ort.SessionOptions()
    # One thread: a 256-sample frame is far too small to split, and the
    # wake loop's whisper wants the other cores.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


_session_factory = _new_session  # swapped in tests


@functools.lru_cache(maxsize=2)
def _session(path: str):
    # InferenceSession.run is thread-safe, so every capture's denoiser
    # shares one; each keeps its own recurrent state.
    return _session_factory(path)


class Denoiser:
    """Cleans int16 mono 16 kHz audio chunk by chunk. `process` returns as
    many samples as it was given, delayed by LAG samples (32 ms); state
    carries across calls until `reset`."""

    def __init__(self, session) -> None:
        self._sess = session
        self.reset()

    def reset(self) -> None:
        self._caches = {k: np.zeros(v, dtype=np.float32) for k, v in _CACHE_SHAPES.items()}
        self._frame = np.zeros(N_FFT, dtype=np.float32)
        self._ola = np.zeros(N_FFT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._out = np.zeros(HOP, dtype=np.float32)   # the extra hop of LAG
        self._env = AGC_TARGET / AGC_MAX_GAIN            # start at full gain: a quiet room
        self._hpf = np.zeros((_HPF.shape[0], 2))

    def _gain(self) -> float:
        level = float(np.sqrt(np.mean(self._frame * self._frame)))
        rate = AGC_ATTACK if level > self._env else AGC_RELEASE
        self._env += rate * (level - self._env)
        return min(AGC_MAX_GAIN, max(1.0, AGC_TARGET / max(self._env, 1e-9)))

    def _hop(self, x: np.ndarray) -> np.ndarray:
        self._frame = np.concatenate([self._frame[HOP:], x])
        gain = self._gain()
        spec = np.fft.rfft(self._frame * (_WINDOW * gain))
        mix = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)[None, :, None, :]
        enh, conv, tra, inter = self._sess.run(None, {"mix": mix, **self._caches})
        self._caches = {"conv_cache": conv, "tra_cache": tra, "inter_cache": inter}
        y = np.fft.irfft(enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1], N_FFT).astype(np.float32) * (_WINDOW / gain)
        self._ola += y
        out = self._ola[:HOP].copy()
        self._ola = np.concatenate([self._ola[HOP:], np.zeros(HOP, dtype=np.float32)])
        return out

    def process(self, pcm16: np.ndarray) -> np.ndarray:
        n = pcm16.size
        if n == 0:
            return np.zeros(0, dtype=np.int16)
        hp, self._hpf = sosfilt(_HPF, pcm16.astype(np.float64) / 32768.0, zi=self._hpf)
        x = np.concatenate([self._pending, hp.astype(np.float32)])
        hops = x.size // HOP
        outs = [self._out] + [self._hop(x[i * HOP:(i + 1) * HOP]) for i in range(hops)]
        self._pending = x[hops * HOP:]
        out = np.concatenate(outs)
        self._out = out[n:]
        return np.clip(np.round(out[:n] * 32768.0), -32768, 32767).astype(np.int16)

    def process_bytes(self, frame: bytes) -> bytes:
        return self.process(np.frombuffer(frame, dtype=np.int16)).tobytes()


_missing_logged: set[str] = set()


def make_denoiser(settings: Settings) -> Denoiser | None:
    """A fresh Denoiser when noise suppression is on and its model is on
    disk; None otherwise (the caller then uses the raw mic audio). A
    missing model is logged once per path; `prepare_model` gets it."""
    if not settings.noise_suppression:
        return None
    path = settings.models_dir / models.GTCRN.name
    if not path.exists():
        if str(path) not in _missing_logged:
            _missing_logged.add(str(path))
            log.warning("noise suppression is on but %s is missing; using raw mic audio "
                        "(run scripts/download_models.py, or restart online)", path)
        return None
    try:
        return Denoiser(_session(str(path)))
    except Exception:
        if str(path) not in _missing_logged:
            _missing_logged.add(str(path))
            log.exception("noise suppression model failed to load; using raw mic audio")
        return None


def prepare_model(settings: Settings) -> Path | None:
    """Download the model if suppression is on and it's missing, and load
    it, so the first capture doesn't wait on either (startup runs this off
    the main thread). Never raises: offline just means raw audio until the
    next launch."""
    if not settings.noise_suppression:
        return None
    try:
        path = models.ensure(models.GTCRN, settings.models_dir)
        _session(str(path))
        return path
    except Exception as e:
        log.warning("couldn't prepare the noise suppression model: %s", e)
        return None
