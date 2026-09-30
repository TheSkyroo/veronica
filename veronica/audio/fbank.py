"""Kaldi-compatible log-mel filterbank features in numpy.

The speaker-embedding model was trained on Kaldi `compute-fbank-feats`
features (80 bins, 25 ms / 10 ms frames, Hamming window, no dither), so the
input has to match them closely; this reproduces kaldi's defaults (DC
removal, 0.97 pre-emphasis, 512-point power spectrum, mel banks from 20 Hz
to Nyquist on kaldi's 1127·ln(1 + f/700) scale, log floored at float eps)
without pulling in torchaudio or a native feature library."""
import functools

import numpy as np

_EPS = float(np.finfo(np.float32).eps)


@functools.lru_cache(maxsize=4)
def _mel_banks(num_bins: int, n_fft: int, sample_rate: int, low: float = 20.0) -> np.ndarray:
    """(num_bins, n_fft // 2 + 1) triangular filters, kaldi's layout: the
    Nyquist bin gets no weight."""
    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    high = sample_rate / 2
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (num_bins + 1)
    bin_mel = mel(np.arange(n_fft // 2) * sample_rate / n_fft)
    banks = np.zeros((num_bins, n_fft // 2 + 1), dtype=np.float64)
    for i in range(num_bins):
        left, center, right = lo + i * delta, lo + (i + 1) * delta, lo + (i + 2) * delta
        up = (bin_mel - left) / (center - left)
        down = (right - bin_mel) / (right - center)
        banks[i, : n_fft // 2] = np.maximum(0.0, np.minimum(up, down))
    return banks.astype(np.float32)


@functools.lru_cache(maxsize=4)
def _window(kind: str, n: int) -> np.ndarray:
    a = 2 * np.pi * np.arange(n) / (n - 1)
    if kind == "povey":
        return (0.5 - 0.5 * np.cos(a)) ** 0.85
    return 0.54 - 0.46 * np.cos(a)  # hamming


def fbank(
    samples: np.ndarray,
    sample_rate: int = 16000,
    num_bins: int = 80,
    frame_ms: float = 25.0,
    shift_ms: float = 10.0,
    window: str = "hamming",
) -> np.ndarray:
    """(frames, num_bins) float32 log-mel energies of float `samples` (in
    whatever scale the model expects: kaldi's own tools feed int16 values).
    Frames are snipped at the edges (kaldi's snip_edges=true); audio shorter
    than one frame gives an empty (0, num_bins) array."""
    x = np.asarray(samples, dtype=np.float64)
    flen = int(sample_rate * frame_ms / 1000)
    fshift = int(sample_rate * shift_ms / 1000)
    if x.size < flen:
        return np.zeros((0, num_bins), dtype=np.float32)
    n_frames = 1 + (x.size - flen) // fshift
    idx = np.arange(flen)[None, :] + fshift * np.arange(n_frames)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    # pre-emphasis, kaldi style: the first sample is emphasised against itself
    frames = np.concatenate([frames[:, :1] * (1 - 0.97), frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    frames = frames * _window(window, flen)
    n_fft = 1 << (flen - 1).bit_length()
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    mel = power.astype(np.float32) @ _mel_banks(num_bins, n_fft, sample_rate).T
    return np.log(np.maximum(mel, _EPS)).astype(np.float32)
