"""Helpers for HUD events: audio levels and envelopes."""
import math

import numpy as np

EVENT_KINDS = frozenset({"state", "heard", "heard_partial", "sentence", "tool", "prompt", "mic", "voice", "warm", "hud"})


def rms(pcm16: np.ndarray) -> float:
    """RMS of int16 audio scaled to 0..1."""
    if pcm16.size == 0:
        return 0.0
    x = pcm16.astype(np.float32) / 32768.0
    return float(min(1.0, math.sqrt(float(np.mean(x * x)))))


def envelope(samples: np.ndarray, sample_rate: int, step_ms: int = 50) -> list[float]:
    """RMS per step_ms window, normalized so the loudest window is 1.0."""
    step = max(1, sample_rate * step_ms // 1000)
    n = samples.size
    if n == 0:
        return [0.0]
    out = []
    for i in range(0, n, step):
        w = samples[i : i + step].astype(np.float32)
        out.append(float(math.sqrt(float(np.mean(w * w)))) if w.size else 0.0)
    peak = max(out)
    return [v / peak for v in out] if peak > 0 else [0.0 for _ in out]
