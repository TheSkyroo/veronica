import numpy as np


def tone(freq_hz: float, ms: int, sample_rate: int = 24000, volume: float = 0.3) -> np.ndarray:
    """Short sine burst with 5 ms fades, float32 mono."""
    n = sample_rate * ms // 1000
    t = np.arange(n, dtype=np.float32) / sample_rate
    y = np.sin(2 * np.pi * freq_hz * t).astype(np.float32) * volume
    fade = max(1, sample_rate * 5 // 1000)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y
