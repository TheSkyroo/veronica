import math
import numpy as np

from veronica.ui.events import EVENT_KINDS, envelope, rms


def test_rms_range():
    assert rms(np.zeros(480, dtype=np.int16)) == 0.0
    full = np.full(480, 32767, dtype=np.int16)
    assert 0.99 <= rms(full) <= 1.0
    assert rms(np.zeros(0, dtype=np.int16)) == 0.0


def test_envelope_shape_and_normalization():
    sr = 24000
    x = np.zeros(sr, dtype=np.float32)          # 1 s
    x[sr // 2 : sr // 2 + 1200] = 0.5            # 50 ms burst in the middle
    env = envelope(x, sr, 50)
    assert len(env) == math.ceil(sr / (sr * 50 // 1000))
    assert max(env) == 1.0 and min(env) == 0.0
    assert envelope(np.zeros(10, dtype=np.float32), sr) == [0.0]


def test_event_kinds():
    assert EVENT_KINDS == {"state", "heard", "heard_partial", "sentence", "tool", "prompt", "mic", "voice", "warm"}
