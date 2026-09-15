import numpy as np

from veronica.audio.chime import tone


def test_tone_shape_and_range():
    t = tone(880, 120)
    assert t.dtype == np.float32
    assert len(t) == 24000 * 120 // 1000
    assert np.abs(t).max() <= 0.3 + 1e-6
    assert abs(t[0]) < 1e-3 and abs(t[-1]) < 1e-3  # faded ends
