import numpy as np
import pytest

from veronica.audio.fbank import fbank

# Frame 5, bins 0/10/25/40/60/79 of kaldi-native-fbank 1.x (dither 0, 80
# bins, snip edges) on the signal below: the reference the speaker model's
# features have to match.
KALDI = {
    "hamming": [11.3897, 11.1322, 13.415, 12.0018, 13.5496, 14.0411],
    "povey": [7.8033, 14.7872, 7.5509, 3.7603, 5.7384, 6.8633],
}


def signal():
    t = np.arange(3200) / 16000
    return 8000 * np.sin(2 * np.pi * 440 * t) + 2000 * np.sin(2 * np.pi * 3000 * t) + 500 * np.sin(2 * np.pi * 7000 * t)


@pytest.mark.parametrize("window", ["hamming", "povey"])
def test_matches_kaldi(window):
    f = fbank(signal(), window=window)
    assert f.shape == (18, 80) and f.dtype == np.float32
    assert f[5, [0, 10, 25, 40, 60, 79]] == pytest.approx(KALDI[window], abs=5e-3)


def test_too_short_for_a_frame_is_empty():
    assert fbank(np.zeros(399)).shape == (0, 80)


def test_silence_is_floored_not_minus_infinity():
    f = fbank(np.zeros(1600))
    assert np.isfinite(f).all()
