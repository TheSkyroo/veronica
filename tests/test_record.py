import numpy as np

from veronica.audio.record import Recorder
from veronica.config import Settings

FRAME = 480  # 30 ms @ 16 kHz


class FakeVad:
    """Speech if frame is non-zero."""

    def __init__(self, level):
        pass

    def is_speech(self, frame_bytes, sample_rate):
        return any(frame_bytes)


def frames(pattern):
    """pattern: string of 's' (speech) / '.' (silence), one char per 30 ms frame."""
    for ch in pattern:
        val = 1000 if ch == "s" else 0
        yield np.full(FRAME, val, dtype=np.int16).tobytes()
    while True:
        yield np.zeros(FRAME, dtype=np.int16).tobytes()


def make(pattern, monkeypatch, **over):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1, **over)
    return Recorder(s, frames=lambda: frames(pattern))


async def test_returns_speech_then_stops_on_silence(monkeypatch):
    r = make("....ssssss.........", monkeypatch)
    pcm = await r.capture()
    assert pcm is not None
    # 6 speech frames + 3 silence frames (90 ms) captured
    assert len(pcm) == FRAME * 9


async def test_too_short_speech_returns_none(monkeypatch):
    r = make("..s....", monkeypatch)
    assert await r.capture() is None


async def test_no_speech_before_max_returns_none(monkeypatch):
    r = make(".......................................", monkeypatch)
    assert await r.capture(max_s=1) is None


async def test_max_utterance_cap(monkeypatch):
    r = make("s" * 200, monkeypatch)  # 6 s of speech, cap 1 s
    pcm = await r.capture()
    assert len(pcm) <= 16000 + FRAME
