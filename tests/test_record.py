import asyncio
import threading

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
    defaults = {"vad_silence_ms": 90, "min_speech_ms": 60, "max_utterance_s": 1}
    defaults.update(over)
    s = Settings(**defaults)
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


async def test_max_s_does_not_cap_utterance(monkeypatch):
    # 1 s wait budget, speech starts at frame 20 (600 ms) and runs 50 frames (1.5 s)
    r = make("." * 20 + "s" * 50 + "....", monkeypatch, max_utterance_s=3)
    pcm = await r.capture(max_s=1)
    assert pcm is not None
    assert len(pcm) == FRAME * 53   # 50 speech + 3 silence frames


async def test_stop_returns_none_and_is_consumed(monkeypatch):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1)
    state = {"pattern": ""}  # infinite silence via frames()'s trailing `while True: yield zeros`
    started = threading.Event()

    def frame_source():
        for i, frame in enumerate(frames(state["pattern"])):
            if i == 0:
                started.set()   # synchronize: capture has begun before we stop() it
            yield frame

    r = Recorder(s, frames=frame_source)

    task = asyncio.create_task(r.capture(max_s=None))
    await asyncio.to_thread(started.wait, 2)
    r.stop()
    pcm = await asyncio.wait_for(task, timeout=2)
    assert pcm is None

    # stop() is one-shot: a subsequent capture on the same recorder is unaffected.
    state["pattern"] = "....ssssss........."
    pcm2 = await r.capture()
    assert pcm2 is not None


async def test_stop_when_not_capturing_is_noop(monkeypatch):
    r = make("....ssssss.........", monkeypatch)
    r.stop()   # no capture in flight: must not affect the next capture
    pcm = await r.capture()
    assert pcm is not None


async def test_on_level_called_per_frame(monkeypatch):
    levels = []
    r = make("..sss..", monkeypatch)
    r._on_level = levels.append          # constructor kwarg is on_level=; set after make() for simplicity
    await r.capture()
    assert len(levels) >= 7 and all(0.0 <= v <= 1.0 for v in levels)
    assert max(levels) > 0.0            # speech frames are non-zero


# -- pre-roll handoff (item 1) -------------------------------------------------

async def test_preroll_speech_captured_then_endpointed_by_live_silence(monkeypatch):
    r = make("", monkeypatch)  # live frames: pure silence forever
    preroll = np.full(FRAME * 6, 1000, dtype=np.int16)
    pcm = await r.capture(preroll=preroll)
    assert pcm is not None
    # 6 preroll speech frames + 3 live silence frames to endpoint (90 ms / 30 ms)
    assert len(pcm) == FRAME * 9
    assert np.all(pcm[: FRAME * 6] == 1000)


async def test_preroll_all_silence_then_live_speech_works_as_before(monkeypatch):
    r = make("....ssssss.........", monkeypatch)
    preroll = np.zeros(FRAME * 5, dtype=np.int16)
    pcm = await r.capture(preroll=preroll)
    assert pcm is not None
    assert len(pcm) == FRAME * 9  # same result as without any preroll at all


def test_has_speech(monkeypatch):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings()
    r = Recorder(s, frames=lambda: iter([]))
    assert r.has_speech(np.full(FRAME * 3, 1000, dtype=np.int16)) is True
    assert r.has_speech(np.zeros(FRAME * 3, dtype=np.int16)) is False
    assert r.has_speech(np.zeros(0, dtype=np.int16)) is False
    assert r.has_speech(None) is False

