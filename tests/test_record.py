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


async def test_preroll_tail_only_waits_for_real_live_onset(monkeypatch):
    """A pre-roll onset with too few speech frames to count as a real
    utterance (e.g. just the tail end of the wake word itself) must not
    give up — it should keep waiting for genuine live speech."""
    # 2 speech frames in the preroll (below min_speech_frames=2? use a
    # higher min so 2 preroll frames alone don't already qualify), then
    # 1 s of live silence (well past vad_silence_ms), then live speech.
    r = make(
        "." * 33 + "ssssss.........",  # ~1s silence (33*30ms), then real speech + endpoint
        monkeypatch, min_speech_ms=120,  # 4 frames needed; the 2 preroll frames alone don't qualify
    )
    preroll = np.full(FRAME * 2, 1000, dtype=np.int16)
    pcm = await r.capture(preroll=preroll)
    assert pcm is not None
    # returns the live speech (6 frames) + 3 silence frames to endpoint; the
    # false-started preroll frames were discarded.
    assert len(pcm) == FRAME * 9
    assert np.all(pcm[: FRAME * 6] == 1000)  # the live speech, not the discarded preroll


async def test_preroll_tail_only_returns_none_if_no_live_onset_within_max_s(monkeypatch):
    r = make("." * 100, monkeypatch, min_speech_ms=120)  # plenty of live silence, no speech ever
    preroll = np.full(FRAME * 2, 1000, dtype=np.int16)
    pcm = await r.capture(max_s=1, preroll=preroll)
    assert pcm is None


def test_has_speech(monkeypatch):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings()
    r = Recorder(s, frames=lambda: iter([]))
    # >= 5 speech frames (150 ms) required
    assert r.has_speech(np.full(FRAME * 5, 1000, dtype=np.int16)) is True
    assert r.has_speech(np.full(FRAME * 3, 1000, dtype=np.int16)) is False  # too few frames
    assert r.has_speech(np.zeros(FRAME * 3, dtype=np.int16)) is False
    assert r.has_speech(np.zeros(0, dtype=np.int16)) is False
    assert r.has_speech(None) is False


def test_has_speech_uses_a_separate_vad_instance(monkeypatch):
    """Thread safety: has_speech() must not share the Vad instance capture()
    uses on the worker thread."""
    created = []

    class TrackingVad(FakeVad):
        def __init__(self, level):
            super().__init__(level)
            created.append(self)

    monkeypatch.setattr(Recorder, "_vad_cls", TrackingVad)
    s = Settings()
    r = Recorder(s, frames=lambda: iter([]))
    assert len(created) == 1  # the constructor's own instance
    r.has_speech(np.full(FRAME * 5, 1000, dtype=np.int16))
    assert len(created) == 2  # has_speech made its own, didn't reuse self._vad
    assert created[1] is not r._vad


# -- live partial transcript (item 3) ------------------------------------------

async def test_on_audio_called_after_hop_of_speech_not_before(monkeypatch):
    calls = []
    # 30 ms/frame; partial_hop_s=0.09 -> 3 frames per hop
    r = make("ssssssssss..........", monkeypatch, partial_hop_s=0.09)
    r.on_audio = calls.append
    await r.capture(partial=True)
    assert len(calls) >= 1
    # not called before speech starts: every call carries only started-buffer audio
    for c in calls:
        assert c.size > 0
        assert np.all(c == 1000) or 1000 in c
    # first call happens after >= 3 speech frames (0.09s), not on frame 1
    assert calls[0].size >= FRAME * 3


async def test_on_audio_not_called_when_partial_is_false(monkeypatch):
    calls = []
    r = make("ssssssssss..........", monkeypatch, partial_hop_s=0.09)
    r.on_audio = calls.append
    await r.capture()  # partial defaults to False (e.g. confirm()'s yes/no capture)
    assert calls == []


async def test_on_audio_hop_counts_only_speech_frames(monkeypatch):
    """Trailing silence must not advance the partial hop counter: only 2
    speech frames (short of the 3-frame hop) followed by many silence
    frames (far more than 3 total buffered frames) must never fire
    on_audio, since the hop only ever sees 2 speech frames."""
    calls = []
    r = make("ss" + "." * 10, monkeypatch, partial_hop_s=0.09, vad_silence_ms=90)
    r.on_audio = calls.append
    await r.capture(partial=True)
    assert calls == []


# -- follow-up window: a brief false onset (e.g. Veronica's own audio tail) --

async def test_short_live_speech_then_real_speech_within_wait_budget(monkeypatch):
    """A brief false onset (2 speech frames, below min_speech_frames) that
    endpoints on silence must not give up — with wait budget left, it should
    keep waiting for a real onset, here arriving ~1.5s in."""
    pattern = "ss" + "." * 48 + "ssssss" + "....."
    r = make(pattern, monkeypatch, min_speech_ms=120, max_utterance_s=5)
    pcm = await r.capture(max_s=4)
    assert pcm is not None
    assert len(pcm) == FRAME * 9   # 6 real speech frames + 3 silence to endpoint
    assert np.all(pcm[: FRAME * 6] == 1000)


async def test_short_live_speech_with_no_wait_budget_returns_none(monkeypatch):
    """Same pattern, but with no wait budget (max_s=None): old behavior —
    give up and return None once the brief false onset endpoints."""
    pattern = "ss" + "." * 48 + "ssssss" + "....."
    r = make(pattern, monkeypatch, min_speech_ms=120, max_utterance_s=5)
    pcm = await r.capture(max_s=None)
    assert pcm is None


async def test_skip_ms_drops_leading_live_frames(monkeypatch):
    """skip_ms discards the first skip_ms of *live* frames before the VAD
    ever sees them — speech in those frames is ignored entirely."""
    # 10 frames (300 ms) of speech, then silence, then real speech.
    pattern = "s" * 10 + "." * 5 + "ssssss" + "....."
    r = make(pattern, monkeypatch, min_speech_ms=60, max_utterance_s=5)
    pcm = await r.capture(max_s=4, skip_ms=300)
    assert pcm is not None
    assert len(pcm) == FRAME * 9   # only the real speech (6) + 3 silence to endpoint
    assert np.all(pcm[: FRAME * 6] == 1000)


async def test_repeated_false_onsets_bounded_by_hard_wait_cap(monkeypatch):
    """Repeated false onsets (e.g. a bursty noise source alternating short
    speech bursts with gaps) must not let the reset-and-keep-waiting logic
    inflate the real wait far past max_s: a hard cap on total live-frame
    elapsed time (wait_frames + max_frames) always wins, regardless of
    started/reset state."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(
        vad_silence_ms=90, min_speech_ms=150, max_utterance_s=1,
    )
    fm = s.frame_ms
    wait_frames = 1000 // fm       # max_s=1
    max_frames = s.max_utterance_s * 1000 // fm
    consumed = []

    def infinite_burst_gap():
        while True:
            for _ in range(2):
                consumed.append(1)
                yield np.full(FRAME, 1000, dtype=np.int16).tobytes()
            for _ in range(4):
                consumed.append(1)
                yield np.zeros(FRAME, dtype=np.int16).tobytes()

    r = Recorder(s, frames=infinite_burst_gap)
    pcm = await r.capture(max_s=1)
    assert pcm is None
    assert len(consumed) <= wait_frames + max_frames + 1
