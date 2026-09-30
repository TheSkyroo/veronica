import asyncio
import threading

import numpy as np
import pytest

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
    elapsed time spent waiting for a real onset (wait_frames + extra_frames)
    always wins, regardless of reset state — as long as no utterance is
    actually in progress (`not started`)."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(
        vad_silence_ms=90, min_speech_ms=150, max_utterance_s=1, capture_extra_s=3.0,
    )
    fm = s.frame_ms
    wait_frames = 1000 // fm       # max_s=1
    extra_frames = int(s.capture_extra_s * 1000 // fm)
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
    assert len(consumed) <= wait_frames + extra_frames + 1


async def test_real_speech_in_progress_not_cut_by_onset_wait_cap(monkeypatch):
    """Once real speech has started within the wait budget, the onset-wait
    hard cap (wait_frames + extra_frames) must not apply — the utterance is
    bounded only by max_utterance_s, same as before."""
    s_ms = 30
    wait_s = 1
    wait_frames = wait_s * 1000 // s_ms
    # speech starts one frame before the wait budget would expire, and lasts
    # 2 s (well past wait_frames + extra_frames worth of *silence*, but this
    # is speech, not silence, so the cap must not fire).
    speech_frames = 2000 // s_ms
    pattern = "." * (wait_frames - 1) + "s" * speech_frames + "....."
    r = make(pattern, monkeypatch, max_utterance_s=5, capture_extra_s=0.5)
    pcm = await r.capture(max_s=wait_s)
    assert pcm is not None
    # full speech run + 3 silence frames (90 ms) to endpoint
    assert len(pcm) == FRAME * (speech_frames + 3)


# -- hold mode / finish() (A2 push-to-talk) ------------------------------------
#
# These gate the frame generator on a threading.Event (rather than an
# arbitrary frame count) so the background capture thread is genuinely
# blocked — not just "probably still running" — at the moment the test
# asserts task.done()/calls finish(): the generator runs on the same
# to_thread worker thread as _capture(), so blocking it deterministically
# pauses capture() without any wall-clock race.

def _gated_frames(pattern, block_after_index, started: threading.Event, gate: threading.Event):
    for i, frame in enumerate(frames(pattern)):
        yield frame
        if i == block_after_index:
            started.set()
            gate.wait()


async def test_hold_mode_ignores_silence_endpoint_until_finish(monkeypatch):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    started, gate = threading.Event(), threading.Event()
    # speech (4 frames) then silence well past the 90ms/3-frame endpoint
    # that would have ended a normal (non-hold) capture by frame index 5.
    pattern = "ssss" + "." * 50
    r = Recorder(s, frames=lambda: _gated_frames(pattern, 5, started, gate))

    task = asyncio.create_task(r.capture(max_s=None, hold=True))
    await asyncio.to_thread(started.wait, 2)
    assert not task.done()   # normal-mode silence endpoint would have ended this by now
    r.finish()
    gate.set()
    pcm = await asyncio.wait_for(task, timeout=2)
    assert pcm is not None
    assert len(pcm) >= FRAME * 4  # at least the 4 speech frames captured


async def test_hold_mode_waits_for_onset_and_finish_before_any_speech(monkeypatch):
    """Releasing the PTT key before any speech was captured returns None,
    without hanging (max_s=None normally waits forever for onset)."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    started, gate = threading.Event(), threading.Event()
    r = Recorder(s, frames=lambda: _gated_frames("." * 500, 3, started, gate))

    task = asyncio.create_task(r.capture(max_s=None, hold=True))
    await asyncio.to_thread(started.wait, 2)
    assert not task.done()
    r.finish()
    gate.set()
    pcm = await asyncio.wait_for(task, timeout=2)
    assert pcm is None


async def test_hold_mode_returns_short_speech_below_min_speech_ms(monkeypatch):
    """Unlike normal mode, hold mode doesn't discard a too-short utterance —
    the user explicitly ended it via finish()."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=600, max_utterance_s=10)
    started, gate = threading.Event(), threading.Event()
    # block right after the single speech frame is captured (index 0)
    r = Recorder(s, frames=lambda: _gated_frames("s" + "." * 500, 0, started, gate))

    task = asyncio.create_task(r.capture(max_s=None, hold=True))
    await asyncio.to_thread(started.wait, 2)
    r.finish()
    gate.set()
    pcm = await asyncio.wait_for(task, timeout=2)
    assert pcm is not None
    assert len(pcm) == FRAME  # the single speech frame, well under min_speech_ms


async def test_finish_when_not_capturing_is_noop(monkeypatch):
    r = make("....ssssss.........", monkeypatch)
    r.finish()  # no capture in flight
    pcm = await r.capture()
    assert pcm is not None


async def test_finish_does_not_affect_non_hold_capture(monkeypatch):
    """finish() only matters in hold mode; a plain capture() must ignore it
    — and must clear a stale flag on entry (I4) so it can't linger into
    the *next* hold capture and cut it short."""
    r = make("....ssssss.........", monkeypatch)
    r._finish.set()
    pcm = await r.capture()
    assert pcm is not None
    assert len(pcm) == FRAME * 9
    assert not r._finish.is_set()


async def test_finish_during_non_hold_capture_is_dropped(monkeypatch):
    """finish() aimed at a normal capture doesn't set the flag at all: the
    capture ends on its own silence endpoint and nothing lingers."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    started, gate = threading.Event(), threading.Event()
    r = Recorder(s, frames=lambda: _gated_frames("ssssss.........", 2, started, gate))
    task = asyncio.create_task(r.capture(max_s=2))
    await asyncio.to_thread(started.wait, 2)
    r.finish()
    assert not r._finish.is_set()
    gate.set()
    pcm = await asyncio.wait_for(task, timeout=2)
    assert len(pcm) == FRAME * 9
    assert not r._finish.is_set()


async def test_stale_stop_and_finish_cleared_at_capture_start(monkeypatch):
    """A stop()/finish() that raced a previous capture's natural end (and so
    was never consumed) must not abort the next capture (I4)."""
    r = make("....ssssss.........", monkeypatch)
    r._stop.set()
    r._finish.set()
    pcm = await r.capture()
    assert pcm is not None and len(pcm) == FRAME * 9
    assert not r._stop.is_set() and not r._finish.is_set()


async def test_hold_capture_all_silence_ends_within_max_s(monkeypatch):
    """Regression (reviewer repro_hold): a hold capture with no speech and
    a lost key-up must not run forever — it's capped at max_s of live
    audio (C3)."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    consumed = 0

    def silence():
        nonlocal consumed
        for f in frames(""):
            consumed += 1
            yield f

    r = Recorder(s, frames=silence)
    r.finish()   # release arrived before capture() started (quick tap during chime): ignored
    pcm = await asyncio.wait_for(r.capture(max_s=1, hold=True), timeout=3)
    assert pcm is None
    assert consumed <= 1000 // 30 + 2   # ~1 s of 30 ms frames


async def test_hold_capture_without_max_s_is_capped_by_max_utterance_s(monkeypatch):
    """Reviewer repro_hold verbatim: max_s=None, max_utterance_s=1, all
    silence, a finish() that landed before the capture started -> returns
    within a bounded number of frames instead of running unbounded."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(max_utterance_s=1)
    consumed = 0

    def silence():
        nonlocal consumed
        for f in frames(""):
            consumed += 1
            yield f

    r = Recorder(s, frames=silence)
    r.finish()
    pcm = await asyncio.wait_for(r.capture(max_s=None, hold=True), timeout=3)
    assert pcm is None
    assert consumed <= 1000 // 30 + 2


async def test_hold_capture_total_time_capped_even_with_speech(monkeypatch):
    """The hold cap counts from capture start (onset wait + recording), so
    speech that starts late still returns at max_s with what was captured."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    # 20 silent frames (600 ms), then speech forever; cap = 1 s = 33 frames
    r = Recorder(s, frames=lambda: frames("." * 20 + "s" * 500))
    pcm = await asyncio.wait_for(r.capture(max_s=1, hold=True), timeout=3)
    assert pcm is not None
    assert FRAME * 10 <= len(pcm) <= FRAME * 15   # ~13 speech frames before the cap


async def test_arm_makes_same_iteration_finish_honored(monkeypatch):
    """capture() is a coroutine, so its own flag arming only runs on the
    task's first step. A finish() issued in the same loop iteration as
    ensure_future(capture(...)) must still be honored when the caller
    arm()ed synchronously first — otherwise a push-to-talk quick tap
    leaves the mic open for the whole max_s."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    gate = threading.Event()

    def blocking_frames():
        gate.wait(5)   # the worker blocks here until the test releases it
        yield from frames("." * 1000)

    r = Recorder(s, frames=blocking_frames)
    r.arm(hold=True)
    task = asyncio.ensure_future(r.capture(max_s=30, hold=True))
    r.finish()                       # same iteration: capture() body hasn't run yet
    assert r._finish.is_set()        # honored, not dropped
    gate.set()
    pcm = await asyncio.wait_for(task, timeout=2)
    assert pcm is None               # ended immediately on the pre-armed finish


async def test_without_arm_same_iteration_finish_is_dropped(monkeypatch):
    """Documents why arm() exists: the plain coroutine can't see a finish()
    issued before its first step (it's a no-op since _capturing is False,
    and capture() then clears any stale flag anyway)."""
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    started, gate = threading.Event(), threading.Event()
    r = Recorder(s, frames=lambda: _gated_frames("." * 500, 0, started, gate))
    task = asyncio.ensure_future(r.capture(max_s=30, hold=True))
    r.finish()
    assert not r._finish.is_set()
    await asyncio.to_thread(started.wait, 2)
    assert not task.done()           # still capturing: the finish was lost
    r.finish()
    gate.set()
    assert await asyncio.wait_for(task, timeout=2) is None


async def test_arm_then_stop_same_iteration_returns_none(monkeypatch):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=10)
    gate = threading.Event()

    def blocking_frames():
        gate.wait(5)
        yield from frames("ssssssss" + "." * 100)

    r = Recorder(s, frames=blocking_frames)
    r.arm()
    task = asyncio.ensure_future(r.capture(max_s=5))
    r.stop()
    gate.set()
    assert await asyncio.wait_for(task, timeout=2) is None


async def test_arm_is_consumed_by_one_capture_and_capture_self_arms(monkeypatch):
    r = make("....ssssss.........", monkeypatch)
    r.arm(hold=True)
    assert r._armed and r._hold
    pcm = await r.capture()          # consumes the arm; capture's own hold=False wins nothing here
    assert pcm is not None
    assert not r._armed and not r._capturing
    r._finish.set()                  # stale
    pcm = await r.capture()          # self-arms: stale flag cleared
    assert pcm is not None and not r._finish.is_set()


def test_disarm_resets_arm_flags(monkeypatch):
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    rec = Recorder(Settings(), frames=lambda: iter([]))
    rec.arm(hold=True)
    assert rec._capturing and rec._armed and rec._hold
    rec.disarm()
    assert not rec._capturing and not rec._armed and not rec._hold


def test_recorder_registers_capture_in_flight_as_device_busy():
    # (devices.reset() in the autouse fixture clears the hook between tests)
    from veronica.audio import devices

    r = Recorder(Settings(), frames=lambda: iter([]))
    assert devices.busy() is False
    r.arm()
    assert devices.busy() is True
    r.disarm()
    assert devices.busy() is False


async def test_followup_skip_grows_with_input_latency(monkeypatch):
    """A Bluetooth mic reports ~300 ms latency: skip_ms=300 must become
    latency+100 so her own tail (arriving that late) is dropped."""
    r = make("sssss", monkeypatch, vad_silence_ms=90, min_speech_ms=60)
    r.input_latency_s = 0.3
    seen = {"n": 0}
    orig = r._frames
    def counting():
        for f in orig():
            seen["n"] += 1
            yield f
    r._frames = counting
    await r.capture(max_s=1, skip_ms=300)
    # 400 ms / 30 ms frames = 13 frames skipped before the VAD sees anything,
    # plus whatever the capture itself consumed
    assert seen["n"] >= 13 + 5
    r2 = make("sssss", monkeypatch, vad_silence_ms=90, min_speech_ms=60)
    r2.input_latency_s = 0.0
    seen2 = {"n": 0}
    orig2 = r2._frames
    def counting2():
        for f in orig2():
            seen2["n"] += 1
            yield f
    r2._frames = counting2
    await r2.capture(max_s=1, skip_ms=300)
    assert seen2["n"] < seen["n"]


# -- voice isolation: energy gate and noise suppression ---------------------------
def quiet_frames(pattern, level):
    """Like frames(), with 's' frames at `level` (int16 amplitude)."""
    for ch in pattern:
        yield np.full(FRAME, level if ch == "s" else 0, dtype=np.int16).tobytes()
    while True:
        yield np.zeros(FRAME, dtype=np.int16).tobytes()


class PassDenoiser:
    """Suppression that changes nothing: the level floor applies, the audio
    is what went in."""

    def process_bytes(self, frame):
        return frame


def with_suppression(monkeypatch, den=None):
    from veronica.audio import denoise

    monkeypatch.setattr(denoise, "make_denoiser", lambda settings: den if den is not None else PassDenoiser())


async def test_frames_under_the_energy_gate_do_not_open_a_capture(monkeypatch):
    # The VAD calls every non-zero frame speech; at amplitude 20 (rms ~0.0006)
    # it is room noise to the gate.
    with_suppression(monkeypatch)
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1, vad_min_rms=0.002)
    r = Recorder(s, frames=lambda: quiet_frames("...ssssss......", 20))
    assert await r.capture(max_s=1) is None


async def test_energy_gate_zero_lets_the_vad_decide_alone(monkeypatch):
    with_suppression(monkeypatch)
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1, vad_min_rms=0.0)
    r = Recorder(s, frames=lambda: quiet_frames("...ssssss......", 20))
    assert await r.capture(max_s=1) is not None


async def test_without_suppression_the_floor_is_off_as_before(monkeypatch):
    # (conftest: no denoiser) — quiet speech the VAD hears must still count.
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1, vad_min_rms=0.002)
    r = Recorder(s, frames=lambda: quiet_frames("...ssssss......", 20))
    assert await r.capture(max_s=1) is not None


def test_has_speech_is_the_vad_alone(monkeypatch):
    # the pre-roll is raw audio: no floor on it, as before
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    r = Recorder(Settings(vad_min_rms=0.002), frames=lambda: frames(""))
    assert r.has_speech(np.full(FRAME * 6, 20, dtype=np.int16))


class HalvingDenoiser:
    def __init__(self):
        self.calls = 0

    def process_bytes(self, frame):
        self.calls += 1
        return (np.frombuffer(frame, dtype=np.int16) // 2).tobytes()


async def test_suppression_decides_speech_but_the_capture_is_the_raw_audio(monkeypatch):
    from veronica.audio import denoise

    levels = []
    den = HalvingDenoiser()
    monkeypatch.setattr(denoise, "make_denoiser", lambda settings: den)
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1)
    r = Recorder(s, frames=lambda: frames("..ssssss...."), on_level=levels.append)
    pcm = await r.capture()
    assert den.calls > 0
    assert pcm.max() == 1000                     # what STT and the speaker check get
    assert max(levels) == pytest.approx(1000 / 32768, rel=1e-3)  # the HUD meter shows the mic as it is


async def test_suppression_that_removes_a_noise_burst_keeps_it_from_opening(monkeypatch):
    from veronica.audio import denoise

    class Silencer:
        def process_bytes(self, frame):
            return bytes(len(frame))

    monkeypatch.setattr(denoise, "make_denoiser", lambda settings: Silencer())
    r = make("..ssssss....", monkeypatch)
    assert await r.capture(max_s=1) is None


class DelayDenoiser:
    """Behaves like the real one in time: output is input delayed by LAG."""

    def __init__(self):
        from veronica.audio import denoise

        self.tail = np.zeros(denoise.LAG, dtype=np.int16)

    def process_bytes(self, frame):
        x = np.concatenate([self.tail, np.frombuffer(frame, dtype=np.int16)])
        self.tail = x[-self.__class__._lag():]
        return x[: x.size - self.__class__._lag()].tobytes()

    @staticmethod
    def _lag():
        from veronica.audio import denoise

        return denoise.LAG


async def test_the_onset_is_not_clipped_by_the_suppressors_delay(monkeypatch):
    """Onset is seen in the delayed suppressed audio; the raw frames that
    hold it come before that frame and must be in the capture."""
    with_suppression(monkeypatch, DelayDenoiser())
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=300, min_speech_ms=60, max_utterance_s=2)
    raw = "......" + "s" * 10 + "." * 30
    r = Recorder(s, frames=lambda: frames(raw))
    pcm = await r.capture(max_s=2)
    assert pcm is not None
    assert np.count_nonzero(pcm) == 10 * FRAME          # all of the speech, from its first sample
    first = int(np.flatnonzero(pcm)[0])
    assert first <= FRAME * 2                           # a little lead-in, not a clipped start


async def test_a_suppressor_that_raises_falls_back_to_raw_and_turns_itself_off(monkeypatch, caplog):
    from veronica.audio import denoise

    class Broken:
        calls = 0

        def process_bytes(self, frame):
            Broken.calls += 1
            raise RuntimeError("onnxruntime: bad state")

    made = []

    def make(settings):
        if denoise._disabled:
            return None
        made.append(Broken())
        return made[-1]

    monkeypatch.setattr(denoise, "make_denoiser", make)
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1)
    r = Recorder(s, frames=lambda: frames("..ssssss...."))
    pcm = await r.capture()
    assert pcm is not None and pcm.max() == 1000     # the turn still got its audio
    assert Broken.calls == 1                         # tried once, then raw for the rest
    assert caplog.text.count("noise suppression failed") == 1
    assert await r.capture() is not None
    assert len(made) == 1                            # off until restart


# -- the real mic path: a stream that dies mid-capture ---------------------------

class DyingStream:
    """A RawInputStream stand-in: hands out `chunks` (int16 sample counts,
    speech) as they become "buffered", then goes silent for good, the way
    PortAudio's input unit does after -10863. A read for more than is
    buffered would block forever on a real stream, so it fails here."""

    instances = []

    def __init__(self, *, chunks=(), close_raises=False, **kw):
        self.chunks = [np.full(c, 1000, dtype=np.int16).tobytes() for c in chunks]
        self.close_raises = close_raises
        self.latency = 0.01
        self.closed = False
        DyingStream.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True
        if self.close_raises:
            raise RuntimeError("PortAudio: stream already stopped")

    @property
    def read_available(self):
        return len(self.chunks[0]) // 2 if self.chunks else 0

    def read(self, frames):
        if not self.chunks or frames > len(self.chunks[0]) // 2:
            raise AssertionError(f"blocking read of {frames} frames on a dead stream")
        return self.chunks.pop(0), False


def dying_recorder(monkeypatch, chunks, **kw):
    from veronica.audio import record as record_mod
    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    monkeypatch.setattr(record_mod, "STALL_S", 0.3)
    DyingStream.instances = []
    monkeypatch.setattr(record_mod.sd, "RawInputStream", lambda **a: DyingStream(chunks=chunks, **kw))
    return Recorder(Settings(vad_silence_ms=300, min_speech_ms=60, max_utterance_s=5))


async def test_a_stream_that_dies_mid_capture_ends_it_with_what_was_heard(monkeypatch, caplog):
    # uneven PortAudio reads (700 + 740 samples) are re-cut into 480-sample frames
    rec = dying_recorder(monkeypatch, [700, 740])
    t0 = asyncio.get_running_loop().time()
    pcm = await asyncio.wait_for(rec.capture(max_s=5), timeout=3)
    assert asyncio.get_running_loop().time() - t0 < 2
    assert pcm is not None and pcm.size == 3 * FRAME
    assert DyingStream.instances[0].closed and rec._capturing is False
    assert "no audio" in caplog.text


async def test_a_dead_stream_before_any_speech_returns_none(monkeypatch):
    rec = dying_recorder(monkeypatch, [])
    assert await asyncio.wait_for(rec.capture(max_s=30), timeout=3) is None


async def test_the_refresh_lock_is_free_while_waiting_on_a_dead_stream(monkeypatch):
    from veronica.audio import devices
    rec = dying_recorder(monkeypatch, [480])
    task = asyncio.ensure_future(rec.capture(max_s=5))
    await asyncio.sleep(0.15)                  # capture thread is polling the dead stream

    def grab():
        if not devices.refresh_lock.acquire(timeout=0.1):
            return False
        devices.refresh_lock.release()
        return True

    assert await asyncio.to_thread(grab)
    await asyncio.wait_for(task, timeout=3)


async def test_stop_while_the_stream_is_dead_returns_none(monkeypatch):
    rec = dying_recorder(monkeypatch, [480] * 3)
    task = asyncio.ensure_future(rec.capture(max_s=5))
    await asyncio.sleep(0.1)
    rec.stop()
    assert await asyncio.wait_for(task, timeout=3) is None


async def test_a_dead_stream_that_complains_on_close_still_returns(monkeypatch):
    rec = dying_recorder(monkeypatch, [480] * 3, close_raises=True)
    pcm = await asyncio.wait_for(rec.capture(max_s=5), timeout=3)
    assert pcm is not None and pcm.size == 3 * FRAME
