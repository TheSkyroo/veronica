import asyncio

import numpy as np
import pytest

from veronica.config import Settings
from veronica.orchestrator import Orchestrator


class Wake:
    async def wait(self, threshold=None, suppress=None): pass
    def stop(self): pass
    def take_preroll(self): return np.zeros(0, dtype=np.int16)

class Rec:
    def __init__(self, pcms, has_speech=False):
        self.pcms = list(pcms)
        self._has_speech = has_speech
        self.preroll_calls = []
    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0):
        self.preroll_calls.append(preroll)
        return self.pcms.pop(0) if self.pcms else None
    def has_speech(self, pcm): return self._has_speech

class STT:
    def __init__(self, texts): self.texts = list(texts)
    async def atranscribe(self, pcm): return self.texts.pop(0) if self.texts else None

class Brain:
    def __init__(self): self.asked = []
    async def ask(self, text):
        self.asked.append(text)
        yield "Sure."
        yield "Done."

class TTS:
    def __init__(self): self.said = []
    async def asynth(self, text):
        self.said.append(text)
        return np.zeros(10, dtype=np.float32), 24000

class Player:
    def __init__(self): self.played = 0; self.stops = 0; self.resets = 0
    async def play(self, s): self.played += 1
    def stop(self): self.stops += 1
    def reset(self): self.resets += 1


def build(rec_pcms=(), stt_texts=()):
    states = []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec(rec_pcms), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
    )
    return o, states


async def test_handle_text_speaks_each_sentence():
    o, states = build()
    out = await o.handle_text("hello")
    assert out == ["Sure.", "Done."]
    assert o.tts.said == ["Sure.", "Done."]
    assert o.player.played == 2 and o.player.resets == 1
    assert states[:2] == ["thinking", "speaking"]


async def test_chime_after_wake_and_on_followup():
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    await o.one_turn()
    # chime samples go through player.play like speech; count plays: wake chime + 2 sentences + followup chime
    assert o.player.played == 4


async def test_wake_chime_skipped_when_preroll_has_speech():
    o, _ = build()
    o.recorder = Rec([np.zeros(1, np.int16), None], has_speech=True)
    o.stt = STT(["hi"])
    await o.one_turn()
    # wake chime skipped (preroll already has speech), so: 2 sentences + followup chime
    assert o.player.played == 3


async def test_wake_chime_played_when_preroll_has_no_speech():
    o, _ = build()
    o.recorder = Rec([np.zeros(1, np.int16), None], has_speech=False)
    o.stt = STT(["hi"])
    await o.one_turn()
    # wake chime + 2 sentences + followup chime
    assert o.player.played == 4


async def test_preroll_is_handed_to_recorder_capture():
    rec = Rec([np.zeros(1, np.int16), None])
    o, _ = build()
    o.recorder = rec
    o.wake = Wake()
    o.stt = STT(["hi"])
    await o.one_turn()
    assert len(rec.preroll_calls) >= 1
    assert isinstance(rec.preroll_calls[0], np.ndarray)  # the wake-triggered listen gets the wake engine's preroll


async def test_chime_skipped_when_muted():
    o, _ = build()
    o.muted = True
    await o.chime(880, 120)
    assert o.player.played == 0


async def test_confirm_yes_and_no():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)], stt_texts=["Yes, do it", "nah"])
    assert await o.confirm("Bash: ls") is True
    assert await o.confirm("Bash: rm") is False
    assert o.tts.said[0] == "Run Bash: ls?"


async def test_confirm_question_override_replaces_default_prompt():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    assert await o.confirm("Quit Veronica", question="Quit Veronica?") is True
    assert o.tts.said == ["Quit Veronica?"]
    assert ("prompt", "Quit Veronica?") in ev
    tools = [p for k, p in ev if k == "tool"]
    # the "ask" tool event still carries the original summary, independent
    # of the spoken question override
    assert tools[0]["summary"] == "Quit Veronica"


async def test_confirm_no_speech_is_deny():
    o, _ = build(rec_pcms=[None])
    assert await o.confirm("Write file a") is False


async def test_confirm_when_muted_denies_silently():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    o.muted = True
    assert await o.confirm("Bash: rm x") is False
    assert o.tts.said == []


async def test_confirming_state_emitted():
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    await o.confirm("Bash: rm x")
    assert "confirming" in states


async def test_wake_retry_returns_to_idle(monkeypatch):
    o, states = build()

    class W:
        n = 0

        async def wait(self, threshold=None, suppress=None):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("no mic")
            raise asyncio.CancelledError

        def stop(self):
            pass

    o.wake = W()

    async def nosleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", nosleep)
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert states[-2:] == ["error", "idle"]


async def test_empty_transcript_prompts_retry():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=[""])
    await o.one_turn()
    assert o.tts.said == ["Sorry, didn't catch that."]
    assert o.brain.asked == []


async def test_full_turn():
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what time is it"])
    await o.one_turn()
    assert o.brain.asked == ["what time is it"]
    assert states == ["listening", "thinking", "speaking", "followup", "idle"]


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("yes", True),
        ("Yes, do it", True),
        ("yes please", True),
        ("go ahead please", True),
        ("sure", True),
        ("sure thing", True),
        ("not sure", False),
        ("go away", False),
        ("go", False),
        ("yesterday", False),
        ("no", False),
        ("yes no wait", False),
        ("", False),
        ("Don't do it", False),
        ("don't", False),
        ("No, don't do it.", False),
        ("do it", True),
    ],
)
def test_is_confirmation(heard, expected):
    assert Orchestrator.is_confirmation(heard) is expected


async def test_run_forever_survives_reporting_failure():
    class RaisingBrain:
        async def ask(self, text):
            raise RuntimeError("brain boom")
            yield  # pragma: no cover - makes this an async generator function

    class RaisingTTS:
        async def asynth(self, text):
            raise RuntimeError("tts boom")

    class WakeOnceThenCancel:
        """calls counts only the outer main-loop wait()s (threshold=None); a
        barge-listener wait() (threshold set) never barges and just blocks
        until stop()."""
        def __init__(self):
            self.calls = 0
            self._barge_ev = asyncio.Event()

        async def wait(self, threshold=None, suppress=None):
            if threshold is not None:
                await self._barge_ev.wait()
                self._barge_ev.clear()
                return False
            self.calls += 1
            if self.calls > 1:
                raise asyncio.CancelledError()
            return True

        def stop(self):
            self._barge_ev.set()

        def take_preroll(self):
            return __import__("numpy").zeros(0, dtype="int16")

    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=WakeOnceThenCancel(),
        recorder=Rec([np.zeros(1, np.int16)]),
        stt=STT(["do something"]),
        brain=RaisingBrain(),
        tts=RaisingTTS(),
        player=Player(),
    )
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert o.state == "idle"


# -- finding 2: confirm() must not race handle_text's speech ------------------

class TrackingTTS:
    def __init__(self):
        self.said = []
        self.in_progress = 0
        self.max_in_progress = 0
        self._texts_by_index = []

    async def asynth(self, text):
        self.in_progress += 1
        self.max_in_progress = max(self.max_in_progress, self.in_progress)
        self.said.append(text)
        idx = len(self._texts_by_index)
        self._texts_by_index.append(text)
        await asyncio.sleep(0)
        self.in_progress -= 1
        # encode the index into the sample array so the player can recover
        # which text a given sample array corresponds to, since the pipeline
        # may synthesize sentences out of play order.
        return np.full(1, idx, dtype=np.float32), 24000


class TrackingPlayer:
    def __init__(self, tts):
        self.tts = tts
        self.in_progress = 0
        self.max_in_progress = 0
        self.played_texts = []

    async def play(self, s):
        self.in_progress += 1
        self.max_in_progress = max(self.max_in_progress, self.in_progress)
        idx = int(s[0])
        self.played_texts.append(self.tts._texts_by_index[idx])
        await asyncio.sleep(0)
        self.in_progress -= 1

    def stop(self): pass
    def reset(self): pass


class ConfirmingBrain:
    """Yields a sentence, awaits orch.confirm() mid-stream (as the SDK's
    can_use_tool callback would), then yields another sentence."""

    def __init__(self, orch):
        self.orch = orch

    async def ask(self, text):
        yield "First."
        assert await self.orch.confirm("Bash: ls") is True
        yield "Second."


async def test_confirm_does_not_race_handle_text_speech():
    tts = TrackingTTS()
    player = TrackingPlayer(tts)
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16)]), stt=STT(["yes"]),
        brain=None, tts=tts, player=player,
    )
    o.brain = ConfirmingBrain(o)

    out = await o.handle_text("hello")

    assert out == ["First.", "Second."]
    # With the pipeline, the producer may synthesize "Second." ahead of the
    # confirm prompt's synth, so synth order is not guaranteed. What must be
    # guaranteed is *play* order (nothing is heard out of sequence) and that
    # playback never overlaps (confirm's speech + handle_text's speech share
    # _speech_lock).
    assert player.played_texts == ["First.", "Run Bash: ls?", "Second."]
    assert player.max_in_progress == 1


# -- finding 3: first capture after wake has an onset timeout -----------------

class RecArgs:
    def __init__(self, pcms):
        self.pcms = list(pcms)
        self.max_s_calls = []

    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0):
        self.max_s_calls.append(max_s)
        return self.pcms.pop(0) if self.pcms else None

    def has_speech(self, pcm): return False


async def test_first_capture_uses_listen_wait_s_and_no_speech_goes_idle():
    states = []
    rec = RecArgs([None])
    o = Orchestrator(
        Settings(listen_wait_s=6, followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=rec, stt=STT([]), brain=Brain(), tts=TTS(), player=Player(),
        on_state=states.append,
    )
    await o.one_turn()
    assert rec.max_s_calls == [6]
    assert states == ["listening", "idle"]
    assert o.tts.said == []


# -- finding 6: mute must actually mute ---------------------------------------

async def test_muted_handle_text_speaks_nothing():
    o, _ = build()
    o.muted = True
    out = await o.handle_text("hello")
    assert out == ["Sure.", "Done."]
    # The pipeline synthesizes ahead of playback regardless of mute (the
    # producer doesn't know the mute state is meant to silence output), but
    # nothing actually reaches the speaker: the consumer skips player.play().
    assert o.player.played == 0


# -- finding 7: wake failures are retried, not fatal; login errors speak ------

async def test_wake_failure_retries_and_continues_into_a_turn(monkeypatch):
    sleep_calls = []

    async def fake_sleep(secs):
        sleep_calls.append(secs)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    class FlakyWake:
        """calls counts only the outer main-loop wait()s (threshold=None); a
        barge-listener wait() (threshold set) never barges and just blocks
        until stop()."""
        def __init__(self):
            self.calls = 0
            self._barge_ev = asyncio.Event()

        async def wait(self, threshold=None, suppress=None):
            if threshold is not None:
                await self._barge_ev.wait()
                self._barge_ev.clear()
                return False
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("no audio device")
            if self.calls > 2:
                raise asyncio.CancelledError()
            return True

        def stop(self):
            self._barge_ev.set()

        def take_preroll(self):
            return __import__("numpy").zeros(0, dtype="int16")

    states = []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=FlakyWake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(["hi"]),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
    )
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert "error" in states
    assert sleep_calls == [10]
    assert "listening" in states


async def test_turn_error_mentioning_login_speaks_specific_message():
    class LoginFailBrain:
        async def ask(self, text):
            raise RuntimeError("Not logged in: please run claude login")
            yield  # pragma: no cover - makes this an async generator function

    class WakeOnceThenCancel:
        """calls counts only the outer main-loop wait()s (threshold=None); a
        barge-listener wait() (threshold set) never barges and just blocks
        until stop()."""
        def __init__(self):
            self.calls = 0
            self._barge_ev = asyncio.Event()

        async def wait(self, threshold=None, suppress=None):
            if threshold is not None:
                await self._barge_ev.wait()
                self._barge_ev.clear()
                return False
            self.calls += 1
            if self.calls > 1:
                raise asyncio.CancelledError()
            return True

        def stop(self):
            self._barge_ev.set()

        def take_preroll(self):
            return __import__("numpy").zeros(0, dtype="int16")

    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=WakeOnceThenCancel(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(["hi"]),
        brain=LoginFailBrain(), tts=TTS(), player=Player(),
    )
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert "Claude Code isn't logged in." in o.tts.said


# -- task 5: warm-up + pipelined TTS ------------------------------------------

async def test_warmup_touches_tts_and_stt():
    o, _ = build()
    await o.warmup()
    assert o.tts.said == ["ok"] and o.ready is True


async def test_warmup_without_stt():
    o, _ = build()
    o.stt = None
    await o.warmup()
    assert o.ready is True


async def test_pipelined_tts_preserves_order_and_overlaps_synth_with_play():
    import asyncio

    order = []

    class SlowTTS:
        async def asynth(self, text):
            order.append(("synth", text))
            await asyncio.sleep(0.01)
            return np.zeros(10, dtype=np.float32), 24000

    class SlowPlayer:
        def __init__(self): self.played = []; self.resets = 0
        async def play(self, s):
            order.append(("play", len(self.played))); self.played.append(s); await asyncio.sleep(0.02)
        def stop(self): pass
        def reset(self): self.resets += 1

    class Brain3:
        async def ask(self, text):
            for s in ["A.", "B.", "C."]:
                yield s

    o, _ = build()
    o.tts, o.player, o.brain = SlowTTS(), SlowPlayer(), Brain3()
    out = await o.handle_text("x")
    assert out == ["A.", "B.", "C."]
    assert [t for k, t in order if k == "synth"] == ["A.", "B.", "C."]
    assert len(o.player.played) == 3
    # synth of B must start before play of A finishes: synth B appears before play 1
    assert order.index(("synth", "B.")) < order.index(("play", 1))


# -- task 5 fix round 1: cancellation-safe pipeline, yield-time ordering -----

async def test_handle_text_cancel_with_full_queue_does_not_hang():
    class FastTTS:
        async def asynth(self, text):
            return np.zeros(10, dtype=np.float32), 24000

    class SlowTTS:
        """Synth that stays pending for a full second — long enough that,
        with maxsize=2 and a brain producing faster than synth completes,
        the producer will be blocked mid-`queue.put()` with an
        already-created-but-not-yet-queued synth future at cancel time
        (exercising the orphaned-future fix), and other futures will still
        be genuinely in-flight (not just already-done) when drained."""
        def __init__(self): self.started = 0; self.finished = 0
        async def asynth(self, text):
            self.started += 1
            try:
                await asyncio.sleep(1)
            finally:
                self.finished += 1
            return np.zeros(10, dtype=np.float32), 24000

    class FastPlayer:
        def __init__(self): self.played = 0
        async def play(self, s):
            self.played += 1
        def stop(self): pass
        def reset(self): pass

    class Brain5:
        async def ask(self, text):
            for s in ["A.", "B.", "C.", "D.", "E."]:
                yield s

    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    slow_tts = SlowTTS()
    o.tts, o.brain, o.player = slow_tts, Brain5(), FastPlayer()

    turn = asyncio.create_task(o.handle_text("x"))
    await asyncio.sleep(0.05)
    q = o._speech_queue
    assert q is not None
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(turn, 1)

    assert o._speech_queue is None
    # unfinished_tasks balanced back to zero: nothing left that would hang
    # a confirm() blocked in queue.join().
    await asyncio.wait_for(q.join(), 0.1)

    # give cancelled synth futures (and the producer task) one tick to
    # actually unwind, then confirm none are still pending — the orphaned-
    # future bug left one of these running to completion, untracked.
    await asyncio.sleep(0.05)
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    assert pending == []

    o.tts, o.brain, o.player = FastTTS(), Brain5(), FastPlayer()
    out = await asyncio.wait_for(o.handle_text("y"), 1)
    assert out == ["A.", "B.", "C.", "D.", "E."]

    assert await asyncio.wait_for(o.confirm("Bash: rm x"), 1) is True


async def test_confirm_from_concurrent_task_waits_for_yielded_sentence():
    texts_by_index: list[str] = []

    class DelayedTTS:
        async def asynth(self, text):
            idx = len(texts_by_index)
            texts_by_index.append(text)
            await asyncio.sleep(0.05)
            return np.full(1, idx, dtype=np.float32), 24000

    class OrderPlayer:
        def __init__(self): self.played_texts = []
        async def play(self, s):
            self.played_texts.append(texts_by_index[int(s[0])])
        def stop(self): pass
        def reset(self): pass

    holder = {}

    class ConcurrentConfirmBrain:
        """Mirrors real SDK ordering: the assistant message ("First.") is
        yielded before the control request (confirm) that follows it is
        spawned as a separate task the brain then awaits."""
        def __init__(self, orch): self.orch = orch
        async def ask(self, text):
            yield "First."
            holder["task"] = asyncio.create_task(self.orch.confirm("Bash: ls"))
            assert await holder["task"] is True
            yield "Second."

    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    o.tts, o.player = DelayedTTS(), OrderPlayer()
    o.brain = ConcurrentConfirmBrain(o)

    out = await o.handle_text("hello")

    assert out == ["First.", "Second."]
    assert o.player.played_texts == ["First.", "Run Bash: ls?", "Second."]


# -- task 7: barge-in ----------------------------------------------------------

class BargeWake:
    """wait() returns True after `after` calls when barge=True, else blocks until stop()."""
    def __init__(self, barge_on_call=None):
        self.calls = 0; self.stops = 0; self.barge_on_call = barge_on_call
        self.received_suppress = None
        self._ev = __import__("asyncio").Event()
    async def wait(self, threshold=None, suppress=None):
        self.calls += 1
        self.received_suppress = suppress
        if self.barge_on_call == self.calls:
            await __import__("asyncio").sleep(0.005)
            return True
        await self._ev.wait(); self._ev.clear(); return False
    def stop(self):
        self.stops += 1; self._ev.set()
    def take_preroll(self): return np.zeros(0, dtype=np.int16)


async def test_barge_listener_receives_suppress_callback():
    """The barge listener must be handed a callable that yields the sentence
    currently being spoken, so the whisper wake engine can ignore Veronica's
    own speech (e.g. "I'm Veronica") instead of self-interrupting."""
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    o.wake = BargeWake(barge_on_call=None)
    await o.one_turn()
    assert callable(o.wake.received_suppress)
    # immediately after speaking, the post-playback suppression window (see
    # test_now_speaking_set_during_play_and_cleared_after / _finished_speaking)
    # is still active, so the last-spoken text is still offered
    assert "Done." in o.wake.received_suppress()
    # once that window has elapsed, suppress() reverts to just _now_speaking
    o._last_spoken_until = 0.0
    assert o.wake.received_suppress() == ""


async def test_now_speaking_set_during_play_and_cleared_after():
    """_now_speaking must reflect the sentence text for the duration of
    player.play() (both the pipeline consumer and _say_unlocked), and be
    cleared once play() returns."""
    seen = []

    class RecordingPlayer(Player):
        async def play(self, s):
            seen.append(o._now_speaking)
            await super().play(s)

    o, _ = build()
    o.player = RecordingPlayer()
    await o.say("Hello there.")
    assert seen == ["Hello there."]
    assert o._now_speaking == ""

    seen.clear()
    await o.handle_text("anything")
    assert seen == ["Sure.", "Done."]
    assert o._now_speaking == ""

    seen.clear()
    o.recorder = Rec([None])
    await o.confirm("do a thing")
    assert seen == ["Run do a thing?"]
    assert o._now_speaking == ""


async def test_suppress_stays_active_for_wake_window_after_playback():
    """The mic's rolling wake-analysis window (wake_window_s + wake_hop_s) can
    still hold the tail of a just-finished sentence after _now_speaking is
    cleared, so suppress() must keep returning text mentioning it until that
    window has elapsed."""
    o, _ = build()
    now = [1000.0]
    o._clock = lambda: now[0]
    await o.say("I am Veronica.")
    assert o._now_speaking == ""

    # still within wake_window_s (1.6) + wake_hop_s (0.6) = 2.2 s of playback ending
    now[0] += 1.0
    assert "veronica" in o._suppress_text().lower()

    # past the window: no longer suppressed
    now[0] += 2.0  # total 3.0s elapsed
    assert o._suppress_text() == ""


class SlowBrain:
    def __init__(self): self.interrupts = 0
    async def ask(self, text):
        yield "One."
        await __import__("asyncio").sleep(0.05)
        yield "Two."
    async def interrupt(self): self.interrupts += 1


async def test_barge_in_stops_speech_and_relistens():
    events = []

    class LoggingRec(Rec):
        """Skips logging the initial pre-turn capture; logs only the
        re-listen capture that follows a barge, for ordering assertions."""
        def __init__(self, pcms):
            super().__init__(pcms)
            self.n = 0

        async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0):
            self.n += 1
            if self.n > 1:
                events.append("capture")
            return await super().capture(max_s=max_s, preroll=preroll, partial=partial)

    class LoggingSlowBrain(SlowBrain):
        async def interrupt(self):
            events.append("interrupt")
            await super().interrupt()

    # call 1 = main wake (we call one_turn directly, so calls start at the barge listener)
    o, states = build(rec_pcms=[], stt_texts=["first"])
    o.recorder = LoggingRec([np.zeros(1, np.int16), None])
    o.wake = BargeWake(barge_on_call=1)
    o.brain = LoggingSlowBrain()
    await o.one_turn()
    assert o.brain.interrupts == 1
    assert o.player.stops >= 1
    assert "listening" in states[states.index("speaking") + 1:]     # re-listened after barge
    assert o.tts.said == ["One."]     # "Two." is never reached: barged before its 50 ms sleep
    assert events == ["interrupt", "capture"]     # interrupt happens before the re-listen capture


class StoppableRec:
    """capture() returns queued pcms normally, except a BLOCK sentinel, which
    blocks until stop() is called and then returns None (mirroring the real
    Recorder's consume-on-use stop())."""
    BLOCK = object()

    def __init__(self, pcms):
        self.pcms = list(pcms)
        self.stops = 0
        self._ev = asyncio.Event()

    def stop(self):
        self.stops += 1
        self._ev.set()

    def has_speech(self, pcm):
        return False

    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0):
        item = self.pcms.pop(0) if self.pcms else None
        if item is self.BLOCK:
            await self._ev.wait()
            self._ev.clear()
            return None
        return item


class ConfirmDuringBargeBrain:
    """Yields a sentence, then awaits orch.confirm() mid-stream, as the SDK's
    can_use_tool callback would during a tool call."""
    def __init__(self, orch):
        self.orch = orch
        self.results = []

    async def ask(self, text):
        yield "First."
        self.results.append(await self.orch.confirm("Bash: rm x"))
        yield "Second."

    async def interrupt(self):
        pass


async def test_barge_during_confirm_stops_capture():
    o, states = build(rec_pcms=[], stt_texts=["first"])
    rec = StoppableRec([np.zeros(1, np.int16), StoppableRec.BLOCK, None])
    o.recorder = rec
    o.wake = BargeWake(barge_on_call=1)
    o.brain = ConfirmDuringBargeBrain(o)
    await o.one_turn()
    assert o.brain.results == [False]                 # confirm() returned False, not orphaned
    assert rec.stops == 1                              # in-flight confirm capture was stopped
    assert "listening" in states[states.index("speaking") + 1:]     # re-listened after barge
    assert o.tts.said == ["First.", "Run Bash: rm x?"]


async def test_barge_during_confirm_prompt_aborts_confirm():
    """A barge that lands while confirm() is still speaking (or waiting to
    speak) its "Run X?" prompt must not fall through to capture() and eat
    the user's follow-up — confirm() should observe the barge and bail."""
    release_ev = asyncio.Event()

    class PromptBlockingTTS:
        """asynth for the confirm prompt blocks until player.stop() (called
        by the barge branch) releases it — simulating the barge interrupting
        the prompt while it's being spoken."""
        def __init__(self):
            self.said = []

        async def asynth(self, text):
            self.said.append(text)
            if text.startswith("Run "):
                await release_ev.wait()
            return np.zeros(10, dtype=np.float32), 24000

    class ReleasingPlayer:
        def __init__(self):
            self.played = 0
            self.stops = 0

        async def play(self, s):
            self.played += 1

        def stop(self):
            self.stops += 1
            release_ev.set()

        def reset(self):
            pass

    class CountingRec:
        def __init__(self, pcms):
            self.pcms = list(pcms)
            self.captures = 0

        async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0):
            self.captures += 1
            return self.pcms.pop(0) if self.pcms else None

        def has_speech(self, pcm):
            return False

        def stop(self):
            pass   # never reached: confirm() bails before it would capture

    o, states = build(rec_pcms=[], stt_texts=["first"])
    o.tts = PromptBlockingTTS()
    o.player = ReleasingPlayer()
    rec = CountingRec([np.zeros(1, np.int16), None])
    o.recorder = rec
    o.wake = BargeWake(barge_on_call=1)
    o.brain = ConfirmDuringBargeBrain(o)
    await o.one_turn()

    assert o.brain.results == [False]                 # confirm() bailed, didn't hang or raise
    assert rec.captures == 2                           # initial listen + post-barge re-listen only
    assert "listening" in states[states.index("speaking") + 1:]


async def test_no_barge_listener_stopped_when_turn_ends():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    o.wake = BargeWake()  # never barges
    await o.one_turn()
    assert o.wake.stops == 1  # listener stopped once when handle_text finished


async def test_barge_uses_barge_threshold():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    seen = []
    class W(BargeWake):
        async def wait(self, threshold=None, suppress=None):
            seen.append(threshold); return await super().wait(threshold)
    o.wake = W()
    await o.one_turn()
    assert seen == [0.8]


async def test_barge_listener_task_not_left_pending():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    o.wake = BargeWake()  # never barges
    await o.one_turn()
    current = asyncio.current_task()
    assert [t for t in asyncio.all_tasks() if t is not current] == []


async def test_barge_while_confirm_waits_for_lock_skips_prompt():
    """A barge landing while confirm() is blocked *acquiring* _speech_lock
    (as opposed to while it's speaking the prompt, already covered by
    test_barge_during_confirm_prompt_aborts_confirm, or while waiting for
    the handle_text queue to drain, covered by the pre-lock _barged check)
    must still be observed before the prompt is spoken.

    _speech_queue is None here (no handle_text pipeline is running), so
    confirm()'s only wait is the lock acquisition itself — this isolates
    the "immediately after acquiring _speech_lock" check from the
    pre-existing "after queue.join()" one."""
    release_ev = asyncio.Event()

    class SlowPlayer:
        def __init__(self):
            self.played = 0
            self.stops = 0

        async def play(self, s):
            self.played += 1
            await release_ev.wait()

        def stop(self):
            # mirrors the real player: barge-in releases whatever play()
            # call currently holds the speech lock.
            self.stops += 1
            release_ev.set()

        def reset(self):
            pass

    o, _ = build()
    o.player = SlowPlayer()

    # A concurrent say() (standing in for a running handle_text's own
    # player.play under _speech_lock) holds _speech_lock via a slow
    # player.play() that blocks until player.stop() releases it.
    holder = asyncio.create_task(o.say("Holding."))
    await asyncio.sleep(0.02)
    assert o.player.played == 1        # holder is inside player.play(), lock held
    assert o._speech_queue is None     # no handle_text pipeline: only the lock gates confirm()

    confirm_task = asyncio.create_task(o.confirm("Bash: rm x"))
    await asyncio.sleep(0.02)   # confirm() is now blocked acquiring _speech_lock

    o._barged = True
    o.player.stop()   # releases player.play() -> lock is freed -> confirm() acquires it

    result = await asyncio.wait_for(confirm_task, 1)
    await asyncio.wait_for(holder, 1)

    assert result is False
    assert not any(t.startswith("Run") for t in o.tts.said)


async def test_barge_listener_failure_does_not_cancel_good_turn(caplog):
    class RaisingWake:
        def __init__(self): self.stops = 0
        async def wait(self, threshold=None, suppress=None):
            raise RuntimeError("mic hiccup")
        def stop(self):
            self.stops += 1
        def take_preroll(self):
            return np.zeros(0, dtype=np.int16)

    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    o.wake = RaisingWake()
    import logging
    with caplog.at_level(logging.ERROR, logger="veronica.orchestrator"):
        await o.one_turn()
    assert o.tts.said == ["Sure.", "Done."]
    assert not any("Something went wrong" in r.message for r in caplog.records)


async def test_play_exception_propagates_and_cleans_up():
    class RaisingPlayer:
        def __init__(self): self.calls = 0
        async def play(self, s):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("play boom")
        def stop(self): pass
        def reset(self): pass

    o, _ = build()
    o.player = RaisingPlayer()

    with pytest.raises(RuntimeError):
        await asyncio.wait_for(o.handle_text("hello"), 1)

    assert o._speech_queue is None


def build3(rec_pcms=(), stt_texts=()):
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec(rec_pcms), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)),
    )
    return o, states, events


async def test_events_full_turn():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what time is it"])
    await o.one_turn()
    kinds = [k for k, _ in ev]
    assert ("heard", "what time is it") in ev
    assert [p for k, p in ev if k == "sentence"] == ["Sure.", "Done."]
    voices = [p for k, p in ev if k == "voice"]
    assert len(voices) == 2 and voices[0]["step_ms"] == 50 and isinstance(voices[0]["levels"], list)
    # every sentence is preceded by its voice envelope
    assert kinds.index("voice") < kinds.index("sentence")
    assert ("state", "listening") in ev and ("state", "idle") in ev


async def test_events_confirm_ask_then_allowed_and_declined():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)], stt_texts=["yes", "no"])
    assert await o.confirm("Bash: rm x", "Bash: rm -rf x") is True
    assert await o.confirm("Bash: rm y", "Bash: rm -rf y") is False
    tools = [p for k, p in ev if k == "tool"]
    assert tools == [
        {"summary": "Bash: rm x", "detail": "Bash: rm -rf x", "decision": "ask", "timeout_ms": 0},
        {"summary": "Bash: rm x", "decision": "allowed"},
        {"summary": "Bash: rm y", "detail": "Bash: rm -rf y", "decision": "ask", "timeout_ms": 0},
        {"summary": "Bash: rm y", "decision": "declined"},
    ]
    assert ("prompt", "Run Bash: rm x?") in ev
    kinds = [k for k, _ in ev]
    # the "ask" tool event is emitted after the question is spoken (the
    # "prompt" event), not at confirm()'s entry.
    first_ask_idx = next(i for i, (k, p) in enumerate(ev) if k == "tool" and p.get("decision") == "ask")
    first_prompt_idx = kinds.index("prompt")
    assert first_prompt_idx < first_ask_idx


async def test_events_confirm_no_speech_declined():
    o, _, ev = build3(rec_pcms=[None])
    assert await o.confirm("Bash: rm x") is False
    assert [p["decision"] for k, p in ev if k == "tool"] == ["ask", "declined"]


async def test_events_warm():
    o, _, ev = build3()
    await o.warmup()
    assert [p for k, p in ev if k == "warm"] == [{"ready": False}, {"ready": True}]


async def test_on_event_errors_are_swallowed():
    def boom(k, p): raise RuntimeError("x")
    o = Orchestrator(Settings(), wake=Wake(), recorder=Rec([]), stt=STT([]), brain=Brain(), tts=TTS(), player=Player(), on_event=boom)
    await o.say("hi")   # must not raise


# -- item 3: live partial transcript -------------------------------------------

class PartialSTT:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    async def atranscribe(self, pcm):
        self.calls += 1
        return self.texts.pop(0) if self.texts else ""


class RecWithOnAudio(Rec):
    """Simulates the real Recorder firing on_audio during capture()."""
    def __init__(self, pcms, audio_chunks):
        super().__init__(pcms)
        self.audio_chunks = audio_chunks
        self.on_audio = None

    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0):
        if partial and self.on_audio is not None:
            for chunk in self.audio_chunks:
                self.on_audio(chunk)
            # give the loop.call_soon_threadsafe-scheduled callback (and the
            # task it creates) time to actually run before this returns.
            await asyncio.sleep(0.02)
        return await super().capture(max_s=max_s, preroll=preroll, partial=partial)


async def test_partial_transcript_emitted_during_capture_then_final_heard():
    events = []
    audio_chunks = [np.zeros(10, dtype=np.int16)]
    rec = RecWithOnAudio([np.zeros(1, np.int16), None], audio_chunks)
    partial = PartialSTT(["what time"])
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=rec, stt=STT(["what time is it"]),
        partial_stt=partial,
        brain=Brain(), tts=TTS(), player=Player(),
        on_event=lambda k, p: events.append((k, p)),
    )
    await o.one_turn()
    await asyncio.sleep(0.05)

    kinds = [k for k, _ in events]
    assert [p for k, p in events if k == "heard_partial"] == ["what time"]
    assert ("heard", "what time is it") in events
    assert kinds.index("heard_partial") < kinds.index("heard")


async def test_no_partial_transcript_without_partial_stt():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    assert o.partial_stt is None
    # Rec (the plain fake) has no on_audio attribute set by Orchestrator
    # since partial_stt is None.
    await o.one_turn()  # must not raise


async def test_partial_transcription_coalesces_while_one_in_flight():
    """A second on_audio callback that arrives while a partial transcription
    is still running must be dropped, not queued."""
    running = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class SlowPartial:
        async def atranscribe(self, pcm):
            calls.append(pcm)
            running.set()
            await release.wait()
            return "slow result"

    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([]), stt=STT([]),
        partial_stt=SlowPartial(),
        brain=Brain(), tts=TTS(), player=Player(),
    )
    o._loop = asyncio.get_running_loop()

    o._on_recorder_audio(np.zeros(4, dtype=np.int16))
    await asyncio.wait_for(running.wait(), 1)
    # second callback while the first is still in flight: coalesced away
    o._on_recorder_audio(np.zeros(4, dtype=np.int16))
    await asyncio.sleep(0.01)
    assert len(calls) == 1

    release.set()
    await asyncio.sleep(0.01)


# -- fix round: confirm() must not be contaminated by a stray partial ---------

async def test_confirm_capture_does_not_fire_on_audio():
    """confirm()'s yes/no capture must never invoke on_audio — a partial
    transcription of "yes" would otherwise overwrite the HUD's You row."""
    audio_chunks = [np.zeros(10, dtype=np.int16)]
    rec = RecWithOnAudio([np.zeros(1, np.int16)], audio_chunks)
    partial = PartialSTT(["should never appear"])
    events = []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=rec, stt=STT(["yes"]),
        partial_stt=partial,
        brain=Brain(), tts=TTS(), player=Player(),
        on_event=lambda k, p: events.append((k, p)),
    )
    o._loop = asyncio.get_running_loop()
    rec.on_audio = o._on_recorder_audio

    assert await o.confirm("Bash: rm x") is True
    await asyncio.sleep(0.05)
    assert [p for k, p in events if k == "heard_partial"] == []
    assert partial.calls == 0


# -- fix round: stale partial dropped after the turn's final heard ------------

async def test_run_partial_drops_result_if_gen_advanced_while_transcribing():
    """The gen re-check right before _emit: even if a partial transcription
    wasn't (or couldn't be) cancelled in time, a result computed for a
    generation that's no longer current must not be emitted."""
    events = []

    class SlowPartial:
        async def atranscribe(self, pcm):
            return "late text"

    o = Orchestrator(
        Settings(), wake=Wake(), recorder=Rec([]), stt=STT([]),
        partial_stt=SlowPartial(),
        brain=Brain(), tts=TTS(), player=Player(),
        on_event=lambda k, p: events.append((k, p)),
    )
    stale_gen = o._partial_gen
    o._partial_gen += 1  # simulate the owning capture() having already ended

    await o._run_partial(np.zeros(4, dtype=np.int16), stale_gen)

    assert [p for k, p in events if k == "heard_partial"] == []


def test_schedule_partial_drops_stale_gen():
    o = Orchestrator(
        Settings(), wake=Wake(), recorder=Rec([]), stt=STT([]),
        partial_stt=object(), brain=Brain(), tts=TTS(), player=Player(),
    )
    stale_gen = o._partial_gen
    o._partial_gen += 1

    o._schedule_partial(np.zeros(4, dtype=np.int16), stale_gen)

    assert o._partial_task is None  # never scheduled: the gen was already stale


async def test_partial_task_cancelled_when_capture_returns():
    o, _ = build()
    o.partial_stt = object()  # unused; we drive _partial_task directly
    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    o._partial_task = asyncio.ensure_future(fut)
    o._end_partial_window()
    await asyncio.sleep(0)
    assert o._partial_task.cancelled()


async def test_end_partial_window_bumps_gen_each_call():
    o, _ = build()
    g0 = o._partial_gen
    o._end_partial_window()
    assert o._partial_gen == g0 + 1
    o._end_partial_window()
    assert o._partial_gen == g0 + 2


# -- item 4: stop eavesdropping (shorter follow-up, spoken end phrases) --------

async def test_end_phrase_thanks_veronica_says_okay_and_goes_idle():
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["thanks veronica"])
    await o.one_turn()
    assert o.tts.said == ["Okay."]
    assert o.brain.asked == []
    assert states[-1] == "idle"
    assert "followup" not in states


async def test_end_phrase_thank_you_veronica_says_okay():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["Thank you, Veronica!"])
    await o.one_turn()
    assert o.tts.said == ["Okay."]
    assert o.brain.asked == []


@pytest.mark.parametrize(
    "heard",
    ["that's all", "thats all", "that is all", "stop", "goodbye", "never mind", "nevermind"],
)
async def test_end_phrase_silent_variants_go_idle_without_speaking(heard):
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=[heard])
    await o.one_turn()
    assert o.tts.said == []
    assert o.brain.asked == []
    assert states[-1] == "idle"


async def test_end_phrase_matches_case_and_punctuation_insensitively():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["Stop."])
    await o.one_turn()
    assert o.tts.said == []
    assert o.brain.asked == []


async def test_non_end_phrase_is_not_treated_as_end():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["stop the timer"])
    await o.one_turn()
    assert o.brain.asked == ["stop the timer"]


async def test_followup_empty_transcript_ends_silently():
    """An empty transcript on a follow-up capture (not the first listen
    after wake) must not say "Sorry, didn't catch that." nor reopen another
    follow-up window — just go idle."""
    o, states = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)],
        stt_texts=["hi", ""],
    )
    await o.one_turn()
    assert o.tts.said == ["Sure.", "Done."]
    assert states[-1] == "idle"
    assert states.count("followup") == 1


async def test_first_capture_empty_transcript_keeps_sorry_and_followup():
    """The first capture after wake (not a follow-up) keeps today's
    behavior: "Sorry, didn't catch that." then a follow-up window."""
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=[""])
    await o.one_turn()
    assert o.tts.said == ["Sorry, didn't catch that."]
    assert o.brain.asked == []


async def test_empty_transcript_after_barge_relisten_keeps_sorry():
    """The re-listen after a barge-in is not a follow-up capture: an empty
    transcript there should still get "Sorry, didn't catch that."."""
    o, states = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), None],
        stt_texts=["hi", ""],
    )
    o.wake = BargeWake(barge_on_call=1)
    o.brain = SlowBrain()
    await o.one_turn()
    assert "Sorry, didn't catch that." in o.tts.said


# -- commit 2: local voice intents --------------------------------------------

async def test_end_intent_emits_hud_hide():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["stop"])
    await o.one_turn()
    assert ("hud", {"mode": "hide"}) in ev
    assert o.tts.said == []


async def test_end_intent_spoken_variant_says_okay_and_hides_hud():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["thanks veronica"])
    await o.one_turn()
    assert o.tts.said == ["Okay."]
    assert ("hud", {"mode": "hide"}) in ev


async def test_end_intent_new_phrase_go_idle():
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["go idle"])
    await o.one_turn()
    assert o.tts.said == []
    assert o.brain.asked == []
    assert states[-1] == "idle"


async def test_hud_hide_intent_is_silent_and_goes_idle():
    o, states, ev = build3(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["hide"])
    await o.one_turn()
    assert o.tts.said == []
    assert o.brain.asked == []
    assert states[-1] == "idle"
    assert ("hud", {"mode": "hide"}) in ev
    assert "followup" not in states


async def test_hud_mini_intent_says_okay_emits_hud_and_continues_followup():
    o, states, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["shrink"])
    await o.one_turn()
    assert o.tts.said == ["Okay."]
    assert ("hud", {"mode": "mini"}) in ev
    assert "followup" in states
    assert o.brain.asked == []


async def test_hud_full_intent_says_okay_emits_hud_and_continues_followup():
    o, states, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["expand"])
    await o.one_turn()
    assert o.tts.said == ["Okay."]
    assert ("hud", {"mode": "full"}) in ev
    assert "followup" in states


async def test_non_intent_text_goes_to_brain():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what time is it"])
    await o.one_turn()
    assert o.brain.asked == ["what time is it"]


async def test_followup_window_default_is_four_seconds():
    assert Settings().followup_window_s == 4


async def test_vad_silence_ms_default_is_600():
    assert Settings().vad_silence_ms == 600


# -- announce() ---------------------------------------------------------------

class _WakeBlocksThenCancel:
    """First call() blocks until stop() (mirroring the real wake listener,
    which a stop() interrupts); every later call raises CancelledError to
    break the run_forever loop for the test."""
    def __init__(self):
        self.calls = 0
        self.stops = 0
        self._ev = None

    async def wait(self, threshold=None, suppress=None):
        self.calls += 1
        if self.calls > 1:
            raise asyncio.CancelledError()
        self._ev = asyncio.Event()
        await self._ev.wait()
        return False

    def stop(self):
        self.stops += 1
        if self._ev is not None:
            self._ev.set()

    def take_preroll(self):
        return np.zeros(0, dtype=np.int16)


class _WakeOnceThenCancel:
    """First call() returns True immediately (triggers one_turn); a
    barge-listener call (threshold set) blocks until stop(); every later
    main-loop call raises CancelledError to break the loop for the test."""
    def __init__(self):
        self.calls = 0
        self.stops = 0
        self._barge_ev = None

    async def wait(self, threshold=None, suppress=None):
        if threshold is not None:
            self._barge_ev = asyncio.Event()
            await self._barge_ev.wait()
            return False
        self.calls += 1
        if self.calls > 1:
            raise asyncio.CancelledError()
        return True

    def stop(self):
        self.stops += 1
        if self._barge_ev is not None:
            self._barge_ev.set()

    def take_preroll(self):
        return np.zeros(0, dtype=np.int16)


async def test_announce_queues():
    o, _ = build()
    assert o._announce_queue.empty()
    await o.announce("Timer done")
    assert not o._announce_queue.empty()
    assert o._announce_queue.get_nowait() == "Timer done"


async def test_announce_delivered_while_idle():
    """Queued while idle: the wake listener is stopped, the announcement is
    chimed + spoken, then the wake loop resumes (a fresh wait() call)."""
    o, states = build()
    wake = _WakeBlocksThenCancel()
    o.wake = wake

    async def deliver_soon():
        await asyncio.sleep(0.01)
        await o.announce("Timer done")

    asyncio.ensure_future(deliver_soon())
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert "Timer done" in o.tts.said
    assert wake.stops >= 1
    assert "speaking" in states
    # wake.wait() was called again after delivering (loop resumed)
    assert wake.calls >= 2


async def test_announce_delivered_after_turn_ends():
    """Queued mid-turn: not spoken until the in-flight turn's own sentences
    are done and the orchestrator is idle again."""
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["do something"])
    o.wake = _WakeOnceThenCancel()

    class AnnouncingBrain:
        def __init__(self, orch):
            self.orch = orch

        async def ask(self, text):
            yield "Sure."
            await self.orch.announce("Timer done")
            yield "Done."

    o.brain = AnnouncingBrain(o)

    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert o.tts.said == ["Sure.", "Done.", "Timer done"]
    assert "followup" in states


async def test_wake_and_announce_same_tick_prefers_wake_and_requeues():
    """If the wake word and a queued announcement both resolve in the same
    tick, the wake path wins (a real detection must not be swallowed) and
    the announcement is put back to be delivered after the turn."""
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hello"])
    wake = _WakeOnceThenCancel()
    o.wake = wake
    # Scheduled (not put_nowait'd) before run_forever starts, so it lands in
    # the queue during run_forever's first real suspension — after the
    # top-of-loop drain already found the queue empty, but before wake_task
    # and signal_task run their (synchronous, non-suspending) first step —
    # landing both of them in the same `done` set.
    asyncio.ensure_future(o.announce("Timer done"))

    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert o.brain.asked == ["hello"]
    assert "Sure." in o.tts.said and "Timer done" in o.tts.said
    assert o.tts.said.index("Timer done") > o.tts.said.index("Sure.")


async def test_muted_announcement_held_no_flicker():
    """Queued while muted: no state change, no chime/say (both no-ops while
    muted), and it's still in the queue afterward."""
    o, states = build()
    o.muted = True
    wake = _WakeBlocksThenCancel()
    o.wake = wake

    async def deliver_soon():
        await asyncio.sleep(0.01)
        await o.announce("Timer done")
        await asyncio.sleep(0.05)
        wake.stop()  # break the test out via the blocked wake_task

    asyncio.ensure_future(deliver_soon())
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert o.tts.said == []
    assert "speaking" not in states
    assert not o._announce_queue.empty()


async def test_muted_announcement_delivered_on_unmute():
    o, states = build()
    o.muted = True
    wake = _WakeBlocksThenCancel()
    o.wake = wake

    async def deliver_then_unmute():
        await asyncio.sleep(0.01)
        await o.announce("Timer done")
        await asyncio.sleep(0.01)
        assert o.tts.said == []  # still held while muted
        o.muted = False

    asyncio.ensure_future(deliver_then_unmute())
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()

    assert o.tts.said == ["Timer done"]
    assert "speaking" in states


# -- memory (remember/forget intents, turn logging) --------------------------

class FakeStore:
    def __init__(self):
        self.turns = []
        self.facts = []

    def add_turn(self, heard, reply):
        self.turns.append((heard, reply))

    def add_fact(self, text):
        self.facts.append(text)
        return len(self.facts)

    def delete_fact_matching(self, text):
        before = len(self.facts)
        self.facts = [f for f in self.facts if text.lower() not in f.lower()]
        return before - len(self.facts)


async def test_handle_text_logs_turn_when_store_present():
    store = FakeStore()
    o, _ = build()
    o.store = store
    out = await o.handle_text("hello")
    assert out == ["Sure.", "Done."]
    assert store.turns == [("hello", "Sure. Done.")]


async def test_handle_text_skips_logging_without_store():
    o, _ = build()
    assert o.store is None
    await o.handle_text("hello")  # no store -> no error


async def test_handle_text_skips_logging_when_memory_disabled():
    store = FakeStore()
    o, _ = build()
    o.store = store
    o.s.memory_enabled = False
    await o.handle_text("hello")
    assert store.turns == []


class EmptyBrain:
    async def ask(self, text):
        return
        yield  # pragma: no cover - makes this an async generator


async def test_handle_text_skips_logging_when_nothing_spoken():
    store = FakeStore()
    o, _ = build()
    o.store = store
    o.brain = EmptyBrain()
    out = await o.handle_text("...")
    assert out == []
    assert store.turns == []


async def test_remember_intent_stores_fact_and_says_got_it():
    store = FakeStore()
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["remember that I like tea"])
    o.store = store
    await o.one_turn()
    assert store.facts == ["I like tea"]
    assert "Got it." in o.tts.said


async def test_remember_intent_skips_claude():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["remember that I like tea"])
    await o.one_turn()
    assert o.brain.asked == []


async def test_forget_intent_deletes_and_says_forgotten():
    store = FakeStore()
    store.facts = ["I like tea"]
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["forget that I like tea"])
    o.store = store
    await o.one_turn()
    assert store.facts == []
    assert "Forgotten." in o.tts.said


async def test_forget_intent_no_match_says_didnt_have_that():
    store = FakeStore()
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["forget that I like tea"])
    o.store = store
    await o.one_turn()
    assert "I didn't have that." in o.tts.said


# -- commit: voice mute/unmute/quit intents -----------------------------------

async def test_mute_intent_says_muted_sets_muted_and_hides_hud():
    o, states, ev = build3(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["mute yourself"])
    await o.one_turn()
    assert o.tts.said == ["Muted."]
    assert o.muted is True
    assert ("hud", {"mode": "hide"}) in ev
    assert states[-1] == "idle"
    assert "followup" not in states
    assert o.brain.asked == []


async def test_muted_wake_capture_unmute_says_im_back():
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["unmute"])
    o.wake = _WakeOnceThenCancel()
    o.muted = True
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert o.muted is False
    assert o.tts.said == ["I'm back."]
    assert o.brain.asked == []


async def test_muted_wake_capture_other_text_ignored_silently():
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["what time is it"])
    o.wake = _WakeOnceThenCancel()
    o.muted = True
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert o.muted is True
    assert o.tts.said == []
    assert o.brain.asked == []


async def test_muted_wake_capture_no_speech_stays_muted():
    o, _ = build(rec_pcms=[None])
    o.wake = _WakeOnceThenCancel()
    o.muted = True
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert o.muted is True
    assert o.tts.said == []


async def test_quit_intent_yes_says_goodbye_and_calls_on_quit():
    called = []
    o, states = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)],
        stt_texts=["quit veronica", "yes"],
    )
    o._on_quit = lambda: called.append(True)
    await o.one_turn()
    assert called == [True]
    assert "Goodbye." in o.tts.said
    assert states[-1] == "idle"
    # natural prompt wording, not the generic confirm() "Run {summary}?" form
    assert o.tts.said[0] == "Quit Veronica?"


async def test_quit_intent_no_does_not_quit_and_continues_turn():
    called = []
    o, states = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), None],
        stt_texts=["quit veronica", "no"],
    )
    o._on_quit = lambda: called.append(True)
    await o.one_turn()
    assert called == []
    assert "Goodbye." not in o.tts.said
    assert "followup" in states


async def test_on_quit_defaults_to_noop():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)], stt_texts=["quit", "yes"])
    await o.one_turn()  # must not raise even with no on_quit provided
    assert "Goodbye." in o.tts.said


async def test_remember_without_store_still_says_got_it():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["remember that I like tea"])
    assert o.store is None
    await o.one_turn()
    assert "Got it." in o.tts.said


# -- batch A: screen awareness fast path --------------------------------------

import veronica.orchestrator as orchestrator_mod


class ImageBrain:
    def __init__(self):
        self.asked = []

    async def ask(self, text, images=()):
        self.asked.append((text, tuple(images)))
        yield "I see a browser."


async def test_screen_intent_captures_and_sends_image(monkeypatch):
    monkeypatch.setattr(
        orchestrator_mod, "capture_screenshot", lambda region: (b"PNGDATA", "/tmp/x.png")
    )
    o, states, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what's on my screen"])
    o.brain = ImageBrain()
    await o.one_turn()
    assert o.brain.asked == [("what's on my screen", (b"PNGDATA",))]
    assert ("tool", {"summary": "Look at screen", "decision": "auto"}) in ev
    assert "I see a browser." in o.tts.said


async def test_screen_intent_capture_failure_falls_back_to_text_only():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["look at my screen"])

    async def fake_to_thread(fn, *a, **k):
        return "denied"

    orig_to_thread = asyncio.to_thread
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "to_thread", fake_to_thread)
        await o.one_turn()
    assert o.brain.asked == ["look at my screen"]


async def test_handle_text_with_images_calls_brain_ask_with_images():
    o, _ = build()
    o.brain = ImageBrain()
    await o.handle_text("describe this", images=[b"abc"])
    assert o.brain.asked == [("describe this", (b"abc",))]


async def test_handle_text_without_images_omits_kwarg():
    """A Brain.ask(text) fake without an images param must keep working when
    handle_text is called with no images (the common, non-screen path)."""
    o, _ = build()
    await o.handle_text("hi")
    assert o.brain.asked == ["hi"]
