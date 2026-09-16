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
        self.hold_calls = []
        self.finish_calls = 0
        self.stop_calls = 0
    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0, hold=False):
        self.preroll_calls.append(preroll)
        self.hold_calls.append(hold)
        return self.pcms.pop(0) if self.pcms else None
    def has_speech(self, pcm): return self._has_speech
    def finish(self): self.finish_calls += 1
    def stop(self): self.stop_calls += 1

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
    async def asynth(self, text, lang=None):
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
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
    await o.one_turn()
    # chime samples go through player.play like speech; count plays: wake chime + 2 sentences + followup chime
    assert o.player.played == 4


async def test_wake_chime_skipped_when_preroll_has_speech():
    o, _ = build()
    o.recorder = Rec([np.zeros(1, np.int16), None], has_speech=True)
    o.stt = STT(["tell me a joke"])
    await o.one_turn()
    # wake chime skipped (preroll already has speech), so: 2 sentences + followup chime
    assert o.player.played == 3


async def test_wake_chime_played_when_preroll_has_no_speech():
    o, _ = build()
    o.recorder = Rec([np.zeros(1, np.int16), None], has_speech=False)
    o.stt = STT(["tell me a joke"])
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
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
    await o.one_turn()
    assert o.brain.asked == ["tell me a joke"]
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
        async def asynth(self, text, lang=None):
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

    async def asynth(self, text, lang=None):
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
        self.calls = []

    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0, hold=False):
        self.max_s_calls.append(max_s)
        self.calls.append({"max_s": max_s, "partial": partial, "skip_ms": skip_ms, "hold": hold})
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
        wake=WakeOnceThenCancel(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(["tell me a joke"]),
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
        async def asynth(self, text, lang=None):
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
        async def asynth(self, text, lang=None):
            return np.zeros(10, dtype=np.float32), 24000

    class SlowTTS:
        """Synth that stays pending for a full second — long enough that,
        with maxsize=2 and a brain producing faster than synth completes,
        the producer will be blocked mid-`queue.put()` with an
        already-created-but-not-yet-queued synth future at cancel time
        (exercising the orphaned-future fix), and other futures will still
        be genuinely in-flight (not just already-done) when drained."""
        def __init__(self): self.started = 0; self.finished = 0
        async def asynth(self, text, lang=None):
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
        async def asynth(self, text, lang=None):
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
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
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

    # still within wake_window_s (1.2) + wake_hop_s (0.25) = 1.45 s of playback ending
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
    o, states, ev = build3(rec_pcms=[], stt_texts=["first"])
    rec = StoppableRec([np.zeros(1, np.int16), StoppableRec.BLOCK, None])
    o.recorder = rec
    o.wake = BargeWake(barge_on_call=1)
    o.brain = ConfirmDuringBargeBrain(o)
    await o.one_turn()
    assert rec.stops == 1                              # in-flight confirm capture was stopped
    # confirm() (awaited inside the turn's own producer task here) is torn
    # down with the turn — "declined", never left orphaned in capture(),
    # and the brain never gets to continue past it ("Second." unreached).
    assert o.brain.results == []
    assert ("tool", {"summary": "Bash: rm x", "decision": "declined"}) in ev
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

        async def asynth(self, text, lang=None):
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
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
    o.wake = BargeWake()  # never barges
    await o.one_turn()
    assert o.wake.stops == 1  # listener stopped once when handle_text finished


async def test_barge_uses_barge_threshold():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
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

    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
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
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
    await o.one_turn()
    kinds = [k for k, _ in ev]
    assert ("heard", "tell me a joke") in ev
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
        stt_texts=["tell me a joke", ""],
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
        stt_texts=["tell me a joke", ""],
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
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
    await o.one_turn()
    assert o.brain.asked == ["tell me a joke"]


async def test_followup_window_default_is_four_seconds():
    assert Settings().followup_window_s == 4


async def test_vad_silence_ms_default_is_1200():
    assert Settings().vad_silence_ms == 1200


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
    assert o._announce_queue.get_nowait() == ("Timer done", None)


async def test_announce_queue_stores_expiry_and_skips_stale():
    """A queued announcement whose expires_at has passed is dropped at
    delivery time (a "starts in 5 minutes" nudge after the meeting began);
    one that's still in the future, or has no expiry, is spoken."""
    import datetime as dt
    o, states = build()
    wake = _WakeBlocksThenCancel()
    o.wake = wake
    past = dt.datetime.now() - dt.timedelta(minutes=1)
    future = dt.datetime.now() + dt.timedelta(hours=1)
    await o.announce("Stale nudge", expires_at=past)
    await o.announce("Fresh nudge", expires_at=future)
    await o.announce("Timer done")
    assert o._announce_queue.qsize() == 3
    task = asyncio.ensure_future(o.run_forever())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "Stale nudge" not in o.tts.said
    assert o.tts.said == ["Fresh nudge", "Timer done"]


async def test_stale_announcement_via_signal_path_is_skipped():
    """Same expiry check on the direct (idle-wait) delivery path."""
    import datetime as dt
    o, states = build()
    wake = _WakeBlocksThenCancel()
    o.wake = wake

    async def deliver_soon():
        await asyncio.sleep(0.01)
        await o.announce("Stale nudge", expires_at=dt.datetime.now() - dt.timedelta(seconds=1))

    asyncio.ensure_future(deliver_soon())
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert o.tts.said == []
    assert "speaking" not in states
    assert wake.calls >= 2


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
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
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

    assert o.brain.asked == ["tell me a joke"]
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
        orchestrator_mod, "capture_screenshot", lambda region: (b"PNGDATA", "/tmp/x.png", "image/png")
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


# -- batch A: music fast path --------------------------------------------------

from veronica.tools import music as music_tools_mod


async def test_music_pause_intent_speaks_result(monkeypatch):
    async def fake_pause(args):
        return {"content": [{"type": "text", "text": "Paused."}]}

    monkeypatch.setattr(music_tools_mod.music_pause, "handler", fake_pause)
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["pause"])
    await o.one_turn()
    assert "Paused." in o.tts.said
    assert ("tool", {"summary": "Pause music", "decision": "auto"}) in ev
    assert o.brain.asked == []


async def test_music_now_playing_intent_speaks_result(monkeypatch):
    async def fake_now_playing(args):
        return {"content": [{"type": "text", "text": "Now playing Foo by Bar."}]}

    monkeypatch.setattr(music_tools_mod.music_now_playing, "handler", fake_now_playing)
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what's playing"])
    await o.one_turn()
    assert "Now playing Foo by Bar." in o.tts.said


async def test_music_intent_error_speaks_fallback(monkeypatch):
    async def fake_next(args):
        return {"content": [{"type": "text", "text": "error: no player"}], "is_error": True}

    monkeypatch.setattr(music_tools_mod.music_next, "handler", fake_next)
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["skip"])
    await o.one_turn()
    assert "Sorry, I couldn't do that." in o.tts.said


# -- batch A: push-to-talk (A2) ------------------------------------------------
#
# PTT is a *signal* into run_forever / the in-flight turn (never a concurrent
# turn): ptt_start() sets an event that run_forever races with the wake
# listener while idle, and that _run_with_barge / every capture race during a
# turn. These fixtures let the tests observe concurrency: EventWake blocks
# until stop() (one-shot False) or a queued True; SlowRec's capture() blocks
# until finish()/stop() and counts overlapping captures.


class EventWake:
    """wake.wait() blocks on an asyncio queue; stop() -> False one-shot."""
    def __init__(self):
        self.q = asyncio.Queue()
        self.waits = 0
        self.stops = 0
        self.suppress_seen = []

    async def wait(self, threshold=None, suppress=None):
        self.waits += 1
        self.suppress_seen.append(suppress)
        return await self.q.get()

    def stop(self):
        self.stops += 1
        self.q.put_nowait(False)

    def take_preroll(self):
        return np.zeros(0, dtype=np.int16)


class SlowRec(Rec):
    """capture() blocks until finish()/stop() so concurrency is observable."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.active = 0
        self.max_active = 0
        self.ev = asyncio.Event()
        self.capture_kw = []

    async def capture(self, max_s=None, preroll=None, partial=False, skip_ms=0, hold=False):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.hold_calls.append(hold)
        self.capture_kw.append({"max_s": max_s, "hold": hold, "skip_ms": skip_ms})
        try:
            await self.ev.wait()
            self.ev.clear()
        finally:
            self.active -= 1
        return self.pcms.pop(0) if self.pcms else None

    def finish(self):
        self.finish_calls += 1
        self.ev.set()

    def stop(self):
        self.stop_calls += 1
        self.ev.set()


class InterruptibleBrain(Brain):
    def __init__(self):
        super().__init__()
        self.interrupts = 0

    async def interrupt(self):
        self.interrupts += 1


def build_ptt(stt_texts=(), pcms=None):
    o, states = build(stt_texts=stt_texts)
    o.ready = True
    o.wake = EventWake()
    o.brain = InterruptibleBrain()
    o.recorder = SlowRec(pcms if pcms is not None else [np.zeros(1, np.int16)] * 4)
    o.events = []
    o._on_event = lambda k, p: o.events.append((k, p))
    return o, states


async def _settle(n=5):
    for _ in range(n):
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)


async def _cancel(task):
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_ptt_start_is_ignored_before_ready_and_while_muted():
    o, _ = build()
    o.ptt_start()
    assert not o._ptt_event.is_set() and not o._ptt_held
    o.ready = True
    o.muted = True
    o.ptt_start()
    assert not o._ptt_event.is_set() and not o._ptt_held
    o.muted = False
    o.ptt_start()
    assert o._ptt_event.is_set() and o._ptt_held


def test_ptt_start_repeat_while_held_is_noop_and_end_without_start_is_noop():
    o, _ = build()
    o.ready = True
    o.ptt_end()                      # release with no press: nothing
    assert o.recorder.finish_calls == 0
    o.ptt_start()
    o._ptt_event.clear()             # as _listen_after_ptt would
    o.ptt_start()                    # key-repeat: must not re-raise the signal
    assert not o._ptt_event.is_set()
    o.ptt_end()
    assert not o._ptt_held
    assert o.recorder.finish_calls == 0   # capture wasn't in flight: nothing to finish


def test_ptt_end_finishes_recorder_only_while_hold_capture_in_flight():
    o, _ = build()
    o.ready = True
    o.ptt_start()
    o._ptt_capturing = True
    o.ptt_end()
    assert o.recorder.finish_calls == 1


async def test_ptt_while_idle_runs_one_turn_inside_run_forever():
    """Regression (reviewer repro_ptt: scenario_wake_word_during_ptt). PTT
    while idle with run_forever live: exactly one capture in flight, no
    second one_turn, and the wake listener is NOT re-armed without
    suppress during the answer (only the barge listener is alive)."""
    o, states = build_ptt(stt_texts=["ptt text"], pcms=[np.zeros(1, np.int16), None])
    rf = asyncio.create_task(o.run_forever())
    await _settle()
    assert o.wake.waits == 1                      # idle wake listener armed
    o.ptt_start()
    await _settle()
    assert o.wake.stops == 1                      # idle listener stopped...
    assert o.wake.waits == 1                      # ...and NOT re-armed while capturing
    assert o.recorder.active == 1 and o.recorder.hold_calls == [True]
    assert o.state == "listening"
    o.ptt_end()                                    # key up -> capture finishes
    await _settle()
    # answering now: the only live listener is the barge listener, with suppress
    assert o.wake.waits == 2
    assert callable(o.wake.suppress_seen[-1])
    assert o.recorder.max_active == 1
    assert o.brain.asked == ["ptt text"]
    assert "Sure." in o.tts.said
    # follow-up capture is a normal (non-hold) capture
    assert o.recorder.hold_calls == [True, False]
    o.recorder.stop()                              # end the follow-up window (None)
    await _settle()
    assert o.state == "idle"
    assert o.wake.waits == 3                       # idle listener re-armed after the turn
    assert o.wake.suppress_seen[-1] is None
    await _cancel(rf)


async def test_wake_word_during_ptt_capture_does_not_start_second_turn():
    """The reviewer's original repro: with the old design the idle wake
    listener got re-armed during the PTT capture and a wake detection
    started a concurrent one_turn (two captures at once)."""
    o, _ = build_ptt(stt_texts=["ptt text", "wake text"])
    rf = asyncio.create_task(o.run_forever())
    await _settle()
    o.ptt_start()
    await _settle()
    assert o.wake.waits == 1
    o.wake.q.put_nowait(True)                      # a wake detection arrives anyway
    await _settle()
    assert o.recorder.max_active == 1
    assert o.recorder.hold_calls == [True]
    assert o.brain.asked == []
    await _cancel(rf)
    o.recorder.finish()
    await _settle()


async def test_ptt_while_speaking_cancels_turn_and_hold_captures():
    """Regression (reviewer repro_barge): PTT while speaking -> the turn is
    cancelled, brain.interrupt() called, NO follow-up capture from the old
    turn, the hold capture proceeds, then the new answer."""
    o, states = build_ptt(stt_texts=["tell me a joke", "ptt text"])
    o.recorder = SlowRec([np.zeros(1, np.int16), np.zeros(1, np.int16), None])
    interrupts = []

    class SlowBrain2:
        def __init__(self):
            self.asked = []
            self.ev = asyncio.Event()

        async def ask(self, text, images=()):
            self.asked.append(text)
            yield "First sentence."
            if text == "tell me a joke":
                await self.ev.wait()       # "thinking" about sentence 2 until interrupted

        async def interrupt(self):
            interrupts.append(True)
            self.ev.set()

    o.brain = SlowBrain2()
    t = asyncio.create_task(o.one_turn())
    await _settle()
    o.recorder.finish()                            # first listen returns -> handle_text under _run_with_barge
    await _settle()
    assert o.state == "speaking"
    assert o.wake.waits == 1                       # barge listener up
    o.ptt_start()
    await _settle()
    assert interrupts == [True]
    assert o.player.stops >= 1
    assert o._barged is True
    assert o.wake.stops == 1                       # barge listener torn down
    assert o.state == "listening"
    assert o.recorder.active == 1 and o.recorder.hold_calls == [False, True]
    assert o.recorder.max_active == 1
    o.ptt_end()
    await _settle()
    assert o.brain.asked == ["tell me a joke", "ptt text"]
    assert o.recorder.hold_calls == [False, True, False]   # then a normal follow-up window
    o.recorder.stop()
    await _settle()
    assert t.done() and t.exception() is None
    assert o.state == "idle"


async def test_ptt_during_followup_capture_switches_to_hold_capture():
    """Regression (reviewer repro_ptt: scenario_ptt_during_followup): with
    the old design a PTT press during the follow-up window opened a second
    capture beside the follow-up one and left a stale wake.stop() behind."""
    o, states = build_ptt(stt_texts=["tell me a joke", "ptt text"])
    o.recorder = SlowRec([np.zeros(1, np.int16), None, np.zeros(1, np.int16), None])
    t = asyncio.create_task(o.one_turn())
    await _settle()
    o.recorder.finish()                            # first listen returns
    await _settle()
    assert o.state == "followup"
    o.ptt_start()
    await _settle()
    assert o.recorder.stop_calls == 1              # follow-up capture was stopped...
    assert o.recorder.hold_calls == [False, False, True]   # ...and a hold capture opened
    assert o.recorder.max_active == 1
    assert o.wake.stops == 1                       # only the barge listener's own stop; no stale one
    o.ptt_end()
    await _settle()
    assert o.brain.asked == ["tell me a joke", "ptt text"]
    o.recorder.stop()
    await _settle()
    assert t.done() and t.exception() is None
    assert o.state == "idle"


async def test_ptt_during_initial_listen_switches_to_hold_capture():
    o, _ = build_ptt(stt_texts=["ptt text"])
    o.recorder = SlowRec([None, np.zeros(1, np.int16), None])
    t = asyncio.create_task(o.one_turn())
    await _settle()
    assert o.state == "listening" and o.recorder.hold_calls == [False]
    o.ptt_start()
    await _settle()
    assert o.recorder.stop_calls == 1
    assert o.recorder.hold_calls == [False, True]
    assert o.recorder.max_active == 1
    o.ptt_end()
    await _settle()
    assert o.brain.asked == ["ptt text"]
    o.recorder.stop()
    await t


async def test_ptt_while_confirming_declines_confirm_then_runs_ptt_turn():
    """PTT during a confirmation prompt is a barge (confirm -> declined),
    never the answer to the question."""
    o, states = build_ptt(stt_texts=["do the thing", "ptt text"])
    o.recorder = SlowRec([np.zeros(1, np.int16), None, np.zeros(1, np.int16), None])
    results = []
    interrupts = []

    class ConfirmBrain:
        def __init__(self, orch):
            self.orch = orch
            self.asked = []

        async def ask(self, text, images=()):
            self.asked.append(text)
            if text == "do the thing":
                results.append(await self.orch.confirm("Bash: rm x"))
            yield "Okay."

        async def interrupt(self):
            interrupts.append(True)

    o.brain = ConfirmBrain(o)
    t = asyncio.create_task(o.one_turn())
    await _settle()
    o.recorder.finish()                            # first listen returns
    await _settle()
    assert o.state == "confirming"
    assert o.recorder.active == 1 and o.recorder.hold_calls == [False, False]
    o.ptt_start()
    await _settle()
    assert o.recorder.stop_calls == 1              # the yes/no capture was unblocked, not orphaned
    # confirm() is torn down with the turn: declined, and the brain never
    # continues past it (results stays empty — "Okay." for "do the thing"
    # is never reached)
    assert results == []
    assert ("tool", {"summary": "Bash: rm x", "decision": "declined"}) in o.events
    assert interrupts == [True]
    assert o.recorder.hold_calls == [False, False, True]
    assert o.recorder.max_active == 1
    o.ptt_end()
    await _settle()
    assert o.brain.asked == ["do the thing", "ptt text"]
    o.recorder.stop()
    await t
    assert o.state == "idle"


async def test_ptt_quick_tap_released_during_chime_leaves_nothing_stuck():
    """Regression (reviewer repro_ptt: scenario_quick_tap): a release that
    lands before the capture started (e.g. during the chime) must not
    leave _ptt_capturing/_ptt_held stuck or the mic open."""
    o, states = build_ptt(stt_texts=["x"])
    o.recorder = SlowRec([None])
    rf = asyncio.create_task(o.run_forever())
    await _settle()
    o.ptt_start()
    o.ptt_end()                                    # released before run_forever even woke up
    await _settle()
    assert o.recorder.hold_calls == []             # mic never opened
    assert o.recorder.active == 0
    assert not o._ptt_capturing and not o._ptt_held
    assert not o._ptt_event.is_set()
    assert o.state == "idle"
    assert o.brain.asked == []
    assert o.wake.waits == 2                       # idle listener re-armed
    # and a real press afterwards still works
    o.ptt_start()
    await _settle()
    assert o.recorder.hold_calls == [True] and o.recorder.active == 1
    o.ptt_end()
    await _settle()
    await _cancel(rf)


async def test_ptt_release_during_chime_ends_capture_that_already_started():
    """The capture is started before the chime is awaited, so a release
    that lands while the chime is still playing finishes a real capture."""
    o, _ = build_ptt(stt_texts=["tell me a joke"])
    o.recorder = SlowRec([np.zeros(1, np.int16), None])

    class SlowPlayer(Player):
        async def play(self, s):
            await asyncio.sleep(0.05)
            await super().play(s)

    o.player = SlowPlayer()
    o.ptt_start()
    t = asyncio.create_task(o.one_turn(ptt=True))
    await asyncio.sleep(0.01)                      # chime in progress, capture in flight
    assert o.recorder.active == 1 and o._ptt_capturing
    o.ptt_end()
    assert o.recorder.finish_calls == 1
    await _settle()
    assert not o._ptt_capturing
    o.recorder.stop()
    await asyncio.wait_for(t, 2)
    assert o.brain.asked == ["tell me a joke"]


async def test_ptt_capture_uses_ptt_max_s_hold_and_partial():
    o, _ = build_ptt(stt_texts=["hello"])
    o.s = Settings(followup_window_s=0, confirm_listen_s=0, ptt_max_s=7)
    o.recorder = SlowRec([np.zeros(1, np.int16), None])
    o.ptt_start()
    t = asyncio.create_task(o.one_turn(ptt=True))
    await _settle()
    assert o.recorder.capture_kw[0] == {"max_s": 7, "hold": True, "skip_ms": 0}
    o.ptt_end()
    await _settle()
    o.recorder.stop()
    await t


async def test_ptt_no_speech_goes_idle_without_asking_brain():
    o, states = build_ptt()
    o.recorder = SlowRec([None])
    o.ptt_start()
    t = asyncio.create_task(o.one_turn(ptt=True))
    await _settle()
    o.ptt_end()
    await t
    assert o.brain.asked == []
    assert states[-1] == "idle"


async def test_ptt_turn_error_is_reported_like_a_wake_turn():
    """I8: a failing PTT turn runs inside run_forever's error handling."""
    o, states = build_ptt(stt_texts=["boom"])
    o.recorder = SlowRec([np.zeros(1, np.int16)])

    class BoomBrain:
        asked = []

        async def ask(self, text):
            raise RuntimeError("kaboom")
            yield  # noqa: unreachable, makes this an async generator

        async def interrupt(self):
            pass

    o.brain = BoomBrain()
    rf = asyncio.create_task(o.run_forever())
    await _settle()
    o.ptt_start()
    await _settle()
    o.ptt_end()
    await _settle()
    assert "Something went wrong, check the log." in o.tts.said
    assert o.state == "idle"
    assert o.wake.waits == 3                       # idle, barge listener, idle again: loop resumed
    assert o.wake.suppress_seen[-1] is None
    await _cancel(rf)


async def test_ptt_can_barge_her_own_ptt_answer():
    """Spec A2: PTT works while she's speaking — including the answer to a
    previous PTT question (so _ptt_capturing must be scoped to the capture,
    not the whole turn)."""
    o, _ = build_ptt(stt_texts=["one", "two"])
    o.recorder = SlowRec([np.zeros(1, np.int16), np.zeros(1, np.int16), None])
    interrupts = []

    class HangBrain:
        def __init__(self):
            self.asked = []
            self.ev = asyncio.Event()

        async def ask(self, text):
            self.asked.append(text)
            yield "Sure."
            if text == "one":
                await self.ev.wait()

        async def interrupt(self):
            interrupts.append(True)
            self.ev.set()

    o.brain = HangBrain()
    o.ptt_start()
    t = asyncio.create_task(o.one_turn(ptt=True))
    await _settle()
    o.ptt_end()
    await _settle()
    assert o.state == "speaking"
    o.ptt_start()                                  # second press during her answer
    await _settle()
    assert interrupts == [True]
    assert o.recorder.hold_calls == [True, True]
    assert o.recorder.max_active == 1
    o.ptt_end()
    await _settle()
    assert o.brain.asked == ["one", "two"]
    o.recorder.stop()
    await t


async def test_ptt_and_announcement_same_tick_requeues_announcement():
    o, _ = build_ptt(stt_texts=["ptt text"])
    o.recorder = SlowRec([np.zeros(1, np.int16), None])
    rf = asyncio.create_task(o.run_forever())
    await _settle()
    o.ptt_start()
    o._announce_queue.put_nowait(("Timer done", None))
    await _settle()
    assert "Timer done" not in o.tts.said          # held until the PTT turn ends
    o.ptt_end()
    await _settle()
    o.recorder.stop()
    await _settle()
    assert o.tts.said.index("Timer done") > o.tts.said.index("Sure.")
    await _cancel(rf)


async def test_barge_during_dictation_stops_recorder_no_overlap():
    """Regression: a wake-word barge during dictation must stop the
    in-flight dictation capture (I6) so its thread isn't orphaned, and the
    re-listen must not overlap it."""
    o, states = build_ptt(stt_texts=["dictate", "after"])
    o.recorder = SlowRec([np.zeros(1, np.int16), None, np.zeros(1, np.int16), None])
    t = asyncio.create_task(o.one_turn())
    await _settle()
    o.recorder.finish()                            # "dictate" heard -> dictation turn
    await _settle()
    assert o.state == "listening" and o.recorder.active == 1
    assert o.wake.waits == 1                       # barge listener up during dictation
    o.wake.q.put_nowait(True)                      # wake-word barge
    await _settle()
    assert o.recorder.stop_calls == 1              # dictation capture unblocked
    assert o.recorder.max_active == 1
    assert o.state == "listening"                  # re-listening after the barge
    assert o.recorder.hold_calls == [False, False, False]
    o.recorder.finish()
    await _settle()
    assert o.brain.asked == ["after"]
    o.recorder.stop()
    await t


async def test_ptt_during_dictation_stops_recorder_and_hold_captures():
    o, _ = build_ptt(stt_texts=["dictate", "ptt text"])
    o.recorder = SlowRec([np.zeros(1, np.int16), None, np.zeros(1, np.int16), None])
    t = asyncio.create_task(o.one_turn())
    await _settle()
    o.recorder.finish()
    await _settle()
    assert o.state == "listening" and o.recorder.active == 1
    o.ptt_start()
    await _settle()
    assert o.recorder.stop_calls == 1
    assert o.recorder.hold_calls == [False, False, True]
    assert o.recorder.max_active == 1
    o.ptt_end()
    await _settle()
    assert o.brain.asked == ["ptt text"]
    o.recorder.stop()
    await t


async def test_run_with_barge_returns_none_wake_or_ptt():
    o, _ = build()
    o.wake = EventWake()

    async def quick():
        return 1

    assert await o._run_with_barge(quick()) is None

    async def hang():
        await asyncio.Event().wait()

    async def interrupt():
        pass

    o.brain.interrupt = interrupt
    fut = asyncio.ensure_future(o._run_with_barge(hang()))
    await _settle()
    o.wake.q.put_nowait(True)
    assert await fut == "wake"

    o.wake = EventWake()
    fut = asyncio.ensure_future(o._run_with_barge(hang()))
    await _settle()
    o._ptt_event.set()
    assert await fut == "ptt"
    o._ptt_event.clear()


async def test_barge_listener_failure_still_lets_ptt_barge():
    o, _ = build()
    interrupts = []

    async def interrupt():
        interrupts.append(True)

    o.brain.interrupt = interrupt

    class FailingWake(EventWake):
        async def wait(self, threshold=None, suppress=None):
            raise RuntimeError("mic")

    o.wake = FailingWake()

    async def hang():
        await asyncio.Event().wait()

    fut = asyncio.ensure_future(o._run_with_barge(hang()))
    await _settle()
    assert not fut.done()                          # turn keeps running despite listener failure
    o._ptt_event.set()
    assert await fut == "ptt"
    assert interrupts == [True]
    o._ptt_event.clear()


# -- batch A: notes & dictation (A4) -------------------------------------------

from veronica.tools import pim as pim_tools_mod
from veronica.tools import mac as mac_tools_mod


async def test_note_intent_creates_note_and_says_noted(monkeypatch):
    calls = []

    async def fake_notes_create(args):
        calls.append(args)
        return {"content": [{"type": "text", "text": "Created note 'x'"}]}

    monkeypatch.setattr(pim_tools_mod.notes_create, "handler", fake_notes_create)
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["take a note: buy milk"])
    await o.one_turn()
    assert "Noted." in o.tts.said
    assert calls[0]["body"] == "buy milk"
    assert calls[0]["title"].startswith("buy milk")
    assert any(k == "tool" and p.get("decision") == "auto" for k, p in ev)
    assert o.brain.asked == []


async def test_note_intent_error_says_sorry(monkeypatch):
    async def fake_notes_create(args):
        return {"content": [{"type": "text", "text": "error: nope"}], "is_error": True}

    monkeypatch.setattr(pim_tools_mod.notes_create, "handler", fake_notes_create)
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["note that call mom"])
    await o.one_turn()
    assert "Sorry, I couldn't save that note." in o.tts.said


async def test_note_title_is_first_40_chars_plus_timestamp(monkeypatch):
    calls = []

    async def fake_notes_create(args):
        calls.append(args)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(pim_tools_mod.notes_create, "handler", fake_notes_create)
    long_body = "x" * 100
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=[f"note that {long_body}"])
    await o.one_turn()
    title = calls[0]["title"]
    assert title.startswith("x" * 40)
    assert "x" * 41 not in title.split(" — ")[0] + " "  # body portion capped at 40 chars


async def test_dictation_captures_until_stop_and_types(monkeypatch):
    typed = []

    def fake_dictate_type(text):
        typed.append(text)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(mac_tools_mod, "dictate_type", fake_dictate_type)
    o, _, ev = build3(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), np.zeros(1, np.int16), None],
        stt_texts=["dictate", "hello there", "how are you", "stop dictation"],
    )
    await o.one_turn()
    assert "Go ahead." in o.tts.said
    # M12: committed per utterance (so a barge mid-dictation keeps what was
    # already said), later chunks space-separated from the previous one
    assert typed == ["hello there", " how are you"]
    assert "Done." in o.tts.said
    assert o.brain.asked == []


async def test_dictation_ends_on_silence_without_stop_phrase(monkeypatch):
    typed = []
    monkeypatch.setattr(
        mac_tools_mod, "dictate_type",
        lambda text: (typed.append(text), {"content": [{"type": "text", "text": "ok"}]})[1],
    )
    o, _ = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), None],
        stt_texts=["dictate", "just one line"],
    )
    await o.one_turn()
    assert typed == ["just one line"]
    assert "Done." in o.tts.said


async def test_dictation_nothing_said_speaks_fallback():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["start dictation"])
    await o.one_turn()
    assert "I didn't catch anything." in o.tts.said


async def test_dictation_type_error_speaks_fallback(monkeypatch):
    monkeypatch.setattr(
        mac_tools_mod, "dictate_type",
        lambda text: {"content": [{"type": "text", "text": "error: no access"}], "is_error": True},
    )
    o, _ = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), None],
        stt_texts=["dictate", "hello"],
    )
    await o.one_turn()
    assert "Sorry, I couldn't type that." in o.tts.said


async def test_dictation_skips_go_ahead_echo_on_first_capture_only(monkeypatch):
    monkeypatch.setattr(mac_tools_mod, "dictate_type", lambda text: {"content": [{"type": "text", "text": "ok"}]})
    o, _ = build(stt_texts=["dictate", "one", "two"])
    o.recorder = RecArgs([np.zeros(1, np.int16), np.zeros(1, np.int16), np.zeros(1, np.int16), None])
    await o.one_turn()
    dictation_calls = o.recorder.calls[1:4]
    assert dictation_calls[0]["skip_ms"] == o.s.followup_skip_ms
    assert dictation_calls[1]["skip_ms"] == 0
    assert dictation_calls[2]["skip_ms"] == 0


async def test_dictation_strips_trailing_stop_phrase_from_last_utterance(monkeypatch):
    typed = []
    monkeypatch.setattr(
        mac_tools_mod, "dictate_type",
        lambda text: (typed.append(text), {"content": [{"type": "text", "text": "ok"}]})[1],
    )
    o, _ = build(
        rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), np.zeros(1, np.int16), None],
        stt_texts=["dictate", "hello there", "see you soon, stop dictation."],
    )
    await o.one_turn()
    assert typed == ["hello there", " see you soon"]
    assert "Done." in o.tts.said
    # initial listen + 2 dictation captures + the follow-up window: no third
    # dictation capture after the trailing stop phrase
    assert len(o.recorder.hold_calls) == 4


@pytest.mark.parametrize("text, expected", [
    ("hello there stop dictation", ("hello there", True)),
    ("hello there, stop dictating!", ("hello there", True)),
    ("hello. End dictation", ("hello", True)),
    ("stop dictation", ("", True)),
    ("hello there", ("hello there", False)),
    ("please stop dictation software crashes", ("please stop dictation software crashes", False)),
])
def test_strip_trailing_stop_dictation(text, expected):
    assert Orchestrator._strip_trailing_stop_dictation(text) == expected


async def test_dictation_typing_error_mid_way_keeps_earlier_text(monkeypatch):
    typed = []

    def fake_type(text):
        typed.append(text)
        if len(typed) == 2:
            return {"content": [{"type": "text", "text": "error: no access"}], "is_error": True}
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(mac_tools_mod, "dictate_type", fake_type)
    o, _ = build(
        rec_pcms=[np.zeros(1, np.int16)] * 3 + [None],
        stt_texts=["dictate", "first", "second", "third"],
    )
    await o.one_turn()
    assert typed == ["first", " second"]           # stopped at the failure
    assert "Sorry, I couldn't type that." in o.tts.said


async def test_orchestrator_capture_arms_recorder_synchronously():
    """Orchestrator._capture() must arm() the recorder before scheduling
    capture(), in the same iteration, so a ptt_end()/stop() landing before
    the coroutine's first step is honored."""
    o, _ = build()
    events = []

    class ArmRec(Rec):
        def arm(self, hold=False):
            events.append(("arm", hold))

        async def capture(self, **kw):
            events.append(("capture", kw.get("hold", False)))
            return None

    o.recorder = ArmRec([])

    async def run():
        # _capture() is awaited inline (as _listen_after_ptt does), so its
        # body runs synchronously up to its first await; the arm must
        # already have happened by the time control first yields.
        fut = asyncio.ensure_future(o._capture(max_s=3, hold=True))
        await asyncio.sleep(0)           # one step: _capture arms + schedules capture()
        assert events == [("arm", True)]  # capture() body has NOT stepped yet
        await fut

    await run()
    assert events == [("arm", True), ("capture", True)]


async def test_ptt_quick_tap_with_real_recorder_ends_immediately(monkeypatch):
    """End to end with the real Recorder: press, then release in the same
    iteration the hold capture is scheduled -> the capture ends at once
    instead of running for ptt_max_s."""
    import threading
    from veronica.audio.record import Recorder
    from test_record import FakeVad, frames

    monkeypatch.setattr(Recorder, "_vad_cls", FakeVad)
    gate = threading.Event()

    def blocking_frames():
        gate.wait(5)
        yield from frames("." * 2000)

    o, states = build()
    o.ready = True
    o.recorder = Recorder(Settings(ptt_max_s=30, max_utterance_s=60), frames=blocking_frames)
    o.ptt_start()
    t = asyncio.ensure_future(o._listen_after_ptt())
    await asyncio.sleep(0)           # _listen_after_ptt runs up to the shielded await: capture armed
    assert o._ptt_capturing
    o.ptt_end()                      # capture() body may not have stepped yet: must still be honored
    gate.set()
    pcm = await asyncio.wait_for(t, timeout=2)
    assert pcm is None
    assert not o._ptt_capturing and not o._ptt_held


async def test_listen_after_ptt_resets_held_flag_when_capture_ends_on_cap():
    o, _ = build_ptt()
    o.recorder = SlowRec([None])
    o.ptt_start()
    t = asyncio.ensure_future(o._listen_after_ptt())
    await _settle()
    o.recorder.finish()              # simulate the capture ending on ptt_max_s with the key still down
    await t
    assert not o._ptt_held           # so the eventual key-up is a no-op and the next press is fresh
    o.ptt_end()
    assert o.recorder.finish_calls == 1


async def test_barge_teardown_logs_turn_exception(caplog):
    o, _ = build()

    async def interrupt():
        pass

    o.brain.interrupt = interrupt

    async def boom():
        raise RuntimeError("kaboom")

    turn = asyncio.ensure_future(boom())
    await asyncio.sleep(0)
    with caplog.at_level("ERROR", logger="veronica.orchestrator"):
        await o._barge_teardown(turn)
    assert any("torn down" in r.message for r in caplog.records)


# -- batch B: voice / speed --------------------------------------------------

from veronica import prefs as prefs_mod


class TTS2(TTS):
    def __init__(self):
        super().__init__()
        self.voice = "af_sarah"
        self.speed = 1.0
        self.spoken_with = []   # (text, voice, speed) at synth time

    async def asynth(self, text, lang=None):
        self.spoken_with.append((text, self.voice, self.speed))
        return await super().asynth(text)


def build_voice(stt_texts, monkeypatch):
    saved = []
    monkeypatch.setattr(prefs_mod, "save", lambda d: saved.append(d))
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(stt_texts),
        brain=Brain(), tts=TTS2(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)),
    )
    return o, saved, events


async def test_voice_intent_switches_voice_and_saves(monkeypatch):
    o, saved, ev = build_voice(["use a british male voice"], monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "bm_george"
    assert o.tts.spoken_with[-1] == ("Okay, this is George.", "bm_george", 1.0)
    assert {"tts_voice": "bm_george"} in saved
    assert ("tool", {"summary": "Voice: George", "decision": "auto"}) in ev
    assert o.brain.asked == []


async def test_voice_intent_unknown_lists_voices(monkeypatch):
    o, saved, _ = build_voice(["use a robot voice"], monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "af_sarah"
    assert saved == []
    assert o.tts.said[-1].startswith("I don't have that voice. I have Sarah, Bella")
    assert o.tts.said[-1].endswith("George, Lewis, and in Hindi Alpha, Beta, Omega and Psi.")


async def test_voice_intent_next_cycles(monkeypatch):
    o, saved, _ = build_voice(["change your voice"], monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "af_bella"
    assert {"tts_voice": "af_bella"} in saved


async def test_speed_faster_and_clamp(monkeypatch):
    o, saved, _ = build_voice(["speak faster"], monkeypatch)
    await o.one_turn()
    assert o.tts.speed == pytest.approx(1.15)
    assert o.tts.spoken_with[-1][0] == "Like this?"
    assert any(abs(d.get("tts_speed", 0) - 1.15) < 1e-9 for d in saved)

    o.tts.speed = 1.5
    o.stt = STT(["speak faster"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.speed == 1.5
    assert o.tts.said[-1] == "That's as fast as I go."


async def test_speed_slower_normal(monkeypatch):
    o, saved, _ = build_voice(["slow down"], monkeypatch)
    await o.one_turn()
    assert o.tts.speed == pytest.approx(0.85)
    o.stt = STT(["normal speed"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.speed == 1.0
    assert o.tts.said[-1] == "Like this?"
    o.stt = STT(["normal speed"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.said[-1] == "Already at normal speed."


# -- batch B: proactive -------------------------------------------------------

from veronica import proactive as pr_mod


class FakeProactive:
    def __init__(self):
        self.schedule = pr_mod.Schedule()
        self.started = 0
    async def start(self): self.started += 1
    def stop(self): pass
    async def build_briefing(self): return "Good morning, Manik. Nothing on your calendar today."


class _WakeBlocks(Wake):
    """Never detects: blocks until cancelled, so run_forever parks in its
    idle wait instead of spinning."""
    async def wait(self, threshold=None, suppress=None):
        await asyncio.Event().wait()


def build_pro(stt_texts, monkeypatch, wake=None):
    saved = []
    monkeypatch.setattr(pr_mod, "save_schedule", lambda s, save=None: saved.append(s.to_prefs()))
    states, events = [], []
    p = FakeProactive()
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=wake or Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)), proactive=p,
    )
    return o, p, saved, events


async def test_brief_now_speaks_briefing(monkeypatch):
    o, _, saved, ev = build_pro(["brief me"], monkeypatch)
    await o.one_turn()
    assert o.tts.said == ["Good morning, Manik. Nothing on your calendar today."]
    assert ("tool", {"summary": "Briefing", "decision": "auto"}) in ev
    assert o.brain.asked == [] and saved == []


async def test_brief_now_thinks_before_speaking(monkeypatch):
    o, _, _, _ = build_pro(["brief me"], monkeypatch)
    states = []
    o._on_state = states.append
    await o.one_turn()
    assert "thinking" in states and "speaking" in states
    assert states.index("thinking") < states.index("speaking")
    assert o.tts.said == ["Good morning, Manik. Nothing on your calendar today."]


async def test_brief_now_can_be_barged(monkeypatch):
    """A long briefing runs under _run_with_barge: the wake word cuts it
    off and Veronica re-listens instead of finishing the briefing."""
    class SlowProactive(FakeProactive):
        async def build_briefing(self):
            await asyncio.Event().wait()
            return "never"

    class BargeWake(Wake):
        def __init__(self): self.barges = 0
        async def wait(self, threshold=None, suppress=None):
            if threshold is not None:
                self.barges += 1
                return True
            return True

    class InterruptibleBrain(Brain):
        def __init__(self): super().__init__(); self.interrupts = 0
        async def interrupt(self): self.interrupts += 1

    o, _, _, _ = build_pro(["brief me", "tell me a joke"], monkeypatch, wake=BargeWake())
    o.proactive = SlowProactive()
    o.brain = InterruptibleBrain()
    o.recorder = Rec([np.zeros(1, np.int16), np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.wake.barges >= 1
    assert "never" not in o.tts.said
    assert o.brain.asked == ["tell me a joke"]


async def test_time_spoken_12h():
    ts = Orchestrator._time_spoken
    assert ts("18:00") == "6 pm"
    assert ts("07:30") == "7:30 am"
    assert ts("12:00") == "12 pm"
    assert ts("00:15") == "12:15 am"
    assert ts("12:45") == "12:45 pm"


async def test_briefing_on_with_time_saves_and_confirms(monkeypatch):
    o, p, saved, ev = build_pro(["give me a briefing every morning at 7:30 am"], monkeypatch)
    await o.one_turn()
    assert p.schedule.briefing_enabled and p.schedule.briefing_time == "07:30"
    assert saved[-1]["briefing_time"] == "07:30"
    assert o.tts.said[-1] == "Okay, I'll brief you every day at 7:30 am."
    assert ("tool", {"summary": "Update briefing schedule", "decision": "auto"}) in ev
    assert o.brain.asked == []


async def test_briefing_on_without_time_keeps_stored(monkeypatch):
    o, p, _, _ = build_pro(["turn on the morning briefing"], monkeypatch)
    p.schedule.briefing_time = "09:15"
    await o.one_turn()
    assert p.schedule.briefing_enabled and p.schedule.briefing_time == "09:15"
    assert o.tts.said[-1] == "Okay, I'll brief you every day at 9:15 am."


async def test_briefing_off_nudges_on_off(monkeypatch):
    o, p, saved, _ = build_pro(["stop the morning briefing"], monkeypatch)
    p.schedule.briefing_enabled = True
    await o.one_turn()
    assert not p.schedule.briefing_enabled and o.tts.said[-1] == "Okay, no more morning briefings."

    o.stt = STT(["warn me 10 minutes before my meetings"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert p.schedule.nudges_enabled and p.schedule.nudge_minutes == 10
    assert o.tts.said[-1] == "Okay, I'll warn you 10 minutes before each event."

    o.stt = STT(["turn off nudges"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert not p.schedule.nudges_enabled and o.tts.said[-1] == "Okay, no more meeting nudges."
    assert len(saved) == 3


async def test_nudges_on_out_of_range_minutes_keeps_stored(monkeypatch):
    o, p, _, _ = build_pro(["warn me 90 minutes before my meetings"], monkeypatch)
    p.schedule.nudge_minutes = 7
    await o.one_turn()
    assert p.schedule.nudges_enabled and p.schedule.nudge_minutes == 7
    assert o.tts.said[-1] == "Okay, I'll warn you 7 minutes before each event."


async def test_proactive_intent_without_proactive_says_unavailable(monkeypatch):
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["brief me"])
    await o.one_turn()
    assert o.tts.said[-1] == "Briefings aren't available right now."
    assert o.brain.asked == []


async def test_nudges_on_default_minutes(monkeypatch):
    o, p, saved, _ = build_pro(["turn on nudges"], monkeypatch)
    await o.one_turn()
    assert p.schedule.nudges_enabled and p.schedule.nudge_minutes == 5
    assert o.tts.said == ["Okay, I'll warn you 5 minutes before each event."]
    assert o.brain.asked == [] and saved == [p.schedule.to_prefs()]


async def test_run_forever_starts_proactive(monkeypatch):
    o, p, _, _ = build_pro([], monkeypatch, wake=_WakeBlocks())
    task = asyncio.ensure_future(o.run_forever())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert p.started == 1


async def test_run_forever_without_proactive_is_fine():
    o, _ = build()
    o.wake = _WakeBlocks()
    task = asyncio.ensure_future(o.run_forever())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert o.proactive is None


# -- batch C: quick replies ----------------------------------------------------

async def test_quick_time_reply_skips_brain_and_logs():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what day is it"])
    o.store = FakeStore()
    o.s = Settings(followup_window_s=0, confirm_listen_s=0, memory_enabled=True)
    await o.one_turn()
    import datetime as _dt
    assert o.tts.said == [f"It's {_dt.datetime.now().strftime('%A')}."]
    assert o.brain.asked == []
    assert ("tool", {"summary": "Quick reply", "decision": "auto"}) in ev
    assert o.store.turns[-1] == ("what day is it", o.tts.said[0])


async def test_quick_battery_reads_pmset(monkeypatch):
    monkeypatch.setattr(mac_tools_mod, "read_battery", lambda: (72, "charging"))
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["battery level"])
    await o.one_turn()
    assert o.tts.said == ["Battery is at 72 percent and charging."]


async def test_quick_battery_failure_copy(monkeypatch):
    monkeypatch.setattr(mac_tools_mod, "read_battery", lambda: (None, None))
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["battery kitni hai"])
    await o.one_turn()
    assert o.tts.said == ["Battery level नहीं मिल पाया।"]


async def test_quick_battery_unknown_state_reports_percent_only(monkeypatch):
    monkeypatch.setattr(mac_tools_mod, "read_battery", lambda: (98, None))
    o, *_ = build_lang(["battery level"], langs=["en"], mode="en")
    await o.one_turn()
    assert o.tts.said[-1] == "Battery is at 98 percent."
    o, *_ = build_lang(["battery kitni hai"], langs=["en"], mode="en")
    await o.one_turn()
    assert o.tts.said[-1] == "Battery 98 percent है।"


async def test_quick_volume_uses_mac_tool(monkeypatch):
    async def fake_get(args):
        return {"content": [{"type": "text", "text": "40"}]}
    monkeypatch.setattr(mac_tools_mod.volume_get, "handler", fake_get)
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what's the volume"])
    await o.one_turn()
    assert o.tts.said == ["Volume is at 40 percent."]


async def test_quick_volume_error_copy(monkeypatch):
    async def fake_get(args):
        return {"content": [{"type": "text", "text": "error: boom"}], "is_error": True}
    monkeypatch.setattr(mac_tools_mod.volume_get, "handler", fake_get)
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["volume kitna hai"])
    await o.one_turn()
    assert o.tts.said == ["Volume नहीं मिल पाया।"]


async def test_quick_uses_utterance_lang_for_english_phrase():
    # whisper labelled the utterance Hindi (an English phrase said in a
    # Hindi sentence's flow): the reply comes back in Hinglish.
    o, *_ = build_lang(["what day is it"], langs=["hi"], mode="auto")
    await o.one_turn()
    assert o._utterance_lang == "hi"
    import datetime as _dt
    assert o.tts.said == [f"आज {_dt.datetime.now().strftime('%A')} है।"]


async def test_quick_does_not_shadow_local_intents_or_brain():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hello can you open safari"])
    await o.one_turn()
    assert o.brain.asked == ["hello can you open safari"]


async def test_quick_does_not_run_when_local_intent_matched():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["remember that time is 5"])
    o.store = FakeStore()
    await o.one_turn()
    assert o.store.facts == ["time is 5"]
    assert o.tts.said == ["Got it."]


def test_read_battery_parses_pmset():
    class R:
        def __init__(self, out):
            self.stdout = out
            self.returncode = 0
    run = lambda *a, **k: R("Now drawing from 'AC Power'\n -InternalBattery-0 (id=123)\t72%; charging; 0:45 remaining present: true\n")
    assert mac_tools_mod.read_battery(run=run) == (72, "charging")
    run = lambda *a, **k: R(" -InternalBattery-0\t100%; charged; 0:00 remaining\n")
    assert mac_tools_mod.read_battery(run=run) == (100, "charged")
    run = lambda *a, **k: R(" -InternalBattery-0\t35%; discharging; 3:10 remaining\n")
    assert mac_tools_mod.read_battery(run=run) == (35, "discharging")
    # plugged in but not charging (battery-health hold): the state is unknown, not "charging"
    run = lambda *a, **k: R(" -InternalBattery-0\t98%; AC attached; not charging present: true\n")
    assert mac_tools_mod.read_battery(run=run) == (98, None)
    run = lambda *a, **k: R(" -InternalBattery-0\t80%; finishing charge; 0:05 remaining present: true\n")
    assert mac_tools_mod.read_battery(run=run) == (80, "charging")
    run = lambda *a, **k: R("Now drawing from 'AC Power'\n")
    assert mac_tools_mod.read_battery(run=run) == (None, None)
    run = lambda *a, **k: (_ for _ in ()).throw(OSError("no pmset"))
    assert mac_tools_mod.read_battery(run=run) == (None, None)


@pytest.mark.parametrize("heard,ok", [
    ("haan", True), ("haanji", True), ("ji haan", True), ("theek hai", True), ("karo", True), ("ji", True),
    # transcribed laughter must never approve a tool
    ("ha", False), ("ha ha", False),
    ("nahi", False), ("nahin", False), ("mat karo", False), ("rehne do", False), ("haan nahi", False),
    # Devanagari (pinned hi mode makes whisper emit the script)
    ("हाँ", True), ("हां", True), ("जी", True), ("जी हाँ", True), ("ठीक है", True), ("करो", True),
    ("हाँ, करो।", True),
    ("नहीं", False), ("नही", False), ("मत करो", False), ("रहने दो", False), ("हाँ नहीं", False),
    # trailing danda must not glue onto the word
    ("हाँ।", True), ("ठीक है।", True), ("नहीं।", False), ("हाँ, नहीं।", False), ("करो॥", True),
])
def test_is_confirmation_hinglish(heard, ok):
    assert Orchestrator.is_confirmation(heard) is ok


# -- batch C: language mode -----------------------------------------------------

class STT2(STT):
    def __init__(self, texts, langs=None):
        super().__init__(texts); self.langs = list(langs or []); self.language = "en"; self.model_name = "small.en"
    async def atranscribe_detailed(self, pcm):
        t = await self.atranscribe(pcm)
        return t, (self.langs.pop(0) if self.langs else "en")
    def set_language(self, lang): self.language = lang


class TTS3(TTS):
    def __init__(self):
        super().__init__(); self.voice = "af_sarah"; self.speed = 1.0; self.hindi_voice = "hf_alpha"; self.langs = []
    async def asynth(self, text, lang=None):
        self.langs.append((text, lang)); return await super().asynth(text)


def build_lang(stt_texts, langs=(), mode="en", monkeypatch=None):
    saved = []
    if monkeypatch:
        monkeypatch.setattr(prefs_mod, "save", lambda d: saved.append(d))
    made = []
    def factory(model, language):
        s = STT2([], []); s.model_name = model; s.language = language; made.append((model, language)); return s
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT2(stt_texts, langs),
        brain=Brain(), tts=TTS3(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)), stt_factory=factory, language=mode,
    )
    return o, saved, made, events


async def test_utterance_lang_from_detection_and_script():
    o, *_ = build_lang(["kal meeting hai"], langs=["hi"], mode="auto")
    await o.one_turn()
    assert o._utterance_lang == "hi"
    assert o.tts.langs[-1][1] == "hi"                      # brain reply spoken with the Hindi voice
    o, *_ = build_lang(["कल मीटिंग है"], langs=["en"], mode="en")   # Devanagari wins even if detector says en
    await o.one_turn()
    assert o._utterance_lang == "hi"
    o, *_ = build_lang(["what time is it"], langs=["en"], mode="auto")
    await o.one_turn()
    assert o._utterance_lang == "en" and o.tts.langs[-1][1] == "en"


async def test_hinglish_phrase_in_auto_mode_counts_as_hindi():
    o, *_ = build_lang(["shukriya"], langs=["en"], mode="auto")
    await o.one_turn()
    assert o.tts.said[-1] in {"कोई बात नहीं।", "हमेशा।"} and o.tts.langs[-1][1] == "hi"


async def test_hinglish_phrase_in_english_mode_stays_english():
    o, *_ = build_lang(["shukriya"], langs=["en"], mode="en")
    await o.one_turn()
    assert o._utterance_lang == "en"
    # ...but the Hindi reply to a Hinglish phrase is still voiced in Hindi
    assert o.tts.said[-1] in {"कोई बात नहीं।", "हमेशा।"} and o.tts.langs[-1][1] == "hi"


async def test_english_quick_reply_in_english_mode_is_voiced_in_english():
    o, *_ = build_lang(["thanks"], langs=["en"], mode="en")
    await o.one_turn()
    assert o.tts.langs[-1][1] == "en"


async def test_language_switch_turn_swaps_models_and_saves(monkeypatch):
    o, saved, made, ev = build_lang(["speak hindi"], mode="en", monkeypatch=monkeypatch)
    await o.one_turn()
    assert o.language == "hi"
    assert made[-2:] == [("small", "hi"), ("tiny", "hi")]     # main + partial
    assert o.stt.model_name == "small" and o.partial_stt.model_name == "tiny"
    assert {"language": "hi"} in saved
    assert o.tts.said[:2] == ["एक मिनट, हिंदी load कर रही हूँ।", "अब हिंदी में बात करते हैं।"]
    assert o.tts.langs[:2] == [("एक मिनट, हिंदी load कर रही हूँ।", "hi"), ("अब हिंदी में बात करते हैं।", "hi")]
    assert ("tool", {"summary": "Language: hi", "decision": "auto"}) in ev
    assert o.brain.asked == []

    o.stt = STT2(["speak english"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.language == "en" and made[-2:] == [("small.en", "en"), ("tiny.en", "en")]
    assert o.tts.said[-2:] == ["One moment, switching to English.", "Okay, English it is."]
    assert o.tts.langs[-1] == ("Okay, English it is.", "en")

    o.stt = STT2(["dono bhasha"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.language == "auto" and made[-2:] == [("small", None), ("tiny", None)]
    assert o.tts.said[-1] == "ठीक है, दोनों चलेगा।"


async def test_language_switch_same_model_only_sets_language(monkeypatch):
    o, saved, made, _ = build_lang(["speak hindi"], mode="auto", monkeypatch=monkeypatch)
    o.stt.model_name = "small"; o.partial_stt = STT2([]); o.partial_stt.model_name = "tiny"
    await o.one_turn()
    assert made == [] and o.stt.language == "hi" and o.partial_stt.language == "hi"
    assert o.tts.said == ["अब हिंदी में बात करते हैं।"]
    assert {"language": "hi"} in saved


async def test_language_switch_already_active_just_confirms(monkeypatch):
    o, saved, made, _ = build_lang(["speak hindi"], mode="hi", monkeypatch=monkeypatch)
    o.stt.model_name = "small"; o.stt.language = "hi"
    o.partial_stt = STT2([]); o.partial_stt.model_name = "tiny"; o.partial_stt.language = "hi"
    await o.one_turn()
    assert made == [] and o.language == "hi"
    assert o.tts.said == ["अब हिंदी में बात करते हैं।"]


async def test_language_switch_without_factory_only_sets_language(monkeypatch):
    saved = []
    monkeypatch.setattr(prefs_mod, "save", lambda d: saved.append(d))
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT2(["speak hindi"]),
        brain=Brain(), tts=TTS3(), player=Player(),
    )
    await o.one_turn()
    assert o.language == "hi" and o.stt.language == "hi" and o.stt.model_name == "small.en"
    assert o.tts.said == ["अब हिंदी में बात करते हैं।"]


async def test_stt_without_detailed_api_defaults_to_english():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["tell me a joke"])
    await o.one_turn()
    assert o._utterance_lang == "en"
    assert o.brain.asked == ["tell me a joke"]


async def test_announcements_speak_without_forced_lang():
    o, *_ = build_lang([], mode="hi")
    await o._deliver_announcement(("Timer done.", None))
    assert o.tts.langs[-1] == ("Timer done.", None)


async def test_hindi_voice_request_sets_hindi_voice(monkeypatch):
    o, saved, _, ev = build_lang(["use a hindi voice"], mode="en", monkeypatch=monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "af_sarah" and o.tts.hindi_voice == "hf_alpha"
    assert {"tts_hindi_voice": "hf_alpha"} in saved
    assert ("ठीक है, अब मैं ऐसे बोलूँगी।", "hi") in o.tts.langs

    o.stt = STT2(["use the omega voice"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.hindi_voice == "hm_omega" and o.tts.voice == "af_sarah"
    assert {"tts_hindi_voice": "hm_omega"} in saved
    assert ("tool", {"summary": "Voice: Omega", "decision": "auto"}) in ev


async def test_language_switch_load_failure_leaves_everything_untouched(monkeypatch):
    o, saved, made, _ = build_lang(["speak hindi"], mode="en", monkeypatch=monkeypatch)
    original_stt = o.stt
    calls = []
    def factory(model, language):
        calls.append((model, language))
        if len(calls) == 2:
            raise RuntimeError("offline")   # main loaded, partial download failed
        s = STT2([]); s.model_name = model; s.language = language; return s
    o.stt_factory = factory
    await o.one_turn()
    assert calls == [("small", "hi"), ("tiny", "hi")]
    assert o.stt is original_stt and o.stt.model_name == "small.en" and o.stt.language == "en"
    assert o.partial_stt is None
    assert o.language == "en" and saved == []
    assert o.tts.said == ["एक मिनट, हिंदी load कर रही हूँ।", "हिंदी load नहीं हो पाई, बाद में try करो।"]
    assert o.tts.langs[-1][1] == "hi"
    assert o.brain.asked == []


async def test_language_switch_load_failure_english_line(monkeypatch):
    o, saved, made, _ = build_lang(["speak english"], mode="hi", monkeypatch=monkeypatch)
    o.stt.model_name = "small"
    def factory(model, language):
        raise RuntimeError("disk full")
    o.stt_factory = factory
    await o.one_turn()
    assert o.language == "hi" and saved == [] and o.stt.model_name == "small"
    assert o.tts.said[-1] == "Couldn't switch language, check the log."


async def test_hindi_voice_pick_in_english_mode_switches_to_auto(monkeypatch):
    o, saved, made, _ = build_lang(["use a hindi voice"], mode="en", monkeypatch=monkeypatch)
    await o.one_turn()
    assert o.tts.hindi_voice == "hf_alpha"
    assert o.language == "auto"
    assert made[-2:] == [("small", None), ("tiny", None)]
    assert {"language": "auto"} in saved


async def test_hindi_voice_pick_in_auto_mode_keeps_mode(monkeypatch):
    o, saved, made, _ = build_lang(["use a hindi male voice"], mode="auto", monkeypatch=monkeypatch)
    o.stt.model_name = "small"; o.partial_stt = STT2([]); o.partial_stt.model_name = "tiny"
    await o.one_turn()
    assert o.tts.hindi_voice == "hm_omega" and o.language == "auto" and made == []


@pytest.mark.parametrize("heard,ok", [
    # last decisive phrase wins
    ("no no, I said yes, do it", True), ("yes… actually no", False), ("not now", False),
    ("yes", True), ("no", False), ("", False), ("maybe", False),
    # negated confirms
    ("not okay", False), ("don't do it", False), ("mat karo", False), ("that's not fine", False),
    # new affirmatives
    ("ok", True), ("okay", True), ("yup", True), ("yeah yeah", True), ("alright", True), ("fine", True),
    ("absolutely", True), ("please do", True), ("go for it", True), ("of course", True), ("correct", True),
    ("bilkul", True), ("haan haan", True), ("kar do", True), ("ठीक", True), ("बिल्कुल", True), ("कर दो", True),
    # still never on laughter / stop
    ("ha ha", False), ("stop", False), ("okay stop", False), ("cancel that, yes", True),
])
def test_is_confirmation_last_decisive_wins(heard, ok):
    assert Orchestrator.is_confirmation(heard) is ok


# -- Batch D: settings / version / update turns ---------------------------------

from veronica import version as version_mod
from veronica.updater import UpdateStatus


def _status(kind):
    return UpdateStatus(available=kind != "none", kind=kind, detail=f"{kind} detail",
                        running_sha="aaa", head_sha="bbb", remote_sha=None)


def build_d(stt_texts, langs=(), mode="en", **kw):
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT2(stt_texts, langs),
        brain=Brain(), tts=TTS3(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)), language=mode, **kw,
    )
    return o, states, events


async def test_settings_intent_emits_event_and_skips_brain():
    o, _, ev = build_d(["open settings"])
    await o.one_turn()
    assert ("settings", {"open": True, "tab": "general"}) in ev
    assert o.tts.said == ["Here you go."]
    assert o.tts.langs[-1] == ("Here you go.", None)
    assert o.brain.asked == []
    assert o.player.resets >= 1


async def test_history_intent_opens_history_tab():
    o, _, ev = build_d(["show my history"])
    await o.one_turn()
    assert ("settings", {"open": True, "tab": "history"}) in ev
    assert o.tts.said == ["Here you go."]


async def test_settings_intent_in_hindi_replies_in_hindi():
    o, _, ev = build_d(["settings kholo"], langs=["en"], mode="auto")
    await o.one_turn()
    assert ("settings", {"open": True, "tab": "general"}) in ev
    assert o.tts.langs[-1] == ("यह लीजिए।", "hi")


async def test_version_intent_speaks_describe(monkeypatch):
    monkeypatch.setattr(version_mod, "describe", lambda info=None: "Veronica 9.9.9 (abc1234, 1 Jan)")
    o, _, _ = build_d(["what version are you"])
    await o.one_turn()
    assert o.tts.said == ["Veronica 9.9.9 (abc1234, 1 Jan)"]
    assert o.brain.asked == []


async def test_version_intent_uses_injected_describe_off_loop(monkeypatch):
    import threading

    calls = []

    def boom(info=None):
        raise AssertionError("module describe must not be used when one is injected")

    monkeypatch.setattr(version_mod, "describe", boom)
    o, _, _ = build_d(["version"], version_describe=lambda: calls.append(threading.current_thread()) or "Veronica 1.2.3 (cafe123, 2 Feb)")
    await o.one_turn()
    assert o.tts.said == ["Veronica 1.2.3 (cafe123, 2 Feb)"]
    assert len(calls) == 1


async def test_version_intent_default_describe_runs_in_thread(monkeypatch):
    import threading

    main = threading.current_thread()
    seen = []

    def describe(info=None):
        seen.append(threading.current_thread())
        return "Veronica 9.9.9 (abc1234, 1 Jan)"

    monkeypatch.setattr(version_mod, "describe", describe)
    o, _, _ = build_d(["what version are you"])
    await o.one_turn()
    assert o.tts.said == ["Veronica 9.9.9 (abc1234, 1 Jan)"]
    assert seen and seen[0] is not main   # git runs off the event loop


async def test_update_intent_unavailable_in_text_mode():
    o, _, _ = build_d(["update yourself"])
    await o.one_turn()
    assert o.tts.said == ["Updates aren't available in this mode."]
    assert o.brain.asked == []


async def test_update_intent_already_latest():
    calls = []
    o, _, ev = build_d(
        ["check for updates"], updater_check=lambda: _status("none"),
        updater_update=lambda st: calls.append(st) or "log", relaunch=lambda: calls.append("relaunch") or True,
    )
    await o.one_turn()
    assert o.tts.said == ["You're already on the latest."]
    assert calls == []
    assert not any(k == "tool" for k, _ in ev)


@pytest.mark.parametrize("kind", ["remote", "local"])
async def test_update_intent_updates_and_relaunches(kind):
    calls = []
    st = _status(kind)
    o, _, ev = build_d(
        ["update yourself"], updater_check=lambda: st,
        updater_update=lambda s: calls.append(("update", s)) or "log",
        relaunch=lambda: calls.append(("relaunch",)) or True,
    )
    await o.one_turn()
    assert o.tts.said == ["Updating, back in a moment."]
    assert ("tool", {"summary": "Update Veronica", "decision": "auto"}) in ev
    assert calls == [("update", st), ("relaunch",)]


async def test_update_intent_failure_speaks_and_does_not_relaunch(caplog):
    calls = []

    def boom(_st):
        raise RuntimeError("git pull exploded")

    o, _, _ = build_d(
        ["update now"], updater_check=lambda: _status("remote"), updater_update=boom,
        relaunch=lambda: calls.append("relaunch") or True,
    )
    with caplog.at_level("ERROR", logger="veronica.orchestrator"):
        await o.one_turn()
    assert o.tts.said == ["Updating, back in a moment.", "The update failed, check the log."]
    assert calls == []
    assert "git pull exploded" in caplog.text


async def test_update_intent_check_failure_speaks(caplog):
    def boom():
        raise RuntimeError("no network")

    o, _, _ = build_d(["update yourself"], updater_check=boom, updater_update=lambda s: "", relaunch=lambda: True)
    with caplog.at_level("WARNING", logger="veronica.orchestrator"):
        await o.one_turn()
    assert o.tts.said == ["Couldn't check for updates, check the log."]


async def test_update_phrase_inside_longer_request_goes_to_brain():
    # "update my calendar" is not the update intent: it goes to the brain
    o, _, _ = build_d(["update my calendar"])
    await o.one_turn()
    assert o.brain.asked == ["update my calendar"]


async def test_update_intent_already_running_speaks():
    from veronica.updater import UpdateInProgress

    def busy(_st):
        raise UpdateInProgress("Updating already.")

    calls = []
    o, _, ev = build_d(["update yourself"], updater_check=lambda: _status("remote"), updater_update=busy,
                       relaunch=lambda: calls.append("relaunch") or True)
    await o.one_turn()
    assert o.tts.said == ["An update is already running."]
    assert calls == []
    assert not any(k == "tool" for k, _ in ev)


async def test_update_intent_is_not_cancelled_by_barge():
    """The pull/build must never be orphaned by a wake-word barge: the update
    turn runs outside the barge race, so a wake during it changes nothing
    and the relaunch still happens."""
    class BargingWake(Wake):
        def __init__(self): self.waits = 0
        async def wait(self, threshold=None, suppress=None):
            self.waits += 1
            return True                          # would barge immediately

    class Brain2(Brain):
        def __init__(self): super().__init__(); self.interrupts = 0
        async def interrupt(self): self.interrupts += 1

    calls = []

    def slow_update(st):
        import time
        time.sleep(0.05)
        calls.append(("update", st))
        return "log"

    o, _, _ = build_d(["update yourself"], updater_check=lambda: _status("remote"), updater_update=slow_update,
                      relaunch=lambda: calls.append(("relaunch",)) or True)
    o.wake = BargingWake(); o.brain = Brain2()
    await o.one_turn()
    assert [c[0] for c in calls] == ["update", "relaunch"]
    assert o.brain.interrupts == 0
    assert o.wake.waits == 0                     # no barge listener ran during the update


async def test_update_intent_relaunch_not_scheduled_speaks_restart_hint():
    o, _, _ = build_d(["update yourself"], updater_check=lambda: _status("local"), updater_update=lambda s: "log",
                      relaunch=lambda: False)
    await o.one_turn()
    assert o.tts.said == ["Updating, back in a moment.", "Update installed. Restart me from the terminal."]


async def test_update_intent_without_bundle_speaks_before_quitting():
    """Dev run (no .app to reopen): the hint is spoken BEFORE relaunch()
    quits the process, so the speech isn't torn down mid-sentence."""
    order = []

    def relaunch():
        order.append(("relaunch", list(o.tts.said)))
        return False

    o, _, _ = build_d(["update yourself"], updater_check=lambda: _status("local"), updater_update=lambda s: "log",
                      relaunch=relaunch, can_relaunch=lambda: False)
    await o.one_turn()
    assert o.tts.said == ["Updating, back in a moment.", "Update installed. Restart me from the terminal."]
    assert order == [("relaunch", ["Updating, back in a moment.", "Update installed. Restart me from the terminal."])]


async def test_update_intent_with_bundle_relaunches_silently():
    calls = []
    o, _, _ = build_d(["update yourself"], updater_check=lambda: _status("local"), updater_update=lambda s: "log",
                      relaunch=lambda: calls.append("relaunch") or True, can_relaunch=lambda: True)
    await o.one_turn()
    assert o.tts.said == ["Updating, back in a moment."]
    assert calls == ["relaunch"]
