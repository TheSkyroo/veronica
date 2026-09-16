import asyncio

import numpy as np
import pytest

from veronica.config import Settings
from veronica.orchestrator import Orchestrator


class Wake:
    async def wait(self, threshold=None): pass
    def stop(self): pass

class Rec:
    def __init__(self, pcms): self.pcms = list(pcms)
    async def capture(self, max_s=None): return self.pcms.pop(0) if self.pcms else None

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

        async def wait(self, threshold=None):
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

        async def wait(self, threshold=None):
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

    async def capture(self, max_s=None):
        self.max_s_calls.append(max_s)
        return self.pcms.pop(0) if self.pcms else None


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

        async def wait(self, threshold=None):
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

        async def wait(self, threshold=None):
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
        self._ev = __import__("asyncio").Event()
    async def wait(self, threshold=None):
        self.calls += 1
        if self.barge_on_call == self.calls:
            await __import__("asyncio").sleep(0.005)
            return True
        await self._ev.wait(); self._ev.clear(); return False
    def stop(self):
        self.stops += 1; self._ev.set()


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

        async def capture(self, max_s=None):
            self.n += 1
            if self.n > 1:
                events.append("capture")
            return await super().capture(max_s=max_s)

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

    async def capture(self, max_s=None):
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

        async def capture(self, max_s=None):
            self.captures += 1
            return self.pcms.pop(0) if self.pcms else None

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
        async def wait(self, threshold=None):
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
        async def wait(self, threshold=None):
            raise RuntimeError("mic hiccup")
        def stop(self):
            self.stops += 1

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
