import asyncio

import numpy as np
import pytest

from veronica.config import Settings
from veronica.orchestrator import Orchestrator


class Wake:
    async def wait(self): pass

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
        def __init__(self):
            self.calls = 0

        async def wait(self):
            self.calls += 1
            if self.calls > 1:
                raise asyncio.CancelledError()

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
        def __init__(self):
            self.calls = 0

        async def wait(self):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("no audio device")
            if self.calls > 2:
                raise asyncio.CancelledError()

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
        def __init__(self):
            self.calls = 0

        async def wait(self):
            self.calls += 1
            if self.calls > 1:
                raise asyncio.CancelledError()

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

    class SlowPlayer:
        def __init__(self): self.played = 0
        async def play(self, s):
            self.played += 1
            await asyncio.sleep(10)
        def stop(self): pass
        def reset(self): pass

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
    o.tts, o.brain = FastTTS(), Brain5()
    o.player = SlowPlayer()

    turn = asyncio.create_task(o.handle_text("x"))
    await asyncio.sleep(0.05)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(turn, 1)

    assert o._speech_queue is None

    o.player = FastPlayer()
    o.brain = Brain5()
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
        def __init__(self, orch): self.orch = orch
        async def ask(self, text):
            holder["task"] = asyncio.create_task(self.orch.confirm("Bash: ls"))
            await asyncio.sleep(0)
            yield "First."
            assert await holder["task"] is True
            yield "Second."

    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    o.tts, o.player = DelayedTTS(), OrderPlayer()
    o.brain = ConcurrentConfirmBrain(o)

    out = await o.handle_text("hello")

    assert out == ["First.", "Second."]
    assert o.player.played_texts == ["First.", "Run Bash: ls?", "Second."]


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
