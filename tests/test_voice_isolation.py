"""The orchestrator side of "only listen to my voice": a capture the speaker
check rejects is silence — never a request, a follow-up, a dictated line or
a confirm answer — and the voice-profile turns."""
import asyncio

import numpy as np
import pytest

from tests.test_orchestrator import STT, TTS, Brain, Player, Rec, Wake
from veronica.config import Settings
from veronica.orchestrator import Orchestrator

ME, THEM = 1, 2


def said(value: int, n: int = 1) -> np.ndarray:
    """A capture in voice `value` (FakeSpeaker reads the voice off sample 0)."""
    return np.full(n, value, dtype=np.int16)


class FakeSpeaker:
    def __init__(self, active=True, ready=True, prepare_ok=True, agreement=0.9):
        self.active = active
        self.ready = ready
        self.prepare_ok = prepare_ok
        self.agreement = agreement
        self.checked: list[tuple[str, int]] = []
        self.enrolled: list[list[np.ndarray]] = []
        self.forgot = 0

    def check(self, pcm, where):
        ok = int(pcm[0]) == ME
        self.checked.append((where, int(pcm[0])))
        return ok, 0.8 if ok else 0.1

    def model_ready(self):
        return self.ready

    def prepare(self):
        return self.prepare_ok

    def enrol(self, clips):
        self.enrolled.append(clips)
        return (object() if self.agreement >= 0.3 else None), self.agreement

    def forget(self):
        self.forgot += 1
        return True


def build(rec_pcms=(), stt_texts=(), speaker=None, **settings):
    states = []
    o = Orchestrator(
        Settings(**{"followup_window_s": 0, "confirm_listen_s": 0, **settings}),
        wake=Wake(), recorder=Rec(rec_pcms), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
        speaker=speaker if speaker is not None else FakeSpeaker(),
    )
    return o, states


# -- requests and follow-ups ----------------------------------------------------------
async def test_a_request_in_another_voice_is_silence():
    o, states = build(rec_pcms=[said(THEM)], stt_texts=["delete everything"])
    await o.one_turn()
    assert o.brain.asked == []
    assert o.tts.said == []                          # no "didn't catch that"
    assert o.stt.texts == ["delete everything"]      # never even transcribed
    assert o.speaker.checked == [("request", THEM)]
    assert states[-1] == "idle"


async def test_my_request_goes_through():
    o, _ = build(rec_pcms=[said(ME), None], stt_texts=["tell me a joke"])
    await o.one_turn()
    assert o.brain.asked == ["tell me a joke"]


async def test_a_follow_up_in_another_voice_ends_the_turn():
    o, states = build(rec_pcms=[said(ME), said(THEM)], stt_texts=["tell me a joke", "and another"],
                      followup_window_s=4)
    await o.one_turn()
    assert o.brain.asked == ["tell me a joke"]
    assert o.speaker.checked == [("request", ME), ("follow-up", THEM)]
    assert states[-1] == "idle"


async def test_no_profile_means_no_checks():
    o, _ = build(rec_pcms=[said(THEM), None], stt_texts=["tell me a joke"], speaker=FakeSpeaker(active=False))
    await o.one_turn()
    assert o.brain.asked == ["tell me a joke"]
    assert o.speaker.checked == []


async def test_push_to_talk_skips_the_check():
    o, _ = build(rec_pcms=[said(THEM), None], stt_texts=["tell me a joke"])
    o.ready = True
    o.ptt_start()
    await o.one_turn(ptt=True)
    assert o.brain.asked == ["tell me a joke"]
    assert o.speaker.checked == []


# -- the confirm gate ------------------------------------------------------------------
async def test_another_voice_saying_yes_never_approves(caplog):
    caplog.set_level("INFO", logger="veronica.orchestrator")
    o, _ = build(rec_pcms=[said(THEM), said(THEM)], stt_texts=["yes", "yes"])
    r = await o.confirm("Bash: rm -rf build")
    assert r.outcome == "denied" and not r
    assert o.stt.texts == ["yes", "yes"]             # neither was transcribed
    assert o.tts.said[-1] == "Okay, skipping that."
    assert "confirm heard another voice; listening again" in caplog.text


async def test_another_voice_is_never_a_redirect():
    o, _ = build(rec_pcms=[said(THEM), None], stt_texts=["Ich küsse, küsse, küsse."])
    r = await o.confirm("Bash: ls")
    assert r.outcome == "denied" and r.heard == ""


async def test_my_answer_after_another_voice_counts():
    o, _ = build(rec_pcms=[said(THEM), said(ME)], stt_texts=["yes"])
    r = await o.confirm("Bash: ls")
    assert r.outcome == "approved"
    assert o.speaker.checked == [("confirm", THEM), ("confirm", ME)]


async def test_my_no_is_a_no():
    o, _ = build(rec_pcms=[said(ME)], stt_texts=["no"])
    assert (await o.confirm("Bash: ls")).outcome == "denied"


# -- dictation and unmute --------------------------------------------------------------
async def test_dictation_skips_another_voice(monkeypatch):
    from veronica import orchestrator as orch_mod

    typed = []
    monkeypatch.setattr(orch_mod.mac_tools, "dictate_type", lambda t: typed.append(t) or {})
    o, _ = build(rec_pcms=[said(THEM), said(ME), None], stt_texts=["hello there"])
    await o._dictation_turn()
    assert typed == ["hello there"]


async def test_unmute_in_another_voice_stays_muted():
    o, _ = build(rec_pcms=[said(THEM)], stt_texts=["unmute"])
    o.muted = True
    await o._muted_capture()
    assert o.muted is True


# -- the voice profile turns -----------------------------------------------------------
LONG = 16320  # 34 whole 30 ms frames, just over 1 s: enough to enrol


async def test_learn_my_voice_enrols_three_raw_clips():
    clips = [said(ME, LONG) for _ in range(3)]
    o, _ = build(rec_pcms=[said(ME), *clips, None], stt_texts=["Veronica, learn my voice."])
    await o.one_turn()
    assert len(o.speaker.enrolled) == 1 and len(o.speaker.enrolled[0]) == 3
    for line in Orchestrator.ENROL_LINES:
        assert line in o.tts.said
    assert o.tts.said[-1] == "Got it. I'll only listen to you now."
    assert o.brain.asked == []


async def test_enrolment_retries_a_short_line_once_then_gives_up():
    o, _ = build(rec_pcms=[said(ME, LONG), said(ME, 100), None], speaker=FakeSpeaker())
    await o._speaker_turn("enrol")
    assert o.tts.said.count("Once more, a little louder.") == 2
    assert o.tts.said[-1] == "I couldn't hear you. Let's try later."
    assert o.speaker.enrolled == []


async def test_enrolment_that_disagrees_is_not_saved():
    o, _ = build(rec_pcms=[said(ME, LONG)] * 3, speaker=FakeSpeaker(agreement=0.1))
    await o._speaker_turn("enrol")
    assert o.tts.said[-1] == "Those didn't sound like one voice. Let's try again somewhere quieter."


async def test_enrolment_fetches_the_model_first_and_says_when_it_cant():
    o, _ = build(speaker=FakeSpeaker(ready=False, prepare_ok=False))
    await o._speaker_turn("enrol")
    assert o.tts.said == ["One moment, getting the voice model.", "Couldn't get the voice model, check the log."]


async def test_enrolment_with_the_check_switched_off_says_so():
    o, _ = build(rec_pcms=[said(ME, LONG)] * 3, speaker_verification=False)
    await o._speaker_turn("enrol")
    assert o.tts.said[-1] == "Got your voice. Turn on the voice check in Settings to use it."


@pytest.mark.parametrize("utterance", ["Forget my voice.", "meri awaaz bhool jao"])
async def test_forget_my_voice_is_not_a_memory_fact(utterance):
    class Store:
        def delete_fact_matching(self, arg):
            raise AssertionError("went to memory")

    o, _ = build(rec_pcms=[said(ME), None], stt_texts=[utterance])
    o.store = Store()
    await o.one_turn()
    assert o.speaker.forgot == 1
    assert o.tts.said[0] in ("Done, I'll listen to anyone now.", "ठीक है, अब मैं सबकी सुनूँगी।")


async def test_without_a_speaker_gate_she_says_she_cant():
    o = Orchestrator(Settings(followup_window_s=0), wake=Wake(), recorder=Rec([]), stt=STT([]),
                     brain=Brain(), tts=TTS(), player=Player())
    await o._speaker_turn("enrol")
    assert o.tts.said == ["I can't learn voices here."]


async def test_queued_turn_runs_when_idle():
    o, states = build()
    ran = asyncio.Event()

    async def turn():
        ran.set()

    await o.queue_turn(turn)
    await o._deliver_announcement(o._announce_queue.get_nowait())
    assert ran.is_set() and states[-1] == "idle"


async def test_enrolment_counts_the_voice_not_the_trailing_silence():
    blip = np.concatenate([np.full(4800, ME, np.int16) * 3000, np.zeros(19200, np.int16)])   # 0.3 s + 1.2 s quiet
    blip[0] = ME
    o, _ = build(rec_pcms=[blip, None])
    await o._speaker_turn("enrol")
    assert o.tts.said.count("Once more, a little louder.") == 2
    assert o.speaker.enrolled == []
