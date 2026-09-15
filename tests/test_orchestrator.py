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
    async def atranscribe(self, pcm): return self.texts.pop(0)

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
