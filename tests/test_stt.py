import numpy as np
import pytest
import soundfile as sf

from veronica.speech.stt import Transcriber


class FakeSeg:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self, name, device, compute_type):
        self.name = name

    def transcribe(self, audio, beam_size, language, vad_filter):
        return iter([FakeSeg(" hello "), FakeSeg("world")]), None


def test_transcribe_joins_segments(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    t = Transcriber("base.en")
    assert t.transcribe(np.zeros(16000, dtype=np.int16)) == "hello world"


async def test_atranscribe(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    t = Transcriber("base.en")
    assert await t.atranscribe(np.zeros(16000, dtype=np.int16)) == "hello world"


@pytest.mark.live
def test_real_whisper_on_fixture(monkeypatch):
    from faster_whisper import WhisperModel
    monkeypatch.setattr(Transcriber, "_model_cls", WhisperModel)
    pcm, sr = sf.read("tests/fixtures/speech_1s.wav", dtype="int16")
    text = Transcriber("base.en").transcribe(pcm).lower()
    assert "time" in text
