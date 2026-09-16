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


def test_transcribe_detailed_reports_language(monkeypatch):
    calls = []

    class Info:
        language = "hi"

    class Seg:
        def __init__(self, t):
            self.text = t

    class M:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **kw):
            calls.append(kw)
            return iter([Seg(" नमस्ते ")]), Info()

    monkeypatch.setattr(Transcriber, "_model_cls", M)
    t = Transcriber("small", language=None)
    assert t.transcribe_detailed(np.zeros(16000, dtype=np.int16)) == ("नमस्ते", "hi")
    assert calls[-1]["language"] is None
    t.set_language("hi")
    t.transcribe(np.zeros(16000, dtype=np.int16))
    assert calls[-1]["language"] == "hi"
    t2 = Transcriber("small.en")
    t2.transcribe(np.zeros(16000, dtype=np.int16))
    assert calls[-1]["language"] == "en"


async def test_atranscribe_detailed(monkeypatch):
    monkeypatch.setattr(Transcriber, "_model_cls", FakeModel)
    t = Transcriber("base.en")
    assert await t.atranscribe_detailed(np.zeros(16000, dtype=np.int16)) == ("hello world", "en")


@pytest.mark.live
def test_real_whisper_on_fixture(monkeypatch):
    from faster_whisper import WhisperModel
    monkeypatch.setattr(Transcriber, "_model_cls", WhisperModel)
    pcm, sr = sf.read("tests/fixtures/speech_1s.wav", dtype="int16")
    text = Transcriber("base.en").transcribe(pcm).lower()
    assert "time" in text
