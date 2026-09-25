import numpy as np
import pytest

from veronica.speech.tts import Synthesizer


class FakeKokoro:
    def __init__(self, model_path, voices_path):
        self.model_path = model_path
        self.calls = []

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice))
        return np.zeros(240, dtype=np.float32), 24000


def test_synth_returns_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(Synthesizer, "_kokoro_cls", FakeKokoro)
    s = Synthesizer(voice="af_sarah", models_dir=tmp_path)
    samples, sr = s.synth("hello")
    assert sr == 24000
    assert samples.dtype == np.float32
    assert s._engine.calls == [("hello", "af_sarah")]


async def test_asynth(tmp_path, monkeypatch):
    monkeypatch.setattr(Synthesizer, "_kokoro_cls", FakeKokoro)
    s = Synthesizer(voice="af_sarah", models_dir=tmp_path)
    samples, sr = await s.asynth("hi")
    assert len(samples) == 240


@pytest.mark.live
def test_real_kokoro_speaks():
    from veronica.config import settings
    s = Synthesizer(voice=settings.kokoro_voice, models_dir=settings.models_dir)
    samples, sr = s.synth("Hello, I am Veronica.")
    assert sr == 24000 and len(samples) > sr  # > 1 s of audio


def test_synth_passes_current_voice_and_speed(monkeypatch, tmp_path):
    calls = []

    class FakeKokoro:
        def __init__(self, *a): pass
        def create(self, text, voice, speed, lang):
            calls.append((text, voice, speed, lang))
            return [0.0, 0.0], 24000

    monkeypatch.setattr(Synthesizer, "_kokoro_cls", FakeKokoro)
    s = Synthesizer("af_sarah", tmp_path)
    assert s.speed == 1.0
    s.voice = "am_adam"
    s.speed = 1.3
    s.synth("hi")
    assert calls == [("hi", "am_adam", 1.3, "en-us")]


def test_synth_picks_hindi_voice_for_devanagari_or_lang(monkeypatch, tmp_path):
    calls = []

    class FakeKokoro:
        def __init__(self, *a):
            pass

        def create(self, text, voice, speed, lang):
            calls.append((voice, lang))
            return [0.0], 24000

    monkeypatch.setattr(Synthesizer, "_kokoro_cls", FakeKokoro)
    s = Synthesizer("af_sarah", tmp_path)
    assert s.hindi_voice == "hf_alpha"
    s.synth("नमस्ते")
    assert calls[-1] == ("hf_alpha", "hi")
    s.synth("kal teen baje", lang="hi")
    assert calls[-1] == ("hf_alpha", "hi")
    s.synth("hello", lang="en")
    assert calls[-1] == ("af_sarah", "en-us")
    s.synth("hello")
    assert calls[-1] == ("af_sarah", "en-us")
    s.hindi_voice = "hm_omega"
    s.synth("ठीक है")
    assert calls[-1] == ("hm_omega", "hi")


def test_devanagari_reply_uses_the_hindi_voice_even_on_an_english_turn(monkeypatch):
    """A brain that answers in Hindi to an English question must still be
    spoken by the Hindi voice — the English one just mangles the script."""
    from veronica.speech.tts import Synthesizer
    used = {}

    s = Synthesizer.__new__(Synthesizer)
    s.voice, s.hindi_voice, s.speed = "af_bella", "hf_alpha", 1.0
    s._engine = type("E", (), {
        "create": lambda self, text, voice, speed, lang: (used.update(voice=voice, lang=lang), ([0.0], 24000))[1]
    })()
    s.synth("समझ गया।", lang="en")
    assert used == {"voice": "hf_alpha", "lang": "hi"}
    s.synth("Got it.", lang="en")
    assert used == {"voice": "af_bella", "lang": "en-us"}
