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


def test_synth_returns_samples(tmp_path):
    Synthesizer._kokoro_cls = FakeKokoro
    s = Synthesizer(voice="af_sarah", models_dir=tmp_path)
    samples, sr = s.synth("hello")
    assert sr == 24000
    assert samples.dtype == np.float32
    assert s._engine.calls == [("hello", "af_sarah")]


async def test_asynth(tmp_path):
    Synthesizer._kokoro_cls = FakeKokoro
    s = Synthesizer(voice="af_sarah", models_dir=tmp_path)
    samples, sr = await s.asynth("hi")
    assert len(samples) == 240


@pytest.mark.live
def test_real_kokoro_speaks():
    from veronica.config import settings
    s = Synthesizer(voice=settings.kokoro_voice, models_dir=settings.models_dir)
    samples, sr = s.synth("Hello, I am Veronica.")
    assert sr == 24000 and len(samples) > sr  # > 1 s of audio
