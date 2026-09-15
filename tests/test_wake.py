import numpy as np
import pytest

from veronica.audio.wake import WakeWord
from veronica.config import Settings

CHUNK = 1280


class FakeModel:
    def __init__(self, wakeword_models, inference_framework):
        self.name = wakeword_models[0]

    def predict(self, chunk):
        return {self.name: 0.9 if chunk.max() > 0 else 0.0}

    def reset(self):
        pass


def frames(pattern):
    for ch in pattern:
        yield np.full(CHUNK, 1000 if ch == "w" else 0, dtype=np.int16).tobytes()
    while True:
        yield np.zeros(CHUNK, dtype=np.int16).tobytes()


@pytest.mark.asyncio
async def test_wait_returns_on_detection(monkeypatch):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    w = WakeWord(Settings(), frames=lambda: frames("...w"))
    await w.wait()  # must return, not hang


@pytest.mark.asyncio
async def test_threshold_respected(monkeypatch):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    seen = []

    def f():
        for ch in "..w":
            seen.append(ch)
            yield np.full(CHUNK, 1000 if ch == "w" else 0, dtype=np.int16).tobytes()

    w = WakeWord(Settings(wake_threshold=0.5), frames=f)
    await w.wait()
    assert seen == [".", ".", "w"]
