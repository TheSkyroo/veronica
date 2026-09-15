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
    assert await w.wait() is True  # must return, not hang


@pytest.mark.asyncio
async def test_threshold_respected(monkeypatch):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    seen = []

    def f():
        for ch in "..w":
            seen.append(ch)
            yield np.full(CHUNK, 1000 if ch == "w" else 0, dtype=np.int16).tobytes()

    w = WakeWord(Settings(wake_threshold=0.5), frames=f)
    assert await w.wait() is True
    assert seen == [".", ".", "w"]


async def test_wait_returns_true_on_detection(monkeypatch):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    w = WakeWord(Settings(), frames=lambda: frames("..w"))
    assert await w.wait() is True


async def test_stop_returns_false(monkeypatch):
    import asyncio
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    w = WakeWord(Settings(), frames=lambda: frames("." * 100000))
    task = asyncio.create_task(w.wait())
    w.stop()
    assert await asyncio.wait_for(task, 2) is False


async def test_threshold_override(monkeypatch):
    import asyncio
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)  # scores 0.9 on 'w'
    w = WakeWord(Settings(), frames=lambda: frames("w....."))
    task = asyncio.create_task(w.wait(threshold=0.95))
    w.stop()
    assert await asyncio.wait_for(task, 2) is False   # 0.9 < 0.95 → never detected


async def test_stop_is_consumed(monkeypatch):
    import asyncio
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    w = WakeWord(Settings(), frames=lambda: frames("." * 100000))
    task = asyncio.create_task(w.wait())
    w.stop()
    assert await asyncio.wait_for(task, 2) is False

    # a stale stop flag must not poison the next wait()
    w._frames = lambda: frames("..w")
    assert await w.wait() is True


def test_custom_model_used_when_present(monkeypatch, tmp_home):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    models_dir = tmp_home / "models"
    models_dir.mkdir(parents=True)
    custom_path = models_dir / "hey_veronica.onnx"
    custom_path.write_bytes(b"fake")

    w = WakeWord(Settings(wake_model="hey_veronica"), frames=lambda: frames(""))

    assert w._model.name == str(custom_path)
    assert w._key == "hey_veronica"


def test_falls_back_to_hey_jarvis_when_custom_missing(monkeypatch, tmp_home, caplog):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)

    with caplog.at_level("WARNING", logger="veronica.audio"):
        w = WakeWord(Settings(wake_model="hey_veronica"), frames=lambda: frames(""))

    assert w._model.name == "hey_jarvis"
    assert w._key == "hey_jarvis"
    assert any("hey_veronica" in r.message and "hey_jarvis" in r.message for r in caplog.records)


def test_pretrained_model_passes_through_unchanged(monkeypatch, tmp_home, caplog):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)

    with caplog.at_level("WARNING", logger="veronica.audio"):
        w = WakeWord(Settings(wake_model="hey_jarvis"), frames=lambda: frames(""))

    assert w._model.name == "hey_jarvis"
    assert w._key == "hey_jarvis"
    assert not any(r.levelname == "WARNING" for r in caplog.records)
