import asyncio

import numpy as np
import pytest

from veronica.audio.wake import WakeWord, make_wake
from veronica.audio.wake_whisper import WhisperWake
from veronica.config import Settings

CHUNK = 1280  # 80 ms @ 16 kHz


class FakeSegment:
    def __init__(self, text: str, words=None) -> None:
        self.text = text
        self.words = words or []


class Word:
    def __init__(self, start: float, end: float, word: str) -> None:
        self.start, self.end, self.word = start, end, word


class FakeModel:
    """Returns scripted transcripts keyed by call index; extra calls repeat the last."""

    def __init__(self, model_name, device=None, compute_type=None) -> None:
        self.model_name = model_name
        self.calls = 0

    def transcribe(self, audio, **kw):
        script = getattr(self, "script", [""])
        idx = min(self.calls, len(script) - 1)
        text = script[idx]
        self.calls += 1
        return [FakeSegment(text)] if text else [], None


def scripted_model_cls(script):
    def _factory(model_name, device=None, compute_type=None):
        m = FakeModel(model_name, device=device, compute_type=compute_type)
        m.script = script
        return m
    return staticmethod(_factory)


def const_frames(value: int):
    while True:
        yield np.full(CHUNK, value, dtype=np.int16).tobytes()


def silence_frames():
    while True:
        yield np.zeros(CHUNK, dtype=np.int16).tobytes()


async def _wait_for(coro, timeout):
    return await asyncio.wait_for(coro, timeout)


@pytest.mark.asyncio
async def test_detects_on_scripted_call(monkeypatch):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["", "", "hey veronica please"]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True


@pytest.mark.asyncio
async def test_silence_never_calls_model_and_stop_returns_false(monkeypatch):
    calls = {"n": 0}

    def _factory(model_name, device=None, compute_type=None):
        m = FakeModel(model_name, device=device, compute_type=compute_type)
        m.script = ["hey veronica"]  # would match if ever called

        real_transcribe = m.transcribe

        def counting_transcribe(audio, **kw):
            calls["n"] += 1
            return real_transcribe(audio, **kw)

        m.transcribe = counting_transcribe
        return m

    monkeypatch.setattr(WhisperWake, "_model_cls", staticmethod(_factory))
    w = WhisperWake(Settings(), frames=silence_frames)
    task = asyncio.create_task(w.wait())
    await asyncio.sleep(0.3)
    w.stop()
    assert await _wait_for(task, 3) is False
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_fuzzy_veronika_matches_verona_does_not(monkeypatch):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["veronika"]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True

    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["verona"]))
    w2 = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    task = asyncio.create_task(w2.wait())
    await asyncio.sleep(0.5)
    w2.stop()
    assert await _wait_for(task, 3) is False


@pytest.mark.asyncio
async def test_stop_is_consumed(monkeypatch):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls([""]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    task = asyncio.create_task(w.wait())
    w.stop()
    assert await _wait_for(task, 3) is False

    # a stale stop flag must not poison the next wait()
    monkeypatch.setattr(w, "_transcribe", lambda window: [FakeSegment("hey veronica")])
    assert await _wait_for(w.wait(), 3) is True


@pytest.mark.asyncio
async def test_buffer_cleared_after_match(monkeypatch):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["hey veronica", "", ""]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True

    # after a match the model script continues returning "" — must not re-trigger
    # from stale buffer content; stop() ends it.
    task = asyncio.create_task(w.wait())
    await asyncio.sleep(0.5)
    w.stop()
    assert await _wait_for(task, 3) is False


def test_make_wake_returns_correct_engine(monkeypatch, tmp_home):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls([""]))
    monkeypatch.setattr(WakeWord, "_model_cls", staticmethod(lambda wakeword_models, inference_framework: object()))

    w_whisper = make_wake(Settings(wake_engine="whisper"))
    assert isinstance(w_whisper, WhisperWake)

    w_oww = make_wake(Settings(wake_engine="openwakeword"))
    assert isinstance(w_oww, WakeWord)


@pytest.mark.asyncio
async def test_threshold_accepted_and_ignored(monkeypatch):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["hey veronica"]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(threshold=0.8), 3) is True


@pytest.mark.asyncio
async def test_own_speech_is_suppressed(monkeypatch):
    """A wake match caused by Veronica's own TTS (e.g. "I am Veronica") must be
    dropped rather than returned, so she doesn't self-interrupt; a later,
    genuine match (nothing being spoken at the time) still returns True."""
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["i am veronica", "hey veronica"]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    suppress_texts = iter(["I am Veronica, your assistant.", ""])
    assert await _wait_for(w.wait(suppress=lambda: next(suppress_texts)), 3) is True


def _wordseg_model_cls(words, text):
    def _factory(model_name, device=None, compute_type=None):
        class M:
            def transcribe(self, audio, **kw):
                return [FakeSegment(text, words)], None
        return M()
    return staticmethod(_factory)


@pytest.mark.asyncio
async def test_take_preroll_returns_tail_after_last_wake_word_then_empty(monkeypatch):
    # First hop transcribed is exactly wake_hop_s (0.4s) of buffered audio;
    # "veronica" ends at 0.3s into that window, so the tail from 0.3s to 0.4s
    # (0.1s = 1600 samples at 16 kHz) should become the pre-roll.
    words = [Word(0.0, 0.15, "hey"), Word(0.15, 0.3, "veronica")]
    monkeypatch.setattr(WhisperWake, "_model_cls", _wordseg_model_cls(words, "hey veronica"))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True

    preroll = w.take_preroll()
    assert preroll.size == 1600
    assert np.all(preroll == 1000)
    assert w.take_preroll().size == 0


@pytest.mark.asyncio
async def test_take_preroll_falls_back_without_word_timestamps(monkeypatch):
    # No word-level timestamps at all -> fallback to window_end - 0.3s.
    # window here is 0.4s (wake_hop_s), so fallback end_s = 0.1s -> tail is
    # 0.3s = 4800 samples.
    monkeypatch.setattr(WhisperWake, "_model_cls", _wordseg_model_cls([], "veronica"))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True

    preroll = w.take_preroll()
    assert preroll.size == 4800


def test_wakeword_take_preroll_returns_empty_int16(monkeypatch, tmp_home):
    monkeypatch.setattr(WakeWord, "_model_cls", staticmethod(lambda wakeword_models, inference_framework: object()))
    w = WakeWord(Settings(wake_engine="openwakeword"), frames=lambda: iter([]))
    p = w.take_preroll()
    assert p.size == 0
    assert p.dtype == np.int16


@pytest.mark.asyncio
async def test_own_speech_suppression_is_fuzzy(monkeypatch):
    """The suppress-text check reuses the same _matches() as the wake check
    (substring + fuzzy), so a possessive form like "Veronika's" — punctuation
    stripped to "veronikas", not a substring of any wake phrase but within
    fuzzy ratio (~0.82) of "veronica" — is still recognized as self-speech
    and the match is suppressed."""
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["hey veronica", "hey veronica"]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    suppress_texts = iter(["Veronika's here to help.", ""])
    assert await _wait_for(w.wait(suppress=lambda: next(suppress_texts)), 3) is True
