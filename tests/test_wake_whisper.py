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
    # First hop transcribed is exactly wake_hop_s (0.4s here, set explicitly
    # so the arithmetic doesn't shift with the defaults) of buffered audio;
    # "veronica" ends at 0.3s into that window, so the tail from 0.3s to 0.4s
    # (0.1s = 1600 samples at 16 kHz) should become the pre-roll.
    words = [Word(0.0, 0.15, "hey"), Word(0.15, 0.3, "veronica")]
    monkeypatch.setattr(WhisperWake, "_model_cls", _wordseg_model_cls(words, "hey veronica"))
    w = WhisperWake(Settings(wake_hop_s=0.4, wake_window_s=1.6), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True

    preroll = w.take_preroll()
    assert preroll.size == 1600
    assert np.all(preroll == 1000)
    assert w.take_preroll().size == 0


@pytest.mark.asyncio
async def test_take_preroll_falls_back_without_word_timestamps(monkeypatch):
    # No word-level timestamps at all -> fallback to window_end - 0.3s.
    # window here is 0.4s (wake_hop_s, set explicitly), so fallback
    # end_s = 0.1s -> tail is 0.3s = 4800 samples.
    monkeypatch.setattr(WhisperWake, "_model_cls", _wordseg_model_cls([], "veronica"))
    w = WhisperWake(Settings(wake_hop_s=0.4, wake_window_s=1.6), frames=lambda: const_frames(1000))
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


def test_mic_frames_buffers_while_consumer_stalls(monkeypatch):
    """The mic is read on its own thread into a queue, so a slow consumer
    (whisper taking longer than one hop) never makes PortAudio overflow and
    drop audio: every frame the device produced is delivered, in order."""
    import threading
    import time

    from veronica.audio import devices, mic
    from veronica.audio import wake_whisper as ww

    produced = []

    class FakeStream:
        def __init__(self, **kw):
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, n):
            self.n += 1
            time.sleep(0.001)
            frame = self.n.to_bytes(4, "little") * (n // 2)   # n int16 frames
            produced.append(frame)
            return frame, False

    monkeypatch.setattr(mic.sd, "RawInputStream", FakeStream)
    monkeypatch.setattr(devices, "default_input_id", lambda: None)   # never touch CoreAudio
    w = ww.WhisperWake.__new__(ww.WhisperWake)
    w.s = ww.Settings()
    frames = w._mic_frames()
    got = [next(frames)]
    time.sleep(0.05)              # consumer stalls; producer keeps reading
    for _ in range(20):
        got.append(next(frames))
    assert got == produced[: len(got)]      # nothing dropped, in order
    assert len(produced) > len(got)          # producer ran ahead into the queue
    frames.close()
    time.sleep(0.02)
    assert not any(t.name == "wake-mic" and t.is_alive() for t in threading.enumerate())


async def test_wait_skips_transcription_while_behind(monkeypatch):
    """When the mic reader has more than a hop of frames queued, the loop
    catches up instead of transcribing stale windows."""
    from veronica.audio import wake_whisper as ww

    calls = []

    class Model:
        def __init__(self, *a, **k): pass
        def transcribe(self, audio, **kw):
            calls.append(len(audio))
            return iter([]), None

    monkeypatch.setattr(ww.WhisperWake, "_model_cls", Model)
    loud = (np.ones(ww.CHUNK, dtype=np.int16) * 3000).tobytes()
    n_frames = 60   # 12 hops at the explicit 0.4 s hop below
    w = ww.WhisperWake(ww.Settings(wake_hop_s=0.4, wake_window_s=1.6), frames=lambda: iter([loud] * n_frames))
    hops = {"n": 0}

    def backlog():
        hops["n"] += 1
        # pretend the reader is 10 frames ahead for the first 10 hops
        return 10 if hops["n"] <= 10 else 0

    w._backlog = backlog
    assert await w.wait() is False      # frames exhausted without a match
    assert 0 < len(calls) <= 2      # only the final, caught-up hop(s) are transcribed


@pytest.mark.asyncio
async def test_wait_logs_rms_per_hop_at_debug(monkeypatch, caplog):
    """Every hop logs its window rms against the gate at DEBUG (opt-in via
    VERONICA_LOG_LEVEL=DEBUG) so far-field sensitivity can be diagnosed."""
    import logging

    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["hey veronica"]))
    s = Settings(wake_min_rms=0.01)
    quiet = np.full(CHUNK, 100, dtype=np.int16).tobytes()      # rms ~0.003 < gate
    loud = np.full(CHUNK, 1000, dtype=np.int16).tobytes()      # rms ~0.03 > gate
    hop_frames = int(s.wake_hop_s * s.sample_rate) // CHUNK + 1
    w = WhisperWake(s, frames=lambda: iter([quiet] * hop_frames + [loud] * hop_frames * 4))
    with caplog.at_level(logging.DEBUG, logger="veronica.audio"):
        assert await _wait_for(w.wait(), 3) is True
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("wake hop rms=")]
    assert lines[0] == "wake hop rms=0.0031 gate=0.0100 (below gate)"
    assert any(line.endswith("gate=0.0100") for line in lines[1:])   # the loud hop is not suffixed


def test_engines_delegate_mic_frames_to_shared_reader(monkeypatch, tmp_home):
    """Both engines' default frame source is veronica.audio.mic.mic_frames
    with an InputWatch and the Player-closing refresh hook."""
    from veronica.audio import wake as wake_mod
    from veronica.audio import wake_whisper as ww
    from veronica.audio.devices import InputWatch
    from veronica.audio.play import close_registered_streams

    calls = []

    def fake_mic_frames(settings, chunk, prefix, **kw):
        calls.append((chunk, prefix, kw))
        return iter([])

    monkeypatch.setattr(ww, "mic_frames", fake_mic_frames)
    monkeypatch.setattr(wake_mod, "mic_frames", fake_mic_frames)
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls([""]))
    monkeypatch.setattr(WakeWord, "_model_cls", staticmethod(lambda wakeword_models, inference_framework: object()))

    w = WhisperWake(Settings())
    assert list(w._mic_frames()) == []
    chunk, prefix, kw = calls[-1]
    assert (chunk, prefix) == (CHUNK, "wake")
    assert isinstance(kw["watch"], InputWatch) and kw["before_refresh"] is close_registered_streams
    kw["on_backlog"](lambda: 7)
    assert w._backlog() == 7

    o = WakeWord(Settings(wake_engine="openwakeword"))
    assert list(o._mic_frames()) == []
    chunk, prefix, kw = calls[-1]
    assert (chunk, prefix) == (CHUNK, "wake")
    assert isinstance(kw["watch"], InputWatch) and kw["before_refresh"] is close_registered_streams


@pytest.mark.asyncio
async def test_wait_raises_when_frames_raise(monkeypatch):
    """A dead mic reader (mic_frames raising) must surface from wait() so
    the orchestrator's 'wake listener failed; retrying' backoff applies."""
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls([""]))

    def broken_frames():
        yield b"\x00\x00" * 1280
        raise OSError("no input device")

    w = WhisperWake(Settings(), frames=broken_frames)
    with pytest.raises(OSError, match="no input device"):
        await _wait_for(w.wait(), 2)


def test_strict_match_drops_fuzzy_near_misses():
    from veronica.audio.wake_whisper import _matches
    phrases = ["veronica"]
    # idle: a near miss still wakes her
    assert _matches("veronika are you there", phrases) is True
    # mid-turn: only the real name counts, so a mis-heard word can't cancel
    # the answer the user is waiting for
    assert _matches("veronika are you there", phrases, strict=True) is False
    assert _matches("hey veronica stop", phrases, strict=True) is True


def test_transcribe_never_primes_the_decoder(monkeypatch):
    """Priming tiny.en with the wake phrases makes it spell them out of room
    noise (a false wake) and turns a 0.1 s hop into a ~4 s one, so the loop
    falls behind the mic and skips the hop the name was really in."""
    seen = {}

    class Recorder:
        def transcribe(self, audio, **kw):
            seen.update(kw)
            return [], None

    monkeypatch.setattr(WhisperWake, "_model_cls", staticmethod(lambda *a, **k: Recorder()))
    w = WhisperWake(Settings(), frames=silence_frames)
    w._transcribe(np.zeros(16000, dtype=np.int16))
    assert seen.get("initial_prompt") is None


@pytest.mark.asyncio
async def test_noise_only_a_primed_decoder_hears_as_the_name_never_wakes(monkeypatch):
    """The live false wakes: room noise came back as "Hi. Veronika. Hi.
    Veronika. ..." only because the decoder had been handed the phrases."""
    class Primed:
        def __init__(self, *a, **k) -> None:
            pass

        def transcribe(self, audio, **kw):
            if kw.get("initial_prompt"):
                return [FakeSegment("Hi. Veronika. Hi. Veronika.")], None
            return [], None

    monkeypatch.setattr(WhisperWake, "_model_cls", staticmethod(Primed))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    task = asyncio.ensure_future(w.wait())
    await asyncio.sleep(0.2)
    w.stop()
    assert await _wait_for(task, 3) is False


@pytest.mark.asyncio
async def test_spoken_name_still_wakes_without_the_bias(monkeypatch):
    monkeypatch.setattr(WhisperWake, "_model_cls", scripted_model_cls(["", "veronica"]))
    w = WhisperWake(Settings(), frames=lambda: const_frames(1000))
    assert await _wait_for(w.wait(), 3) is True
