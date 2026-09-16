import asyncio

import numpy as np
import pytest

from veronica.audio import play as play_mod


class FakeStream:
    """Stands in for sd.OutputStream: exposes the callback so a test can pump
    it manually (as the real audio thread would, driving `outdata` in place)."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs.get("finished_callback")
        self.started = 0
        self.stopped = 0
        self.closed = 0
        self.active = True

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1
        self.active = False

    def close(self):
        self.closed += 1


class FakeSD:
    def __init__(self):
        self.streams = []

    def OutputStream(self, **kwargs):
        s = FakeStream(**kwargs)
        self.streams.append(s)
        return s


@pytest.fixture
def fake_sd(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(play_mod, "sd", sd)
    return sd


async def pump(stream, n_blocks, blocksize=1024):
    """Feed n_blocks of blocksize frames through the fake stream's callback,
    the way the real audio thread would, and return what it wrote."""
    collected = []
    for _ in range(n_blocks):
        outdata = np.zeros((blocksize, 1), dtype=np.float32)
        stream.callback(outdata, blocksize, None, None)
        collected.append(outdata[:, 0].copy())
        await asyncio.sleep(0)
    return np.concatenate(collected)


async def test_play_sends_samples_with_fades(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    samples = np.ones(2000, dtype=np.float32)
    task = asyncio.ensure_future(p.play(samples))
    await asyncio.sleep(0.01)

    stream = fake_sd.streams[-1]
    assert stream.started == 1
    assert stream.kwargs["samplerate"] == 24000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "float32"
    assert stream.kwargs["blocksize"] == 1024
    assert stream.kwargs["latency"] == "high"

    got = await pump(stream, 2, blocksize=1024)
    await asyncio.wait_for(task, 1)

    fade_n = 24000 * 3 // 1000
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    expected = samples.copy()
    expected[:fade_n] *= ramp
    expected[-fade_n:] *= ramp[::-1]

    np.testing.assert_allclose(got[:2000], expected, atol=1e-6)
    assert p.is_playing is False


async def test_stop_mid_play_returns_promptly(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    task = asyncio.ensure_future(p.play(np.ones(1_000_000, dtype=np.float32)))
    await asyncio.sleep(0.01)
    assert p.is_playing is True

    p.stop()
    await asyncio.wait_for(task, 1)

    assert p.is_playing is False


async def test_stop_skips_queued(fake_sd):
    p = play_mod.Player()
    p.stop()  # sets stopped flag before play
    await p.play(np.zeros(10, dtype=np.float32))
    assert p.is_playing is False
    assert fake_sd.streams == []  # never opens the device at all


async def test_second_play_after_reset_reuses_stream(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)

    task1 = asyncio.ensure_future(p.play(np.ones(500, dtype=np.float32)))
    await asyncio.sleep(0.01)
    stream1 = fake_sd.streams[-1]
    await pump(stream1, 1)
    await asyncio.wait_for(task1, 1)

    p.stop()
    p.reset()

    task2 = asyncio.ensure_future(p.play(np.ones(500, dtype=np.float32)))
    await asyncio.sleep(0.01)
    await pump(fake_sd.streams[-1], 1)
    await asyncio.wait_for(task2, 1)

    assert len(fake_sd.streams) == 1  # the same stream was reused, not reopened


async def test_play_times_out_if_never_pumped(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    p._timeout_margin_s = 0.05
    task = asyncio.ensure_future(p.play(np.ones(10, dtype=np.float32)))

    await asyncio.wait_for(task, 1)  # returns via the timeout path, not a drain

    stream = fake_sd.streams[-1]
    assert stream.stopped == 1
    assert stream.closed == 1
    assert p._stream is None


async def test_finished_callback_drains_and_marks_stream_dead(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    task = asyncio.ensure_future(p.play(np.ones(1_000_000, dtype=np.float32)))
    await asyncio.sleep(0.01)
    assert p.is_playing is True

    stream = fake_sd.streams[-1]
    stream.finished_callback()  # simulates the device dying under us

    await asyncio.wait_for(task, 1)
    assert p._stream is None


async def test_close_is_idempotent(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    task = asyncio.ensure_future(p.play(np.ones(500, dtype=np.float32)))
    await asyncio.sleep(0.01)
    stream = fake_sd.streams[-1]
    await pump(stream, 1)
    await asyncio.wait_for(task, 1)

    p.close()
    p.close()  # must not raise

    assert stream.stopped == 1
    assert stream.closed == 1
    assert p._stream is None


async def test_underrun_zero_fills(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    task = asyncio.ensure_future(p.play(np.ones(100, dtype=np.float32)))
    await asyncio.sleep(0.01)
    stream = fake_sd.streams[-1]

    got = await pump(stream, 1, blocksize=1024)
    await asyncio.wait_for(task, 1)

    assert np.all(got[100:] == 0.0)  # rest of the block is silence, not garbage


async def test_multi_chunk_queue_plays_back_to_back(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    a = np.full(300, 0.5, dtype=np.float32)
    b = np.full(300, 0.25, dtype=np.float32)

    task_a = asyncio.ensure_future(p.play(a))
    await asyncio.sleep(0.01)
    stream = fake_sd.streams[-1]

    # Enqueue the second chunk before the first has drained, so a single
    # callback block spans both chunks in the queue.
    task_b = asyncio.ensure_future(p.play(b))
    await asyncio.sleep(0.01)

    got = await pump(stream, 1, blocksize=1024)
    await asyncio.wait_for(task_a, 1)
    await asyncio.wait_for(task_b, 1)

    fade_n = 24000 * 3 // 1000
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    exp_a = a.copy(); exp_a[:fade_n] *= ramp; exp_a[-fade_n:] *= ramp[::-1]
    exp_b = b.copy(); exp_b[:fade_n] *= ramp; exp_b[-fade_n:] *= ramp[::-1]

    np.testing.assert_allclose(got[:300], exp_a, atol=1e-6)
    np.testing.assert_allclose(got[300:600], exp_b, atol=1e-6)


async def test_stream_reopened_after_open_error(monkeypatch):
    class FlakySD:
        def __init__(self):
            self.calls = 0
            self.streams = []

        def OutputStream(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("device busy")
            s = FakeStream(**kwargs)
            self.streams.append(s)
            return s

    fsd = FlakySD()
    monkeypatch.setattr(play_mod, "sd", fsd)

    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    with pytest.raises(RuntimeError):
        await p.play(np.ones(10, dtype=np.float32))
    assert p._stream is None

    task = asyncio.ensure_future(p.play(np.ones(500, dtype=np.float32)))
    await asyncio.sleep(0.01)
    stream = fsd.streams[-1]
    await pump(stream, 1)
    await asyncio.wait_for(task, 1)

    assert fsd.calls == 2


async def test_stream_open_retries_after_portaudio_reinit(monkeypatch):
    class SD:
        PortAudioError = play_mod._PortAudioError

        def __init__(self):
            self.calls = 0
            self.reinit = []
            self.streams = []

        def _terminate(self):
            self.reinit.append("terminate")

        def _initialize(self):
            self.reinit.append("initialize")

        def OutputStream(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise play_mod._PortAudioError("Internal PortAudio error", -9986)
            s = FakeStream(**kwargs)
            self.streams.append(s)
            return s

    fsd = SD()
    monkeypatch.setattr(play_mod, "sd", fsd)
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    task = asyncio.ensure_future(p.play(np.ones(500, dtype=np.float32)))
    await asyncio.sleep(0.01)
    assert fsd.reinit == ["terminate", "initialize"]
    assert fsd.calls == 2
    stream = fsd.streams[-1]
    await pump(stream, 1)
    await task


@pytest.mark.asyncio
async def test_close_stream_keeps_playing_and_reopens_on_next_play(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    p._timeout_margin_s = 0.2
    task = asyncio.create_task(p.play(np.ones(2048, dtype=np.float32) * 0.5))
    await asyncio.sleep(0.01)
    stream1 = fake_sd.streams[-1]
    p.close_stream()                    # device switched: drop the old stream
    stream1.finished_callback()         # PortAudio reports the stream finished
    await task                          # in-flight play() returns rather than hangs
    assert stream1.closed >= 1          # (play()'s inactive-stream path may close it again)
    assert p._stopped is False          # not a stop(): playback is still allowed

    task = asyncio.create_task(p.play(np.ones(2048, dtype=np.float32) * 0.5))
    await asyncio.sleep(0.01)
    assert len(fake_sd.streams) == 2    # next play() opened a fresh stream
    await pump(fake_sd.streams[-1], 2)
    await task


@pytest.mark.asyncio
async def test_registered_players_close_streams_on_refresh_hook(fake_sd):
    p = play_mod.Player(sample_rate=24000, blocksize=1024)
    p._timeout_margin_s = 0.2
    play_mod.register_for_refresh(p)
    play_mod.register_for_refresh(p)    # idempotent
    try:
        task = asyncio.create_task(p.play(np.ones(1024, dtype=np.float32)))
        await asyncio.sleep(0.01)
        play_mod.close_registered_streams()
        assert fake_sd.streams[-1].closed == 1
        assert p._stream is None and p._stopped is False
        fake_sd.streams[-1].finished_callback()
        await task
    finally:
        play_mod._registry.remove(p)


def test_close_stream_without_stream_is_noop(fake_sd):
    p = play_mod.Player()
    p.close_stream()
    assert fake_sd.streams == []
