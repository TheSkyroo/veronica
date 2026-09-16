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
        self.started = 0
        self.stopped = 0
        self.closed = 0

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

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
