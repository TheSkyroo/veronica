import asyncio
import numpy as np
import pytest

from veronica.audio import play as play_mod


class FakeSD:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, samples, samplerate):
        self.played.append((len(samples), samplerate))

    def wait(self):
        pass

    def stop(self):
        self.stopped += 1


@pytest.fixture
def fake_sd(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(play_mod, "sd", sd)
    return sd


async def test_play_sends_samples(fake_sd):
    p = play_mod.Player(sample_rate=24000)
    await p.play(np.zeros(2400, dtype=np.float32))
    assert fake_sd.played == [(2400, 24000)]
    assert p.is_playing is False


async def test_stop_calls_sd_stop(fake_sd):
    p = play_mod.Player()
    p.stop()
    assert fake_sd.stopped == 1


async def test_stop_skips_queued(fake_sd):
    p = play_mod.Player()
    p.stop()  # sets stopped flag before play
    await p.play(np.zeros(10, dtype=np.float32))
    assert fake_sd.played == []
