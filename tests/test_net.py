import asyncio
import time

import pytest

from veronica import net


@pytest.fixture(autouse=True)
def _clean():
    net.forget()
    yield
    net.forget()


async def test_online_probes_the_vendor_host():
    seen = []
    assert await net.online("api.openai.com", connect=lambda h, t: seen.append((h, t)) or True)
    assert seen == [("api.openai.com", 1.5)]


async def test_offline_when_the_connect_fails():
    assert await net.online("api.anthropic.com", connect=lambda h, t: False) is False


async def test_the_answer_is_cached_then_re_probed():
    calls, now = [], [100.0]
    probe = lambda h, t: calls.append(h) or True     # noqa: E731
    for _ in range(3):
        await net.online("api.openai.com", connect=probe, clock=lambda: now[0])
    assert calls == ["api.openai.com"]
    now[0] += net.CACHE_S + 1
    await net.online("api.openai.com", connect=probe, clock=lambda: now[0])
    assert len(calls) == 2


async def test_each_host_is_cached_on_its_own():
    calls = []
    for host in ("api.openai.com", "api.anthropic.com", "api.openai.com"):
        await net.online(host, connect=lambda h, t: calls.append(h) or True)
    assert calls == ["api.openai.com", "api.anthropic.com"]


def test_every_networked_brain_has_a_host():
    from veronica.brain.backends import BACKENDS

    assert set(net.VENDOR_HOSTS) == set(BACKENDS) - {"local"}


async def test_a_socket_error_reads_as_offline(monkeypatch):
    def boom(addr, timeout=None):
        raise OSError("nodename nor servname provided")

    monkeypatch.setattr(net.socket, "create_connection", boom)
    assert await net.online("api.openai.com") is False


async def test_the_probe_never_blocks_the_event_loop():
    """The connect is a blocking socket call made once a turn; on dead Wi-Fi
    it costs the whole timeout, and audio, the HUD and the barge listener
    all have to keep running through it."""
    ticks = []

    async def ticker():
        for _ in range(6):
            await asyncio.sleep(0.005)
            ticks.append(1)

    spin = asyncio.ensure_future(ticker())
    assert await net.online("api.openai.com", connect=lambda h, t: time.sleep(0.06) or True)
    await spin
    assert len(ticks) >= 3
