import asyncio

import pytest

from veronica.tools.timers import TimerService


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


async def test_set_fires_on_fire_and_notify(monkeypatch):
    fired = []

    async def on_fire(text):
        fired.append(text)

    notified = []

    async def fake_notify_handler(args):
        notified.append(args)
        return {"content": [{"type": "text", "text": "ok"}]}

    from veronica.tools import mac
    monkeypatch.setattr(mac.notify, "handler", fake_notify_handler)

    svc = TimerService(on_fire=on_fire)
    svc.set(0.05 / 60, label="tea")  # 0.05s
    await asyncio.sleep(0.2)
    assert fired == ["Timer tea done"]
    assert notified and notified[0]["message"] == "Timer tea done"


async def test_set_no_label_message(monkeypatch):
    fired = []

    async def on_fire(text):
        fired.append(text)

    from veronica.tools import mac

    async def fake_notify_handler(args):
        return {"content": [{"type": "text", "text": "ok"}]}
    monkeypatch.setattr(mac.notify, "handler", fake_notify_handler)

    svc = TimerService(on_fire=on_fire)
    svc.set(0.05 / 60)
    await asyncio.sleep(0.2)
    assert fired == ["Timer  done"]


async def test_list_reports_remaining(monkeypatch):
    clock = FakeClock(100.0)

    async def on_fire(text):
        pass

    svc = TimerService(on_fire=on_fire, clock=clock)
    svc.set(1.0, label="x")  # fire_at = 100 + 60 = 160
    listed = svc.list()
    assert len(listed) == 1
    assert listed[0]["label"] == "x"
    assert listed[0]["remaining_s"] == pytest.approx(60.0)
    clock.t = 130.0
    listed = svc.list()
    assert listed[0]["remaining_s"] == pytest.approx(30.0)
    svc.cancel("x")


async def test_cancel_by_label_and_id(monkeypatch):
    async def on_fire(text):
        pass

    svc = TimerService(on_fire=on_fire)
    tid = svc.set(10.0, label="soup")
    assert svc.cancel("soup") is True
    assert svc.list() == []

    tid2 = svc.set(10.0, label="rice")
    assert svc.cancel(tid2) is True
    assert svc.list() == []

    assert svc.cancel("missing") is False


async def test_multiple_timers_independent(monkeypatch):
    fired = []

    async def on_fire(text):
        fired.append(text)

    from veronica.tools import mac

    async def fake_notify_handler(args):
        return {"content": [{"type": "text", "text": "ok"}]}
    monkeypatch.setattr(mac.notify, "handler", fake_notify_handler)

    svc = TimerService(on_fire=on_fire)
    svc.set(0.05 / 60, label="a")
    svc.set(0.05 / 60, label="b")
    await asyncio.sleep(0.2)
    assert sorted(fired) == ["Timer a done", "Timer b done"]


async def test_on_fire_exception_does_not_crash(monkeypatch):
    from veronica.tools import mac

    async def fake_notify_handler(args):
        return {"content": [{"type": "text", "text": "ok"}]}
    monkeypatch.setattr(mac.notify, "handler", fake_notify_handler)

    async def bad_on_fire(text):
        raise RuntimeError("boom")

    svc = TimerService(on_fire=bad_on_fire)
    svc.set(0.05 / 60, label="x")
    await asyncio.sleep(0.2)  # should not raise


@pytest.mark.live
async def test_live_timer_fires_within_3s():
    fired = asyncio.Event()
    texts = []

    async def on_fire(text):
        texts.append(text)
        fired.set()

    svc = TimerService(on_fire=on_fire)
    svc.set(0.02, label="live")
    await asyncio.wait_for(fired.wait(), timeout=3.0)
    assert texts == ["Timer live done"]
