import asyncio

import pytest

from veronica.audio import hotkey
from veronica.audio.hotkey import HotkeyMonitor


class FakeEvent:
    def __init__(self, keycode: int, alt_down: bool):
        self.keycode = keycode
        self.alt_down = alt_down


class FakeQuartz:
    """Minimal fake of the Quartz surface HotkeyMonitor touches."""
    kCGSessionEventTap = 0
    kCGHeadInsertEventTap = 0
    kCGEventTapOptionListenOnly = 0
    kCGEventFlagsChanged = 12
    kCGKeyboardEventKeycode = 9
    kCGEventFlagMaskAlternate = 0x00080000
    kCFRunLoopCommonModes = "common"

    tap_result = object()
    stopped = []
    run_calls = 0

    @classmethod
    def reset(cls, tap_result=None):
        cls.tap_result = tap_result if tap_result is not None else object()
        cls.stopped = []
        cls.run_calls = 0

    @staticmethod
    def CGEventMaskBit(bit):
        return 1 << bit

    @classmethod
    def CGEventTapCreate(cls, *a, **k):
        return cls.tap_result

    @staticmethod
    def CFMachPortCreateRunLoopSource(a, tap, b):
        return object()

    @staticmethod
    def CFRunLoopGetCurrent():
        return "the-run-loop"

    @staticmethod
    def CFRunLoopAddSource(*a):
        pass

    @staticmethod
    def CGEventTapEnable(*a):
        pass

    @classmethod
    def CFRunLoopRun(cls):
        cls.run_calls += 1

    @classmethod
    def CFRunLoopStop(cls, rl):
        cls.stopped.append(rl)

    @staticmethod
    def CGEventGetIntegerValueField(event, field):
        return event.keycode

    @staticmethod
    def CGEventGetFlags(event):
        return FakeQuartz.kCGEventFlagMaskAlternate if event.alt_down else 0


@pytest.fixture
def fake_quartz(monkeypatch):
    FakeQuartz.reset()
    monkeypatch.setattr(HotkeyMonitor, "_import_quartz", staticmethod(lambda: FakeQuartz))
    return FakeQuartz


def test_setup_tap_success_sets_available_true(fake_quartz):
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._setup_tap() is True
    assert mon.available is True


def test_setup_tap_none_result_sets_available_false(fake_quartz):
    fake_quartz.tap_result = None
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._setup_tap() is False
    assert mon.available is False


def test_setup_tap_no_quartz_module_sets_available_false(monkeypatch):
    def boom():
        raise ImportError("no Quartz")

    monkeypatch.setattr(HotkeyMonitor, "_import_quartz", staticmethod(boom))
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._setup_tap() is False
    assert mon.available is False


async def test_callback_dispatches_press_and_release(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    assert mon._setup_tap() is True

    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, True), None)
    await asyncio.sleep(0)
    assert events == ["press"]

    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, False), None)
    await asyncio.sleep(0)
    assert events == ["press", "release"]


async def test_callback_ignores_other_keycodes(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    mon._setup_tap()

    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(60, True), None)
    await asyncio.sleep(0)
    assert events == []


async def test_callback_ignores_repeat_down_events(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    mon._setup_tap()

    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, True), None)
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, True), None)
    await asyncio.sleep(0)
    assert events == ["press"]


async def test_callback_ignores_other_event_types(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    mon._setup_tap()

    mon._callback(None, 99, FakeEvent(61, True), None)
    await asyncio.sleep(0)
    assert events == []


async def test_start_and_stop_real_thread(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon.start()
    assert mon.available is True
    assert fake_quartz.run_calls == 1
    mon.stop()
    assert fake_quartz.stopped == ["the-run-loop"]


async def test_start_unavailable_does_not_hang(monkeypatch):
    def boom():
        raise ImportError("no Quartz")

    monkeypatch.setattr(HotkeyMonitor, "_import_quartz", staticmethod(boom))
    mon = HotkeyMonitor(lambda: None, lambda: None)
    mon.start()
    assert mon.available is False
    mon.stop()  # must not raise even though the tap was never set up
