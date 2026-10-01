import asyncio

import pytest

from veronica.audio import hotkey
from veronica.audio.hotkey import HotkeyMonitor


class FakeEvent:
    """A flags-changed event for `keycode`. `alt_down` is the *right*
    Option key's own device bit (NX_DEVICERALTKEYMASK); `left_alt_down`
    is the left Option key's. Either sets the generic Alternate bit, as
    the real CGEventFlags does."""
    def __init__(self, keycode: int, alt_down: bool, left_alt_down: bool = False):
        self.keycode = keycode
        self.alt_down = alt_down
        self.left_alt_down = left_alt_down


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
    enable_calls = []

    @classmethod
    def reset(cls, tap_result=None):
        cls.tap_result = tap_result if tap_result is not None else object()
        cls.stopped = []
        cls.run_calls = 0
        cls.enable_calls = []

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

    @classmethod
    def CGEventTapEnable(cls, tap, enabled):
        cls.enable_calls.append((tap, enabled))

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
        flags = 0
        if event.alt_down:
            flags |= FakeQuartz.kCGEventFlagMaskAlternate | hotkey.DEVICE_FLAG_MASKS[61]
        if event.left_alt_down:
            flags |= FakeQuartz.kCGEventFlagMaskAlternate | hotkey.DEVICE_FLAG_MASKS[58]
        return flags


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


async def test_left_option_held_does_not_mask_right_option_release(fake_quartz):
    """With left-Option held, the generic Alternate flag stays set when
    right-Option is released; the monitor must read right-Option's own
    device-specific bit (NX_DEVICERALTKEYMASK) so the release is seen."""
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    mon._setup_tap()

    # left option goes down first (keycode 58): not our key, ignored
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(58, False, left_alt_down=True), None)
    # right option down while left is still held
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, True, left_alt_down=True), None)
    # right option up, left still held: Alternate still set, device bit clear
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, False, left_alt_down=True), None)
    await asyncio.sleep(0)
    assert events == ["press", "release"]


async def test_left_option_events_never_toggle_state(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    mon._setup_tap()
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(58, False, left_alt_down=True), None)
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(58, False, left_alt_down=False), None)
    await asyncio.sleep(0)
    assert events == []


def test_unknown_keycode_falls_back_to_generic_alternate_mask(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=99)
    mon._setup_tap()
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(99, False, left_alt_down=True), None)
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(99, False, left_alt_down=False), None)
    assert events == ["press", "release"]


def test_tap_disabled_by_os_is_reenabled_and_logged_once(fake_quartz, caplog):
    mon = HotkeyMonitor(lambda: None, lambda: None)
    mon._setup_tap()
    assert fake_quartz.enable_calls == [(fake_quartz.tap_result, True)]  # initial enable
    with caplog.at_level("WARNING", logger="veronica.audio.hotkey"):
        mon._callback(None, hotkey.TAP_DISABLED_BY_TIMEOUT, None, None)
        mon._callback(None, hotkey.TAP_DISABLED_BY_USER_INPUT, None, None)
    assert fake_quartz.enable_calls == [(fake_quartz.tap_result, True)] * 3
    assert mon.reenable_count == 2
    assert sum("re-enabling" in r.message for r in caplog.records) == 1


def test_start_without_running_loop_calls_back_directly(fake_quartz):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon.start()  # no asyncio loop running in this (sync) test: must not raise
    assert mon._loop is None
    mon._dispatch(True)
    assert events == ["press"]
    mon.stop()


async def test_reenable_tap_releases_a_held_key(fake_quartz):
    """A key-up can be missed while the tap is disabled; re-enabling must
    not leave the monitor (and push-to-talk) believing the key is down."""
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode=61)
    mon._loop = asyncio.get_running_loop()
    mon._setup_tap()
    mon._callback(None, fake_quartz.kCGEventFlagsChanged, FakeEvent(61, True), None)
    mon._callback(None, hotkey.TAP_DISABLED_BY_TIMEOUT, None, None)
    await asyncio.sleep(0)
    assert events == ["press", "release"]
    assert mon._pressed is False
    mon._callback(None, hotkey.TAP_DISABLED_BY_USER_INPUT, None, None)   # nothing held: no extra release
    await asyncio.sleep(0)
    assert events == ["press", "release"]
