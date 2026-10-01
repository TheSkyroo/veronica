import asyncio
import ctypes
import threading

import pytest

from veronica.audio import hotkey
from veronica.audio.hotkey import KBDLLHOOKSTRUCT, HotkeyMonitor


class FakeWin32:
    """Minimal fake of the user32/kernel32 surface HotkeyMonitor touches.
    GetMessageW blocks until PostThreadMessageW(WM_QUIT) arrives."""

    def __init__(self, hook_result=0xBEEF):
        self.hook_result = hook_result
        self.installed = []
        self.unhooked = []
        self.posted = []
        self.next_calls = []
        self.get_calls = 0
        self._quit = threading.Event()

    @staticmethod
    def HOOKPROC(fn):
        return fn

    def SetWindowsHookExW(self, id_hook, proc, hmod, thread_id):
        self.installed.append((id_hook, proc, hmod, thread_id))
        return self.hook_result

    def CallNextHookEx(self, hook, n_code, w_param, l_param):
        self.next_calls.append((n_code, w_param))
        return 0

    def UnhookWindowsHookEx(self, hook):
        self.unhooked.append(hook)
        return 1

    def GetMessageW(self, msg_ref, hwnd, lo, hi):
        self.get_calls += 1
        self._quit.wait(5)
        return 0

    def PostThreadMessageW(self, tid, msg, w, l):
        self.posted.append((tid, msg))
        if msg == hotkey.WM_QUIT:
            self._quit.set()
        return 1

    @staticmethod
    def GetModuleHandleW(name):
        return 0x400000

    @staticmethod
    def GetCurrentThreadId():
        return 4242


@pytest.fixture
def fake_win32(monkeypatch):
    fake = FakeWin32()
    monkeypatch.setattr(HotkeyMonitor, "_import_win32", staticmethod(lambda: fake))
    return fake


def key(mon, message, vk, flags=0, n_code=hotkey.HC_ACTION):
    """Drive the hook proc with a real KBDLLHOOKSTRUCT, as Windows would."""
    kb = KBDLLHOOKSTRUCT(vkCode=vk, scanCode=0, flags=flags, time=0, dwExtraInfo=0)
    return mon._hook_proc(n_code, message, ctypes.addressof(kb))


def down(mon, vk=hotkey.VK_RCONTROL, flags=0):
    return key(mon, hotkey.WM_KEYDOWN, vk, flags)


def up(mon, vk=hotkey.VK_RCONTROL, flags=0):
    return key(mon, hotkey.WM_KEYUP, vk, flags)


def test_default_key_is_right_ctrl():
    assert HotkeyMonitor(lambda: None, lambda: None)._keycode == hotkey.VK_RCONTROL == 0xA3


@pytest.mark.parametrize("value, vk", [
    (0xA5, 0xA5), ("right_alt", hotkey.VK_RMENU), ("Right_Ctrl", hotkey.VK_RCONTROL),
    (" f13 ", hotkey.VK_F13), ("fn", hotkey.VK_RCONTROL), ("bogus", hotkey.VK_RCONTROL),
])
def test_resolve_keycode(value, vk):
    assert hotkey.resolve_keycode(value) == vk


def test_install_hook_success_sets_available_true(fake_win32):
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._install_hook() is True
    assert mon.available is True
    (id_hook, proc, hmod, tid), = fake_win32.installed
    assert id_hook == hotkey.WH_KEYBOARD_LL and hmod == 0x400000 and tid == 0
    assert mon._proc is proc                       # kept alive for the hook's lifetime


def test_install_hook_null_handle_sets_available_false(fake_win32):
    fake_win32.hook_result = None
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._install_hook() is False
    assert mon.available is False


def test_install_hook_no_user32_sets_available_false(monkeypatch):
    def boom():
        raise OSError("no user32")

    monkeypatch.setattr(HotkeyMonitor, "_import_win32", staticmethod(boom))
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._install_hook() is False
    assert mon.available is False


def test_real_win32_unavailable_off_windows():
    # ctypes.WinDLL doesn't exist here: the real import fails, cleanly.
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._install_hook() is False and mon.available is False


async def test_hook_dispatches_press_and_release_and_chains(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon._loop = asyncio.get_running_loop()
    assert mon._install_hook() is True

    down(mon)
    await asyncio.sleep(0)
    assert events == ["press"]
    up(mon)
    await asyncio.sleep(0)
    assert events == ["press", "release"]
    # every event is passed on to the next hook
    assert fake_win32.next_calls == [(0, hotkey.WM_KEYDOWN), (0, hotkey.WM_KEYUP)]


async def test_syskey_messages_count_too(fake_win32):
    """Alt combos arrive as WM_SYSKEYDOWN/UP."""
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), keycode="right_alt")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    key(mon, hotkey.WM_SYSKEYDOWN, hotkey.VK_RMENU)
    key(mon, hotkey.WM_SYSKEYUP, hotkey.VK_RMENU)
    await asyncio.sleep(0)
    assert events == ["press", "release"]


async def test_hook_ignores_other_keys(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon, vk=hotkey.VK_LCONTROL)
    up(mon, vk=hotkey.VK_LCONTROL)
    await asyncio.sleep(0)
    assert events == []


async def test_left_ctrl_held_does_not_mask_right_ctrl_release(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon, vk=hotkey.VK_LCONTROL)
    down(mon)
    up(mon)
    await asyncio.sleep(0)
    assert events == ["press", "release"]


async def test_hook_ignores_autorepeat_down_events(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon)
    down(mon)
    down(mon)
    await asyncio.sleep(0)
    assert events == ["press"]


async def test_hook_ignores_injected_events(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon, flags=hotkey.LLKHF_INJECTED)
    await asyncio.sleep(0)
    assert events == []


async def test_hook_ignores_non_action_codes_but_still_chains(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    key(mon, hotkey.WM_KEYDOWN, hotkey.VK_RCONTROL, n_code=-1)
    await asyncio.sleep(0)
    assert events == []
    assert fake_win32.next_calls == [(-1, hotkey.WM_KEYDOWN)]


def test_hook_survives_callback_errors(fake_win32):
    def bad():
        raise RuntimeError("orchestrator gone")

    mon = HotkeyMonitor(bad, lambda: None)
    mon._install_hook()
    assert down(mon) == 0                            # no exception escapes into Windows
    assert fake_win32.next_calls == [(0, hotkey.WM_KEYDOWN)]


async def test_start_and_stop_real_thread(fake_win32):
    mon = HotkeyMonitor(lambda: None, lambda: None)
    mon.start()
    assert mon.available is True
    assert mon._thread_id == 4242
    mon.stop()
    assert not mon._thread.is_alive()
    assert fake_win32.posted == [(4242, hotkey.WM_QUIT)]
    assert fake_win32.unhooked == [0xBEEF]
    assert fake_win32.get_calls == 1


async def test_start_unavailable_does_not_hang(monkeypatch):
    def boom():
        raise OSError("no user32")

    monkeypatch.setattr(HotkeyMonitor, "_import_win32", staticmethod(boom))
    mon = HotkeyMonitor(lambda: None, lambda: None)
    mon.start()
    assert mon.available is False
    mon.stop()  # must not raise even though the hook was never installed


def test_start_without_running_loop_calls_back_directly(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon.start()  # no asyncio loop running in this (sync) test: must not raise
    assert mon._loop is None
    mon._dispatch(True)
    assert events == ["press"]
    mon.stop()


def test_stop_while_held_releases_the_key(fake_win32):
    """The key-up can never arrive once the hook is gone: don't leave
    push-to-talk believing the key is still down."""
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"))
    mon.start()
    down(mon)
    mon.stop()
    assert events == ["press", "release"]
    assert mon._pressed is False
