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


def test_default_hotkeys_are_copilot_and_ctrl_alt_space():
    mon = HotkeyMonitor(lambda: None, lambda: None)
    assert mon._combos == [(frozenset({"win", "shift"}), 0x86), (frozenset({"ctrl", "alt"}), hotkey.VK_SPACE)]
    assert mon.description == "Copilot key or Ctrl+Alt+Space"


@pytest.mark.parametrize("spec, mods, vk", [
    ("win+space", {"win"}, hotkey.VK_SPACE),
    ("Windows + Space", {"win"}, hotkey.VK_SPACE),
    ("ctrl+alt+p", {"ctrl", "alt"}, ord("P")),
    ("copilot", {"win", "shift"}, 0x86),          # F23
    ("right_ctrl", set(), hotkey.VK_RCONTROL),
    ("f9", set(), 0x78),
    (0xA5, set(), 0xA5),
])
def test_parse_hotkey(spec, mods, vk):
    assert hotkey.parse_hotkey(spec) == (frozenset(mods), vk)


@pytest.mark.parametrize("spec", ["fn", "win+ctrl", "hyper+space", ""])
def test_parse_hotkey_rejects_unusable_keys(spec):
    with pytest.raises(ValueError):
        hotkey.parse_hotkey(spec)


def test_parse_hotkeys_skips_bad_entries_and_falls_back_when_none_left():
    assert hotkey.parse_hotkeys("fn, right_ctrl") == [(frozenset(), hotkey.VK_RCONTROL)]
    assert hotkey.parse_hotkeys("fn") == hotkey.parse_hotkeys(hotkey.DEFAULT_HOTKEY)


@pytest.mark.parametrize("spec, text", [
    ("win+space", "Win+Space"), ("alt+shift+f5", "Alt+Shift+F5"), ("right_ctrl", "Right Ctrl"),
    ("copilot", "Copilot key"), ("win+shift+f23", "Copilot key"),
    ("copilot, ctrl+alt+space", "Copilot key or Ctrl+Alt+Space"),
])
def test_describe_hotkey(spec, text):
    assert hotkey.describe_hotkey(spec) == text


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
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
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
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_alt")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    key(mon, hotkey.WM_SYSKEYDOWN, hotkey.VK_RMENU)
    key(mon, hotkey.WM_SYSKEYUP, hotkey.VK_RMENU)
    await asyncio.sleep(0)
    assert events == ["press", "release"]


async def test_hook_ignores_other_keys(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon, vk=hotkey.VK_LCONTROL)
    up(mon, vk=hotkey.VK_LCONTROL)
    await asyncio.sleep(0)
    assert events == []


async def test_left_ctrl_held_does_not_mask_right_ctrl_release(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon, vk=hotkey.VK_LCONTROL)
    down(mon)
    up(mon)
    await asyncio.sleep(0)
    assert events == ["press", "release"]


async def test_hook_ignores_autorepeat_down_events(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon)
    down(mon)
    down(mon)
    await asyncio.sleep(0)
    assert events == ["press"]


async def test_hook_ignores_injected_events(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    down(mon, flags=hotkey.LLKHF_INJECTED)
    await asyncio.sleep(0)
    assert events == []


async def test_hook_ignores_non_action_codes_but_still_chains(fake_win32):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon._loop = asyncio.get_running_loop()
    mon._install_hook()
    key(mon, hotkey.WM_KEYDOWN, hotkey.VK_RCONTROL, n_code=-1)
    await asyncio.sleep(0)
    assert events == []
    assert fake_win32.next_calls == [(-1, hotkey.WM_KEYDOWN)]


def test_hook_survives_callback_errors(fake_win32):
    def bad():
        raise RuntimeError("orchestrator gone")

    mon = HotkeyMonitor(bad, lambda: None, hotkey="right_ctrl")
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
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon.start()  # no asyncio loop running in this (sync) test: must not raise
    assert mon._loop is None
    mon._dispatch(True)
    assert events == ["press"]
    mon.stop()


def test_stop_while_held_releases_the_key(fake_win32):
    """The key-up can never arrive once the hook is gone: don't leave
    push-to-talk believing the key is still down."""
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey="right_ctrl")
    mon.start()
    down(mon)
    mon.stop()
    assert events == ["press", "release"]
    assert mon._pressed is False


# -- combos (Win+Space) -----------------------------------------------------------
def combo(fake_win32, spec="win+space"):
    events = []
    mon = HotkeyMonitor(lambda: events.append("press"), lambda: events.append("release"), hotkey=spec)
    mon._install_hook()
    mon._thread_id = 4242
    return mon, events


SWALLOWED = 1


def test_win_space_hold_and_release_space(fake_win32):
    mon, events = combo(fake_win32)
    assert down(mon, vk=hotkey.VK_LWIN) == 0                 # Win passes through
    assert down(mon, vk=hotkey.VK_SPACE) == SWALLOWED        # no layout switch
    assert down(mon, vk=hotkey.VK_SPACE) == SWALLOWED        # auto-repeat swallowed too
    assert up(mon, vk=hotkey.VK_SPACE) == SWALLOWED
    assert up(mon, vk=hotkey.VK_LWIN) == 0
    assert events == ["press", "release"]
    # the Start-menu mask was requested once, while Win was down
    assert fake_win32.posted == [(4242, hotkey.WM_APP_MASK)]


def test_releasing_win_first_ends_the_hold(fake_win32):
    mon, events = combo(fake_win32)
    down(mon, vk=hotkey.VK_RWIN)
    down(mon, vk=hotkey.VK_SPACE)
    up(mon, vk=hotkey.VK_RWIN)
    assert events == ["press", "release"]
    assert up(mon, vk=hotkey.VK_SPACE) == SWALLOWED          # its down was ours
    assert events == ["press", "release"]


def test_plain_space_is_never_touched(fake_win32):
    mon, events = combo(fake_win32)
    assert down(mon, vk=hotkey.VK_SPACE) == 0
    assert up(mon, vk=hotkey.VK_SPACE) == 0
    assert events == [] and fake_win32.posted == []


def test_other_win_shortcuts_pass_through(fake_win32):
    mon, events = combo(fake_win32)
    down(mon, vk=hotkey.VK_LWIN)
    assert down(mon, vk=ord("E")) == 0                       # Win+E still opens Explorer
    assert up(mon, vk=ord("E")) == 0
    assert events == []


def test_injected_keys_are_ignored_for_combos(fake_win32):
    mon, events = combo(fake_win32)
    down(mon, vk=hotkey.VK_LWIN, flags=hotkey.LLKHF_INJECTED)
    assert down(mon, vk=hotkey.VK_SPACE) == 0                # Win wasn't really held
    assert events == []


def test_ctrl_combo_needs_no_mask(fake_win32):
    mon, events = combo(fake_win32, "ctrl+shift+space")
    down(mon, vk=hotkey.VK_LCONTROL)
    down(mon, vk=hotkey.VK_RSHIFT)
    assert down(mon, vk=hotkey.VK_SPACE) == SWALLOWED
    up(mon, vk=hotkey.VK_SPACE)
    assert events == ["press", "release"] and fake_win32.posted == []


def test_mask_message_injects_the_mask_key(fake_win32):
    sent = []
    fake_win32.SendInput = lambda n, ptr, size: sent.append((n, size)) or n
    mon, _ = combo(fake_win32)
    mon._send_mask()
    assert sent == [(2, ctypes.sizeof(hotkey._INPUT))]
    inputs = hotkey._mask_inputs()
    assert [i.ki.wVk for i in inputs] == [hotkey.VK_MASK] * 2
    assert [i.ki.dwFlags for i in inputs] == [0, hotkey.KEYEVENTF_KEYUP]


# -- several keys / the Copilot key ---------------------------------------------------

VK_F23 = 0x86


def test_copilot_key_is_a_combo_and_is_swallowed(fake_win32):
    """The Copilot key arrives as LWin + LShift + F23: F23 must not reach
    Windows (it would open Copilot), and the Win release must be masked."""
    mon, events = combo(fake_win32, "copilot")
    down(mon, vk=hotkey.VK_LWIN)
    down(mon, vk=hotkey.VK_LSHIFT)
    assert down(mon, vk=VK_F23) == SWALLOWED
    assert up(mon, vk=VK_F23) == SWALLOWED
    up(mon, vk=hotkey.VK_LSHIFT)
    up(mon, vk=hotkey.VK_LWIN)
    assert events == ["press", "release"]
    assert fake_win32.posted == [(4242, hotkey.WM_APP_MASK)]


def test_either_of_two_hotkeys_works(fake_win32):
    mon, events = combo(fake_win32, "copilot, ctrl+alt+space")
    down(mon, vk=hotkey.VK_LCONTROL)
    down(mon, vk=hotkey.VK_LMENU)
    assert down(mon, vk=hotkey.VK_SPACE) == SWALLOWED
    assert up(mon, vk=hotkey.VK_SPACE) == SWALLOWED
    assert events == ["press", "release"]
    assert fake_win32.posted == [(4242, hotkey.WM_APP_MASK)]     # alt is held: masked


def test_a_second_hotkey_during_a_hold_does_not_double_press(fake_win32):
    mon, events = combo(fake_win32, "right_ctrl, ctrl+alt+space")
    down(mon, vk=hotkey.VK_RCONTROL)                 # right ctrl alone: press
    down(mon, vk=hotkey.VK_LCONTROL)
    down(mon, vk=hotkey.VK_LMENU)
    assert down(mon, vk=hotkey.VK_SPACE) == SWALLOWED  # also matches, but already held
    up(mon, vk=hotkey.VK_SPACE)
    assert events == ["press"]                        # owned by right ctrl, not released
    up(mon, vk=hotkey.VK_RCONTROL)
    assert events == ["press", "release"]
