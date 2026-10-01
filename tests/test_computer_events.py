import pytest

from veronica.tools import computer_events as ce
from veronica.tools.computer_events import KEYEVENTF_EXTENDEDKEY as EXT
from veronica.tools.computer_events import KEYEVENTF_KEYUP as UP
from veronica.tools.computer_events import KEYEVENTF_UNICODE as UNI
from veronica.tools.computer_events import Front, KeyInput, MouseInput

ABS = ce.MOUSEEVENTF_MOVE | ce.MOUSEEVENTF_ABSOLUTE | ce.MOUSEEVENTF_VIRTUALDESK
LDOWN, LUP = ce.MOUSEEVENTF_LEFTDOWN, ce.MOUSEEVENTF_LEFTUP
RDOWN, RUP = ce.MOUSEEVENTF_RIGHTDOWN, ce.MOUSEEVENTF_RIGHTUP
MDOWN, MUP = ce.MOUSEEVENTF_MIDDLEDOWN, ce.MOUSEEVENTF_MIDDLEUP
CTRL, ALT, SHIFT, WIN = 0xA2, 0xA4, 0xA0, 0x5B

# --- fakes -----------------------------------------------------------------


class FakeWin32:
    """Records SendInput batches; answers window/process questions from
    plain dicts. Virtual screen: a 1920×1080 primary plus a 1920×1080
    monitor to its left (x from -1920), like a real two-monitor desk."""

    def __init__(self, vscreen=(-1920, 0, 3840, 1080)):
        self.batches: list[list] = []
        self.vscreen = vscreen
        self.fg = 0
        self.titles: dict[int, str] = {}
        self.classes: dict[int, str] = {}
        self.pids: dict[int, int] = {}
        self.paths: dict[int, str] = {}
        self.names: dict[str, str] = {}
        self.uwp_child: dict[int, int] = {}
        self.elevated: dict[int, bool | None] = {}
        self.me_elevated = False
        self.short_by = 0
        self.dpi_calls = 0
        self.cursor: list[tuple[int, int]] = []

    def set_dpi_awareness(self):
        self.dpi_calls += 1
        return "per-monitor-v2"

    def send_input(self, events):
        self.batches.append(list(events))
        return len(events) - self.short_by

    def set_cursor_pos(self, x, y):
        self.cursor.append((x, y))

    def virtual_screen(self):
        return self.vscreen

    def foreground_window(self):
        return self.fg

    def window_title(self, hwnd):
        return self.titles.get(hwnd, "")

    def window_class(self, hwnd):
        return self.classes.get(hwnd, "")

    def window_pid(self, hwnd):
        return self.pids.get(hwnd, 0)

    def process_path(self, pid):
        return self.paths.get(pid, "")

    def app_name(self, path):
        return self.names.get(path, "")

    def uwp_child_pid(self, hwnd):
        return self.uwp_child.get(hwnd)

    def self_elevated(self):
        return self.me_elevated

    def process_elevated(self, pid):
        return self.elevated.get(pid)

    # helpers
    @property
    def posted(self):
        return [e for b in self.batches for e in b]


@pytest.fixture
def win(monkeypatch):
    w = FakeWin32()
    monkeypatch.setattr(ce, "_win32", lambda: w)
    sleeps = []
    monkeypatch.setattr(ce, "_sleep", sleeps.append)
    w.sleeps = sleeps
    return w


def _mouse(w):
    return [(e.flags & ~ABS, e.x, e.y) for e in w.posted if isinstance(e, MouseInput)]


def _keys(w):
    return [(e.vk, e.flags) for e in w.posted if isinstance(e, KeyInput)]


# --- DPI awareness ---------------------------------------------------------

def test_ensure_dpi_awareness_is_idempotent(monkeypatch):
    w = FakeWin32()
    monkeypatch.setattr(ce, "_win32", lambda: w)
    monkeypatch.setattr(ce, "_dpi_state", None)
    assert ce.ensure_dpi_awareness() == "per-monitor-v2"
    assert ce.ensure_dpi_awareness() == "per-monitor-v2"
    assert w.dpi_calls == 1


def test_ensure_dpi_awareness_never_raises(monkeypatch):
    def boom():
        raise AttributeError("no windll here")
    monkeypatch.setattr(ce, "_win32", boom)
    monkeypatch.setattr(ce, "_dpi_state", None)
    assert ce.ensure_dpi_awareness() == "unavailable"


def test_real_backend_is_unavailable_off_windows(monkeypatch):
    """Nothing Windows-only at import time; the backend fails only when used."""
    import sys
    if sys.platform == "win32":
        pytest.skip("real Win32 here")
    monkeypatch.setattr(ce, "_backend", None)
    with pytest.raises((AttributeError, OSError)):
        ce._win32()
    assert ce._backend is None


# --- coordinates -------------------------------------------------------------

def test_absolute_normalises_over_the_virtual_desktop(win):
    # 3840 px wide from -1920: the primary's (0, 0) is half way across
    nx, ny = ce._absolute(0, 0)
    assert 32768 <= nx < 32768 + 17 and ny < 61
    assert ce._absolute(-1920, 0)[0] < 17
    assert ce._absolute(1919, 1079)[0] <= 65535


@pytest.mark.parametrize("x", [-1920, -1, 0, 1, 777, 1919])
def test_absolute_maps_back_onto_the_same_pixel(win, x):
    """Windows maps n back to a pixel as n * width / 65536 (move() then
    snaps with SetCursorPos anyway, in case it rounds the other way)."""
    vx, _vy, vw, _vh = win.vscreen
    nx, _ = ce._absolute(x, 0)
    assert nx * vw // 65536 + vx == x


def test_absolute_clamps_off_screen_points(win):
    assert ce._absolute(-5000, -5) == (0, 0)
    assert ce._absolute(99999, 99999) == (65535, 65535)


# --- mouse -------------------------------------------------------------------

def test_move_sends_an_absolute_virtual_desk_move_then_snaps_the_cursor(win):
    ce.move(-100.4, 20)
    assert len(win.batches) == 1
    (e,) = win.batches[0]
    assert e.flags == ABS and (e.x, e.y) == (-100.4, 20)
    assert (e.dx, e.dy) == ce._absolute(-100, 20)
    assert win.cursor == [(-100, 20)]


def test_click_moves_then_down_up_at_the_point(win):
    ce.click(100, 200)
    assert _mouse(win) == [(0, 100, 200), (LDOWN, 100, 200), (LUP, 100, 200)]
    # the buttons press wherever the cursor was snapped to
    assert [e.flags for e in win.posted] == [ABS, LDOWN, LUP]
    assert win.cursor == [(100, 200)]


def test_click_right_and_middle_buttons(win):
    ce.click(1, 2, button="right")
    ce.click(3, 4, button="middle")
    assert _mouse(win) == [
        (0, 1, 2), (RDOWN, 1, 2), (RUP, 1, 2),
        (0, 3, 4), (MDOWN, 3, 4), (MUP, 3, 4),
    ]


def test_click_unknown_button(win):
    with pytest.raises(ValueError):
        ce.click(1, 2, button="thumb")
    assert win.posted == []


def test_double_click_sends_two_pairs_with_a_short_gap(win):
    ce.click(5, 6, double=True)
    assert [f for f, *_ in _mouse(win)] == [0, LDOWN, LUP, LDOWN, LUP]
    assert win.sleeps == [ce.CLICK_GAP_S]


def test_drag_posts_down_eight_moves_up(win):
    ce.drag(0, 0, 80, 40)
    ev = _mouse(win)
    assert ev[0] == (0, 0, 0)
    assert ev[1] == (LDOWN, 0, 0)
    moves = ev[2:-2]
    assert len(moves) == 8 and all(f == 0 for f, *_ in moves)
    assert [x for _, x, _ in moves] == pytest.approx([10, 20, 30, 40, 50, 60, 70, 80])
    assert [y for _, _, y in moves] == pytest.approx([5, 10, 15, 20, 25, 30, 35, 40])
    assert ev[-2:] == [(0, 80, 40), (LUP, 80, 40)]   # released on the end point
    assert win.cursor[-1] == (80, 40)
    assert sum(win.sleeps) == pytest.approx(0.2, abs=0.05)


def test_scroll_moves_then_turns_the_wheel_in_whole_notches(win):
    ce.scroll(50, 60, dx=-30, dy=250)
    assert _mouse(win)[0] == (0, 50, 60)
    wheel = win.batches[-1]
    assert [(e.flags, e.data) for e in wheel] == [
        (ce.MOUSEEVENTF_WHEEL, 2 * ce.WHEEL_DELTA),      # 250 px forward → 2 notches up
        (ce.MOUSEEVENTF_HWHEEL, -ce.WHEEL_DELTA),        # a small tilt is still one notch
    ]


def test_scroll_zero_only_moves(win):
    ce.scroll(1, 2)
    assert len(win.batches) == 1


def test_post_raises_when_windows_inserts_fewer_events(win):
    win.short_by = 1
    with pytest.raises(OSError, match="inserted 0 of 1"):
        ce.move(1, 1)


# --- keyboard ----------------------------------------------------------------

def _typed(w) -> str:
    units = [e.scan for e in w.posted if isinstance(e, KeyInput) and e.flags == UNI]
    return b"".join(u.to_bytes(2, "little") for u in units).decode("utf-16-le")


def test_type_text_sends_unicode_units_down_up_in_chunk_batches(win):
    text = "hello world, नमस्ते — आप कैसे हैं? all good here"
    ce.type_text(text)
    assert len(win.batches) == len(ce._chunks(text)) >= 2
    keys = win.posted
    assert all(e.vk == 0 for e in keys)
    assert [e.flags for e in keys[0::2]] == [UNI] * (len(keys) // 2)
    assert [e.flags for e in keys[1::2]] == [UNI | UP] * (len(keys) // 2)
    assert _typed(win) == text
    assert all(len(b) <= 2 * ce.TYPE_CHUNK_UTF16 for b in win.batches)
    assert win.sleeps == [ce.TYPE_CHUNK_GAP_S] * (len(win.batches) - 1)


def test_type_text_astral_characters_are_two_units(win):
    ce.type_text("😀")
    assert [e.scan for e in win.posted] == [0xD83D, 0xD83D, 0xDE00, 0xDE00]
    assert _typed(win) == "😀"


def test_type_text_chunks_never_split_surrogate_pairs():
    text = "a" * 19 + "😀" + "b"
    chunks = ce._chunks(text)
    assert "".join(chunks) == text
    assert all(len(c.encode("utf-16-le")) // 2 <= 20 for c in chunks)
    assert chunks[0] == "a" * 19  # the emoji (2 units) would not fit


def test_type_text_newlines_and_tabs_are_real_key_presses(win):
    ce.type_text("a\r\nb\tc\n")
    vk = [(e.vk, e.flags) for e in win.posted if e.vk]
    assert vk == [(0x0D, 0), (0x0D, UP), (0x09, 0), (0x09, UP), (0x0D, 0), (0x0D, UP)]
    assert _typed(win) == "abc"


def test_type_text_empty_posts_nothing(win):
    ce.type_text("")
    assert win.posted == []


def test_key_combo_downs_then_ups_in_reverse(win):
    ce.key("ctrl+shift+s")
    assert win.batches == [
        [KeyInput(CTRL), KeyInput(SHIFT), KeyInput(ord("S"))],
        [KeyInput(ord("S"), UP), KeyInput(SHIFT, UP), KeyInput(CTRL, UP)],
    ]


def test_cmd_means_ctrl(win):
    ce.key("cmd+c")
    ce.key("command+v")
    assert [vk for vk, f in _keys(win) if not f & UP] == [CTRL, ord("C"), CTRL, ord("V")]


def test_win_combos_and_extended_keys(win):
    ce.key("win+d")
    ce.key("ctrl+alt+left")
    assert win.batches[0] == [KeyInput(WIN, EXT), KeyInput(ord("D"))]
    assert win.batches[2] == [KeyInput(CTRL), KeyInput(ALT), KeyInput(0x25, EXT)]


def test_lone_modifier_presses_that_key(win):
    ce.key("win")
    assert _keys(win) == [(WIN, EXT), (WIN, EXT | UP)]


def test_key_enter_and_aliases(win):
    ce.key("enter")
    ce.key("Return")
    ce.key(" ESC ")
    ce.key("escape")
    assert [vk for vk, f in _keys(win) if not f & UP] == [0x0D, 0x0D, 0x1B, 0x1B]


def test_key_unknown_raises(win):
    with pytest.raises(ValueError):
        ce.key("hyper+s")
    with pytest.raises(ValueError):
        ce.key("ctrl+")
    with pytest.raises(ValueError):
        ce.key("")
    with pytest.raises(ValueError):
        ce.key("ctrl+shift")  # modifiers only, nothing to press
    with pytest.raises(ValueError):
        ce.key("ctrl+nosuchkey")
    assert win.posted == []


def test_keycode_table_spot_checks():
    k = ce.KEYCODES
    assert k["a"] == 0x41 and k["z"] == 0x5A
    assert k["0"] == 0x30 and k["9"] == 0x39
    assert k["enter"] == 0x0D == k["return"]
    assert k["esc"] == 0x1B and k["tab"] == 0x09 and k["space"] == 0x20
    assert k["backspace"] == 0x08 and k["delete"] == 0x2E == k["forwarddelete"]
    assert (k["left"], k["up"], k["right"], k["down"]) == (0x25, 0x26, 0x27, 0x28)
    assert (k["home"], k["end"], k["pageup"], k["pagedown"]) == (0x24, 0x23, 0x21, 0x22)
    assert k["f1"] == 0x70 and k["f12"] == 0x7B and k["f24"] == 0x87
    assert k["-"] == k["minus"] == 0xBD and k["/"] == 0xBF and k["`"] == 0xC0
    assert len({name for name in k if len(name) == 1 and name.isalpha()}) == 26
    assert set(ce.MODIFIERS) == {"ctrl", "alt", "shift", "win"}


# --- permissions -------------------------------------------------------------

def test_accessibility_trusted_is_always_true_on_windows():
    assert ce.accessibility_trusted() is True
    assert ce.accessibility_trusted(prompt=True) is True


def test_input_blocked_only_for_an_elevated_foreground(win):
    win.fg, win.pids[10] = 10, 500
    assert ce.input_blocked() is False
    win.elevated[500] = True
    assert ce.input_blocked() is True
    win.me_elevated = True                 # an elevated Veronica may drive it
    assert ce.input_blocked() is False


def test_input_blocked_false_when_unknown(win, monkeypatch):
    assert ce.input_blocked() is False     # no foreground window
    win.fg, win.pids[10] = 10, 500
    win.elevated[500] = None
    assert ce.input_blocked() is False

    def boom():
        raise OSError("no win32")
    monkeypatch.setattr(ce, "_win32", boom)
    assert ce.input_blocked() is False


def test_permission_hint_mentions_administrator():
    assert "administrator" in ce.PERMISSION_HINT


class _Elem:
    def __init__(self, password):
        self.CurrentIsPassword = password


class _Control:
    def __init__(self, password):
        self.Element = _Elem(password)


class FakeUIA:
    def __init__(self, control=None, raise_on_focus=False):
        self.control, self.raise_on_focus = control, raise_on_focus
        self.inits = 0

    def UIAutomationInitializerInThread(self):
        outer = self

        class Ctx:
            def __enter__(self):
                outer.inits += 1

            def __exit__(self, *a):
                return False
        return Ctx()

    def GetFocusedControl(self):
        if self.raise_on_focus:
            raise RuntimeError("COM error")
        return self.control


def test_focused_is_secure_reads_is_password(monkeypatch):
    uia = FakeUIA(_Control(True))
    monkeypatch.setattr(ce, "_uia", lambda: uia)
    assert ce.focused_is_secure() is True
    assert uia.inits == 1
    monkeypatch.setattr(ce, "_uia", lambda: FakeUIA(_Control(False)))
    assert ce.focused_is_secure() is False


def test_focused_is_secure_false_on_errors(monkeypatch):
    monkeypatch.setattr(ce, "_uia", lambda: FakeUIA(None))
    assert ce.focused_is_secure() is False
    monkeypatch.setattr(ce, "_uia", lambda: FakeUIA(raise_on_focus=True))
    assert ce.focused_is_secure() is False

    def boom():
        raise ImportError("no uiautomation")
    monkeypatch.setattr(ce, "_uia", boom)
    assert ce.focused_is_secure() is False


# --- frontmost -------------------------------------------------------------

def test_frontmost_reports_app_exe_title_and_class(win):
    win.fg = 0x1234
    win.titles[0x1234], win.classes[0x1234], win.pids[0x1234] = "Untitled - Notepad", "Notepad", 42
    win.paths[42] = r"C:\Windows\System32\Notepad.exe"
    win.names[win.paths[42]] = "Notepad"
    assert ce.frontmost() == Front(
        app="Notepad", bundle_id="notepad.exe", window_title="Untitled - Notepad", pid=42,
        window_class="Notepad",
    )


def test_frontmost_name_falls_back_to_the_exe_stem(win):
    win.fg, win.pids[1] = 1, 7
    win.paths[7] = r"C:\Tools\MyTool.EXE"
    front = ce.frontmost()
    assert (front.app, front.bundle_id) == ("MyTool", "mytool.exe")


def test_frontmost_uwp_reports_the_hosted_app(win):
    win.fg, win.pids[1], win.titles[1] = 1, 7, "Settings"
    win.paths[7] = r"C:\Windows\System32\ApplicationFrameHost.exe"
    win.uwp_child[1] = 9
    win.paths[9] = r"C:\Windows\ImmersiveControlPanel\SystemSettings.exe"
    win.names[win.paths[9]] = "Settings"
    front = ce.frontmost()
    assert (front.app, front.bundle_id, front.pid) == ("Settings", "systemsettings.exe", 9)
    assert ce.is_system_dialog(front)


def test_frontmost_without_a_foreground_window(win):
    assert ce.frontmost() == Front(app="", bundle_id="", window_title="", pid=0)


def test_frontmost_survives_failures(win, monkeypatch):
    win.fg, win.titles[1], win.pids[1] = 1, "Doc", 3

    def broken(pid):
        raise OSError("access denied")
    monkeypatch.setattr(win, "process_path", broken)
    front = ce.frontmost()
    assert front.window_title == "Doc" and front.bundle_id == ""

    def boom():
        raise OSError("no win32")
    monkeypatch.setattr(ce, "_win32", boom)
    assert ce.frontmost() == Front(app="", bundle_id="", window_title="", pid=0)


# --- system dialogs ----------------------------------------------------------

@pytest.mark.parametrize("exe, title, cls, expected", [
    ("consent.exe", "", "", True),
    ("credentialuibroker.exe", "Windows Security", "Credential Dialog Xaml Host", True),
    ("CredentialUIBroker.exe", "", "", True),
    ("smartscreen.exe", "", "", True),
    ("sechealthui.exe", "Windows Security", "", True),
    ("systemsettings.exe", "Settings", "", True),        # any page: unconditionally sensitive
    ("logonui.exe", "", "", True),
    ("explorer.exe", "Open File - Security Warning", "#32770", True),
    ("chrome.exe", "Windows  Security", "", True),       # the title alone is enough
    ("foo.exe", "", "Credential Dialog Xaml Host", True),
    ("notepad.exe", "Privacy settings.txt - Notepad", "Notepad", False),
    ("", "", "", False),
])
def test_is_system_dialog(exe, title, cls, expected):
    front = Front(app="x", bundle_id=exe, window_title=title, pid=1, window_class=cls)
    assert ce.is_system_dialog(front) is expected


def test_disallowed_dialog_targets_table():
    assert {"allow", "always allow", "ok", "yes", "continue", "install", "run anyway"} <= ce.DISALLOWED_DIALOG_TARGETS
    assert all(t == t.lower() for t in ce.DISALLOWED_DIALOG_TARGETS)
    assert all(t == t.lower() for t in ce.SYSTEM_DIALOG_APPS | ce.SYSTEM_DIALOG_CLASSES | ce.SYSTEM_DIALOG_TITLES)


# --- release on failure ------------------------------------------------------

class _FailOn:
    """Records every batch on the fake; raises (after recording) on the
    first batch `pred` matches."""

    def __init__(self, w, pred):
        self.w, self.pred, self.tripped = w, pred, False

    def __call__(self, *events):
        self.w.batches.append(list(events))
        if not self.tripped and self.pred(events):
            self.tripped = True
            raise OSError("input gone")


def test_key_releases_everything_when_downs_fail(win, monkeypatch):
    monkeypatch.setattr(ce, "post", _FailOn(win, lambda evs: not evs[0].flags & UP))
    with pytest.raises(OSError, match="input gone"):
        ce.key("ctrl+shift+s")
    assert win.batches[1] == [KeyInput(ord("S"), UP), KeyInput(SHIFT, UP), KeyInput(CTRL, UP)]


def test_click_releases_button_when_down_fails(win, monkeypatch):
    monkeypatch.setattr(ce, "post", _FailOn(win, lambda evs: evs[0].flags & RDOWN))
    with pytest.raises(OSError):
        ce.click(1, 2, button="right")
    assert _mouse(win) == [(0, 1, 2), (RDOWN, 1, 2), (RUP, 1, 2)]


def test_double_click_second_pair_releases_when_down_fails(win, monkeypatch):
    downs = []

    def post(*events):
        win.batches.append(list(events))
        if events[0].flags & LDOWN:
            downs.append(1)
            if len(downs) == 2:
                raise OSError("input gone")
    monkeypatch.setattr(ce, "post", post)
    with pytest.raises(OSError):
        ce.click(5, 6, double=True)
    assert [f for f, *_ in _mouse(win)] == [0, LDOWN, LUP, LDOWN, LUP]


def test_drag_releases_at_end_point_when_a_move_fails(win, monkeypatch):
    monkeypatch.setattr(ce, "post", _FailOn(win, lambda evs: evs[0].flags == ABS and evs[0].x != 0))
    with pytest.raises(OSError):
        ce.drag(0, 0, 80, 40)
    ev = _mouse(win)
    assert [f for f, *_ in ev] == [0, LDOWN, 0, 0, LUP]
    assert ev[-2:] == [(0, 80, 40), (LUP, 80, 40)]


def test_release_failure_is_logged_not_raised_over_original(win, monkeypatch, caplog):
    def always_fail(*events):
        win.batches.append(list(events))
        raise OSError("up failed" if events[0].flags & UP else "down failed")
    monkeypatch.setattr(ce, "post", always_fail)
    with pytest.raises(OSError, match="down failed"):
        ce.key("a")
    assert len(win.batches) == 2
    assert "up failed" in caplog.text


# --- dangerous combos --------------------------------------------------------

@pytest.mark.parametrize("combo, expected", [
    ("ctrl+q", "ctrl+q"),
    ("Command+Q", "ctrl+q"),
    ("shift+option+control+windows+s", "ctrl+alt+shift+win+s"),
    ("Alt+F4", "alt+f4"),
    ("control+alt+del", "ctrl+alt+delete"),
    (" shift + ctrl + Escape ", "ctrl+shift+esc"),
    ("super+L", "win+l"),
    ("ctrl+ctrl+s", "ctrl+s"),
    ("cmd+ctrl+s", "ctrl+s"),
    ("enter", "enter"),
    ("WIN", "win"),
])
def test_normalize_combo(combo, expected):
    assert ce.normalize_combo(combo) == expected


def test_normalize_combo_rejects_malformed():
    for bad in ("", "ctrl+", "+s", "ctrl+shift", "hyper+s"):
        with pytest.raises(ValueError):
            ce.normalize_combo(bad)


@pytest.mark.parametrize("combo", [
    "alt+f4", "Alt+F4", "option+f4", "win+l", "Windows+L", "ctrl+alt+delete", "ctrl+alt+del",
    "control+alt+Delete", "ctrl+alt+end", "ctrl+shift+esc", "shift+ctrl+escape", "win+x",
    "ctrl+q", "cmd+q", "ctrl+shift+q",
])
def test_key_refuses_dangerous_combos(win, combo):
    with pytest.raises(ce.DangerousCombo):
        ce.key(combo)
    assert win.posted == []


def test_dangerous_combo_is_a_value_error_and_table_is_normalised():
    assert issubclass(ce.DangerousCombo, ValueError)
    for c in ce.DANGEROUS_COMBOS:
        assert ce.normalize_combo(c) == c, c


def test_key_still_allows_safe_combos(win):
    ce.key("ctrl+s")
    ce.key("ctrl+shift+z")
    ce.key("alt+tab")
    ce.key("win+d")
    assert len(win.batches) == 8
