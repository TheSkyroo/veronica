import pytest

from veronica.tools import computer_events as ce
from veronica.tools.computer_events import Front

# --- fakes -----------------------------------------------------------------

class FakeEvent:
    def __init__(self, kind, type_, x=None, y=None, button=None, keycode=None, keydown=None):
        self.kind = kind          # "mouse" | "scroll" | "key"
        self.type = type_
        self.x, self.y = x, y
        self.button = button
        self.keycode, self.keydown = keycode, keydown
        self.flags = 0
        self.fields = {}
        self.unicode = None


class FakeQuartz:
    kCGHIDEventTap = 0
    kCGEventMouseMoved = 5
    kCGEventLeftMouseDown = 1
    kCGEventLeftMouseUp = 2
    kCGEventRightMouseDown = 3
    kCGEventRightMouseUp = 4
    kCGEventOtherMouseDown = 25
    kCGEventOtherMouseUp = 26
    kCGEventLeftMouseDragged = 6
    kCGMouseButtonLeft = 0
    kCGMouseButtonRight = 1
    kCGMouseButtonCenter = 2
    kCGMouseEventClickState = 1
    kCGScrollEventUnitPixel = 0
    kCGEventFlagMaskCommand = 1 << 20
    kCGEventFlagMaskShift = 1 << 17
    kCGEventFlagMaskAlternate = 1 << 19
    kCGEventFlagMaskControl = 1 << 18
    kCGWindowListOptionOnScreenOnly = 1
    kCGWindowOwnerPID = "kCGWindowOwnerPID"
    kCGWindowLayer = "kCGWindowLayer"
    kCGWindowName = "kCGWindowName"
    kCGWindowBounds = "kCGWindowBounds"

    def __init__(self, windows=()):
        self.posted: list[FakeEvent] = []
        self.taps: list = []
        self.windows = list(windows)

    def CGEventCreateMouseEvent(self, src, type_, point, button):
        assert src is None
        return FakeEvent("mouse", type_, point[0], point[1], button)

    def CGEventCreateScrollWheelEvent(self, src, unit, count, dy, dx):
        assert src is None and unit == self.kCGScrollEventUnitPixel and count == 2
        ev = FakeEvent("scroll", "wheel")
        ev.dy, ev.dx = dy, dx
        return ev

    def CGEventCreateKeyboardEvent(self, src, keycode, keydown):
        assert src is None
        return FakeEvent("key", "keydown" if keydown else "keyup", keycode=keycode, keydown=keydown)

    def CGEventSetIntegerValueField(self, ev, field, value):
        ev.fields[field] = value

    def CGEventSetFlags(self, ev, flags):
        ev.flags = flags

    def CGEventKeyboardSetUnicodeString(self, ev, length, text):
        assert length == len(text.encode("utf-16-le")) // 2
        ev.unicode = text

    def CGEventPost(self, tap, ev):
        self.taps.append(tap)
        self.posted.append(ev)

    def CGWindowListCopyWindowInfo(self, opts, relative):
        assert opts == self.kCGWindowListOptionOnScreenOnly and relative == 0
        return self.windows


class FakeAX:
    kAXTrustedCheckOptionPrompt = "AXTrustedCheckOptionPrompt"
    kAXFocusedUIElementAttribute = "AXFocusedUIElement"
    kAXRoleAttribute = "AXRole"
    kAXSubroleAttribute = "AXSubrole"

    def __init__(self, trusted=True, role="AXTextField", subrole=None, focus_err=0, raise_on_role=False):
        self.trusted = trusted
        self.role, self.subrole = role, subrole
        self.focus_err = focus_err
        self.raise_on_role = raise_on_role
        self.options = []

    def AXIsProcessTrustedWithOptions(self, options):
        self.options.append(dict(options))
        return self.trusted

    def AXUIElementCreateSystemWide(self):
        return "sys"

    def AXUIElementCopyAttributeValue(self, elem, attr, out):
        assert out is None
        if elem == "sys" and attr == "AXFocusedUIElement":
            return (self.focus_err, None if self.focus_err else "elem")
        if elem == "elem" and attr == "AXRole":
            if self.raise_on_role:
                raise RuntimeError("boom")
            return (0, self.role)
        if elem == "elem" and attr == "AXSubrole":
            return (0, self.subrole)
        return (-25205, None)


class _App:
    def __init__(self, name, bundle, pid):
        self._n, self._b, self._p = name, bundle, pid

    def localizedName(self):
        return self._n

    def bundleIdentifier(self):
        return self._b

    def processIdentifier(self):
        return self._p


class FakeAppKit:
    def __init__(self, app):
        outer = self

        class NSWorkspace:
            @staticmethod
            def sharedWorkspace():
                return outer

        self.NSWorkspace = NSWorkspace
        self._app = app

    def frontmostApplication(self):
        return self._app


@pytest.fixture
def quartz(monkeypatch):
    q = FakeQuartz()
    monkeypatch.setattr(ce, "_quartz", lambda: q)
    sleeps = []
    monkeypatch.setattr(ce, "_sleep", sleeps.append)
    q.sleeps = sleeps
    return q


def _mouse(q):
    return [(e.type, e.x, e.y, e.button) for e in q.posted if e.kind == "mouse"]


# --- mouse -------------------------------------------------------------------

def test_move_posts_mouse_moved_to_hid_tap(quartz):
    ce.move(10.5, 20)
    assert _mouse(quartz) == [(5, 10.5, 20, 0)]
    assert quartz.taps == [0]


def test_click_posts_move_down_up(quartz):
    ce.click(100, 200)
    assert _mouse(quartz) == [(5, 100, 200, 0), (1, 100, 200, 0), (2, 100, 200, 0)]
    assert all(e.fields.get(1, 1) == 1 for e in quartz.posted)


def test_click_right_and_middle_buttons(quartz):
    ce.click(1, 2, button="right")
    ce.click(3, 4, button="middle")
    assert _mouse(quartz) == [
        (5, 1, 2, 0), (3, 1, 2, 1), (4, 1, 2, 1),
        (5, 3, 4, 0), (25, 3, 4, 2), (26, 3, 4, 2),
    ]


def test_click_unknown_button(quartz):
    with pytest.raises(ValueError):
        ce.click(1, 2, button="thumb")


def test_double_click_posts_two_pairs_with_click_state_2(quartz):
    ce.click(5, 6, double=True)
    types = [e.type for e in quartz.posted]
    assert types == [5, 1, 2, 1, 2]
    states = [e.fields.get(1) for e in quartz.posted]
    assert states == [None, None, None, 2, 2]


def test_drag_posts_down_eight_dragged_moves_up(quartz):
    ce.drag(0, 0, 80, 40)
    ev = _mouse(quartz)
    assert ev[0] == (5, 0, 0, 0)
    assert ev[1] == (1, 0, 0, 0)
    dragged = ev[2:-1]
    assert len(dragged) == 8
    assert all(t == 6 for t, *_ in dragged)
    xs = [x for _, x, _, _ in dragged]
    ys = [y for _, _, y, _ in dragged]
    assert xs == pytest.approx([10, 20, 30, 40, 50, 60, 70, 80])
    assert ys == pytest.approx([5, 10, 15, 20, 25, 30, 35, 40])
    assert ev[-1] == (2, 80, 40, 0)
    # ~200 ms spread over the intermediate moves
    assert sum(quartz.sleeps) == pytest.approx(0.2, abs=0.05)


def test_scroll_moves_then_posts_wheel(quartz):
    ce.scroll(50, 60, dx=-3.7, dy=12.2)
    assert _mouse(quartz) == [(5, 50, 60, 0)]
    wheel = [e for e in quartz.posted if e.kind == "scroll"]
    assert len(wheel) == 1
    assert (wheel[0].dy, wheel[0].dx) == (12, -3)
    assert isinstance(wheel[0].dy, int) and isinstance(wheel[0].dx, int)


# --- keyboard ----------------------------------------------------------------

def test_type_text_chunks_utf16_and_posts_down_up_per_chunk(quartz):
    text = "hello world, नमस्ते — आप कैसे हैं? all good here"
    ce.type_text(text)
    keys = [e for e in quartz.posted if e.kind == "key"]
    assert len(keys) == 2 * len(ce._chunks(text))
    downs = keys[0::2]
    ups = keys[1::2]
    assert all(e.type == "keydown" and e.keycode == 0 for e in downs)
    assert all(e.type == "keyup" and e.keycode == 0 for e in ups)
    assert "".join(e.unicode for e in downs) == text
    assert [e.unicode for e in ups] == [e.unicode for e in downs]
    assert all(len(e.unicode.encode("utf-16-le")) // 2 <= 20 for e in downs)
    assert len(downs) >= 2
    assert quartz.sleeps == [ce.TYPE_CHUNK_GAP_S] * (len(downs) - 1)


def test_type_text_chunks_never_split_surrogate_pairs(quartz):
    text = "a" * 19 + "😀" + "b"
    chunks = ce._chunks(text)
    assert "".join(chunks) == text
    assert all(len(c.encode("utf-16-le")) // 2 <= 20 for c in chunks)
    assert chunks[0] == "a" * 19  # the emoji (2 units) would not fit


def test_type_text_empty_posts_nothing(quartz):
    ce.type_text("")
    assert quartz.posted == []


def test_key_combo_sets_keycode_and_flags_on_down_and_up(quartz):
    ce.key("cmd+shift+s")
    keys = [(e.type, e.keycode, e.flags) for e in quartz.posted]
    want = FakeQuartz.kCGEventFlagMaskCommand | FakeQuartz.kCGEventFlagMaskShift
    assert keys == [("keydown", 1, want), ("keyup", 1, want)]


def test_key_enter_and_aliases(quartz):
    ce.key("enter")
    ce.key("Return")
    ce.key(" ESC ")
    assert [(e.type, e.keycode, e.flags) for e in quartz.posted] == [
        ("keydown", 36, 0), ("keyup", 36, 0),
        ("keydown", 36, 0), ("keyup", 36, 0),
        ("keydown", 53, 0), ("keyup", 53, 0),
    ]


def test_key_modifier_aliases(quartz):
    ce.key("option+ctrl+command+a")
    q = FakeQuartz
    want = q.kCGEventFlagMaskAlternate | q.kCGEventFlagMaskControl | q.kCGEventFlagMaskCommand
    assert [e.flags for e in quartz.posted] == [want, want]


def test_key_unknown_raises(quartz):
    with pytest.raises(ValueError):
        ce.key("hyper+s")
    with pytest.raises(ValueError):
        ce.key("cmd+")
    with pytest.raises(ValueError):
        ce.key("")
    with pytest.raises(ValueError):
        ce.key("cmd+shift")  # modifiers only, nothing to press
    assert quartz.posted == []


def test_keycode_table_spot_checks():
    k = ce.KEYCODES
    assert k["a"] == 0 and k["s"] == 1 and k["z"] == 6
    assert k["0"] == 29 and k["1"] == 18 and k["9"] == 25
    assert k["enter"] == 36 == k["return"]
    assert k["esc"] == 53 == k["escape"]
    assert k["tab"] == 48 and k["space"] == 49
    assert k["backspace"] == 51 == k["delete"] and k["forwarddelete"] == 117
    assert (k["up"], k["down"], k["left"], k["right"]) == (126, 125, 123, 124)
    assert (k["home"], k["end"], k["pageup"], k["pagedown"]) == (115, 119, 116, 121)
    assert k["f1"] == 122 and k["f12"] == 111
    assert k["minus"] == 27 and k["equal"] == 24 and k["grave"] == 50
    assert k["-"] == 27 and k["."] == 47 and k["/"] == 44
    assert len({name for name in k if len(name) == 1 and name.isalpha()}) == 26
    assert set(ce.MODIFIERS) == {"cmd", "command", "shift", "alt", "option", "ctrl", "control"}


# --- accessibility ---------------------------------------------------------

def test_accessibility_trusted_passes_prompt_option(monkeypatch):
    ax = FakeAX(trusted=False)
    monkeypatch.setattr(ce, "_ax", lambda: ax)
    assert ce.accessibility_trusted() is False
    assert ce.accessibility_trusted(prompt=True) is False
    assert ax.options == [
        {"AXTrustedCheckOptionPrompt": False},
        {"AXTrustedCheckOptionPrompt": True},
    ]
    ax.trusted = True
    assert ce.accessibility_trusted() is True


def test_accessibility_trusted_false_when_framework_missing(monkeypatch):
    def boom():
        raise ImportError("no ApplicationServices")
    monkeypatch.setattr(ce, "_ax", boom)
    assert ce.accessibility_trusted() is False


def test_permission_hint_mentions_accessibility():
    assert "Accessibility" in ce.PERMISSION_HINT
    assert "Privacy & Security" in ce.PERMISSION_HINT


def test_focused_is_secure_role(monkeypatch):
    monkeypatch.setattr(ce, "_ax", lambda: FakeAX(role="AXSecureTextField"))
    assert ce.focused_is_secure() is True
    monkeypatch.setattr(ce, "_ax", lambda: FakeAX(role="AXTextField"))
    assert ce.focused_is_secure() is False


def test_focused_is_secure_subrole(monkeypatch):
    monkeypatch.setattr(ce, "_ax", lambda: FakeAX(role="AXTextField", subrole="AXSecureTextField"))
    assert ce.focused_is_secure() is True


def test_focused_is_secure_false_on_errors(monkeypatch):
    monkeypatch.setattr(ce, "_ax", lambda: FakeAX(focus_err=-25204))
    assert ce.focused_is_secure() is False
    monkeypatch.setattr(ce, "_ax", lambda: FakeAX(raise_on_role=True))
    assert ce.focused_is_secure() is False

    def boom():
        raise ImportError("nope")
    monkeypatch.setattr(ce, "_ax", boom)
    assert ce.focused_is_secure() is False


# --- frontmost -------------------------------------------------------------

def _win(pid, layer, name, y=100):
    return {
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": layer,
        "kCGWindowName": name,
        "kCGWindowBounds": {"X": 10, "Y": y, "Width": 800, "Height": 600},
    }


def test_frontmost_uses_first_layer0_window_of_front_pid(monkeypatch):
    q = FakeQuartz(windows=[
        _win(999, 0, "Other app"),
        _win(42, 25, "Menu bar thing"),
        _win(42, 0, "Untitled — TextEdit"),
        _win(42, 0, "Second window"),
    ])
    monkeypatch.setattr(ce, "_quartz", lambda: q)
    monkeypatch.setattr(ce, "_appkit", lambda: FakeAppKit(_App("TextEdit", "com.apple.TextEdit", 42)))
    front = ce.frontmost()
    assert front == Front(app="TextEdit", bundle_id="com.apple.TextEdit",
                          window_title="Untitled — TextEdit", pid=42)


def test_frontmost_prefers_on_screen_window(monkeypatch):
    q = FakeQuartz(windows=[
        _win(42, 0, "Parked offscreen", y=-5000),
        _win(42, 0, "Visible one", y=50),
    ])
    monkeypatch.setattr(ce, "_quartz", lambda: q)
    monkeypatch.setattr(ce, "_appkit", lambda: FakeAppKit(_App("X", "com.x", 42)))
    assert ce.frontmost().window_title == "Visible one"


def test_frontmost_falls_back_to_offscreen_window_when_nothing_else(monkeypatch):
    q = FakeQuartz(windows=[_win(42, 0, "Parked offscreen", y=-5000)])
    monkeypatch.setattr(ce, "_quartz", lambda: q)
    monkeypatch.setattr(ce, "_appkit", lambda: FakeAppKit(_App("X", "com.x", 42)))
    assert ce.frontmost().window_title == "Parked offscreen"


def test_frontmost_without_window_or_app(monkeypatch):
    q = FakeQuartz(windows=[_win(1, 0, "Something else")])
    monkeypatch.setattr(ce, "_quartz", lambda: q)
    monkeypatch.setattr(ce, "_appkit", lambda: FakeAppKit(_App("Finder", None, 42)))
    front = ce.frontmost()
    assert front == Front(app="Finder", bundle_id="", window_title="", pid=42)

    monkeypatch.setattr(ce, "_appkit", lambda: FakeAppKit(None))
    assert ce.frontmost() == Front(app="", bundle_id="", window_title="", pid=0)


def test_frontmost_survives_window_list_failure(monkeypatch):
    class Broken(FakeQuartz):
        def CGWindowListCopyWindowInfo(self, *a):
            raise RuntimeError("cg down")
    monkeypatch.setattr(ce, "_quartz", lambda: Broken())
    monkeypatch.setattr(ce, "_appkit", lambda: FakeAppKit(_App("X", "com.x", 7)))
    assert ce.frontmost() == Front(app="X", bundle_id="com.x", window_title="", pid=7)


# --- system dialogs ----------------------------------------------------------

@pytest.mark.parametrize("bundle, title, expected", [
    ("com.apple.SecurityAgent", "", True),
    ("com.apple.UserNotificationCenter", "", True),
    ("com.apple.coreservices.uiagent", "Open?", True),
    ("com.apple.systempreferences", "Privacy & Security", True),
    ("com.apple.systempreferences", "Accessibility — privacy", True),
    ("com.apple.systempreferences", "Login Security", True),
    ("com.apple.systempreferences", "Displays", False),
    ("com.apple.systempreferences", "", False),
    ("com.apple.TextEdit", "Privacy & Security", False),
    ("", "", False),
])
def test_is_system_dialog(bundle, title, expected):
    assert ce.is_system_dialog(Front(app="x", bundle_id=bundle, window_title=title, pid=1)) is expected


def test_disallowed_dialog_targets_table():
    assert {"allow", "always allow", "ok", "open system settings", "continue", "install", "trust"} <= ce.DISALLOWED_DIALOG_TARGETS
    assert all(t == t.lower() for t in ce.DISALLOWED_DIALOG_TARGETS)
    assert "com.apple.SecurityAgent" in ce.SYSTEM_DIALOG_BUNDLES
