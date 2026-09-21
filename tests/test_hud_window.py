import time
import json
import sys
import types

from veronica.config import Settings
from veronica.ui.hud import HudWindow


class FakeWeb:
    def __init__(self): self.js = []
    def evaluateJavaScript_completionHandler_(self, js, cb): self.js.append(js)


class FakeRect:
    def __init__(self, x, y, w, h):
        self.origin = types.SimpleNamespace(x=x, y=y)
        self.size = types.SimpleNamespace(width=w, height=h)


class FakePanel:
    def __init__(self, x=0.0, y=0.0, w=540, h=300):
        self.alpha = 0.0
        self.visible = False
        self.orders = []
        self._rect = FakeRect(x, y, w, h)
        self.setFrame_calls = []

    def setAlphaValue_(self, a): self.alpha = a
    def orderFrontRegardless(self): self.visible = True; self.orders.append("front")
    def orderOut_(self, _): self.visible = False; self.orders.append("out")

    def frame(self):
        return self._rect

    def setFrame_display_animate_(self, frame, display, animate):
        self.setFrame_calls.append((frame.origin.x, frame.origin.y, frame.size.width, frame.size.height))
        self._rect = frame


def _noop_prefs_load():
    return {}


def _noop_prefs_save(_prefs):
    pass


# A single 1440x900 display (visibleFrame origin at 0,0) stands in for
# NSScreen.screens() so geometry tests never touch AppKit.
MAIN_SCREEN = FakeRect(0, 0, 1440, 900)


def make(t=[0.0], mark_loaded=True, screens=None, prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
         subscribe_screen_changes=None, **settings_over):
    web, panel = FakeWeb(), FakePanel()
    frames = [MAIN_SCREEN] if screens is None else list(screens)
    h = HudWindow(
        Settings(hud_hide_after_s=3.0, **settings_over),
        webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        clock=lambda: t[0], main=lambda fn: fn(),
        prefs_load=prefs_load, prefs_save=prefs_save,
        screens=lambda: frames, subscribe_screen_changes=subscribe_screen_changes,
    )
    if mark_loaded:
        # Simulate the page having already finished loading, and clear the
        # replay JS mark_loaded() produces, so tests assert only their own
        # calls — most of this file's tests aren't about the load gate.
        h.mark_loaded()
        web.js = []
    return h, web, panel, t


def test_push_serializes_json():
    h, web, _, _ = make()
    h.push({"kind": "heard", "payload": "héllo \"q\""})
    assert web.js == ['window.hud.push(' + json.dumps({"kind": "heard", "payload": "héllo \"q\""}, ensure_ascii=False) + ')']


def test_show_on_non_idle_and_hide_after_delay():
    h, _, panel, t = make()
    h.on_state("listening")
    assert panel.visible and panel.alpha == 1.0
    h.on_state("idle"); h.tick()
    assert panel.visible                     # not yet
    t[0] = 3.1; h.tick()
    assert not panel.visible and panel.alpha == 0.0


def test_non_idle_cancels_pending_hide():
    h, _, panel, t = make()
    h.on_state("listening"); h.on_state("idle"); t[0] = 2.0
    h.on_state("thinking"); t[0] = 5.0; h.tick()
    assert panel.visible


def test_push_after_close_is_noop():
    h, web, _, _ = make()
    h.close(); h.push({"kind": "mic", "payload": 0.1})
    assert web.js == []


def test_factory_failure_is_soft(caplog):
    def bad(s): raise RuntimeError("no webkit")
    h = HudWindow(Settings(), webview_factory=bad, panel_factory=lambda s, w: FakePanel(), main=lambda fn: fn(),
                  prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save)
    h.push({"kind": "state", "payload": "idle"}); h.on_state("listening")   # no raise
    assert h.available is False


def test_show_calls_set_visible_true_and_hide_calls_set_visible_false():
    h, web, panel, t = make()
    h.show()
    assert web.js[-1] == "window.hud.setVisible(true)"
    h.hide()
    assert web.js[-1] == "window.hud.setVisible(false)"


def test_hide_waits_for_animation_completion_before_ordering_out(monkeypatch):
    # A panel that HAS an animator(), plus a fake AppKit.NSAnimationContext
    # whose endGrouping() does NOT fire the completion handler, exercises the
    # real (non-except) branch of _fade: the panel must not be ordered out
    # until the completion handler set via setCompletionHandler_ is invoked.
    class FakeAnimatorProxy:
        def __init__(self, panel):
            self._panel = panel

        def setAlphaValue_(self, a):
            self._panel.alpha = a

    class FakePanelWithAnimator(FakePanel):
        def animator(self):
            return FakeAnimatorProxy(self)

    class FakeContext:
        def __init__(self):
            self.completion = None

        def setDuration_(self, d):
            pass

        def setCompletionHandler_(self, cb):
            self.completion = cb

    class FakeNSAnimationContext:
        ctx = FakeContext()

        @classmethod
        def beginGrouping(cls):
            pass

        @classmethod
        def currentContext(cls):
            return cls.ctx

        @classmethod
        def endGrouping(cls):
            pass  # deliberately does NOT invoke the completion handler

    fake_appkit = types.SimpleNamespace(NSAnimationContext=FakeNSAnimationContext)
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)

    web, panel = FakeWeb(), FakePanelWithAnimator()
    h = HudWindow(Settings(), webview_factory=lambda s: web, panel_factory=lambda s, w: panel, main=lambda fn: fn(),
                  prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save)

    h.hide()
    assert "out" not in panel.orders
    assert panel.alpha == 0.0

    FakeNSAnimationContext.ctx.completion()
    assert "out" in panel.orders


def test_show_during_fade_prevents_stale_hide_completion_from_ordering_out(monkeypatch):
    # A hide() begins fading out; before its completion handler fires, a
    # show() (e.g. a new turn starting) supersedes it. The stale hide
    # completion must not order the panel out from under the new show.
    class FakeAnimatorProxy:
        def __init__(self, panel):
            self._panel = panel

        def setAlphaValue_(self, a):
            self._panel.alpha = a

    class FakePanelWithAnimator(FakePanel):
        def animator(self):
            return FakeAnimatorProxy(self)

    class FakeContext:
        def __init__(self):
            self.completion = None

        def setDuration_(self, d):
            pass

        def setCompletionHandler_(self, cb):
            self.completion = cb

    class FakeNSAnimationContext:
        ctx = FakeContext()

        @classmethod
        def beginGrouping(cls):
            pass

        @classmethod
        def currentContext(cls):
            return cls.ctx

        @classmethod
        def endGrouping(cls):
            pass  # deliberately does NOT invoke the completion handler

    fake_appkit = types.SimpleNamespace(NSAnimationContext=FakeNSAnimationContext)
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)

    web, panel = FakeWeb(), FakePanelWithAnimator()
    h = HudWindow(Settings(), webview_factory=lambda s: web, panel_factory=lambda s, w: panel, main=lambda fn: fn(),
                  prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save)

    h.hide()
    stale_completion = FakeNSAnimationContext.ctx.completion
    assert "out" not in panel.orders

    h.on_state("listening")  # supersedes the pending hide with a show()
    assert panel.visible

    stale_completion()  # the old hide's fade completion fires late
    assert panel.visible
    assert "out" not in panel.orders


# -- commit 3: mini mode, draggable panel, position persistence ---------------

def test_default_mode_is_full_and_construction_leaves_geometry_untouched():
    h, _, panel, _ = make()
    assert h._mode == "full"
    assert panel.setFrame_calls == []  # factory-built geometry left alone


def test_set_mode_mini_resizes_panel_and_calls_js():
    h, web, panel, _ = make()
    h.set_mode("mini")
    assert h._mode == "mini"
    assert panel.setFrame_calls[-1][2:] == (400, 72)   # width, height
    assert web.js[-1] == 'window.hud.setMode("mini")'


def test_set_mode_full_resizes_back():
    h, web, panel, _ = make()
    h.set_mode("mini")
    h.set_mode("full")
    assert panel.setFrame_calls[-1][2:] == (540, 300)
    assert web.js[-1] == 'window.hud.setMode("full")'


def test_set_mode_ignores_unknown_mode():
    h, web, panel, _ = make()
    h.set_mode("huge")
    assert h._mode == "full"
    assert panel.setFrame_calls == []


def test_set_mode_persists_pref():
    saved = {}
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=_noop_prefs_load,
        prefs_save=lambda p: saved.update(p),
        screens=lambda: [MAIN_SCREEN],
    )
    h.set_mode("mini")
    assert saved == {"hud_mode": "mini"}


def test_construction_applies_saved_mini_mode():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_mode": "mini"}, prefs_save=_noop_prefs_save,
        screens=lambda: [MAIN_SCREEN],
    )
    assert h._mode == "mini"
    assert panel.setFrame_calls[-1][2:] == (400, 72)


def test_construction_falls_back_to_settings_hud_mode_when_no_saved_pref():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0, hud_mode="mini"), webview_factory=lambda s: web,
        panel_factory=lambda s, w: panel, main=lambda fn: fn(),
        prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
        screens=lambda: [MAIN_SCREEN],
    )
    assert h._mode == "mini"


def test_construction_ignores_invalid_saved_mode():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_mode": "gigantic"}, prefs_save=_noop_prefs_save,
        screens=lambda: [MAIN_SCREEN],
    )
    assert h._mode == "full"


def test_construction_applies_saved_position():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_pos_full": [12.0, 34.0]}, prefs_save=_noop_prefs_save,
        screens=lambda: [MAIN_SCREEN],
    )
    assert h._pos["full"] == (12.0, 34.0)
    assert panel.setFrame_calls[-1][:2] == (12.0, 34.0)


def test_hide_persists_panel_position():
    saved = {}
    web, panel = FakeWeb(), FakePanel(x=100.0, y=200.0)
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=_noop_prefs_load,
        prefs_save=lambda p: saved.update(p),
        screens=lambda: [MAIN_SCREEN],
    )
    h.hide()
    assert saved == {"hud_pos_full": [100.0, 200.0]}
    assert h._pos["full"] == (100.0, 200.0)


def test_hide_persists_panel_position_separately_per_mode():
    saved = {}
    web, panel = FakeWeb(), FakePanel(x=5.0, y=6.0)
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=_noop_prefs_load,
        prefs_save=lambda p: saved.update(p),
        screens=lambda: [MAIN_SCREEN],
    )
    h.set_mode("mini")
    # Simulate the user dragging the (now mini) panel to a new spot.
    panel._rect = FakeRect(7.0, 8.0, 400, 72)
    saved.clear()
    h.hide()
    assert saved == {"hud_pos_mini": [7.0, 8.0]}
    assert h._pos["mini"] == (7.0, 8.0)
    assert h._pos["full"] is None


def test_set_mode_mini_defaults_to_top_center_below_menubar():
    """With no saved position for mini mode, switching to mini should place
    the compact bar centered horizontally under the menu bar (a
    notch/Dynamic-Island style default), not the full card's top-right
    corner."""
    h, web, panel, _ = make()   # make()'s fake main screen is 1440x900
    h.set_mode("mini")

    x, y, w, hgt = panel.setFrame_calls[-1]
    assert (w, hgt) == (400, 72)
    assert x == (1440 - 400) / 2
    assert y == 900 - 72 - 8


def test_prefs_load_failure_is_soft(caplog):
    def boom():
        raise RuntimeError("disk error")
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=boom, prefs_save=_noop_prefs_save,
        screens=lambda: [MAIN_SCREEN],
    )
    assert h._mode == "full"   # falls back to settings default
    assert h.available


def test_panel_factory_sets_up_draggable_non_activating_panel(monkeypatch):
    """The real panel factory must make the panel draggable (accepts mouse
    events, movable by its background) while staying a non-activating
    panel."""
    import types as _types

    calls = {}

    class FakeAppKitPanel:
        def __init__(self):
            self.ignores_mouse = None
            self.movable_by_bg = None

        def setOpaque_(self, v): pass
        def setBackgroundColor_(self, v): pass
        def setLevel_(self, v): pass
        def setCollectionBehavior_(self, v): pass
        def setIgnoresMouseEvents_(self, v): self.ignores_mouse = v
        def setMovableByWindowBackground_(self, v): self.movable_by_bg = v
        def setHasShadow_(self, v): pass
        def setAlphaValue_(self, v): pass
        def setContentView_(self, v): pass

    fake_panel_instance = FakeAppKitPanel()

    class FakeAlloc:
        def initWithContentRect_styleMask_backing_defer_(self, *a, **k):
            return fake_panel_instance

    class FakeNSPanel:
        @staticmethod
        def alloc():
            return FakeAlloc()

    class FakeScreen:
        @staticmethod
        def mainScreen():
            frame = _types.SimpleNamespace(
                origin=_types.SimpleNamespace(x=0, y=0),
                size=_types.SimpleNamespace(width=1440, height=900),
            )
            return _types.SimpleNamespace(visibleFrame=lambda: frame)

    fake_appkit = _types.SimpleNamespace(
        NSPanel=FakeNSPanel,
        NSScreen=FakeScreen,
        NSColor=_types.SimpleNamespace(clearColor=lambda: None),
        NSWindowStyleMaskBorderless=0,
        NSWindowStyleMaskNonactivatingPanel=0,
        NSFloatingWindowLevel=0,
        NSWindowCollectionBehaviorCanJoinAllSpaces=0,
        NSWindowCollectionBehaviorStationary=0,
        NSBackingStoreBuffered=0,
    )
    fake_foundation = _types.SimpleNamespace(NSMakeRect=lambda x, y, w, h: (x, y, w, h))
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
    monkeypatch.setitem(sys.modules, "Foundation", fake_foundation)

    from veronica.ui.hud import _real_panel

    panel = _real_panel(Settings(), object())
    assert panel.ignores_mouse is False
    assert panel.movable_by_bg is True


# -- commit: defer HUD JS until the page has loaded ---------------------------

def test_js_before_loaded_is_queued_not_evaluated():
    h, web, _, _ = make(mark_loaded=False)
    h.push({"kind": "mic", "payload": 0.5})
    assert web.js == []


def test_mark_loaded_replays_mode_first_then_flushes_queued_pushes_in_order():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_mode": "mini"}, prefs_save=_noop_prefs_save,
        screens=lambda: [MAIN_SCREEN],
    )
    # Constructing with a persisted mini mode already queued a setMode call
    # (via _apply_geometry) before the page has loaded.
    h.push({"kind": "mic", "payload": 0.2})
    h.push({"kind": "mic", "payload": 0.4})
    assert web.js == []

    h.mark_loaded()

    assert web.js[0] == 'window.hud.setMode("mini")'
    push_calls = [j for j in web.js if j.startswith("window.hud.push(")]
    assert push_calls == [
        "window.hud.push(" + json.dumps({"kind": "mic", "payload": 0.2}, ensure_ascii=False) + ")",
        "window.hud.push(" + json.dumps({"kind": "mic", "payload": 0.4}, ensure_ascii=False) + ")",
    ]


def test_configure_before_loaded_is_queued_then_sent():
    h, web, _, _ = make(mark_loaded=False)
    h.configure({"particles": 1200, "intensity": 0.8})
    assert web.js == []
    h.mark_loaded()
    assert web.js[-1] == 'window.hud.configure({"particles": 1200, "intensity": 0.8})'


def test_configure_after_loaded_evaluates_immediately():
    h, web, _, _ = make(mark_loaded=True)
    h.configure({"particles": 2500, "intensity": 1.5})
    assert web.js == ['window.hud.configure({"particles": 2500, "intensity": 1.5})']


def test_initial_config_pushed_on_load_from_settings():
    h, web, _, _ = make(mark_loaded=False, hud_particles=300, hud_intensity=1.25, hud_detail=0.75)
    h.push({"kind": "mic", "payload": 0.2})
    h.mark_loaded()
    assert web.js[0] == 'window.hud.setMode("full")'
    assert web.js[1] == 'window.hud.configure({"particles": 300, "intensity": 1.25, "detail": 0.75})'
    assert web.js[2].startswith("window.hud.push(")


def test_push_after_loaded_evaluates_immediately():
    h, web, _, _ = make(mark_loaded=True)
    h.push({"kind": "mic", "payload": 0.9})
    assert web.js == [
        "window.hud.push(" + json.dumps({"kind": "mic", "payload": 0.9}, ensure_ascii=False) + ")",
    ]


def test_webview_class_is_draggable_through_the_page(monkeypatch):
    """WKWebView returns NO for mouseDownCanMoveWindow by default, so the
    panel's setMovableByWindowBackground_ never fires for a click landing on
    the web view. _webview_class() must build a WKWebView subclass that
    overrides mouseDownCanMoveWindow (and acceptsFirstMouse_, so the first
    click on the non-activating panel starts a drag rather than just
    activating it) to True."""
    import types as _types

    class FakeWKWebView:
        pass

    fake_webkit = _types.SimpleNamespace(WKWebView=FakeWKWebView)
    monkeypatch.setitem(sys.modules, "WebKit", fake_webkit)

    from veronica.ui.hud import _webview_class

    cls = _webview_class()
    assert issubclass(cls, FakeWKWebView)
    instance = cls.__new__(cls)
    assert instance.mouseDownCanMoveWindow() is True
    assert instance.acceptsFirstMouse_(None) is True


# -- commit: click the HUD orb to open the menu --------------------------------

def test_webview_right_click_calls_super_then_on_menu(monkeypatch):
    class FakeWKWebView:
        def rightMouseDown_(self, event):
            self.super_rightMouseDown_called = event

    fake_webkit = types.SimpleNamespace(WKWebView=FakeWKWebView)
    monkeypatch.setitem(sys.modules, "WebKit", fake_webkit)

    from veronica.ui.hud import _webview_class

    cls = _webview_class()
    instance = cls.__new__(cls)
    seen = []
    instance.on_menu = lambda ev: seen.append(ev)

    instance.rightMouseDown_("evt")

    assert instance.super_rightMouseDown_called == "evt"
    assert seen == ["evt"]


def test_webview_right_click_is_noop_without_on_menu(monkeypatch):
    class FakeWKWebView:
        def rightMouseDown_(self, event):
            pass

    fake_webkit = types.SimpleNamespace(WKWebView=FakeWKWebView)
    monkeypatch.setitem(sys.modules, "WebKit", fake_webkit)

    from veronica.ui.hud import _webview_class

    cls = _webview_class()
    instance = cls.__new__(cls)
    instance.rightMouseDown_("evt")  # must not raise; on_menu defaults to None


class _FakeMenuWindow:
    def __init__(self, x, y):
        self._x, self._y = x, y

    def frame(self):
        return types.SimpleNamespace(origin=types.SimpleNamespace(x=self._x, y=self._y))


def _draggable_instance(monkeypatch):
    class FakeWKWebView:
        def mouseDown_(self, event):
            self.down_called = event

        def mouseUp_(self, event):
            self.up_called = event

    fake_webkit = types.SimpleNamespace(WKWebView=FakeWKWebView)
    monkeypatch.setitem(sys.modules, "WebKit", fake_webkit)

    from veronica.ui.hud import _webview_class

    cls = _webview_class()
    return cls.__new__(cls)


def test_webview_click_without_drag_calls_on_menu(monkeypatch):
    instance = _draggable_instance(monkeypatch)
    window = _FakeMenuWindow(10.0, 20.0)
    instance.window = lambda: window
    seen = []
    instance.on_menu = lambda ev: seen.append(ev)

    instance.mouseDown_("down")
    instance.mouseUp_("up")

    assert instance.down_called == "down"
    assert instance.up_called == "up"
    assert seen == ["up"]


def test_webview_drag_does_not_call_on_menu(monkeypatch):
    instance = _draggable_instance(monkeypatch)
    window = _FakeMenuWindow(0.0, 0.0)
    instance.window = lambda: window
    seen = []
    instance.on_menu = lambda ev: seen.append(ev)

    instance.mouseDown_("down")
    window._x, window._y = 50.0, 5.0   # the panel moved: setMovableByWindowBackground_ dragged it
    instance.mouseUp_("up")

    assert seen == []


def test_webview_click_without_on_menu_is_noop(monkeypatch):
    instance = _draggable_instance(monkeypatch)
    window = _FakeMenuWindow(1.0, 1.0)
    instance.window = lambda: window
    instance.mouseDown_("down")
    instance.mouseUp_("up")  # must not raise; on_menu defaults to None


def test_popup_menu_invokes_on_menu_with_coords():
    h, _, _, _ = make()
    seen = []
    h.on_menu = lambda x, y: seen.append((x, y))
    h._popup_menu(10.0, 20.0)
    assert seen == [(10.0, 20.0)]


def test_popup_menu_noop_when_on_menu_unset():
    h, _, _, _ = make()
    h._popup_menu(1.0, 2.0)  # must not raise


def test_webview_click_triggers_popup_menu_via_screen_coords(monkeypatch):
    fake_point = types.SimpleNamespace(x=111.0, y=222.0)
    fake_appkit = types.SimpleNamespace(NSEvent=types.SimpleNamespace(mouseLocation=lambda: fake_point))
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)

    h, web, _, _ = make()
    seen = []
    h.on_menu = lambda x, y: seen.append((x, y))

    web.on_menu(object())  # simulate the draggable web view's click handler firing

    assert seen == [(111.0, 222.0)]


def test_close_before_load_discards_pending_queue():
    h, web, _, _ = make(mark_loaded=False)
    h.push({"kind": "mic", "payload": 0.1})
    assert web.js == []

    h.close()
    h.mark_loaded()

    assert web.js == []


# -- display changes / off-screen recovery -----------------------------------

def test_saved_position_on_a_vanished_screen_falls_back_to_default_and_clears_pref():
    """The bug: frame=(870,-39 540x300) after a monitor went away. A saved
    origin that no longer lands (>= 25%) on any current screen must be
    discarded — top-right default on the main screen, pref cleared."""
    saved = {}
    h, _, panel, _ = make(
        prefs_load=lambda: {"hud_pos_full": [2000.0, 300.0]},   # was on a second display
        prefs_save=lambda p: saved.update(p),
    )
    x, y, w, hgt = panel.setFrame_calls[-1]
    assert (x, y) == (1440 - 540 - h.s.hud_margin, 900 - 300 - h.s.hud_margin)
    assert h._pos["full"] is None
    assert saved == {"hud_pos_full": None}


def test_half_off_screen_saved_frame_is_clamped_inside_best_screen():
    saved = {}
    h, _, panel, _ = make(
        prefs_load=lambda: {"hud_pos_full": [870.0, -39.0]},
        prefs_save=lambda p: saved.update(p),
    )
    x, y, w, hgt = panel.setFrame_calls[-1]
    assert (x, y, w, hgt) == (870.0, 0.0, 540, 300)
    assert h._pos["full"] == (870.0, -39.0)   # still the user's spot; only the frame is clamped
    assert saved == {}


def test_position_on_secondary_screen_stays_on_secondary():
    secondary = FakeRect(1440, 100, 1920, 1080)
    h, _, panel, _ = make(
        screens=[MAIN_SCREEN, secondary],
        prefs_load=lambda: {"hud_pos_full": [2500.0, 500.0]},
    )
    assert panel.setFrame_calls[-1][:2] == (2500.0, 500.0)
    # Partly past the secondary's right edge: clamped against the secondary,
    # not dragged back to the main display.
    h._pos["full"] = (3200.0, 500.0)
    h._apply_geometry()
    assert panel.setFrame_calls[-1][:2] == (1440 + 1920 - 540, 500.0)


def test_screen_change_callback_repositions_hud(caplog):
    handlers = []
    frames = [FakeRect(0, 0, 2560, 1440)]
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_pos_full": [2000.0, 1100.0]}, prefs_save=_noop_prefs_save,
        screens=lambda: frames, subscribe_screen_changes=handlers.append,
    )
    assert len(handlers) == 1
    assert panel.setFrame_calls[-1][:2] == (2000.0, 1100.0)
    # The big display goes away; only the laptop panel is left.
    frames[:] = [MAIN_SCREEN]
    before = len(panel.setFrame_calls)
    with caplog.at_level("INFO", logger="veronica.ui.hud"):
        # A burst of notifications (a monitor connecting fires dozens) is
        # coalesced: nothing moves until they've been quiet for a moment.
        for _ in range(50):
            handlers[0]()
        h.tick()
        assert len(panel.setFrame_calls) == before
        h._clock = lambda: time.monotonic() + 1.0
        h.tick()
        h.tick()
    assert len(panel.setFrame_calls) == before + 1
    assert panel.setFrame_calls[-1][:2] == (1440 - 540 - h.s.hud_margin, 900 - 300 - h.s.hud_margin)
    assert caplog.text.count("display change: repositioning HUD") == 1


def test_screen_change_subscription_unsubscribes_on_close():
    unsubscribed = []

    def subscribe(handler):
        return lambda: unsubscribed.append(handler)

    h, _, _, _ = make(subscribe_screen_changes=subscribe)
    h.close()
    assert len(unsubscribed) == 1
    # After close the handler is inert (no reposition of a closed panel).
    calls_before = len(h._panel.setFrame_calls)
    unsubscribed[0]()
    h._on_screens_changed()
    h._clock = lambda: time.monotonic() + 1.0
    h.tick()
    assert len(h._panel.setFrame_calls) == calls_before


def test_screen_change_subscription_failure_is_soft():
    def bad(handler):
        raise RuntimeError("no notification center")

    h, _, _, _ = make(subscribe_screen_changes=bad)
    assert h.available


def test_save_position_clamps_before_persisting():
    saved = {}
    h, _, panel, _ = make(prefs_save=lambda p: saved.update(p))
    panel._rect = FakeRect(870.0, -39.0, 540, 300)
    h.hide()
    assert saved == {"hud_pos_full": [870.0, 0.0]}
    assert h._pos["full"] == (870.0, 0.0)


def test_reset_position_clears_prefs_repositions_and_shows():
    saved = {}
    h, web, panel, _ = make(
        prefs_load=lambda: {"hud_pos_full": [12.0, 34.0], "hud_pos_mini": [56.0, 78.0]},
        prefs_save=lambda p: saved.update(p),
    )
    h.reset_position()
    assert h._pos == {"full": None, "mini": None}
    assert saved == {"hud_pos_full": None, "hud_pos_mini": None}
    assert panel.setFrame_calls[-1][:2] == (1440 - 540 - h.s.hud_margin, 900 - 300 - h.s.hud_margin)
    assert panel.visible and panel.alpha == 1.0
    assert "window.hud.setVisible(true)" in web.js


def test_no_screen_info_keeps_saved_position_untouched():
    """Without any screen list (AppKit missing / query failed) there's
    nothing to validate against: don't second-guess the saved spot."""
    h, _, panel, _ = make(screens=[], prefs_load=lambda: {"hud_pos_full": [12.0, 34.0]})
    assert panel.setFrame_calls[-1][:2] == (12.0, 34.0)
    assert h._pos["full"] == (12.0, 34.0)


def test_real_screen_change_subscription_uses_notification_center(monkeypatch):
    """The default subscription path: an NSObject observer registered with
    NSNotificationCenter for NSApplicationDidChangeScreenParametersNotification,
    removed again on close."""
    import types as _types

    registrations, removals = [], []

    class FakeCenter:
        def addObserver_selector_name_object_(self, observer, selector, name, obj):
            registrations.append((observer, selector, name, obj))

        def removeObserver_(self, observer):
            removals.append(observer)

    center = FakeCenter()

    class FakeObserver:
        def __init__(self, callback):
            self.callback = callback

    class FakeObserverCls:
        @staticmethod
        def alloc():
            return _types.SimpleNamespace(initWithCallback_=lambda cb: FakeObserver(cb))

    from veronica.ui import hud as hud_module
    monkeypatch.setattr(hud_module, "_screen_observer_class", lambda: FakeObserverCls)
    monkeypatch.setitem(sys.modules, "Foundation", _types.SimpleNamespace(
        NSNotificationCenter=_types.SimpleNamespace(defaultCenter=lambda: center), NSMakeRect=FakeRect))
    monkeypatch.setitem(sys.modules, "AppKit", _types.SimpleNamespace(
        NSApplicationDidChangeScreenParametersNotification="NSApplicationDidChangeScreenParametersNotification"))

    class RealSubscriptionHud(HudWindow):
        # Test fakes pass a webview_factory, which (like the nav delegate)
        # turns the real subscription off; opt back in explicitly.
        def __init__(self, *a, **k):
            super().__init__(*a, subscribe_screen_changes=self._subscribe_screen_changes_real, **k)

    frames = [MAIN_SCREEN]
    web, panel = FakeWeb(), FakePanel()
    h = RealSubscriptionHud(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
        screens=lambda: frames,
    )
    assert len(registrations) == 1
    observer, selector, name, obj = registrations[0]
    assert isinstance(observer, FakeObserver)
    assert (selector, name, obj) == ("onScreens:", "NSApplicationDidChangeScreenParametersNotification", None)
    assert h._screen_observer is observer   # strong ref kept on the window
    observer.callback()
    h._clock = lambda: time.monotonic() + 1.0
    h.tick()
    assert panel.setFrame_calls[-1][:2] == (1440 - 540 - h.s.hud_margin, 900 - 300 - h.s.hud_margin)
    h.close()
    assert removals == [observer]
