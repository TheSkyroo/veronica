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


def make(t=[0.0], mark_loaded=True, **settings_over):
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0, **settings_over),
        webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        clock=lambda: t[0], main=lambda fn: fn(),
        prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
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
    )
    h.set_mode("mini")
    assert saved == {"hud_mode": "mini"}


def test_construction_applies_saved_mini_mode():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_mode": "mini"}, prefs_save=_noop_prefs_save,
    )
    assert h._mode == "mini"
    assert panel.setFrame_calls[-1][2:] == (400, 72)


def test_construction_falls_back_to_settings_hud_mode_when_no_saved_pref():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0, hud_mode="mini"), webview_factory=lambda s: web,
        panel_factory=lambda s, w: panel, main=lambda fn: fn(),
        prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
    )
    assert h._mode == "mini"


def test_construction_ignores_invalid_saved_mode():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_mode": "gigantic"}, prefs_save=_noop_prefs_save,
    )
    assert h._mode == "full"


def test_construction_applies_saved_position():
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(
        Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
        main=lambda fn: fn(), prefs_load=lambda: {"hud_pos_full": [12.0, 34.0]}, prefs_save=_noop_prefs_save,
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
    )
    h.set_mode("mini")
    # Simulate the user dragging the (now mini) panel to a new spot.
    panel._rect = FakeRect(7.0, 8.0, 400, 72)
    saved.clear()
    h.hide()
    assert saved == {"hud_pos_mini": [7.0, 8.0]}
    assert h._pos["mini"] == (7.0, 8.0)
    assert h._pos["full"] is None


def test_set_mode_mini_defaults_to_top_center_below_menubar(monkeypatch):
    """With no saved position for mini mode, switching to mini should place
    the compact bar centered horizontally under the menu bar (a
    notch/Dynamic-Island style default), not the full card's top-right
    corner."""
    import types as _types

    frame = _types.SimpleNamespace(
        origin=_types.SimpleNamespace(x=0, y=0),
        size=_types.SimpleNamespace(width=1440, height=900),
    )
    fake_appkit = _types.SimpleNamespace(
        NSScreen=_types.SimpleNamespace(mainScreen=lambda: _types.SimpleNamespace(visibleFrame=lambda: frame)),
    )
    monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)

    h, web, panel, _ = make()
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


def test_close_before_load_discards_pending_queue():
    h, web, _, _ = make(mark_loaded=False)
    h.push({"kind": "mic", "payload": 0.1})
    assert web.js == []

    h.close()
    h.mark_loaded()

    assert web.js == []
