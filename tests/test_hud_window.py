import json
import sys
import types

from veronica.config import Settings
from veronica.ui.hud import HudWindow


class FakeWeb:
    def __init__(self): self.js = []
    def evaluateJavaScript_completionHandler_(self, js, cb): self.js.append(js)


class FakePanel:
    def __init__(self): self.alpha = 0.0; self.visible = False; self.orders = []
    def setAlphaValue_(self, a): self.alpha = a
    def orderFrontRegardless(self): self.visible = True; self.orders.append("front")
    def orderOut_(self, _): self.visible = False; self.orders.append("out")


def make(t=[0.0]):
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
                  clock=lambda: t[0], main=lambda fn: fn())
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
    h = HudWindow(Settings(), webview_factory=bad, panel_factory=lambda s, w: FakePanel(), main=lambda fn: fn())
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
    h = HudWindow(Settings(), webview_factory=lambda s: web, panel_factory=lambda s, w: panel, main=lambda fn: fn())

    h.hide()
    assert "out" not in panel.orders
    assert panel.alpha == 0.0

    FakeNSAnimationContext.ctx.completion()
    assert "out" in panel.orders
