import json

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
