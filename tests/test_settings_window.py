"""SettingsWindow with a fake web view/window: show/hide, the JS load gate,
and script-message dispatch to a fake bridge. No AppKit, no WebKit."""
import json

from veronica.config import Settings
from veronica.ui.settings import LONG_COMMANDS, SettingsWindow


class FakeWeb:
    def __init__(self): self.js = []
    def evaluateJavaScript_completionHandler_(self, js, cb): self.js.append(js)


class FakeWindow:
    def __init__(self):
        self.orders = []
        self.visible = False

    def makeKeyAndOrderFront_(self, _): self.visible = True; self.orders.append("front")
    def orderOut_(self, _): self.visible = False; self.orders.append("out")


class FakeBridge:
    def __init__(self):
        self.calls = []
        self.threads = []
        self.on_state_changed = None
        self.state = {"general": {"language": "en"}, "meta": {"restart_required": False, "fields": {}}}
        self.results = {}
        self.raise_on = None

    def get_state(self):
        return self.state

    def handle(self, cmd, args):
        self.calls.append((cmd, args))
        if cmd == self.raise_on:
            raise RuntimeError("boom")
        return self.results.get(cmd, {"ok": True, "message": "", "cmd": cmd})

    def run_thread(self, fn):
        # Record and run inline: tests are synchronous.
        self.threads.append(fn)
        fn()


def make(mark_loaded=True):
    web, win, bridge = FakeWeb(), FakeWindow(), FakeBridge()
    activated = []
    sw = SettingsWindow(
        Settings(), bridge,
        webview_factory=lambda s, owner: web,
        window_factory=lambda s, w, owner: win,
        main=lambda fn: fn(),
        activate=lambda: activated.append(True),
    )
    if mark_loaded:
        sw.mark_loaded()
        web.js = []
    return sw, web, win, bridge, activated


def test_available_and_bridge_wired():
    sw, _, _, bridge, _ = make()
    assert sw.available
    assert bridge.on_state_changed == sw.push_state


def test_factory_failure_leaves_window_unavailable():
    def boom(s, owner):
        raise RuntimeError("no webkit")
    sw = SettingsWindow(Settings(), FakeBridge(), webview_factory=boom,
                        window_factory=lambda s, w, o: FakeWindow(), main=lambda fn: fn(), activate=lambda: None)
    assert not sw.available
    sw.show(); sw.hide(); sw.push_state({})   # all no-ops, never raise


def test_show_orders_front_activates_selects_tab_and_pushes_state():
    sw, web, win, bridge, activated = make()
    sw.show("history")
    assert win.orders == ["front"] and activated == [True]
    assert web.js[0] == 'window.settings.select("history")'
    assert web.js[1] == "window.settings.state(" + json.dumps(bridge.state, ensure_ascii=False) + ")"


def test_show_defaults_to_general():
    sw, web, _, _, _ = make()
    sw.show()
    assert web.js[0] == 'window.settings.select("general")'


def test_hide_orders_out():
    sw, _, win, _, _ = make()
    sw.show(); sw.hide()
    assert win.orders == ["front", "out"] and not win.visible


def test_js_queued_until_loaded():
    sw, web, win, _, _ = make(mark_loaded=False)
    sw.show("voice")
    sw.push_state({"a": 1})
    assert web.js == []                       # page not loaded yet
    assert win.orders == ["front"]            # but the window is up
    sw.mark_loaded()
    assert web.js[0] == 'window.settings.select("voice")'
    assert web.js[-1] == 'window.settings.state({"a": 1})'
    sw.push_state({"b": 2})
    assert web.js[-1] == 'window.settings.state({"b": 2})'   # direct once loaded


def test_push_state_serializes_unicode():
    sw, web, _, _, _ = make()
    sw.push_state({"x": "héllo \"q\""})
    assert web.js == ['window.settings.state(' + json.dumps({"x": "héllo \"q\""}, ensure_ascii=False) + ')']


def test_on_message_dispatches_and_replies_with_id():
    sw, web, _, bridge, _ = make()
    sw._on_message({"id": 7, "cmd": "set", "args": {"section": "brain", "key": "effort", "value": "high"}})
    assert bridge.calls == [("set", {"section": "brain", "key": "effort", "value": "high"})]
    assert bridge.threads == []               # short command: inline
    assert web.js == ['window.settings.reply(7, ' + json.dumps({"ok": True, "message": "", "cmd": "set"}) + ')']


def test_on_message_without_args():
    sw, web, _, bridge, _ = make()
    sw._on_message({"id": 1, "cmd": "restart"})
    assert bridge.calls == [("restart", {})]
    assert web.js[0].startswith("window.settings.reply(1, ")


def test_long_commands_run_on_thread_and_reply_when_done():
    sw, web, _, bridge, _ = make()
    for i, cmd in enumerate(sorted(LONG_COMMANDS), start=1):
        sw._on_message({"id": i, "cmd": cmd, "args": {}})
    assert {"check_update", "update_now"} <= set(LONG_COMMANDS)
    assert len(bridge.threads) == len(LONG_COMMANDS)
    assert [c for c, _ in bridge.calls] == sorted(LONG_COMMANDS)
    assert len(web.js) == len(LONG_COMMANDS)
    assert web.js[0].startswith("window.settings.reply(1, ")


def test_bridge_exception_still_replies():
    sw, web, _, bridge, _ = make()
    bridge.raise_on = "set"
    sw._on_message({"id": 3, "cmd": "set", "args": {}})
    reply = json.loads(web.js[0][len("window.settings.reply(3, "):-1])
    assert reply["ok"] is False and "boom" in reply["message"]


def test_malformed_message_is_ignored():
    sw, web, _, bridge, _ = make()
    sw._on_message("nope")
    sw._on_message({"cmd": "set"})            # no id: nothing to reply to
    sw._on_message({"id": 9})                 # no cmd
    assert bridge.calls == [] and web.js == []


def test_reply_marshals_through_main():
    hops = []
    web, win, bridge = FakeWeb(), FakeWindow(), FakeBridge()

    def main(fn):
        hops.append(fn); fn()
    sw = SettingsWindow(Settings(), bridge, webview_factory=lambda s, o: web,
                        window_factory=lambda s, w, o: win, main=main, activate=lambda: None)
    sw.mark_loaded(); web.js = []; hops.clear()
    sw._on_message({"id": 2, "cmd": "check_update", "args": {}})
    assert hops                                # reply from the worker hopped to main
    assert web.js and web.js[0].startswith("window.settings.reply(2, ")


def test_state_change_from_bridge_pushes_state():
    _sw, web, _, bridge, _ = make()
    bridge.on_state_changed({"meta": {"restart_required": True}})
    assert web.js == ['window.settings.state({"meta": {"restart_required": true}})']


def test_push_state_never_raises_when_main_fails():
    web, win, bridge = FakeWeb(), FakeWindow(), FakeBridge()

    def bad_main(fn):
        raise RuntimeError("no main loop")
    sw = SettingsWindow(Settings(), bridge, webview_factory=lambda s, o: web,
                        window_factory=lambda s, w, o: win, main=bad_main, activate=lambda: None)
    sw.push_state({"a": 1})                    # swallowed, logged
