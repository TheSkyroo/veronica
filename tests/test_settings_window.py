"""SettingsWindow with a fake pywebview window: show/hide, the JS load gate,
close-to-hide, and js_api dispatch to a fake bridge. No pywebview, no WebView2."""
import json
import sys
import types

from veronica.config import Settings
from veronica.ui import settings as settings_module
from veronica.ui.settings import LONG_COMMANDS, SettingsWindow


class FakeWindow:
    def __init__(self):
        self.js = []
        self.orders = []
        self.visible = False
        self.destroyed = False

    def run_js(self, js): self.js.append(js)
    def show(self): self.visible = True; self.orders.append("front")
    def hide(self): self.visible = False; self.orders.append("out")
    def destroy(self): self.destroyed = True


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


def make(mark_loaded=True, show=True):
    """The window is created on first show(): by default show it (and clear
    what that queued), so tests start from an open, loaded window."""
    win, bridge = FakeWindow(), FakeBridge()
    activated, created = [], []

    def factory(s, owner, hidden):
        created.append(hidden)
        win.visible = not hidden
        return win

    sw = SettingsWindow(
        Settings(), bridge,
        window_factory=factory,
        main=lambda fn: fn(),
        activate=lambda w: activated.append(w),
    )
    if show:
        sw.show()
        if mark_loaded:
            sw.mark_loaded()
        win.js = []
        win.orders = []
        activated.clear()
    return sw, win, win, bridge, activated


def test_available_and_bridge_wired():
    sw, _, _, bridge, _ = make()
    assert sw.available
    assert bridge.on_state_changed == sw.push_state


def test_window_created_lazily_on_first_show():
    created = []
    win, bridge = FakeWindow(), FakeBridge()

    def factory(s, owner, hidden):
        created.append((owner, hidden))
        return win

    sw = SettingsWindow(Settings(), bridge, window_factory=factory, main=lambda fn: fn(), activate=lambda w: None)
    assert created == []                      # nothing until it's needed
    sw.show("voice")
    assert created == [(sw, False)]           # created visible: no separate show()
    assert win.orders == []
    sw.hide(); sw.show()
    assert created == [(sw, False)]           # reused afterwards
    assert win.orders == ["out", "front"]


def test_create_hidden_makes_the_window_up_front():
    created = []
    win = FakeWindow()
    sw = SettingsWindow(Settings(), FakeBridge(), window_factory=lambda s, o, hidden: created.append(hidden) or win,
                        main=lambda fn: fn(), activate=lambda w: None)
    sw.create_hidden()
    assert created == [True]
    sw.show()
    assert created == [True] and win.orders == ["front"]


def test_factory_failure_leaves_window_unavailable():
    def boom(s, owner, hidden):
        raise RuntimeError("no webview2")
    sw = SettingsWindow(Settings(), FakeBridge(), window_factory=boom, main=lambda fn: fn(), activate=lambda w: None)
    sw.show()
    assert not sw.available
    sw.show(); sw.hide(); sw.push_state({}); sw.close()   # all no-ops, never raise


def test_show_brings_front_activates_selects_tab_and_pushes_state():
    sw, web, win, bridge, activated = make()
    sw.hide()
    win.orders = []
    sw.show("history")
    assert win.orders == ["front"] and activated == [win]
    assert web.js[0] == 'window.settings.select("history")'
    assert web.js[1] == "window.settings.state(" + json.dumps(bridge.state, ensure_ascii=False) + ")"


def test_show_defaults_to_general():
    sw, web, _, _, _ = make()
    sw.show()
    assert web.js[0] == 'window.settings.select("general")'
    sw.show("nonsense")
    assert web.js[2] == 'window.settings.select("general")'


def test_hide_hides():
    sw, _, win, _, _ = make()
    sw.hide()
    assert win.orders == ["out"] and not win.visible


def test_closing_hides_instead_unless_quitting():
    sw, _, win, _, _ = make()
    assert sw._on_closing() is False          # cancels the close...
    assert win.orders == ["out"]              # ...and hides instead
    sw.close()
    assert win.destroyed
    assert sw._on_closing() is True           # quitting: let it go


def test_closed_window_is_recreated_on_next_show():
    created = []

    def factory(s, owner, hidden):
        w = FakeWindow()
        created.append(w)
        return w

    sw = SettingsWindow(Settings(), FakeBridge(), window_factory=factory, main=lambda fn: fn(), activate=lambda w: None)
    sw.show(); sw.mark_loaded()
    sw._on_closed()
    sw.push_state({"a": 1})                   # no window: dropped, never raises
    sw.show()
    assert len(created) == 2
    assert created[1].js == []                # new page not loaded yet: queued
    sw.mark_loaded()
    assert created[1].js[0] == 'window.settings.select("general")'


def test_js_queued_until_loaded():
    sw, web, win, _, _ = make(show=False)
    sw.show("voice")
    sw.push_state({"a": 1})
    assert web.js == []                       # page not loaded yet
    assert win.visible                        # but the window is up
    sw.mark_loaded()
    assert web.js[0] == 'window.settings.select("voice")'
    assert web.js[-1] == 'window.settings.state({"a": 1})'
    sw.push_state({"b": 2})
    assert web.js[-1] == 'window.settings.state({"b": 2})'   # direct once loaded


def test_push_state_before_the_window_exists_is_dropped():
    sw, web, _, _, _ = make(show=False)
    sw.push_state({"a": 1})
    sw.show(); sw.mark_loaded()
    assert 'window.settings.state({"a": 1})' not in web.js


def test_push_state_serializes_unicode():
    sw, web, _, _, _ = make()
    sw.push_state({"x": "héllo \"q\""})
    assert web.js == ['window.settings.state(' + json.dumps({"x": "héllo \"q\""}, ensure_ascii=False) + ')']


def test_js_api_handle_dispatches_and_replies_with_id():
    sw, web, _, bridge, _ = make()
    assert sw.api.handle(7, "set", {"section": "brain", "key": "effort", "value": "high"}) is None
    assert bridge.calls == [("set", {"section": "brain", "key": "effort", "value": "high"})]
    assert bridge.threads == []               # short command: inline
    assert web.js == ['window.settings.reply(7, ' + json.dumps({"ok": True, "message": "", "cmd": "set"}) + ')']


def test_js_api_handle_hops_to_the_ui_thread():
    hops = []
    win, bridge = FakeWindow(), FakeBridge()
    sw = SettingsWindow(Settings(), bridge, window_factory=lambda s, o, h: win,
                        main=lambda fn: hops.append(fn), activate=lambda w: None)
    sw.api.handle(1, "restart", None)
    assert bridge.calls == [] and len(hops) == 1
    hops[0]()
    assert bridge.calls == [("restart", {})]


def test_js_api_exposes_only_handle():
    sw, _, _, _, _ = make()
    assert [n for n in dir(sw.api) if not n.startswith("_")] == ["handle"]


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
    sw._on_message({"id": 9, "cmd": "set", "args": "not a dict"})
    assert bridge.calls == [("set", {})]      # bad args become {}
    assert len(web.js) == 1


def test_reply_marshals_through_main():
    hops = []
    win, bridge = FakeWindow(), FakeBridge()

    def main(fn):
        hops.append(fn); fn()
    sw = SettingsWindow(Settings(), bridge, window_factory=lambda s, o, h: win, main=main, activate=lambda w: None)
    sw.show(); sw.mark_loaded(); win.js = []; hops.clear()
    sw._on_message({"id": 2, "cmd": "check_update", "args": {}})
    assert hops                                # reply from the worker hopped to the UI thread
    assert win.js and win.js[0].startswith("window.settings.reply(2, ")


def test_state_change_from_bridge_pushes_state():
    _sw, web, _, bridge, _ = make()
    bridge.on_state_changed({"meta": {"restart_required": True}})
    assert web.js == ['window.settings.state({"meta": {"restart_required": true}})']


def test_push_state_never_raises_when_main_fails():
    win, bridge = FakeWindow(), FakeBridge()

    def bad_main(fn):
        raise RuntimeError("no UI thread")
    sw = SettingsWindow(Settings(), bridge, window_factory=lambda s, o, h: win, main=bad_main, activate=lambda w: None)
    sw.push_state({"a": 1})                    # swallowed, logged
    sw.api.handle(1, "set", {})                # likewise


def test_js_failure_is_soft():
    class BrokenWindow(FakeWindow):
        def run_js(self, js): raise RuntimeError("destroyed")

    win = BrokenWindow()
    sw = SettingsWindow(Settings(), FakeBridge(), window_factory=lambda s, o, h: win, main=lambda fn: fn(),
                        activate=lambda w: None)
    sw.show(); sw.mark_loaded()
    sw.push_state({"a": 1})                    # logged, not raised


def test_plain_converts_mapping_and_sequence_bodies():
    from collections import UserDict

    from veronica.ui.settings import _plain

    body = UserDict({"id": 3, "cmd": "set", "args": UserDict({"value": ("a", "b"), "n": 1.5, "ok": True, "none": None})})
    out = _plain(body)
    assert out == {"id": 3, "cmd": "set", "args": {"value": ["a", "b"], "n": 1.5, "ok": True, "none": None}}
    assert type(out) is dict and type(out["args"]) is dict and type(out["args"]["value"]) is list
    assert _plain(b"bytes") == "bytes"


def test_on_message_accepts_userdict_body():
    from collections import UserDict

    sw, web, _, bridge, _ = make()
    sw._on_message(UserDict({"id": 4, "cmd": "set", "args": UserDict({"section": "brain", "key": "effort", "value": "low"})}))
    assert bridge.calls == [("set", {"section": "brain", "key": "effort", "value": "low"})]
    assert type(bridge.calls[0][1]) is dict
    assert web.js[0].startswith("window.settings.reply(4, ")


def test_real_window_is_a_normal_titled_window_wired_to_the_api(monkeypatch):
    class Event:
        def __init__(self): self.handlers = []
        def __iadd__(self, fn): self.handlers.append(fn); return self

    created = {}
    window = types.SimpleNamespace(events=types.SimpleNamespace(loaded=Event(), closing=Event(), closed=Event()))

    def create_window(title, **kw):
        created.update(kw, title=title)
        return window

    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=create_window))
    sw = SettingsWindow(Settings(), FakeBridge(), main=lambda fn: fn(), activate=lambda w: None)
    sw.create_hidden()
    assert created["title"] == settings_module.TITLE
    assert created["js_api"] is sw.api
    assert created["hidden"] is True
    assert (created["width"], created["height"]) == (settings_module.WIDTH, settings_module.HEIGHT)
    assert created["min_size"] == settings_module.MIN_SIZE
    assert created["url"].startswith("file:") and created["url"].endswith("/settings/index.html")
    assert window.events.closing.handlers == [sw._on_closing]
    assert window.events.closed.handlers == [sw._on_closed]
    window.events.loaded.handlers[0]()
    assert sw._loaded
