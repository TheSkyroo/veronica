import asyncio
import importlib
import logging
import sys
import threading
import types

import pytest
import rumps as real_rumps


class FakeMenuItem:
    def __init__(self, title, callback=None):
        self.title = title
        self.callback = callback
        self.state = False
        self.children = []  # submenu entries; None models rumps' add(None) separator

    def add(self, item):
        self.children.append(item)


class FakeTimer:
    def __init__(self, callback, interval):
        self.callback = callback
        self.interval = interval

    def start(self):
        pass


class FakeApp:
    def __init__(self, title, quit_button=None):
        self.title = title
        self.quit_button = quit_button

    def run(self):
        pass


class FakeRumps:
    App = FakeApp
    MenuItem = FakeMenuItem
    Timer = FakeTimer

    def __init__(self):
        self.quit_called = False

    def quit_application(self):
        self.quit_called = True


class FakePlayer:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeHud:
    def __init__(self, *_a, **_k):
        self.pushed = []
        self.states = []
        self.ticks = 0
        self.closed = False
        self.available = True
        self.log = []  # combined order of on_state/push calls
        self._mode = "full"
        self.mode_calls = []
        self.hide_calls = 0

    def push(self, event):
        self.pushed.append(event)
        self.log.append(("push", event["kind"]))

    def on_state(self, state):
        self.states.append(state)
        self.log.append(("state", state))

    def tick(self):
        self.ticks += 1

    def close(self):
        self.closed = True

    def hide(self):
        self.hide_calls += 1

    def set_mode(self, mode):
        self._mode = mode
        self.mode_calls.append(mode)


class FakeOrch:
    def __init__(self, on_state=None):
        self.player = FakePlayer()
        self._on_state = on_state
        self.started = threading.Event()

    async def warmup(self):
        # mirrors the real Orchestrator.warmup(), which ends by emitting
        # on_state("idle") once models are loaded — menubar now sets
        # app._state = "warming" itself before build_orchestrator runs, so
        # tests that assert the post-construction state need this to flip
        # back to idle the way the real orchestrator would.
        if self._on_state is not None:
            self._on_state("idle")

    async def run_forever(self):
        self.started.set()
        await asyncio.Event().wait()


class FakeHotkeyMonitor:
    """Stand-in for veronica.audio.hotkey.HotkeyMonitor: no real Quartz
    CGEventTap, no real thread — just records what it was asked to do so
    tests can drive on_press/on_release directly."""
    instances = []
    available_on_start = True

    def __init__(self, on_press, on_release, keycode=61):
        self.on_press = on_press
        self.on_release = on_release
        self.keycode = keycode
        self.available = True
        self.started_with_loop = None
        self.stopped = False
        FakeHotkeyMonitor.instances.append(self)

    def start(self, loop=None):
        self.started_with_loop = loop
        self.available = FakeHotkeyMonitor.available_on_start

    def stop(self):
        self.stopped = True


@pytest.fixture
def fake_env(monkeypatch, tmp_home, request):
    # `class VeronicaApp(rumps.App)` binds its base class at class-definition
    # time (i.e. first import), so a plain setattr on the already-imported
    # module wouldn't swap the base class rumps.App is derived from. Patch
    # sys.modules with the fake *before* (re)importing/reloading the module
    # so the class statement picks up the fake App/MenuItem/Timer, and no
    # real AppKit machinery is ever touched.
    fake_rumps = FakeRumps()
    monkeypatch.setitem(sys.modules, "rumps", fake_rumps)
    if "veronica.ui.menubar" in sys.modules:
        menubar = importlib.reload(sys.modules["veronica.ui.menubar"])
    else:
        import veronica.ui.menubar as menubar
    assert menubar.rumps is fake_rumps

    def _restore_real_rumps():
        # fixture finalizers run before the fixtures they depend on (here,
        # monkeypatch) are torn down, so do this ourselves rather than rely
        # on monkeypatch's own sys.modules undo: reinstate the real module
        # and reload menubar so it binds back to it, leaving no fake behind
        # for tests/imports that run after this fixture is torn down.
        sys.modules["rumps"] = real_rumps
        importlib.reload(menubar)
        assert menubar.rumps is real_rumps

    request.addfinalizer(_restore_real_rumps)

    # Written on the main thread, before VeronicaApp() ever starts its
    # background thread, so _make_app() waiting on it never races the key
    # itself — only the .set() (done from the background thread once
    # build_orchestrator actually ran) is awaited.
    orch_holder = {"ready": threading.Event()}

    def fake_build_orchestrator(s, on_state=None, on_event=None, *, audio=True, on_quit=None):
        orch = FakeOrch(on_state=on_state)
        orch.on_quit = on_quit
        orch_holder["orch"] = orch
        # VeronicaApp() (on the main thread) can return before the
        # background thread it starts has run build_orchestrator and
        # populated orch_holder — signal readiness explicitly rather than
        # racing a bare dict read.
        orch_holder["ready"].set()
        return orch

    monkeypatch.setattr(menubar, "build_orchestrator", fake_build_orchestrator)
    monkeypatch.setattr(menubar, "HudWindow", FakeHud)
    FakeHotkeyMonitor.instances = []
    FakeHotkeyMonitor.available_on_start = True
    monkeypatch.setattr(menubar, "HotkeyMonitor", FakeHotkeyMonitor)
    return menubar, fake_rumps, orch_holder


def _make_app(menubar, orch_holder):
    app = menubar.VeronicaApp()
    # VeronicaApp() can return before the background thread it starts has
    # reached build_orchestrator and populated orch_holder — wait for that
    # explicitly instead of racing a bare dict read (this was the source of
    # an intermittent KeyError: 'orch').
    assert orch_holder["ready"].wait(2), "build_orchestrator was not called within 2s"
    orch = orch_holder["orch"]
    assert orch.started.wait(2), "background loop did not start within 2s"
    return app, orch


def _quit_and_join(app):
    app.quit(None)
    app._thread.join(timeout=2)


def test_state_is_warming_during_build_orchestrator(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    seen = {}
    original = menubar.build_orchestrator

    def wrapped(s, on_state=None, on_event=None, *, audio=True, on_quit=None):
        # on_state is the VeronicaApp instance's bound _on_state method, so
        # __self__ recovers the app without racing its constructor's
        # `app = VeronicaApp()` assignment on the main thread.
        app = on_state.__self__
        seen["state"] = app._state
        return original(s, on_state=on_state, on_event=on_event, audio=audio, on_quit=on_quit)

    monkeypatch.setattr(menubar, "build_orchestrator", wrapped)
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert seen.get("state") == "warming"
    finally:
        _quit_and_join(app)


def test_construct_starts_loop_and_initial_refresh(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._refresh(None)
        assert app.title == "V ◯"
    finally:
        _quit_and_join(app)


def test_on_state_listening_updates_title(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._on_state("listening")
        app._refresh(None)
        assert app.title == "V ◉"
    finally:
        _quit_and_join(app)


def test_on_state_warming_updates_title(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._on_state("warming")
        app._refresh(None)
        assert app.title == "V …"
    finally:
        _quit_and_join(app)


def test_on_state_confirming_updates_title(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._on_state("confirming")
        app._refresh(None)
        assert app.title == "V ?"
    finally:
        _quit_and_join(app)


def test_toggle_mute(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        mute_item = app.menu[0]

        app.toggle_mute(mute_item)
        assert app._muted is True
        assert orch.player.stopped is True
        assert orch.muted is True
        app._refresh(None)
        assert app.title == "V zz"

        app.toggle_mute(mute_item)
        assert app._muted is False
        assert orch.muted is False
        app._refresh(None)
        assert app.title == "V ◯"
    finally:
        _quit_and_join(app)


def test_quit_calls_rumps_quit_application(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    _quit_and_join(app)
    assert fake_rumps.quit_called is True


def test_quit_does_not_log_error(fake_env, caplog):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    with caplog.at_level(logging.ERROR, logger="veronica.ui"):
        _quit_and_join(app)
    error_records = [r for r in caplog.records if r.name == "veronica.ui" and r.levelno >= logging.ERROR]
    assert error_records == []
    assert app._state != "error"


def test_events_drained_to_hud(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    app._events.put(("mic", 0.1)); app._events.put(("mic", 0.9)); app._events.put(("state", "listening")); app._events.put(("heard", "hi"))
    app._drain(None)
    kinds = [e["kind"] for e in app._hud.pushed]
    assert kinds.count("mic") == 1 and app._hud.pushed[[i for i, e in enumerate(app._hud.pushed) if e["kind"] == "mic"][0]]["payload"] == 0.9
    assert app._hud.states == ["listening"] and app._hud.ticks == 1
    # exact push order: state, then heard, then the coalesced mic last
    assert kinds == ["state", "heard", "mic"]
    # on_state("listening") is recorded before the corresponding state push
    state_call_idx = app._hud.log.index(("state", "listening"))
    state_push_idx = app._hud.log.index(("push", "state"))
    assert state_call_idx < state_push_idx
    _quit_and_join(app)


def test_drain_overflow_drops_mic(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    # "heard" is queued first so it's within the first 64 popped by a single
    # _drain() call, alongside enough mic events (queued after it) to push
    # qsize() over the 1000 overflow threshold at the start of that call.
    app._events.put(("heard", "hi"))
    for i in range(1100):
        app._events.put(("mic", i / 1100))
    app._drain(None)
    kinds = [e["kind"] for e in app._hud.pushed]
    assert "mic" not in kinds
    assert "heard" in kinds
    _quit_and_join(app)


def test_quit_closes_hud(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    app.quit(None)
    assert app._hud.closed
    assert orch.player.closed


# -- commit 3: HUD mini/full menu toggle + hud event drain --------------------

def test_hud_menu_item_starts_labeled_full(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert app._hud_mode_item.title == "HUD: Full"
    finally:
        _quit_and_join(app)


def test_toggle_hud_mode_switches_label_and_calls_set_mode(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        item = app._hud_mode_item
        app.toggle_hud_mode(item)
        assert app._hud.mode_calls == ["mini"]
        assert app._hud_mode_item.title == "HUD: Mini"

        app.toggle_hud_mode(item)
        assert app._hud.mode_calls == ["mini", "full"]
        assert app._hud_mode_item.title == "HUD: Full"
    finally:
        _quit_and_join(app)


def test_drain_hud_mini_event_calls_set_mode_and_updates_menu_label(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._events.put(("hud", {"mode": "mini"}))
        app._drain(None)
        assert app._hud.mode_calls == ["mini"]
        assert app._hud_mode_item.title == "HUD: Mini"
        # the "hud" event is a control action, not forwarded to hud.push()
        assert "hud" not in [e["kind"] for e in app._hud.pushed]
    finally:
        _quit_and_join(app)


def test_drain_hud_hide_event_calls_hud_hide(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._events.put(("hud", {"mode": "hide"}))
        app._drain(None)
        assert app._hud.hide_calls == 1
        assert app._hud.mode_calls == []
    finally:
        _quit_and_join(app)


def test_noop_hud_supports_set_mode_and_hide_without_error():
    # When the HUD is disabled/unavailable, drain must still be able to call
    # set_mode()/hide() on the _NoopHud stand-in without raising.
    hud = menubar_module()._NoopHud()
    hud.set_mode("mini")
    hud.hide()
    assert hud._mode == "full"


def menubar_module():
    import veronica.ui.menubar as menubar
    return menubar


# -- commit: "Start at Login" menu item ---------------------------------

def test_login_item_disabled_when_not_running_from_bundle(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    monkeypatch.setattr(menubar.login_item, "bundle_app_path", lambda: None)
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert app._login_item_item.title == "Start at Login (build the app first)"
        assert app._login_item_item.callback is None
    finally:
        _quit_and_join(app)


def test_login_item_enabled_when_running_from_bundle(fake_env, monkeypatch, tmp_path):
    menubar, fake_rumps, orch_holder = fake_env
    app_path = tmp_path / "Veronica.app"
    monkeypatch.setattr(menubar.login_item, "bundle_app_path", lambda: app_path)
    monkeypatch.setattr(menubar.login_item, "is_enabled", lambda: False)
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert app._login_item_item.title == "Start at Login"
        assert app._login_item_item.callback is not None
        assert app._login_item_item.state is False
    finally:
        _quit_and_join(app)


def test_toggle_login_item_enables_and_disables(fake_env, monkeypatch, tmp_path):
    menubar, fake_rumps, orch_holder = fake_env
    app_path = tmp_path / "Veronica.app"
    calls = {"enabled": False}

    monkeypatch.setattr(menubar.login_item, "bundle_app_path", lambda: app_path)
    monkeypatch.setattr(menubar.login_item, "is_enabled", lambda: calls["enabled"])

    def fake_enable(p):
        assert p == app_path
        calls["enabled"] = True

    def fake_disable():
        calls["enabled"] = False

    monkeypatch.setattr(menubar.login_item, "enable", fake_enable)
    monkeypatch.setattr(menubar.login_item, "disable", fake_disable)

    app, orch = _make_app(menubar, orch_holder)
    try:
        item = app._login_item_item
        app.toggle_login_item(item)
        assert calls["enabled"] is True
        assert item.state is True

        app.toggle_login_item(item)
        assert calls["enabled"] is False
        assert item.state is False
    finally:
        _quit_and_join(app)


# -- commit: voice "quit" intent -> menu bar quit ------------------------------

def test_build_orchestrator_receives_on_quit_callback(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert orch.on_quit == app._schedule_quit
    finally:
        _quit_and_join(app)


def test_schedule_quit_calls_quit_via_apphelper(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    from PyObjCTools import AppHelper
    calls = []
    monkeypatch.setattr(AppHelper, "callAfter", lambda fn: calls.append(fn))
    app._schedule_quit()
    assert len(calls) == 1
    calls[0]()
    app._thread.join(timeout=2)
    assert fake_rumps.quit_called is True


# -- commit: click the HUD orb to open the menu --------------------------------

class _FakeMenuItemNS:
    def __init__(self, title, action, key):
        self.title = title
        self.action = action
        self.key = key
        self.target = None
        self.state = 0
        self.enabled = True

    def setTarget_(self, target):
        self.target = target

    def setState_(self, state):
        self.state = state

    def setEnabled_(self, enabled):
        self.enabled = enabled

    def setSubmenu_(self, submenu):
        self.submenu = submenu

    def setRepresentedObject_(self, obj):
        self._represented = obj

    def representedObject(self):
        return getattr(self, "_represented", None)


class _FakeSeparator:
    title = "-"
    action = None


class _FakeNSMenu:
    def __init__(self, title=""):
        self.title = title
        self.items = []
        self.popups = []

    def addItem_(self, item):
        self.items.append(item)

    def popUpMenuPositioningItem_atLocation_inView_(self, item, point, view):
        self.popups.append((item, point, view))


def _fake_appkit_for_menu():
    menu_holder = {}

    def new_menu():
        m = _FakeNSMenu()
        menu_holder["last"] = m
        return m

    return types.SimpleNamespace(
        NSMenu=types.SimpleNamespace(
            alloc=lambda: types.SimpleNamespace(init=new_menu, initWithTitle_=lambda title: _FakeNSMenu(title))
        ),
        NSMenuItem=types.SimpleNamespace(
            alloc=lambda: types.SimpleNamespace(
                initWithTitle_action_keyEquivalent_=lambda title, action, key: _FakeMenuItemNS(title, action, key)
            ),
            separatorItem=lambda: _FakeSeparator(),
        ),
    ), menu_holder


def test_hud_on_menu_wired_to_popup_menu_at(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert app._hud.on_menu == app._popup_menu_at
    finally:
        _quit_and_join(app)


def test_popup_menu_uses_live_rumps_menu_when_available(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        sentinel = object()
        app.menu = types.SimpleNamespace(_menu=sentinel)
        assert app._build_popup_menu() is sentinel
    finally:
        _quit_and_join(app)


def test_build_popup_menu_fallback_has_five_titles_and_actions(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        fake_appkit, _ = _fake_appkit_for_menu()
        monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)

        menu = app._build_popup_menu()

        assert [i.title for i in menu.items] == [
            "Mute", "HUD: Full", "Voice", "Start at Login (build the app first)", "Quit",
        ]
        assert [i.action for i in menu.items] == [
            "onMute:", "onToggleHud:", None, "onToggleLogin:", "onQuit:",
        ]
        assert all(i.target is not None for i in menu.items if i.action is not None)
        # login item is disabled (no callback) when not running from a bundle
        assert menu.items[3].enabled is False
    finally:
        _quit_and_join(app)


def test_build_popup_menu_reflects_mute_state(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app.toggle_mute(app.menu[0])
        fake_appkit, _ = _fake_appkit_for_menu()
        monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
        menu = app._build_popup_menu()
        assert menu.items[0].state == 1
    finally:
        _quit_and_join(app)


def test_popup_menu_at_shows_menu_at_screen_point(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        fake_menu = _FakeNSMenu()
        monkeypatch.setattr(app, "_build_popup_menu", lambda: fake_menu)
        fake_foundation = types.SimpleNamespace(
            NSMakePoint=lambda x, y: types.SimpleNamespace(x=x, y=y)
        )
        monkeypatch.setitem(sys.modules, "Foundation", fake_foundation)

        app._popup_menu_at(12.0, 34.0)

        assert len(fake_menu.popups) == 1
        item, point, view = fake_menu.popups[0]
        assert item is None and view is None
        assert (point.x, point.y) == (12.0, 34.0)
    finally:
        _quit_and_join(app)


def test_popup_menu_handler_forwards_to_app_callbacks(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    handler_cls = menubar._make_menu_handler_class()
    handler = handler_cls.alloc().initWithApp_(app)

    assert app._muted is False
    handler.onMute_(None)
    assert app._muted is True
    assert orch.player.stopped is True

    handler.onToggleHud_(None)
    assert app._hud.mode_calls == ["mini"]
    assert app._hud_mode_item.title == "HUD: Mini"

    handler.onQuit_(None)
    app._thread.join(timeout=2)
    assert fake_rumps.quit_called is True


# -- commit: Voice submenu (voices, faster/slower/normal) ----------------------

VOICE_NAMES = ["Sarah", "Bella", "Nicole", "Sky", "Adam", "Michael", "Emma", "Isabella", "George", "Lewis"]


class _ResettablePlayer(FakePlayer):
    def __init__(self):
        super().__init__()
        self.resets = 0

    def reset(self):
        self.resets += 1


class _VoiceOrch:
    """Minimal orchestrator stand-in for the Voice menu: records the
    (kind, arg) actions passed to _voice_turn and player.reset() calls."""

    def __init__(self, voice="af_sarah"):
        self.calls = []
        self.tts = types.SimpleNamespace(voice=voice, speed=1.0)
        self.player = _ResettablePlayer()

    async def _voice_turn(self, action):
        self.calls.append(action)


def test_voice_submenu_lists_voices_and_speed(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        sub = app._voice_menu
        assert sub.title == "Voice"
        assert sub in app.menu
        assert app.menu.index(sub) == app.menu.index(app._hud_mode_item) + 1
        titles = [i.title if i is not None else None for i in sub.children]
        assert titles[:10] == VOICE_NAMES
        assert titles[10] is None  # separator
        assert titles[-3:] == ["Faster", "Slower", "Normal speed"]
        assert list(app._voice_items) == VOICE_NAMES
        assert list(app._speed_items) == ["Faster", "Slower", "Normal speed"]
        assert all(i.callback == app._pick_voice for i in app._voice_items.values())
        assert all(i.callback == app._speed for i in app._speed_items.values())
    finally:
        _quit_and_join(app)


def test_voice_menu_click_schedules_voice_turn(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    _quit_and_join(app)  # stop the background loop so a fresh, non-running loop drives the test
    vo = _VoiceOrch()
    app._orch = vo
    app._loop = asyncio.new_event_loop()
    try:
        app._pick_voice(app._voice_items["Adam"])
        app._loop.run_until_complete(asyncio.sleep(0))
        assert vo.calls == [("voice", "adam")]
        assert vo.player.resets == 1
        app._speed(app._speed_items["Faster"])
        app._loop.run_until_complete(asyncio.sleep(0))
        assert vo.calls[-1] == ("speed", "faster")
        assert vo.player.resets == 2
        app._speed(app._speed_items["Normal speed"])
        app._loop.run_until_complete(asyncio.sleep(0))
        assert vo.calls[-1] == ("speed", "normal")
    finally:
        app._loop.close()


def test_voice_menu_click_threadsafe_on_running_loop(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        done = threading.Event()
        vo = _VoiceOrch()

        async def _voice_turn(action):
            vo.calls.append(action)
            done.set()

        vo._voice_turn = _voice_turn
        app._orch = vo
        app._pick_voice(app._voice_items["George"])  # loop is running on the background thread
        assert done.wait(2)
        assert vo.calls == [("voice", "george")]
    finally:
        _quit_and_join(app)


def test_voice_menu_click_noop_without_orch(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app = menubar.VeronicaApp.__new__(menubar.VeronicaApp)
    app._voice_items = {"Adam": fake_rumps.MenuItem("Adam")}
    app._speed_items = {"Faster": fake_rumps.MenuItem("Faster")}
    app._pick_voice(app._voice_items["Adam"])  # must not raise: no self._orch set
    app._speed(app._speed_items["Faster"])
    app._refresh_voice_menu()
    assert app._voice_items["Adam"].state == 0


def test_refresh_voice_menu_checks_current(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="bm_george")
        app._refresh_voice_menu()
        assert app._voice_items["George"].state == 1
        assert app._voice_items["Sarah"].state == 0
        app._orch.tts.voice = "af_sarah"
        app._refresh(None)  # the 0.25 s timer keeps the checkmark in sync after a voice change by voice
        assert app._voice_items["George"].state == 0
        assert app._voice_items["Sarah"].state == 1
    finally:
        _quit_and_join(app)


def test_refresh_does_not_raise_before_orch_or_tts(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._refresh(None)  # FakeOrch has no .tts
        assert all(i.state == 0 for i in app._voice_items.values())
    finally:
        _quit_and_join(app)


def test_popup_menu_voice_submenu_mirrors_menu_bar(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="am_adam")
        fake_appkit, _ = _fake_appkit_for_menu()
        monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
        menu = app._build_popup_menu()
        voice_item = menu.items[2]
        assert voice_item.title == "Voice"
        sub = voice_item.submenu
        assert sub.title == "Voice"
        titles = [i.title for i in sub.items]
        assert titles[:10] == VOICE_NAMES
        assert titles[10] == "-"
        assert titles[-3:] == ["Faster", "Slower", "Normal speed"]
        assert [i.action for i in sub.items[:10]] == ["onPickVoice:"] * 10
        assert [i.action for i in sub.items[-3:]] == ["onSpeed:"] * 3
        assert [i.representedObject() for i in sub.items[:10]] == VOICE_NAMES
        assert [i.representedObject() for i in sub.items[-3:]] == ["Faster", "Slower", "Normal speed"]
        assert all(i.target is not None for i in sub.items if i.action is not None)
        assert [i.state for i in sub.items[:10]] == [1 if n == "Adam" else 0 for n in VOICE_NAMES]
    finally:
        _quit_and_join(app)


def test_popup_menu_handler_forwards_voice_and_speed(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        picked = []
        monkeypatch_pick = lambda item: picked.append(("voice", item.title))
        monkeypatch_speed = lambda item: picked.append(("speed", item.title))
        app._pick_voice = monkeypatch_pick
        app._speed = monkeypatch_speed
        handler = menubar._make_menu_handler_class().alloc().initWithApp_(app)
        handler.onPickVoice_(_represented("Adam"))
        handler.onSpeed_(_represented("Slower"))
        assert picked == [("voice", "Adam"), ("speed", "Slower")]
    finally:
        _quit_and_join(app)


def _represented(name):
    item = _FakeMenuItemNS(name, None, "")
    item.setRepresentedObject_(name)
    return item


# -- push-to-talk (A2) --------------------------------------------------------

def test_ptt_hotkey_started_with_background_loop_when_enabled(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    monkeypatch.setattr(menubar.settings, "ptt_enabled", True)
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert len(FakeHotkeyMonitor.instances) == 1
        mon = FakeHotkeyMonitor.instances[0]
        assert mon.started_with_loop is app._loop
        assert mon.keycode == menubar.settings.ptt_keycode
        assert app._ptt_item is None  # available: no "enable accessibility" item
    finally:
        _quit_and_join(app)
    assert mon.stopped is True


def test_ptt_hotkey_not_created_when_disabled(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    monkeypatch.setattr(menubar.settings, "ptt_enabled", False)
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert FakeHotkeyMonitor.instances == []
        assert app._hotkey is None
        assert app._ptt_item is None
    finally:
        _quit_and_join(app)


def test_ptt_unavailable_adds_accessibility_menu_item(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    monkeypatch.setattr(menubar.settings, "ptt_enabled", True)
    FakeHotkeyMonitor.available_on_start = False
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert app._ptt_item is not None
        assert "Input Monitoring" in app._ptt_item.title
        assert app._ptt_item in app.menu
    finally:
        _quit_and_join(app)


def test_open_accessibility_settings_calls_open(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    calls = []
    monkeypatch.setattr(menubar.subprocess, "run", lambda argv, **kw: calls.append(argv))
    app, orch = _make_app(menubar, orch_holder)
    try:
        app.open_accessibility_settings(None)
        assert calls == [["open", menubar.ACCESSIBILITY_PANE_URL]]
    finally:
        _quit_and_join(app)


def test_ptt_press_and_release_call_orchestrator(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        started = threading.Event()
        ended = threading.Event()

        def ptt_start():
            started.set()

        def ptt_end():
            ended.set()

        orch.ptt_start = ptt_start
        orch.ptt_end = ptt_end
        mon = FakeHotkeyMonitor.instances[0]

        app._loop.call_soon_threadsafe(mon.on_press)
        assert started.wait(2)
        app._loop.call_soon_threadsafe(mon.on_release)
        assert ended.wait(2)
    finally:
        _quit_and_join(app)


def test_ptt_callbacks_noop_before_orch_exists(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app = menubar.VeronicaApp.__new__(menubar.VeronicaApp)
    app._on_ptt_press()   # must not raise: no self._orch set
    app._on_ptt_release()


def test_real_rumps_restored_after_fixture_teardown():
    # Must run after the fake_env-using tests above (default pytest order is
    # file/definition order). Confirms the fixture's finalizer put the real
    # rumps module back on veronica.ui.menubar so nothing downstream (other
    # test modules, the actual app) sees the fake.
    import veronica.ui.menubar as menubar

    assert menubar.rumps.__name__ == "rumps"
    assert menubar.rumps is real_rumps
