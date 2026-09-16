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

    def set_callback(self, callback, key=None):
        self.callback = callback


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
        self.notifications = []
        self.notification_error = None   # set to an exception to mimic no CFBundleIdentifier

    def quit_application(self):
        self.quit_called = True

    def notification(self, title, subtitle, message, **kw):
        if self.notification_error is not None:
            raise self.notification_error
        self.notifications.append((title, subtitle, message))


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
        self.announced = []
        self.state = "idle"
        from veronica.config import Settings
        self.s = Settings()          # bridge.get_state() reads the live settings off the orchestrator

    async def announce(self, text, expires_at=None):
        self.announced.append(text)

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


class FakeSettingsWindow:
    """Stand-in for veronica.ui.settings.SettingsWindow: no WKWebView, just
    records show()/hide() calls and the bridge it was given."""

    def __init__(self, settings, bridge, **_k):
        self.s = settings
        self.bridge = bridge
        self.available = True
        self.shown = []
        self.hidden = 0
        bridge.on_state_changed = self.push_state
        self.states = []

    def push_state(self, state):
        self.states.append(state)

    def show(self, tab="general"):
        self.shown.append(tab)

    def hide(self):
        self.hidden += 1


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

    def fake_build_orchestrator(s, on_state=None, on_event=None, *, audio=True, on_quit=None, **kw):
        orch = FakeOrch(on_state=on_state)
        orch.on_quit = on_quit
        orch.build_kwargs = kw
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
    monkeypatch.setattr(menubar, "SettingsWindow", FakeSettingsWindow)
    # No AppKit main thread under test: run marshalled callbacks inline.
    monkeypatch.setattr(menubar, "_main_thread", lambda fn: fn())
    # Never shell out to git for the About item / update check.
    monkeypatch.setattr(menubar.version, "build_info", lambda *a, **k: {
        "sha": "abc1234", "built_at": "2026-09-17T10:00:00+05:30", "dirty": False, "source": "git",
    })
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

    def wrapped(s, on_state=None, on_event=None, *, audio=True, on_quit=None, **kw):
        # on_state is the VeronicaApp instance's bound _on_state method, so
        # __self__ recovers the app without racing its constructor's
        # `app = VeronicaApp()` assignment on the main thread.
        app = on_state.__self__
        seen["state"] = app._state
        return original(s, on_state=on_state, on_event=on_event, audio=audio, on_quit=on_quit, **kw)

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
        mute_item = app._mute_item

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
            "About Veronica — Veronica 0.1.0 (abc1234, 17 Sep)", "Settings…", "-",
            "Mute", "HUD: Full", "Voice", "Start at Login (build the app first)", "Quit",
        ]
        assert [i.action for i in menu.items] == [
            None, "onSettings:", None, "onMute:", "onToggleHud:", None, "onToggleLogin:", "onQuit:",
        ]
        assert all(i.target is not None for i in menu.items if i.action is not None)
        assert menu.items[0].enabled is False
        # login item is disabled (no callback) when not running from a bundle
        assert menu.items[6].enabled is False
    finally:
        _quit_and_join(app)


def test_build_popup_menu_reflects_mute_state(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app.toggle_mute(app._mute_item)
        fake_appkit, _ = _fake_appkit_for_menu()
        monkeypatch.setitem(sys.modules, "AppKit", fake_appkit)
        menu = app._build_popup_menu()
        assert menu.items[3].state == 1
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
HINDI_VOICE_NAMES = ["Alpha", "Beta", "Omega", "Psi"]
ALL_VOICE_NAMES = VOICE_NAMES + HINDI_VOICE_NAMES


class _ResettablePlayer(FakePlayer):
    def __init__(self):
        super().__init__()
        self.resets = 0

    def reset(self):
        self.resets += 1


class _VoiceOrch:
    """Minimal orchestrator stand-in for the Voice menu: records the
    (kind, arg) actions passed to _voice_turn and player.reset() calls."""

    def __init__(self, voice="af_sarah", hindi_voice="hf_alpha"):
        self.calls = []
        self.tts = types.SimpleNamespace(voice=voice, speed=1.0, hindi_voice=hindi_voice)
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
        assert titles[11:15] == HINDI_VOICE_NAMES
        assert titles[15] is None  # separator
        assert titles[-3:] == ["Faster", "Slower", "Normal speed"]
        assert len(titles) == 19
        assert list(app._voice_items) == ALL_VOICE_NAMES
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
        voice_item = next(i for i in menu.items if i.title == "Voice")
        sub = voice_item.submenu
        assert sub.title == "Voice"
        titles = [i.title for i in sub.items]
        assert titles[:10] == VOICE_NAMES
        assert titles[10] == "-"
        assert titles[11:15] == HINDI_VOICE_NAMES
        assert titles[15] == "-"
        assert titles[-3:] == ["Faster", "Slower", "Normal speed"]
        voice_items = sub.items[:10] + sub.items[11:15]
        assert [i.action for i in voice_items] == ["onPickVoice:"] * 14
        assert [i.action for i in sub.items[-3:]] == ["onSpeed:"] * 3
        assert [i.representedObject() for i in voice_items] == ALL_VOICE_NAMES
        assert [i.representedObject() for i in sub.items[-3:]] == ["Faster", "Slower", "Normal speed"]
        assert all(i.target is not None for i in sub.items if i.action is not None)
        assert [i.state for i in voice_items] == [1 if n in ("Adam", "Alpha") else 0 for n in ALL_VOICE_NAMES]
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


# -- batch C: Hindi voices in the Voice submenu --------------------------------

def test_voice_menu_hindi_pick_schedules_voice_turn(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    _quit_and_join(app)  # stop the background loop so a fresh, non-running loop drives the test
    vo = _VoiceOrch()
    app._orch = vo
    app._loop = asyncio.new_event_loop()
    try:
        app._pick_voice(app._voice_items["Omega"])
        app._loop.run_until_complete(asyncio.sleep(0))
        assert vo.calls == [("voice", "omega")]
        assert vo.player.resets == 1
    finally:
        app._loop.close()


def test_refresh_voice_menu_checks_english_and_hindi_voices(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="bm_george", hindi_voice="hm_omega")
        app._refresh_voice_menu()
        checked = [n for n, i in app._voice_items.items() if i.state == 1]
        assert checked == ["George", "Omega"]
        app._orch.tts.hindi_voice = "hf_beta"
        app._refresh(None)
        checked = [n for n, i in app._voice_items.items() if i.state == 1]
        assert checked == ["George", "Beta"]
    finally:
        _quit_and_join(app)


def test_refresh_voice_menu_tolerates_tts_without_hindi_voice(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="af_sky")
        del app._orch.tts.hindi_voice
        app._refresh_voice_menu()
        checked = [n for n, i in app._voice_items.items() if i.state == 1]
        assert checked == ["Sky"]
    finally:
        _quit_and_join(app)


# -- Batch D: About / Settings… / Check for Updates… / settings event -----------

def _titles(app):
    return [None if i is None else i.title for i in app.menu]


def _status(kind):
    from veronica.updater import UpdateStatus
    return UpdateStatus(available=kind != "none", kind=kind, detail=f"{kind} detail",
                        running_sha="abc1234", head_sha="def5678", remote_sha=None)


def test_menu_starts_with_about_settings_and_updates(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        assert _titles(app)[:6] == [
            "About Veronica — Veronica 0.1.0 (abc1234, 17 Sep)", "Settings…", "Check for Updates…",
            menubar.UPDATE_UNCHECKED_TITLE, None, "Mute",
        ]
        assert app.menu[0].callback is None                      # About is a label
        assert app.menu[1].callback == app.open_settings
        assert app.menu[2].callback == app.check_for_updates
        assert app._update_item.callback is None                 # nothing to install yet
        assert app.menu[5] is app._mute_item
    finally:
        _quit_and_join(app)


def test_settings_item_shows_window_on_general(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app.open_settings(app.menu[1])
        assert app._settings.shown == ["general"]
    finally:
        _quit_and_join(app)


def test_settings_event_shows_window_on_tab(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._events.put(("settings", {"open": True, "tab": "history"}))
        app._events.put(("settings", {"open": True}))
        app._events.put(("settings", "garbage"))
        app._drain(None)
        assert app._settings.shown == ["history", "general", "general"]
        # not forwarded to the HUD as a transcript event
        assert not any(e["kind"] == "settings" for e in app._hud.pushed)
    finally:
        _quit_and_join(app)


def test_bridge_and_window_wired(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        bridge = app._bridge
        assert app._settings.bridge is bridge
        assert bridge._get_orch() is orch
        assert bridge._run_on_loop == app._schedule
        assert bridge._repo == menubar.version.REPO
        assert bridge._bundle_path == menubar.login_item.bundle_app_path()
        calls = []
        monkeypatch.setattr(menubar, "relaunch", lambda bundle, quit: calls.append((bundle, quit)) or True)
        assert bridge._relaunch() is True
        assert calls == [(app._bundle_path, app._schedule_quit)]
    finally:
        _quit_and_join(app)


def test_store_is_late_bound_to_orchestrator(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        # FakeOrch has no store: history reads fail politely instead of crashing
        res = app._bridge.handle("history", {})
        assert res["ok"] is False and res["message"] == "Memory is off."

        class Store:
            def turns(self, limit=200, offset=0, query=""):
                return [{"id": 1, "heard": "hi", "reply": "hello", "limit": limit, "query": query}]

            def delete_turn(self, id):
                return id == 1

            def clear_turns(self):
                return 3

            def close(self):
                pass

        orch.store = Store()
        res = app._bridge.handle("history", {"limit": 5, "query": "h"})
        assert res["ok"] is True and res["items"][0]["limit"] == 5 and res["items"][0]["query"] == "h"
        assert app._bridge.handle("forget_turn", {"id": 1})["ok"] is True
        assert app._bridge.handle("clear_history", {})["count"] == 3
    finally:
        _quit_and_join(app)


def test_store_before_orchestrator_exists_is_none(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._orch = None
        assert app._bridge._history_store() is None
        assert app._bridge.handle("history", {})["message"] == "Memory is off."
    finally:
        _quit_and_join(app)


def test_build_orchestrator_receives_updater_hooks(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        kw = orch.build_kwargs
        assert set(kw) == {"updater_check", "updater_update", "relaunch", "can_relaunch", "version_describe"}
        assert kw["can_relaunch"]() == (app._bundle_path is not None)
        assert kw["version_describe"]() == menubar.version.describe(app._build_info)
        seen = []
        monkeypatch.setattr(menubar.updater, "check", lambda repo, run=None, info=None: seen.append(("check", repo, info)) or _status("none"))
        monkeypatch.setattr(menubar.updater, "update", lambda repo, st, **k: seen.append(("update", repo, st)) or "log")
        st = kw["updater_check"]()
        assert st.kind == "none"
        assert kw["updater_update"](st) == "log"
        assert seen[0][0] == "check" and seen[0][1] == menubar.version.REPO and seen[0][2]["sha"] == "abc1234"
        assert seen[1] == ("update", menubar.version.REPO, st)
        assert kw["relaunch"] == app._relaunch
        assert app._bridge.get_state()["about"]["updating"] is False   # slot released after success
    finally:
        _quit_and_join(app)


def test_voice_update_hook_owns_the_bridge_update_slot(fake_env, monkeypatch):
    from veronica.updater import UpdateInProgress
    from veronica.ui.settings import bridge as bridge_mod
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        hook = orch.build_kwargs["updater_update"]
        st = _status("remote")
        seen = []

        def slow_update(repo, status, **k):
            # while the voice update runs, nothing else may start one
            seen.append(app._bridge.get_state()["about"]["updating"])
            assert app._bridge.begin_update() is False
            assert app._bridge.update_now() == {"ok": False, "message": bridge_mod.BUSY}
            with pytest.raises(UpdateInProgress):
                hook(status)
            assert app._update_item.title == menubar.UPDATE_UPDATING_TITLE
            return "log"

        monkeypatch.setattr(menubar.updater, "update", slow_update)
        assert hook(st) == "log"
        assert seen == [True]
        assert app._bridge.get_state()["about"]["updating"] is False

        def boom(repo, status, **k):
            raise RuntimeError("build exploded")

        monkeypatch.setattr(menubar.updater, "update", boom)
        with pytest.raises(RuntimeError, match="build exploded"):
            hook(st)
        assert app._bridge.get_state()["about"]["updating"] is False
        assert app._bridge.get_state()["about"]["update"]["detail"] == bridge_mod.UPDATE_FAILED
        assert app._update_item.title == menubar.UPDATE_FAILED_TITLE
    finally:
        _quit_and_join(app)


def test_check_for_updates_available_enables_item_and_notifies(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        monkeypatch.setattr(menubar.updater, "check", lambda repo, run=None, info=None: _status("remote"))
        app._bridge.run_thread = lambda fn: fn()
        app.check_for_updates(app.menu[2])
        assert app._update_item.title == menubar.UPDATE_AVAILABLE_TITLE
        assert app._update_item.callback == app.update_now
        assert fake_rumps.notifications == [("Veronica", "Update available", "remote detail")]
    finally:
        _quit_and_join(app)


def test_check_for_updates_latest_and_failure(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        app._bridge.run_thread = lambda fn: fn()
        monkeypatch.setattr(menubar.updater, "check", lambda repo, run=None, info=None: _status("none"))
        app.check_for_updates(app.menu[2])
        assert app._update_item.title == menubar.UPDATE_LATEST_TITLE
        assert app._update_item.callback is None
        assert fake_rumps.notifications[-1] == ("Veronica", "Up to date", "none detail")

        def boom(repo, run=None, info=None):
            raise RuntimeError("offline")

        monkeypatch.setattr(menubar.updater, "check", boom)
        app.check_for_updates(app.menu[2])
        assert app._update_item.title == menubar.UPDATE_CHECK_FAILED_TITLE
        assert app._update_item.callback is None
        assert fake_rumps.notifications[-1] == ("Veronica", "Couldn't check for updates", "Couldn't check: offline")
    finally:
        _quit_and_join(app)


def test_check_for_updates_runs_on_bridge_thread_and_dedupes(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        queued = []
        app._bridge.run_thread = lambda fn: queued.append(fn)
        monkeypatch.setattr(menubar.updater, "check", lambda repo, run=None, info=None: _status("local"))
        app.check_for_updates(app.menu[2])
        app.check_for_updates(app.menu[2])      # already checking: not queued twice
        assert len(queued) == 1
        assert app._update_item.title == menubar.UPDATE_UNCHECKED_TITLE
        queued[0]()
        assert app._update_item.title == menubar.UPDATE_AVAILABLE_TITLE
        app.check_for_updates(app.menu[2])      # a later click checks again
        assert len(queued) == 2
    finally:
        _quit_and_join(app)


def test_hourly_update_timer_checks_silently(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        timer = app._update_timer
        assert isinstance(timer, FakeTimer)
        assert timer.interval == 3600
        assert timer.callback == app._hourly_update_check
        app._bridge.run_thread = lambda fn: fn()
        monkeypatch.setattr(menubar.updater, "check", lambda repo, run=None, info=None: _status("remote"))
        timer.callback(timer)
        assert app._update_item.title == menubar.UPDATE_AVAILABLE_TITLE
        assert app._update_item.callback == app.update_now
        assert fake_rumps.notifications == []
    finally:
        _quit_and_join(app)


def test_update_item_click_runs_bridge_update_now(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        calls = []
        monkeypatch.setattr(app._bridge, "update_now", lambda: calls.append("update_now") or {"ok": True, "message": "Updating, back in a moment."})
        app.update_now(app._update_item)
        assert calls == ["update_now"]
        assert app._update_item.title == menubar.UPDATE_UPDATING_TITLE
        assert app._update_item.callback is None

        monkeypatch.setattr(app._bridge, "update_now", lambda: {"ok": False, "message": "Busy, try again in a moment."})
        app._update_item.title = menubar.UPDATE_AVAILABLE_TITLE
        app._update_item.set_callback(app.update_now)
        app.update_now(app._update_item)
        assert fake_rumps.notifications[-1] == ("Veronica", "", "Busy, try again in a moment.")
        assert app._update_item.title == menubar.UPDATE_AVAILABLE_TITLE   # still installable
    finally:
        _quit_and_join(app)


def test_update_failure_pushed_from_bridge_resets_item(fake_env, monkeypatch):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        from veronica.ui.settings import bridge as bridge_mod
        def push(update, updating=False):
            app._bridge.on_state_changed({"about": {"update": update, "updating": updating}})

        push({"available": True, "detail": "remote detail"}, updating=True)
        assert app._update_item.title == menubar.UPDATE_UPDATING_TITLE     # in progress wins
        assert app._update_item.callback is None
        push({"available": True, "detail": bridge_mod.UPDATE_FAILED})
        assert app._update_item.title == menubar.UPDATE_FAILED_TITLE
        assert app._update_item.callback is None
        # the window still gets the state
        assert app._settings.states[-1]["about"]["update"]["detail"] == bridge_mod.UPDATE_FAILED
        # a check from the settings window's "Check now" reflects in the menu too
        push({"available": True, "detail": "remote detail"})
        assert app._update_item.title == menubar.UPDATE_AVAILABLE_TITLE
        assert app._update_item.callback == app.update_now
        push({"available": False, "detail": "none detail"})
        assert app._update_item.title == menubar.UPDATE_LATEST_TITLE
        # a real bridge push (nothing checked yet) leaves the item alone
        app._update_item.title = menubar.UPDATE_UNCHECKED_TITLE
        app._bridge.on_state_changed(app._bridge.get_state())
        assert app._update_item.title == menubar.UPDATE_UNCHECKED_TITLE
    finally:
        _quit_and_join(app)


def test_popup_menu_handler_forwards_settings(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    handler_cls = menubar._make_menu_handler_class()
    handler = handler_cls.alloc().initWithApp_(app)
    try:
        handler.onSettings_(None)
        assert app._settings.shown == ["general"]
    finally:
        _quit_and_join(app)


def test_quit_hides_settings_window(fake_env):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    _quit_and_join(app)
    assert app._settings.hidden == 1


def _wait_for(cond, timeout=2.0):
    import time
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_notify_falls_back_to_spoken_announce_when_notification_center_unavailable(fake_env, monkeypatch, caplog):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        fake_rumps.notification_error = RuntimeError('Failed to setup the notification center: missing "CFBundleIdentifier"')
        app._bridge.run_thread = lambda fn: fn()
        monkeypatch.setattr(menubar.updater, "check", lambda repo, run=None, info=None: _status("remote"))
        with caplog.at_level(logging.INFO, logger="veronica.ui"):
            app.check_for_updates(app.menu[2])          # must not raise
        assert app._update_item.title == menubar.UPDATE_AVAILABLE_TITLE
        assert fake_rumps.notifications == []
        _wait_for(lambda: len(orch.announced) == 1)        # announce() ran on the background loop
        assert orch.announced == ["An update is ready. Say update yourself, or use the Settings window."]
        assert "remote detail" in caplog.text

        monkeypatch.setattr(app._bridge, "update_now", lambda: {"ok": False, "message": "Busy, try again in a moment."})
        app.update_now(app._update_item)                # must not raise either
        _wait_for(lambda: len(orch.announced) == 2)
        assert orch.announced[-1] == "Busy, try again in a moment."
    finally:
        _quit_and_join(app)


def test_notify_without_orchestrator_only_logs(fake_env, caplog):
    menubar, fake_rumps, orch_holder = fake_env
    app, orch = _make_app(menubar, orch_holder)
    try:
        fake_rumps.notification_error = RuntimeError("no bundle")
        app._orch = None
        with caplog.at_level(logging.INFO, logger="veronica.ui"):
            app._notify("Up to date", "none detail", spoken="You're already on the latest.")
        assert "none detail" in caplog.text
    finally:
        app._orch = orch
        _quit_and_join(app)
