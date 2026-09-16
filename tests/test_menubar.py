import asyncio
import importlib
import logging
import sys
import threading

import pytest
import rumps as real_rumps


class FakeMenuItem:
    def __init__(self, title, callback=None):
        self.title = title
        self.callback = callback
        self.state = False


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


def test_real_rumps_restored_after_fixture_teardown():
    # Must run after the fake_env-using tests above (default pytest order is
    # file/definition order). Confirms the fixture's finalizer put the real
    # rumps module back on veronica.ui.menubar so nothing downstream (other
    # test modules, the actual app) sees the fake.
    import veronica.ui.menubar as menubar

    assert menubar.rumps.__name__ == "rumps"
    assert menubar.rumps is real_rumps
