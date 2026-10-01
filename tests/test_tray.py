"""VeronicaApp (the system tray app) with a fake pystray, fake HUD/Settings
windows and a fake orchestrator: no Win32, no pywebview, no real timers."""
import asyncio
import logging
import threading
import types

import pytest

from veronica.brain.backends import BACKENDS, Availability


# -- fake pystray ------------------------------------------------------------------

class FakePystrayMenu:
    SEPARATOR = object()

    def __init__(self, *items):
        self.items = list(items)


class FakePystrayItem:
    def __init__(self, text, action, checked=None, radio=False, default=False, visible=True, enabled=True):
        self._text = text
        self.action = action
        self._checked = checked
        self.default = default
        self._enabled = enabled

    @property
    def text(self):
        return self._text(self) if callable(self._text) else self._text

    @property
    def checked(self):
        return self._checked(self) if callable(self._checked) else self._checked

    @property
    def enabled(self):
        return self._enabled(self) if callable(self._enabled) else self._enabled

    @property
    def submenu(self):
        return self.action if isinstance(self.action, FakePystrayMenu) else None

    def click(self, icon=None):
        self.action(icon, self)


class FakeIcon:
    HAS_NOTIFICATION = True

    def __init__(self, name, icon=None, title=None, menu=None):
        self.name = name
        self.icon = icon
        self.title = title
        self.menu = menu
        self.ran = threading.Event()
        self.notifications = []
        self.notify_error = None
        self.menu_updates = 0
        self.stopped = False
        self._hwnd = 4242

    def run(self, setup=None):
        self.ran.set()

    def notify(self, message, title=None):
        if self.notify_error is not None:
            raise self.notify_error
        self.notifications.append((title, message))

    def update_menu(self):
        self.menu_updates += 1

    def stop(self):
        self.stopped = True


class FakePystray:
    Menu = FakePystrayMenu
    MenuItem = FakePystrayItem

    def __init__(self):
        self.icons = []

    def Icon(self, *a, **k):  # noqa: N802 — mirrors pystray.Icon
        icon = FakeIcon(*a, **k)
        self.icons.append(icon)
        return icon


class FakeWindow:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class FakeWebview:
    def __init__(self):
        self.windows = [FakeWindow(), FakeWindow()]
        self.started = []

    def start(self, *a, **k):
        self.started.append(k)


class FakeTimer:
    def __init__(self, interval, fn):
        self.interval = interval
        self.fn = fn
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


# -- fake app collaborators ------------------------------------------------------------

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
        self.reset_calls = 0
        self.configure_calls = []
        self.on_menu = None

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

    def reset_position(self):
        self.reset_calls += 1

    def configure(self, cfg):
        self.configure_calls.append(cfg)


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
        # on_state("idle") once models are loaded — the tray app sets
        # app._state = "warming" itself before build_orchestrator runs.
        if self._on_state is not None:
            self._on_state("idle")

    async def run_forever(self):
        self.started.set()
        await asyncio.Event().wait()


class FakeSettingsWindow:
    """Stand-in for veronica.ui.settings.SettingsWindow: records show()/
    hide()/close() calls and the bridge it was given."""

    def __init__(self, settings, bridge, **_k):
        self.s = settings
        self.bridge = bridge
        self.available = True
        self.shown = []
        self.hidden = 0
        self.closed = 0
        self.created_hidden = 0
        bridge.on_state_changed = self.push_state
        self.states = []

    def push_state(self, state):
        self.states.append(state)

    def show(self, tab="general"):
        self.shown.append(tab)

    def hide(self):
        self.hidden += 1

    def close(self):
        self.closed += 1

    def create_hidden(self):
        self.created_hidden += 1


class FakeHotkeyMonitor:
    """Stand-in for veronica.audio.hotkey.HotkeyMonitor: no keyboard hook,
    no thread — records what it was asked to do so tests can drive
    on_press/on_release directly."""
    instances = []
    available_on_start = True

    def __init__(self, on_press, on_release, keycode=0xA3):
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
def fake_env(monkeypatch, tmp_home):
    import veronica.ui.tray as tray

    pystray = FakePystray()
    webview = FakeWebview()
    monkeypatch.setattr(tray, "_import_pystray", lambda: pystray)
    monkeypatch.setattr(tray, "_import_webview", lambda: webview)
    monkeypatch.setattr(tray, "state_image", lambda state, muted=False, **k: ("img", state, muted))
    timers = []

    def every(interval, fn):
        t = FakeTimer(interval, fn)
        timers.append(t)
        return t

    monkeypatch.setattr(tray, "_every", every)
    # No UI thread under test: run marshalled callbacks inline.
    monkeypatch.setattr(tray, "_main_thread", lambda fn: fn())
    posted = []
    monkeypatch.setattr(tray.win32, "post_message", lambda *a: posted.append(a) or True)

    # Written before VeronicaApp() ever starts its background thread, so
    # _make_app() waiting on it never races the key itself — only the
    # .set() (done from the background thread once build_orchestrator
    # actually ran) is awaited.
    orch_holder = {"ready": threading.Event()}

    def fake_build_orchestrator(s, on_state=None, on_event=None, *, audio=True, on_quit=None, **kw):
        orch = FakeOrch(on_state=on_state)
        orch.on_quit = on_quit
        orch.build_kwargs = kw
        orch_holder["orch"] = orch
        orch_holder["ready"].set()
        return orch

    monkeypatch.setattr(tray, "build_orchestrator", fake_build_orchestrator)
    monkeypatch.setattr(tray, "HudWindow", FakeHud)
    FakeHotkeyMonitor.instances = []
    FakeHotkeyMonitor.available_on_start = True
    monkeypatch.setattr(tray, "HotkeyMonitor", FakeHotkeyMonitor)
    monkeypatch.setattr(tray, "SettingsWindow", FakeSettingsWindow)
    # The Brain submenu's availability check looks at PATH and ~: every
    # brain is "ready" under test unless a test fakes it otherwise.
    monkeypatch.setattr(tray, "check_backend", lambda name: Availability(True, "ok"))
    # Not running from the built exe unless a test says so.
    monkeypatch.setattr(tray.login_item, "app_exe_path", lambda: None)
    # Never shell out to git for the About item / update check.
    monkeypatch.setattr(tray.version, "build_info", lambda *a, **k: {
        "sha": "abc1234", "built_at": "2026-09-17T10:00:00+05:30", "dirty": False, "source": "git",
    })
    env = types.SimpleNamespace(tray=tray, pystray=pystray, webview=webview, timers=timers, posted=posted)
    return env, orch_holder


def _make_app(env, orch_holder):
    app = env.tray.VeronicaApp()
    assert orch_holder["ready"].wait(2), "build_orchestrator was not called within 2s"
    orch = orch_holder["orch"]
    assert orch.started.wait(2), "background loop did not start within 2s"
    return app, orch


def _quit_and_join(app):
    app.quit(None)
    app._thread.join(timeout=2)


def _icon(env):
    assert len(env.pystray.icons) == 1
    return env.pystray.icons[0]


def test_state_is_warming_during_build_orchestrator(fake_env, monkeypatch):
    env, orch_holder = fake_env
    seen = {}
    original = env.tray.build_orchestrator

    def wrapped(s, on_state=None, on_event=None, *, audio=True, on_quit=None, **kw):
        app = on_state.__self__
        seen["state"] = app._state
        return original(s, on_state=on_state, on_event=on_event, audio=audio, on_quit=on_quit, **kw)

    monkeypatch.setattr(env.tray, "build_orchestrator", wrapped)
    app, orch = _make_app(env, orch_holder)
    try:
        assert seen.get("state") == "warming"
    finally:
        _quit_and_join(app)


@pytest.mark.parametrize("state, tooltip", [
    ("idle", "Veronica — Idle"), ("listening", "Veronica — Listening"),
    ("warming", "Veronica — Starting up"), ("confirming", "Veronica — Waiting for your answer"),
])
def test_state_sets_tooltip_and_icon(fake_env, state, tooltip):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._on_state(state)
        app._refresh()
        icon = _icon(env)
        assert app.title == tooltip and icon.title == tooltip
        assert icon.icon == ("img", state, False)
    finally:
        _quit_and_join(app)


def test_error_state_tooltip_carries_the_error(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._state, app._error = "error", "no microphone"
        app._refresh()
        assert app.title == "Veronica — Error: no microphone"
        assert _icon(env).icon == ("img", "error", False)
    finally:
        _quit_and_join(app)


def test_tray_icon_runs_on_its_own_thread_with_the_menu(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        icon = _icon(env)
        assert icon.ran.wait(2)
        assert app._icon_thread.daemon
        texts = [None if i is FakePystrayMenu.SEPARATOR else i.text for i in icon.menu.items]
        assert texts == [None if i is None else i.title for i in app.menu]
        settings_item = icon.menu.items[1]
        assert settings_item.default is True                # left-click on the icon opens Settings
        assert icon.menu.items[0].enabled is False          # About is a label
        assert icon.menu.items[0].checked is None           # not a checkbox
        mute = icon.menu.items[5]
        assert mute.text == "Mute" and mute.checked is False and mute.enabled is True
        voice = next(i for i in icon.menu.items if i is not FakePystrayMenu.SEPARATOR and i.text == "Voice")
        assert voice.submenu is not None
        assert voice.submenu.items[0].text == "Sarah"
    finally:
        _quit_and_join(app)


def test_tray_menu_reads_live_item_state(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        icon = _icon(env)
        mute = icon.menu.items[5]
        app.toggle_mute(app._mute_item)
        assert mute.checked is True
        app._update_item.title = "Something else"
        assert icon.menu.items[3].text == "Something else"
    finally:
        _quit_and_join(app)


def test_tray_click_runs_callback_on_ui_thread(fake_env, monkeypatch):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        hops = []
        monkeypatch.setattr(env.tray, "_main_thread", lambda fn: hops.append(fn) or fn())
        _icon(env).menu.items[1].click()                    # Settings…
        assert len(hops) == 1
        assert app._settings.shown == ["general"]
    finally:
        _quit_and_join(app)


def test_refresh_updates_tray_menu_only_when_something_changed(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        icon = _icon(env)
        app._refresh()
        n = icon.menu_updates
        app._refresh()
        assert icon.menu_updates == n                       # nothing changed
        app.toggle_mute(app._mute_item)
        app._refresh()
        assert icon.menu_updates == n + 1
        assert icon.icon == ("img", "idle", True)
        assert app.title == "Veronica — Muted"
    finally:
        _quit_and_join(app)


def test_tray_unavailable_is_soft(fake_env, monkeypatch, caplog):
    env, orch_holder = fake_env

    def boom():
        raise ImportError("no pystray")

    monkeypatch.setattr(env.tray, "_import_pystray", boom)
    with caplog.at_level(logging.WARNING, logger="veronica.ui"):
        app, orch = _make_app(env, orch_holder)
    try:
        assert app._icon is None
        app._refresh()                                      # no raise
        assert "system tray unavailable" in caplog.text
    finally:
        _quit_and_join(app)


def test_timers_scheduled_on_the_ui_thread(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        by_interval = {t.interval: t.fn for t in env.timers}
        assert by_interval[0.25] == app._refresh
        assert by_interval[1 / 30] == app._drain
        assert by_interval[3600] == app._hourly_update_check
    finally:
        _quit_and_join(app)


def test_toggle_mute(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        mute_item = app._mute_item

        app.toggle_mute(mute_item)
        assert app._muted is True
        assert orch.player.stopped is True
        assert orch.muted is True
        app._refresh()
        assert app.title == "Veronica — Muted"

        app.toggle_mute(mute_item)
        assert app._muted is False
        assert orch.muted is False
        app._refresh()
        assert app.title == "Veronica — Idle"
    finally:
        _quit_and_join(app)


def test_quit_tears_everything_down(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    _quit_and_join(app)
    assert not app._thread.is_alive()
    assert app._hud.closed
    assert app._settings.closed == 1
    assert orch.player.closed
    assert _icon(env).stopped
    assert all(t.cancelled for t in env.timers)
    assert all(w.destroyed for w in env.webview.windows)   # webview.start() returns
    assert app._stopped.is_set()
    app.quit(None)                                           # idempotent
    assert app._settings.closed == 1


def test_quit_does_not_log_error(fake_env, caplog):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    with caplog.at_level(logging.ERROR, logger="veronica.ui"):
        _quit_and_join(app)
    error_records = [r for r in caplog.records if r.name == "veronica.ui" and r.levelno >= logging.ERROR]
    assert error_records == []
    assert app._state != "error"


def test_run_starts_the_gui_loop_on_this_thread(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app.run()
        assert env.webview.started == [{"gui": "edgechromium", "private_mode": True}]
        assert app._settings.created_hidden == 0            # the HUD is the first window
        # webview.start() returned without a Quit (every window gone): quit too
        assert app._quitting and app._stopped.is_set()
    finally:
        app._thread.join(timeout=2)


def test_run_without_hud_creates_the_settings_window_first(fake_env, monkeypatch):
    env, orch_holder = fake_env
    monkeypatch.setattr(env.tray.settings, "hud_enabled", False)
    app, orch = _make_app(env, orch_holder)
    try:
        assert isinstance(app._hud, env.tray._NoopHud)
        app.run()
        assert app._settings.created_hidden == 1
    finally:
        app._thread.join(timeout=2)


def test_run_without_pywebview_waits_for_quit(fake_env, monkeypatch):
    env, orch_holder = fake_env

    def boom():
        raise ImportError("no webview")

    monkeypatch.setattr(env.tray, "_import_webview", boom)
    app, orch = _make_app(env, orch_holder)
    t = threading.Thread(target=app.run)
    t.start()
    try:
        assert t.is_alive()
    finally:
        _quit_and_join(app)
        t.join(timeout=2)
    assert not t.is_alive()


def test_events_drained_to_hud(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    app._events.put(("mic", 0.1)); app._events.put(("mic", 0.9)); app._events.put(("state", "listening")); app._events.put(("heard", "hi"))
    app._drain()
    kinds = [e["kind"] for e in app._hud.pushed]
    assert kinds.count("mic") == 1 and app._hud.pushed[[i for i, e in enumerate(app._hud.pushed) if e["kind"] == "mic"][0]]["payload"] == 0.9
    assert app._hud.states == ["listening"] and app._hud.ticks == 1
    # exact push order: state, then heard, then the coalesced mic last
    assert kinds == ["state", "heard", "mic"]
    # on_state("listening") is recorded before the corresponding state push
    assert app._hud.log.index(("state", "listening")) < app._hud.log.index(("push", "state"))
    _quit_and_join(app)


def test_drain_overflow_drops_mic(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    app._events.put(("heard", "hi"))
    for i in range(1100):
        app._events.put(("mic", i / 1100))
    app._drain()
    kinds = [e["kind"] for e in app._hud.pushed]
    assert "mic" not in kinds
    assert "heard" in kinds
    _quit_and_join(app)


# -- HUD mini/full menu toggle + hud event drain --------------------------------

def test_hud_menu_item_starts_labeled_full(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        assert app._hud_mode_item.title == "HUD: Full"
    finally:
        _quit_and_join(app)


def test_toggle_hud_mode_switches_label_and_calls_set_mode(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
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
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._events.put(("hud", {"mode": "mini"}))
        app._drain()
        assert app._hud.mode_calls == ["mini"]
        assert app._hud_mode_item.title == "HUD: Mini"
        # the "hud" event is a control action, not forwarded to hud.push()
        assert "hud" not in [e["kind"] for e in app._hud.pushed]
    finally:
        _quit_and_join(app)


def test_drain_hud_hide_event_calls_hud_hide(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._events.put(("hud", {"mode": "hide"}))
        app._drain()
        assert app._hud.hide_calls == 1
        assert app._hud.mode_calls == []
    finally:
        _quit_and_join(app)


def test_drain_hud_reset_event_calls_reset_position(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._events.put(("hud", {"mode": "reset"}))
        app._drain()
        assert app._hud.reset_calls == 1
        assert app._hud.mode_calls == [] and app._hud.hide_calls == 0
        assert "hud" not in [e["kind"] for e in app._hud.pushed]
    finally:
        _quit_and_join(app)


def test_drain_hud_config_event_calls_configure(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._events.put(("hud", {"config": {"particles": 1200, "intensity": 0.5}}))
        app._drain()
        assert app._hud.configure_calls == [{"particles": 1200, "intensity": 0.5}]
        assert app._hud.mode_calls == [] and app._hud.hide_calls == 0
        assert "hud" not in [e["kind"] for e in app._hud.pushed]
    finally:
        _quit_and_join(app)


def test_noop_hud_supports_set_mode_and_hide_without_error():
    import veronica.ui.tray as tray

    hud = tray._NoopHud()
    hud.set_mode("mini")
    hud.hide()
    hud.reset_position()
    hud.configure({"particles": 1000, "intensity": 1.0})
    assert hud._mode == "full"


# -- "Start at Login" menu item ---------------------------------

def test_login_item_disabled_when_not_running_from_the_exe(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        assert app._login_item_item.title == "Start at Login (build the app first)"
        assert app._login_item_item.callback is None
    finally:
        _quit_and_join(app)


def test_login_item_enabled_when_running_from_the_exe(fake_env, monkeypatch, tmp_path):
    env, orch_holder = fake_env
    exe = tmp_path / "Veronica.exe"
    monkeypatch.setattr(env.tray.login_item, "app_exe_path", lambda: exe)
    monkeypatch.setattr(env.tray.login_item, "is_enabled", lambda: False)
    app, orch = _make_app(env, orch_holder)
    try:
        assert app._login_item_item.title == "Start at Login"
        assert app._login_item_item.callback is not None
        assert app._login_item_item.state is False
    finally:
        _quit_and_join(app)


def test_toggle_login_item_enables_and_disables(fake_env, monkeypatch, tmp_path):
    env, orch_holder = fake_env
    exe = tmp_path / "Veronica.exe"
    calls = {"enabled": False}

    monkeypatch.setattr(env.tray.login_item, "app_exe_path", lambda: exe)
    monkeypatch.setattr(env.tray.login_item, "is_enabled", lambda: calls["enabled"])

    def fake_enable(p):
        assert p == exe
        calls["enabled"] = True

    def fake_disable():
        calls["enabled"] = False

    monkeypatch.setattr(env.tray.login_item, "enable", fake_enable)
    monkeypatch.setattr(env.tray.login_item, "disable", fake_disable)

    app, orch = _make_app(env, orch_holder)
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


# -- voice "quit" intent -> tray quit ------------------------------

def test_build_orchestrator_receives_on_quit_callback(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        assert orch.on_quit == app._schedule_quit
    finally:
        _quit_and_join(app)


def test_schedule_quit_hops_to_the_ui_thread(fake_env, monkeypatch):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    calls = []
    monkeypatch.setattr(env.tray, "_main_thread", lambda fn: calls.append(fn))
    app._schedule_quit()
    assert len(calls) == 1 and not app._quitting
    monkeypatch.setattr(env.tray, "_main_thread", lambda fn: fn())
    calls[0]()
    app._thread.join(timeout=2)
    assert app._quitting and _icon(env).stopped


# -- click the HUD orb to open the menu --------------------------------

def test_hud_on_menu_wired_to_popup_menu_at(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        assert app._hud.on_menu == app._popup_menu_at
    finally:
        _quit_and_join(app)


def test_popup_menu_at_asks_pystray_to_show_its_menu(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        icon = _icon(env)
        n = icon.menu_updates
        app._popup_menu_at(100.0, 50.0)
        assert icon.menu_updates == n + 1                    # fresh titles/checks first
        assert env.posted == [(4242, env.tray.PYSTRAY_WM_NOTIFY, 0, env.tray.WM_RBUTTONUP)]
    finally:
        _quit_and_join(app)


def test_popup_menu_at_without_tray_window_only_logs(fake_env, caplog):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        _icon(env)._hwnd = None
        with caplog.at_level(logging.INFO, logger="veronica.ui"):
            app._popup_menu_at(1.0, 2.0)
        assert env.posted == []
        assert "HUD menu unavailable" in caplog.text
    finally:
        _quit_and_join(app)


# -- Voice submenu ---------------------------------------------------------------

VOICE_NAMES = ["Sarah", "Bella", "Nicole", "Sky", "Adam", "Michael", "Emma", "Isabella", "George", "Lewis", "Heart"]
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
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        sub = app._voice_menu
        assert sub.title == "Voice"
        assert sub in app.menu
        assert app.menu.index(sub) == app.menu.index(app._hud_mode_item) + 1
        titles = [i.title if i is not None else None for i in sub.children]
        assert titles[:11] == VOICE_NAMES
        assert titles[11] is None  # separator
        assert titles[12:16] == HINDI_VOICE_NAMES
        assert titles[16] is None  # separator
        assert titles[-3:] == ["Faster", "Slower", "Normal speed"]
        assert len(titles) == 20
        assert list(app._voice_items) == ALL_VOICE_NAMES
        assert list(app._speed_items) == ["Faster", "Slower", "Normal speed"]
        assert all(i.callback == app._pick_voice for i in app._voice_items.values())
        assert all(i.callback == app._speed for i in app._speed_items.values())
        # the pystray submenu mirrors it: separators in the same places
        voice = next(i for i in _icon(env).menu.items if i is not FakePystrayMenu.SEPARATOR and i.text == "Voice")
        items = voice.submenu.items
        assert items[11] is FakePystrayMenu.SEPARATOR and items[16] is FakePystrayMenu.SEPARATOR
        assert [i.text for i in items[-3:]] == ["Faster", "Slower", "Normal speed"]
        assert items[-1].checked is None and items[0].checked is not None
    finally:
        _quit_and_join(app)


def test_voice_menu_click_schedules_voice_turn(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
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
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
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
    env, orch_holder = fake_env
    tray = env.tray
    app = tray.VeronicaApp.__new__(tray.VeronicaApp)
    app._voice_items = {"Adam": tray.MenuItem("Adam")}
    app._speed_items = {"Faster": tray.MenuItem("Faster")}
    app._pick_voice(app._voice_items["Adam"])  # must not raise: no self._orch set
    app._speed(app._speed_items["Faster"])
    app._refresh_voice_menu()
    assert app._voice_items["Adam"].state is False


def test_refresh_voice_menu_checks_current(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="bm_george")
        app._refresh_voice_menu()
        assert app._voice_items["George"].state
        assert not app._voice_items["Sarah"].state
        app._orch.tts.voice = "af_sarah"
        app._refresh()  # the 0.25 s timer keeps the checkmark in sync after a voice change by voice
        assert not app._voice_items["George"].state
        assert app._voice_items["Sarah"].state
    finally:
        _quit_and_join(app)


def test_refresh_does_not_raise_before_orch_or_tts(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._refresh()  # FakeOrch has no .tts
        assert not any(i.state for i in app._voice_items.values())
    finally:
        _quit_and_join(app)


def test_voice_menu_hindi_pick_schedules_voice_turn(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    _quit_and_join(app)
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
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="bm_george", hindi_voice="hm_omega")
        app._refresh_voice_menu()
        assert [n for n, i in app._voice_items.items() if i.state] == ["George", "Omega"]
        app._orch.tts.hindi_voice = "hf_beta"
        app._refresh()
        assert [n for n, i in app._voice_items.items() if i.state] == ["George", "Beta"]
    finally:
        _quit_and_join(app)


def test_refresh_voice_menu_tolerates_tts_without_hindi_voice(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._orch = _VoiceOrch(voice="af_sky")
        del app._orch.tts.hindi_voice
        app._refresh_voice_menu()
        assert [n for n, i in app._voice_items.items() if i.state] == ["Sky"]
    finally:
        _quit_and_join(app)


# -- push-to-talk (A2) --------------------------------------------------------

def test_ptt_hotkey_started_with_background_loop_when_enabled(fake_env, monkeypatch):
    env, orch_holder = fake_env
    monkeypatch.setattr(env.tray.settings, "ptt_enabled", True)
    app, orch = _make_app(env, orch_holder)
    try:
        assert len(FakeHotkeyMonitor.instances) == 1
        mon = FakeHotkeyMonitor.instances[0]
        assert mon.started_with_loop is app._loop
        assert mon.keycode == env.tray.settings.ptt_keycode
        assert app._ptt_item is None  # available: no "unavailable" item
    finally:
        _quit_and_join(app)
    assert mon.stopped is True


def test_ptt_hotkey_not_created_when_disabled(fake_env, monkeypatch):
    env, orch_holder = fake_env
    monkeypatch.setattr(env.tray.settings, "ptt_enabled", False)
    app, orch = _make_app(env, orch_holder)
    try:
        assert FakeHotkeyMonitor.instances == []
        assert app._hotkey is None
        assert app._ptt_item is None
    finally:
        _quit_and_join(app)


def test_ptt_unavailable_adds_a_disabled_menu_item(fake_env, monkeypatch):
    env, orch_holder = fake_env
    monkeypatch.setattr(env.tray.settings, "ptt_enabled", True)
    FakeHotkeyMonitor.available_on_start = False
    app, orch = _make_app(env, orch_holder)
    try:
        assert app._ptt_item is not None
        assert app._ptt_item.title == env.tray.PTT_UNAVAILABLE_TITLE
        assert app._ptt_item.callback is None
        assert app.menu.index(app._ptt_item) == len(app.menu) - 2   # just above Quit
    finally:
        _quit_and_join(app)


def test_ptt_press_and_release_call_orchestrator(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        started = threading.Event()
        ended = threading.Event()
        orch.ptt_start = started.set
        orch.ptt_end = ended.set
        mon = FakeHotkeyMonitor.instances[0]
        app._loop.call_soon_threadsafe(mon.on_press)
        assert started.wait(2)
        app._loop.call_soon_threadsafe(mon.on_release)
        assert ended.wait(2)
    finally:
        _quit_and_join(app)


def test_ptt_callbacks_noop_before_orch_exists(fake_env):
    env, orch_holder = fake_env
    app = env.tray.VeronicaApp.__new__(env.tray.VeronicaApp)
    app._on_ptt_press()   # must not raise: no self._orch set
    app._on_ptt_release()


# -- Batch D: About / Settings… / Check for Updates… / settings event -----------

def _titles(app):
    return [None if i is None else i.title for i in app.menu]


def _status(kind):
    from veronica.updater import UpdateStatus
    return UpdateStatus(available=kind != "none", kind=kind, detail=f"{kind} detail",
                        running_sha="abc1234", head_sha="def5678", remote_sha=None)


def test_menu_starts_with_about_settings_and_updates(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        assert _titles(app)[:6] == [
            "About Veronica — Veronica 0.1.0 (abc1234, 17 Sep)", "Settings…", "Check for Updates…",
            env.tray.UPDATE_UNCHECKED_TITLE, None, "Mute",
        ]
        assert _titles(app)[-1] == "Quit"
        assert app.menu[0].callback is None                      # About is a label
        assert app.menu[1].callback == app.open_settings
        assert app.menu[2].callback == app.check_for_updates
        assert app._update_item.callback is None                 # nothing to install yet
        assert app.menu[5] is app._mute_item
    finally:
        _quit_and_join(app)


def test_settings_item_shows_window_on_general(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app.open_settings(app.menu[1])
        assert app._settings.shown == ["general"]
    finally:
        _quit_and_join(app)


def test_settings_event_shows_window_on_tab(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._events.put(("settings", {"open": True, "tab": "history"}))
        app._events.put(("settings", {"open": True}))
        app._events.put(("settings", "garbage"))
        app._drain()
        assert app._settings.shown == ["history", "general", "general"]
        assert not any(e["kind"] == "settings" for e in app._hud.pushed)
    finally:
        _quit_and_join(app)


def test_bridge_and_window_wired(fake_env, monkeypatch):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        bridge = app._bridge
        assert app._settings.bridge is bridge
        assert bridge._get_orch() is orch
        assert bridge._run_on_loop == app._schedule
        assert bridge._repo == env.tray.version.REPO
        assert bridge._exe_path is None and app._exe_path is None
        calls = []
        monkeypatch.setattr(env.tray, "relaunch", lambda exe, quit: calls.append((exe, quit)) or True)
        assert bridge._relaunch() is True
        assert calls == [(app._exe_path, app._schedule_quit)]
    finally:
        _quit_and_join(app)


def test_store_is_late_bound_to_orchestrator(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
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
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._orch = None
        assert app._bridge._history_store() is None
        assert app._bridge.handle("history", {})["message"] == "Memory is off."
    finally:
        _quit_and_join(app)


def test_build_orchestrator_receives_updater_hooks(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        kw = orch.build_kwargs
        assert set(kw) == {"updater_check", "updater_update", "relaunch", "can_relaunch", "version_describe"}
        assert kw["can_relaunch"]() == (app._exe_path is not None)
        assert kw["version_describe"]() == tray.version.describe(app._build_info)
        seen = []
        monkeypatch.setattr(tray.updater, "check", lambda repo, run=None, info=None: seen.append(("check", repo, info)) or _status("none"))
        monkeypatch.setattr(tray.updater, "update", lambda repo, st, **k: seen.append(("update", repo, st)) or "log")
        st = kw["updater_check"]()
        assert st.kind == "none"
        assert kw["updater_update"](st) == "log"
        assert seen[0][0] == "check" and seen[0][1] == tray.version.REPO and seen[0][2]["sha"] == "abc1234"
        assert seen[1] == ("update", tray.version.REPO, st)
        assert kw["relaunch"] == app._relaunch
        assert app._bridge.get_state()["about"]["updating"] is False   # slot released after success
    finally:
        _quit_and_join(app)


def test_voice_update_hook_owns_the_bridge_update_slot(fake_env, monkeypatch):
    from veronica.ui.settings import bridge as bridge_mod
    from veronica.updater import UpdateInProgress
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        hook = orch.build_kwargs["updater_update"]
        st = _status("remote")
        seen = []

        def slow_update(repo, status, **k):
            seen.append(app._bridge.get_state()["about"]["updating"])
            assert app._bridge.begin_update() is False
            assert app._bridge.update_now() == {"ok": False, "message": bridge_mod.BUSY}
            with pytest.raises(UpdateInProgress):
                hook(status)
            assert app._update_item.title == tray.UPDATE_UPDATING_TITLE
            return "log"

        monkeypatch.setattr(tray.updater, "update", slow_update)
        assert hook(st) == "log"
        assert seen == [True]
        assert app._bridge.get_state()["about"]["updating"] is False

        def boom(repo, status, **k):
            raise RuntimeError("build exploded")

        monkeypatch.setattr(tray.updater, "update", boom)
        with pytest.raises(RuntimeError, match="build exploded"):
            hook(st)
        assert app._bridge.get_state()["about"]["updating"] is False
        assert app._bridge.get_state()["about"]["update"]["detail"] == bridge_mod.UPDATE_FAILED
        assert app._update_item.title == tray.UPDATE_FAILED_TITLE
    finally:
        _quit_and_join(app)


def test_check_for_updates_available_enables_item_and_notifies(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        monkeypatch.setattr(tray.updater, "check", lambda repo, run=None, info=None: _status("remote"))
        app._bridge.run_thread = lambda fn: fn()
        app.check_for_updates(app.menu[2])
        assert app._update_item.title == tray.UPDATE_AVAILABLE_TITLE
        assert app._update_item.callback == app.update_now
        assert _icon(env).notifications == [("Veronica — Update available", "remote detail")]
    finally:
        _quit_and_join(app)


def test_check_for_updates_latest_and_failure(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        app._bridge.run_thread = lambda fn: fn()
        monkeypatch.setattr(tray.updater, "check", lambda repo, run=None, info=None: _status("none"))
        app.check_for_updates(app.menu[2])
        assert app._update_item.title == tray.UPDATE_LATEST_TITLE
        assert app._update_item.callback is None
        assert _icon(env).notifications[-1] == ("Veronica — Up to date", "none detail")

        def boom(repo, run=None, info=None):
            raise RuntimeError("offline")

        monkeypatch.setattr(tray.updater, "check", boom)
        app.check_for_updates(app.menu[2])
        assert app._update_item.title == tray.UPDATE_CHECK_FAILED_TITLE
        assert app._update_item.callback is None
        assert _icon(env).notifications[-1] == ("Veronica — Couldn't check for updates", "Couldn't check: offline")
    finally:
        _quit_and_join(app)


def test_check_for_updates_runs_on_bridge_thread_and_dedupes(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        queued = []
        app._bridge.run_thread = lambda fn: queued.append(fn)
        monkeypatch.setattr(tray.updater, "check", lambda repo, run=None, info=None: _status("local"))
        app.check_for_updates(app.menu[2])
        app.check_for_updates(app.menu[2])      # already checking: not queued twice
        assert len(queued) == 1
        assert app._update_item.title == tray.UPDATE_UNCHECKED_TITLE
        queued[0]()
        assert app._update_item.title == tray.UPDATE_AVAILABLE_TITLE
        app.check_for_updates(app.menu[2])      # a later click checks again
        assert len(queued) == 2
    finally:
        _quit_and_join(app)


def test_hourly_update_timer_checks_silently(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        timer = next(t for t in env.timers if t.interval == 3600)
        assert timer.fn == app._hourly_update_check
        app._bridge.run_thread = lambda fn: fn()
        monkeypatch.setattr(tray.updater, "check", lambda repo, run=None, info=None: _status("remote"))
        timer.fn()
        assert app._update_item.title == tray.UPDATE_AVAILABLE_TITLE
        assert app._update_item.callback == app.update_now
        assert _icon(env).notifications == []
    finally:
        _quit_and_join(app)


def test_update_item_click_runs_bridge_update_now(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        calls = []
        monkeypatch.setattr(app._bridge, "update_now", lambda: calls.append("update_now") or {"ok": True, "message": "Updating, back in a moment."})
        app.update_now(app._update_item)
        assert calls == ["update_now"]
        assert app._update_item.title == tray.UPDATE_UPDATING_TITLE
        assert app._update_item.callback is None
        monkeypatch.setattr(app._bridge, "update_now", lambda: {"ok": False, "message": "Busy, try again in a moment."})
        app._update_item.title = tray.UPDATE_AVAILABLE_TITLE
        app._update_item.set_callback(app.update_now)
        app.update_now(app._update_item)
        assert _icon(env).notifications[-1] == ("Veronica", "Busy, try again in a moment.")
        assert app._update_item.title == tray.UPDATE_AVAILABLE_TITLE   # still installable
    finally:
        _quit_and_join(app)


def test_update_failure_pushed_from_bridge_resets_item(fake_env):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        from veronica.ui.settings import bridge as bridge_mod

        def push(update, updating=False):
            app._bridge.on_state_changed({"about": {"update": update, "updating": updating}})

        push({"available": True, "detail": "remote detail"}, updating=True)
        assert app._update_item.title == tray.UPDATE_UPDATING_TITLE     # in progress wins
        assert app._update_item.callback is None
        push({"available": True, "detail": bridge_mod.UPDATE_FAILED})
        assert app._update_item.title == tray.UPDATE_FAILED_TITLE
        assert app._update_item.callback is None
        # the window still gets the state
        assert app._settings.states[-1]["about"]["update"]["detail"] == bridge_mod.UPDATE_FAILED
        # a check from the settings window's "Check now" reflects in the menu too
        push({"available": True, "detail": "remote detail"})
        assert app._update_item.title == tray.UPDATE_AVAILABLE_TITLE
        assert app._update_item.callback == app.update_now
        push({"available": False, "detail": "none detail"})
        assert app._update_item.title == tray.UPDATE_LATEST_TITLE
        # a real bridge push (nothing checked yet) leaves the item alone
        app._update_item.title = tray.UPDATE_UNCHECKED_TITLE
        app._bridge.on_state_changed(app._bridge.get_state())
        assert app._update_item.title == tray.UPDATE_UNCHECKED_TITLE
    finally:
        _quit_and_join(app)


def _wait_for(cond, timeout=2.0):
    import time
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_notify_falls_back_to_spoken_announce_when_notifications_fail(fake_env, monkeypatch, caplog):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        _icon(env).notify_error = OSError("Shell_NotifyIcon failed")
        app._bridge.run_thread = lambda fn: fn()
        monkeypatch.setattr(tray.updater, "check", lambda repo, run=None, info=None: _status("remote"))
        with caplog.at_level(logging.INFO, logger="veronica.ui"):
            app.check_for_updates(app.menu[2])          # must not raise
        assert app._update_item.title == tray.UPDATE_AVAILABLE_TITLE
        assert _icon(env).notifications == []
        _wait_for(lambda: len(orch.announced) == 1)        # announce() ran on the background loop
        assert orch.announced == ["An update is ready. Say update yourself, or use the Settings window."]
        assert "remote detail" in caplog.text

        monkeypatch.setattr(app._bridge, "update_now", lambda: {"ok": False, "message": "Busy, try again in a moment."})
        app.update_now(app._update_item)                # must not raise either
        _wait_for(lambda: len(orch.announced) == 2)
        assert orch.announced[-1] == "Busy, try again in a moment."
    finally:
        _quit_and_join(app)


def test_notify_without_tray_or_orchestrator_only_logs(fake_env, caplog):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._icon, icon = None, app._icon
        app._orch = None
        with caplog.at_level(logging.INFO, logger="veronica.ui"):
            app._notify("Up to date", "none detail", spoken="You're already on the latest.")
        assert "none detail" in caplog.text
    finally:
        app._orch = orch
        app._icon = icon
        _quit_and_join(app)


# -- brains: Brain submenu, "Brain: …" label from hud events -------------------

BRAIN_LABELS = [info.label for info in BACKENDS.values()]


class _BrainOrch:
    """Orchestrator stand-in for the Brain menu: a switcher with the active
    brain's name, and request_brain_switch recording what was asked."""

    def __init__(self, active="codex", preferred="codex", standing_in=False):
        self.switcher = types.SimpleNamespace(
            brain=types.SimpleNamespace(name=active), preferred=preferred, standing_in=standing_in)
        self.requested = []
        self.player = _ResettablePlayer()

    def request_brain_switch(self, name):
        self.requested.append(name)


def test_brain_submenu_lists_backends_after_voice(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        sub = app._brain_menu
        assert sub is app._brain_item and sub.title == "Brain"
        assert app.menu.index(sub) == app.menu.index(app._voice_menu) + 1
        assert [i.title for i in sub.children] == BRAIN_LABELS
        assert list(app._brain_items) == list(BACKENDS)
        assert all(i.callback == app._pick_brain for i in app._brain_items.values())
    finally:
        _quit_and_join(app)


def test_drain_hud_backend_event_sets_brain_title_and_reaches_hud(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._events.put(("hud", {"backend": "Codex"}))
        app._drain()
        assert app._brain_item.title == "Brain: Codex"
        assert app._hud.pushed == [{"kind": "hud", "payload": {"backend": "Codex"}}]
        app._events.put(("hud", {"backend": "Claude (for Codex)"}))
        app._events.put(("hud", {"mode": "mini"}))
        app._drain()
        assert app._brain_item.title == "Brain: Claude (for Codex)"
        assert len(app._hud.pushed) == 2 and app._hud.mode_calls == ["mini"]
        brain = next(i for i in _icon(env).menu.items
                     if i is not FakePystrayMenu.SEPARATOR and i.submenu is not None and i.text.startswith("Brain"))
        assert brain.text == "Brain: Claude (for Codex)"
    finally:
        _quit_and_join(app)


def test_brain_menu_click_requests_switch(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        bo = _BrainOrch(active="codex")
        app._orch = bo
        app._pick_brain(app._brain_items["claude"])
        assert bo.requested == ["claude"]
        app._pick_brain(app._brain_items["copilot"])
        assert bo.requested == ["claude", "copilot"]
    finally:
        _quit_and_join(app)


def test_brain_menu_click_noop_without_orch(fake_env):
    env, orch_holder = fake_env
    tray = env.tray
    app = tray.VeronicaApp.__new__(tray.VeronicaApp)
    app._brain_items = {"codex": tray.MenuItem("Codex")}
    app._brain_avail = {}
    app._brain_checked_at = 0
    app._pick_brain(app._brain_items["codex"])  # must not raise: no self._orch set
    app._refresh_brain_menu()
    assert app._brain_items["codex"].state is False


def test_refresh_brain_menu_checks_active_and_marks_standin(fake_env):
    env, orch_holder = fake_env
    app, orch = _make_app(env, orch_holder)
    try:
        app._orch = _BrainOrch(active="codex")
        app._refresh()
        assert app._brain_items["codex"].state is True
        assert app._brain_items["claude"].state is False
        assert app._brain_items["codex"].title == "Codex"
        # Codex hit its limit, Antigravity is answering for it
        app._orch = _BrainOrch(active="antigravity", preferred="codex", standing_in=True)
        app._refresh()
        assert app._brain_items["antigravity"].state is True
        assert app._brain_items["antigravity"].title == "Antigravity — standing in for Codex"
        assert app._brain_items["codex"].state is False and app._brain_items["codex"].title == "Codex"
        # back on the preferred brain: plain title again
        app._orch = _BrainOrch(active="codex")
        app._refresh()
        assert app._brain_items["antigravity"].title == "Antigravity"
        assert app._brain_items["codex"].state is True
    finally:
        _quit_and_join(app)


def test_refresh_brain_menu_disables_unavailable_from_cached_check(fake_env, monkeypatch):
    env, orch_holder = fake_env
    tray = env.tray
    app, orch = _make_app(env, orch_holder)
    try:
        checks = []

        def fake_check(name):
            checks.append(name)
            if name == "copilot":
                return Availability(False, "not installed", "Copilot isn't installed — …")
            if name == "claude":
                return Availability(False, "not logged in", "Claude isn't logged in — …")
            return Availability(True, "ok")

        monkeypatch.setattr(tray, "check_backend", fake_check)
        app._brain_checked_at = -tray.BRAIN_CHECK_INTERVAL_S    # force the next refresh to re-check
        app._orch = _BrainOrch(active="codex")
        app._refresh()
        assert checks == list(BACKENDS)
        assert app._brain_items["copilot"].title == "Copilot (not installed)"
        assert app._brain_items["copilot"].callback is None
        assert app._brain_items["claude"].title == "Claude (not logged in)"
        assert app._brain_items["claude"].callback is None
        assert app._brain_items["codex"].title == "Codex"
        assert app._brain_items["codex"].callback == app._pick_brain
        # the pystray menu shows the reason and greys the item out
        brain = next(i for i in _icon(env).menu.items
                     if i is not FakePystrayMenu.SEPARATOR and i.submenu is not None and i.text.startswith("Brain"))
        sub = {i.text: i for i in brain.submenu.items}
        assert sub["Copilot (not installed)"].enabled is False
        assert sub["Codex"].enabled is True and sub["Codex"].checked is True
        # the 0.25 s timer doesn't re-run the check until the interval passes
        app._refresh()
        app._refresh()
        assert checks == list(BACKENDS)
        # ...then it does, and a brain that got logged in is enabled again
        monkeypatch.setattr(tray, "check_backend", lambda name: Availability(True, "ok"))
        app._brain_checked_at -= tray.BRAIN_CHECK_INTERVAL_S
        app._refresh()
        assert app._brain_items["copilot"].title == "Copilot"
        assert app._brain_items["copilot"].callback == app._pick_brain
    finally:
        _quit_and_join(app)
