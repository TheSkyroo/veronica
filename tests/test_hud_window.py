"""HudWindow with a fake pywebview window and a fake native panel: no
WebView2, no user32. Screen coordinates are Windows': top-left origin, y down."""
import json
import sys
import types

import pytest

from veronica.config import Settings
from veronica.ui import hud as hud_module
from veronica.ui.hud import FADE_S, SCREEN_CHANGE_SETTLE_S, SCREEN_POLL_S, HudWindow
from veronica.ui.win32 import Rect


class FakeWeb:
    def __init__(self): self.js = []
    def run_js(self, js): self.js.append(js)


class FakePanel:
    def __init__(self, x=0.0, y=0.0, w=540, h=300, scale=1.0):
        self.visible = False
        self.orders = []
        self._rect = Rect(x, y, w, h)
        self.set_frame_calls = []
        self.moves = []
        self._scale = scale

    def show(self): self.visible = True; self.orders.append("front")
    def hide(self): self.visible = False; self.orders.append("out")
    def frame(self): return self._rect
    def scale(self): return self._scale

    def set_frame(self, x, y, w, h):
        self.set_frame_calls.append((x, y, w, h))
        self._rect = Rect(x, y, w, h)

    def move(self, x, y):
        self.moves.append((x, y))
        self._rect = Rect(x, y, self._rect.w, self._rect.h)


def _noop_prefs_load():
    return {}


def _noop_prefs_save(_prefs):
    pass


# A single 1440x900 display (work area at 0,0) stands in for the monitor list.
MAIN_SCREEN = Rect(0, 0, 1440, 900)
MARGIN = Settings().hud_margin
FULL_DEFAULT = (1440 - 540 - MARGIN, MARGIN)          # top-right of the work area
MINI_DEFAULT = ((1440 - 400) / 2, 8)                  # top-center


def make(t=None, mark_loaded=True, screens=None, prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
         panel=None, cursor=None, **settings_over):
    t = [0.0] if t is None else t
    web, panel = FakeWeb(), panel or FakePanel()
    frames = [MAIN_SCREEN] if screens is None else screens
    h = HudWindow(
        Settings(hud_hide_after_s=3.0, **settings_over),
        webview_factory=lambda s, api: web, panel_factory=lambda s, w: panel,
        clock=lambda: t[0], main=lambda fn: fn(),
        prefs_load=prefs_load, prefs_save=prefs_save,
        screens=lambda: frames, cursor=cursor or (lambda: (0, 0)),
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
    assert panel.visible
    h.on_state("idle"); h.tick()
    assert panel.visible                     # not yet
    t[0] = 3.1; h.tick()
    assert panel.visible                     # the page is fading out first
    t[0] = 3.1 + FADE_S; h.tick()
    assert not panel.visible


def test_non_idle_cancels_pending_hide():
    h, _, panel, t = make()
    h.on_state("listening"); h.on_state("idle"); t[0] = 2.0
    h.on_state("thinking"); t[0] = 5.0; h.tick()
    assert panel.visible


def test_show_during_fade_keeps_the_window_up():
    # A hide() starts the fade; a show() (a new turn) before it ends must
    # stop the stale hide from taking the window down under it.
    h, _, panel, t = make()
    h.show(); h.hide()
    t[0] = FADE_S / 2
    h.show()
    t[0] = 10.0; h.tick()
    assert panel.visible and panel.orders == ["front", "front"]


def test_push_after_close_is_noop():
    h, web, _, _ = make()
    h.close(); h.push({"kind": "mic", "payload": 0.1})
    assert web.js == []


def test_close_hides_at_once_and_saves_position():
    saved = {}
    h, _, panel, _ = make(prefs_save=lambda p: saved.update(p))
    h.show()
    panel._rect = Rect(100.0, 200.0, 540, 300)
    h.close()
    assert not panel.visible
    assert saved == {"hud_pos_full": [100.0, 200.0]}


def test_factory_failure_is_soft():
    def bad(s, api): raise RuntimeError("no webview2")
    h = HudWindow(Settings(), webview_factory=bad, panel_factory=lambda s, w: FakePanel(), main=lambda fn: fn(),
                  prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save, screens=lambda: [MAIN_SCREEN])
    h.push({"kind": "state", "payload": "idle"}); h.on_state("listening"); h.tick(); h.hide(); h.close()  # no raise
    assert h.available is False


def test_show_calls_set_visible_true_and_hide_calls_set_visible_false():
    h, web, panel, t = make()
    h.show()
    assert web.js[-1] == "window.hud.setVisible(true)"
    h.hide()
    assert web.js[-1] == "window.hud.setVisible(false)"


def test_panel_failure_on_show_is_soft(caplog):
    class BrokenPanel(FakePanel):
        def show(self): raise RuntimeError("no hwnd")

    h, _, panel, _ = make(panel=BrokenPanel())
    with caplog.at_level("WARNING", logger="veronica.ui.hud"):
        h.show()
    assert "failed to show HUD window" in caplog.text


# -- mode / geometry ---------------------------------------------------------------

def test_default_mode_is_full_and_construction_places_it_top_right():
    h, _, panel, _ = make()
    assert h._mode == "full"
    assert panel.set_frame_calls == [(*FULL_DEFAULT, 540, 300)]


def test_set_mode_mini_resizes_panel_and_calls_js():
    h, web, panel, _ = make()
    h.set_mode("mini")
    assert h._mode == "mini"
    assert panel.set_frame_calls[-1][2:] == (400, 72)   # width, height
    assert web.js[-1] == 'window.hud.setMode("mini")'


def test_set_mode_full_resizes_back():
    h, web, panel, _ = make()
    h.set_mode("mini")
    h.set_mode("full")
    assert panel.set_frame_calls[-1][2:] == (540, 300)
    assert web.js[-1] == 'window.hud.setMode("full")'


def test_set_mode_ignores_unknown_mode():
    h, web, panel, _ = make()
    n = len(panel.set_frame_calls)
    h.set_mode("huge")
    assert h._mode == "full"
    assert len(panel.set_frame_calls) == n


def test_set_mode_persists_pref():
    saved = {}
    h, _, _, _ = make(prefs_save=lambda p: saved.update(p))
    h.set_mode("mini")
    assert saved == {"hud_mode": "mini"}


def test_geometry_scales_with_the_monitor_dpi():
    h, _, panel, _ = make(panel=FakePanel(scale=1.5))
    x, y, w, hgt = panel.set_frame_calls[-1]
    assert (w, hgt) == (810, 450)
    assert (x, y) == (1440 - 810 - MARGIN * 1.5, MARGIN * 1.5)


def test_construction_applies_saved_mini_mode():
    h, _, panel, _ = make(prefs_load=lambda: {"hud_mode": "mini"})
    assert h._mode == "mini"
    assert panel.set_frame_calls[-1] == (*MINI_DEFAULT, 400, 72)


def test_construction_falls_back_to_settings_hud_mode_when_no_saved_pref():
    h, _, _, _ = make(hud_mode="mini")
    assert h._mode == "mini"


def test_construction_ignores_invalid_saved_mode():
    h, _, _, _ = make(prefs_load=lambda: {"hud_mode": "gigantic"})
    assert h._mode == "full"


def test_construction_applies_saved_position():
    h, _, panel, _ = make(prefs_load=lambda: {"hud_pos_full": [12.0, 34.0]})
    assert h._pos["full"] == (12.0, 34.0)
    assert panel.set_frame_calls[-1][:2] == (12.0, 34.0)


def test_hide_persists_panel_position():
    saved = {}
    h, _, panel, _ = make(prefs_save=lambda p: saved.update(p))
    h.show()
    panel._rect = Rect(100.0, 200.0, 540, 300)
    h.hide()
    assert saved == {"hud_pos_full": [100.0, 200.0]}
    assert h._pos["full"] == (100.0, 200.0)


def test_hide_when_never_shown_does_not_persist_a_position():
    saved = {}
    h, _, _, _ = make(prefs_save=lambda p: saved.update(p))
    h.hide()
    assert saved == {}


def test_hide_persists_panel_position_separately_per_mode():
    saved = {}
    h, _, panel, _ = make(prefs_save=lambda p: saved.update(p))
    h.set_mode("mini")
    h.show()
    # Simulate the user dragging the (now mini) window to a new spot.
    panel._rect = Rect(7.0, 8.0, 400, 72)
    saved.clear()
    h.hide()
    assert saved == {"hud_pos_mini": [7.0, 8.0]}
    assert h._pos["mini"] == (7.0, 8.0)
    assert h._pos["full"] is None


def test_set_mode_mini_defaults_to_top_center():
    """With no saved position for mini mode, switching to mini places the
    slim bar centered at the top of the primary display, not in the full
    card's top-right corner."""
    h, web, panel, _ = make()
    h.set_mode("mini")
    assert panel.set_frame_calls[-1] == (*MINI_DEFAULT, 400, 72)


def test_prefs_load_failure_is_soft():
    def boom():
        raise RuntimeError("disk error")
    h, _, _, _ = make(prefs_load=boom)
    assert h._mode == "full"   # falls back to settings default
    assert h.available


# -- JS load gate ------------------------------------------------------------------

def test_js_before_loaded_is_queued_not_evaluated():
    h, web, _, _ = make(mark_loaded=False)
    h.push({"kind": "mic", "payload": 0.5})
    assert web.js == []


def test_mark_loaded_replays_mode_first_then_flushes_queued_pushes_in_order():
    h, web, _, _ = make(mark_loaded=False, prefs_load=lambda: {"hud_mode": "mini"})
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
    h, web, _, _ = make(mark_loaded=False, hud_particles=3000, hud_intensity=1.25)
    h.push({"kind": "mic", "payload": 0.2})
    h.mark_loaded()
    assert web.js[0] == 'window.hud.setMode("full")'
    assert web.js[1] == 'window.hud.configure({"particles": 3000, "intensity": 1.25})'
    assert web.js[-1].startswith("window.hud.push(")


def test_push_after_loaded_evaluates_immediately():
    h, web, _, _ = make(mark_loaded=True)
    h.push({"kind": "mic", "payload": 0.9})
    assert web.js == [
        "window.hud.push(" + json.dumps({"kind": "mic", "payload": 0.9}, ensure_ascii=False) + ")",
    ]


def test_js_failure_is_soft():
    class BrokenWeb(FakeWeb):
        def run_js(self, js): raise RuntimeError("window destroyed")

    web = BrokenWeb()
    h = HudWindow(Settings(), webview_factory=lambda s, api: web, panel_factory=lambda s, w: FakePanel(),
                  main=lambda fn: fn(), prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save,
                  screens=lambda: [MAIN_SCREEN])
    h.mark_loaded()
    h.push({"kind": "mic", "payload": 0.1})   # no raise


def test_close_before_load_discards_pending_queue():
    h, web, _, _ = make(mark_loaded=False)
    h.push({"kind": "mic", "payload": 0.1})
    assert web.js == []

    h.close()
    h.mark_loaded()

    assert web.js == []


# -- drag / click -> menu (js_api) -------------------------------------------------

def test_js_api_exposes_only_the_drag_and_menu_calls():
    h, _, _, _ = make()
    public = sorted(n for n in dir(h.api) if not n.startswith("_"))
    assert public == ["drag_end", "drag_start", "drag_to", "menu"]


def test_drag_moves_the_window_from_where_it_started():
    h, _, panel, _ = make()
    panel._rect = Rect(100.0, 50.0, 540, 300)
    h.api.drag_start()
    h.api.drag_to(10, 5)
    h.api.drag_to("30", "-20")            # JS numbers may arrive as anything json-ish
    assert panel.moves == [(110.0, 55.0), (130.0, 30.0)]


def test_drag_end_saves_the_new_position():
    saved = {}
    h, _, panel, _ = make(prefs_save=lambda p: saved.update(p))
    panel._rect = Rect(100.0, 50.0, 540, 300)
    h.api.drag_start()
    h.api.drag_to(200, 100)
    h.api.drag_end()
    assert saved == {"hud_pos_full": [300.0, 150.0]}
    assert h._pos["full"] == (300.0, 150.0)
    h.api.drag_to(1, 1)                   # no drag in progress: ignored
    assert len(panel.moves) == 1


def test_drag_end_without_start_saves_nothing():
    saved = {}
    h, _, _, _ = make(prefs_save=lambda p: saved.update(p))
    h.api.drag_end()
    assert saved == {}


def test_click_pops_up_the_menu_at_the_cursor():
    seen = []
    h, _, _, _ = make(cursor=lambda: (111, 222))
    h.on_menu = lambda x, y: seen.append((x, y))
    h.api.menu()
    assert seen == [(111.0, 222.0)]


def test_menu_noop_when_on_menu_unset():
    h, _, _, _ = make()
    h.api.menu()   # must not raise


def test_menu_cursor_failure_is_soft():
    def boom():
        raise OSError("no cursor")
    seen = []
    h, _, _, _ = make(cursor=boom)
    h.on_menu = lambda x, y: seen.append((x, y))
    h.api.menu()
    assert seen == []


# -- display changes / off-screen recovery -----------------------------------

def test_saved_position_on_a_vanished_screen_falls_back_to_default_and_clears_pref():
    """A saved origin that no longer lands (>= 25%) on any current screen
    must be discarded — top-right default on the primary screen, pref cleared."""
    saved = {}
    h, _, panel, _ = make(
        prefs_load=lambda: {"hud_pos_full": [2000.0, 300.0]},   # was on a second display
        prefs_save=lambda p: saved.update(p),
    )
    assert panel.set_frame_calls[-1] == (*FULL_DEFAULT, 540, 300)
    assert h._pos["full"] is None
    assert saved == {"hud_pos_full": None}


def test_half_off_screen_saved_frame_is_clamped_inside_best_screen():
    saved = {}
    h, _, panel, _ = make(
        prefs_load=lambda: {"hud_pos_full": [870.0, -39.0]},
        prefs_save=lambda p: saved.update(p),
    )
    assert panel.set_frame_calls[-1] == (870.0, 0.0, 540, 300)
    assert h._pos["full"] == (870.0, -39.0)   # still the user's spot; only the frame is clamped
    assert saved == {}


def test_position_on_secondary_screen_stays_on_secondary():
    secondary = Rect(1440, 100, 1920, 1080)
    h, _, panel, _ = make(
        screens=[MAIN_SCREEN, secondary],
        prefs_load=lambda: {"hud_pos_full": [2500.0, 500.0]},
    )
    assert panel.set_frame_calls[-1][:2] == (2500.0, 500.0)
    # Partly past the secondary's right edge: clamped against the secondary,
    # not dragged back to the main display.
    h._pos["full"] = (3200.0, 500.0)
    h._apply_geometry()
    assert panel.set_frame_calls[-1][:2] == (1440 + 1920 - 540, 500.0)


def test_monitor_layout_change_repositions_hud(caplog):
    frames = [Rect(0, 0, 2560, 1440)]
    h, _, panel, t = make(screens=frames, prefs_load=lambda: {"hud_pos_full": [2000.0, 1100.0]})
    assert panel.set_frame_calls[-1][:2] == (2000.0, 1100.0)
    before = len(panel.set_frame_calls)
    t[0] = SCREEN_POLL_S
    h.tick()                                  # same layout: nothing to do
    assert len(panel.set_frame_calls) == before
    # The big display goes away; only the laptop panel is left.
    frames[:] = [MAIN_SCREEN]
    with caplog.at_level("INFO", logger="veronica.ui.hud"):
        h.tick()                              # not polled again yet
        t[0] = 2 * SCREEN_POLL_S
        h.tick()                              # noticed; waits for things to settle
        assert len(panel.set_frame_calls) == before
        t[0] += SCREEN_CHANGE_SETTLE_S
        h.tick()
        h.tick()
    assert len(panel.set_frame_calls) == before + 1
    assert panel.set_frame_calls[-1] == (*FULL_DEFAULT, 540, 300)
    assert caplog.text.count("display change: repositioning HUD") == 1


def test_empty_monitor_list_is_not_a_layout_change():
    frames = [MAIN_SCREEN]
    h, _, panel, t = make(screens=frames)
    before = len(panel.set_frame_calls)
    frames[:] = []                            # a failed query
    t[0] = SCREEN_POLL_S + SCREEN_CHANGE_SETTLE_S
    h.tick(); h.tick()
    assert len(panel.set_frame_calls) == before


def test_no_layout_polling_after_close():
    frames = [MAIN_SCREEN]
    h, _, panel, t = make(screens=frames)
    h.close()
    before = len(panel.set_frame_calls)
    frames[:] = [Rect(0, 0, 800, 600)]
    t[0] = 10 * SCREEN_POLL_S
    h.tick(); t[0] += 1; h.tick()
    assert len(panel.set_frame_calls) == before


def test_save_position_clamps_before_persisting():
    saved = {}
    h, _, panel, _ = make(prefs_save=lambda p: saved.update(p))
    h.show()
    panel._rect = Rect(870.0, -39.0, 540, 300)
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
    assert panel.set_frame_calls[-1][:2] == FULL_DEFAULT
    assert panel.visible
    assert "window.hud.setVisible(true)" in web.js


def test_no_screen_info_keeps_saved_position_untouched():
    """Without any monitor list (the query failed) there's nothing to
    validate against: don't second-guess the saved spot."""
    h, _, panel, _ = make(screens=[], prefs_load=lambda: {"hud_pos_full": [12.0, 34.0]})
    assert panel.set_frame_calls[-1][:2] == (12.0, 34.0)
    assert h._pos["full"] == (12.0, 34.0)


# -- the real factories, with pywebview / user32 faked ---------------------------------

def test_real_webview_is_a_hidden_frameless_non_focusable_topmost_window(monkeypatch):
    created = {}

    def create_window(title, **kw):
        created.update(kw, title=title)
        return "window"

    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=create_window))
    api = object()
    assert hud_module._real_webview(Settings(), api) == "window"
    assert created["title"] == hud_module.TITLE
    assert created["js_api"] is api
    assert created["url"].startswith("file:") and created["url"].endswith("/hud/index.html")
    for key, want in {"hidden": True, "frameless": True, "focus": False, "on_top": True,
                      "transparent": True, "easy_drag": False, "resizable": False, "shadow": False}.items():
        assert created[key] is want, key


def test_real_webview_hooks_page_load(monkeypatch):
    class Event:
        def __init__(self): self.handlers = []
        def __iadd__(self, fn): self.handlers.append(fn); return self

    class Window(FakeWeb):
        def __init__(self):
            super().__init__()
            self.events = types.SimpleNamespace(loaded=Event())

    window = Window()
    monkeypatch.setattr(hud_module, "_real_webview", lambda s, api: window)
    h = HudWindow(Settings(), panel_factory=lambda s, w: FakePanel(), main=lambda fn: fn(),
                  prefs_load=_noop_prefs_load, prefs_save=_noop_prefs_save, screens=lambda: [MAIN_SCREEN])
    assert len(window.events.loaded.handlers) == 1
    window.events.loaded.handlers[0]()
    assert h._loaded and window.js[0] == 'window.hud.setMode("full")'


@pytest.fixture
def fake_win32(monkeypatch):
    calls = []
    w = hud_module.win32
    monkeypatch.setattr(w, "hwnd_of", lambda window, timeout=20.0: calls.append(("hwnd_of",)) or window.hwnd)
    monkeypatch.setattr(w, "make_floating", lambda hwnd: calls.append(("floating", hwnd)))
    monkeypatch.setattr(w, "show_no_activate", lambda hwnd: calls.append(("show", hwnd)))
    monkeypatch.setattr(w, "hide", lambda hwnd: calls.append(("hide", hwnd)))
    monkeypatch.setattr(w, "window_rect", lambda hwnd: Rect(1, 2, 3, 4))
    monkeypatch.setattr(w, "set_window_rect", lambda hwnd, *a: calls.append(("rect", hwnd, *a)))
    monkeypatch.setattr(w, "dpi_scale", lambda hwnd=None: 1.25)
    return calls


def test_win32_panel_styles_the_window_once_and_never_activates_it(fake_win32):
    panel = hud_module._real_panel(Settings(), types.SimpleNamespace(hwnd=77))
    assert fake_win32 == []                       # nothing until first use (GUI loop not up yet)
    panel.show(); panel.hide(); panel.set_frame(1, 2, 3, 4); panel.move(5, 6)
    assert panel.frame() == Rect(1, 2, 3, 4) and panel.scale() == 1.25
    assert fake_win32 == [("hwnd_of",), ("floating", 77), ("show", 77), ("hide", 77),
                          ("rect", 77, 1, 2, 3, 4), ("rect", 77, 5, 6)]


def test_win32_panel_without_a_handle_fails_fast_after_the_first_wait(fake_win32):
    panel = hud_module._real_panel(Settings(), types.SimpleNamespace(hwnd=None))
    with pytest.raises(RuntimeError):
        panel.show()
    with pytest.raises(RuntimeError):
        panel.hide()
    assert fake_win32.count(("hwnd_of",)) == 1    # waited for the GUI once, not on every call
