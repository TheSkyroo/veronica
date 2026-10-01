"""Floating HUD window hosting the orb web page (pywebview + WebView2).

The window is frameless, transparent, always on top and never takes focus:
once the GUI loop has created its HWND we add WS_EX_NOACTIVATE |
WS_EX_TOOLWINDOW | WS_EX_TOPMOST (no taskbar button, no Alt-Tab entry) and
only ever show it with SW_SHOWNOACTIVATE, so speaking to Veronica never pulls
the keyboard away from the app being typed in.

Python → JS: `window.hud.push(event)` etc. through `window.run_js`, queued
until the page has loaded (`_pending_js`). JS → Python: hud.js calls the
js_api object (`window.pywebview.api.drag_start/drag_to/drag_end/menu`) —
a press-and-drag moves the window (the position is remembered per mode in
prefs.json), a plain click or a right-click pops up the tray's menu at the
cursor (`on_menu`, wired by veronica.ui.tray).

Display changes (a monitor unplugged, a resolution change) are noticed by
polling the monitor layout every SCREEN_POLL_S from `tick()`; after it has
been stable for SCREEN_CHANGE_SETTLE_S the window is re-placed so it can't be
stranded on a display that no longer exists.

`webview_factory`/`panel_factory`/`screens`/`cursor` let tests pass fakes:
pywebview and ctypes' user32 are only touched by the real factories.
"""
import json
import logging
import time
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

from veronica import prefs
from veronica.config import Settings
from veronica.ui import win32
from veronica.ui.dispatch import on_ui_thread
from veronica.ui.win32 import Rect

log = logging.getLogger("veronica.ui.hud")

TITLE = "Veronica HUD"
NON_IDLE = frozenset({"listening", "thinking", "speaking", "followup", "confirming", "error", "warming"})
SCREEN_POLL_S = 2.0            # how often tick() re-reads the monitor layout
SCREEN_CHANGE_SETTLE_S = 0.3   # quiet time after a layout change before repositioning
FADE_S = 0.15                  # hud.css fades the page out; the window is hidden after this

MODES = frozenset({"full", "mini"})

# A saved rect must still land at least this much of its area on some
# screen to be worth keeping; below that it's treated as "on a display
# that's gone" and the default spot is used instead.
MIN_VISIBLE_FRACTION = 0.25

_main_thread = on_ui_thread


class _HudApi:
    """The js_api object hud.js talks to (`window.pywebview.api.*`).
    pywebview calls these on its own short-lived threads; only public
    methods are exposed to the page."""

    def __init__(self, hud: "HudWindow") -> None:
        self._hud = hud

    def drag_start(self) -> None:
        self._hud._drag_start()

    def drag_to(self, dx, dy) -> None:
        self._hud._drag_to(float(dx), float(dy))

    def drag_end(self) -> None:
        self._hud._drag_end()

    def menu(self) -> None:
        self._hud._menu()


def _real_webview(s: Settings, api: _HudApi):
    import webview

    html = files("veronica.ui.hud") / "index.html"
    return webview.create_window(
        TITLE, url=Path(str(html)).as_uri(), js_api=api,
        width=s.hud_width, height=s.hud_height, hidden=True, frameless=True, easy_drag=False,
        shadow=False, focus=False, on_top=True, transparent=True, resizable=False,
        background_color="#000000",
    )


class _Win32Panel:
    """The HUD's native window: show/hide without activating, move/resize
    in physical pixels. The HWND only exists once webview.start()'s GUI loop
    has created the form, so it's resolved (and styled) on first use."""

    def __init__(self, window) -> None:
        self._window = window
        self._hwnd: int | None = None
        self._failed = False

    def _handle(self) -> int:
        if self._hwnd is None:
            # Wait for the GUI loop once; if the window never appeared, fail
            # fast from then on instead of stalling the UI thread every call.
            hwnd = None if self._failed else win32.hwnd_of(self._window)
            if hwnd is None:
                self._failed = True
                raise RuntimeError("HUD window has no native handle (GUI loop not running?)")
            win32.make_floating(hwnd)
            self._hwnd = hwnd
        return self._hwnd

    def show(self) -> None:
        win32.show_no_activate(self._handle())

    def hide(self) -> None:
        win32.hide(self._handle())

    def frame(self) -> Rect:
        return win32.window_rect(self._handle())

    def set_frame(self, x: float, y: float, w: float, h: float) -> None:
        win32.set_window_rect(self._handle(), x, y, w, h)

    def move(self, x: float, y: float) -> None:
        win32.set_window_rect(self._handle(), x, y)

    def scale(self) -> float:
        return win32.dpi_scale(self._handle())


def _real_panel(s: Settings, window) -> _Win32Panel:
    return _Win32Panel(window)


def _real_screens() -> list[Rect]:
    """Work area of every attached display, index 0 = the primary one."""
    return win32.work_areas()


class HudWindow:
    def __init__(self, settings: Settings, *, webview_factory=None, panel_factory=None,
                 clock: Callable[[], float] = time.monotonic, main: Callable[[Callable[[], None]], None] = _main_thread,
                 prefs_load: Callable[[], dict] = prefs.load,
                 prefs_save: Callable[[dict], None] = prefs.save,
                 screens: Callable[[], list] | None = None,
                 cursor: Callable[[], tuple[float, float]] | None = None) -> None:
        """`webview_factory(settings, api)` returns the pywebview window
        (default: webview.create_window, hidden — call before webview.start()
        or from any thread after it); `panel_factory(settings, window)` the
        native-window controller. `screens` returns the work area (Rect) of
        every display, primary first; `cursor` the mouse position."""
        self.s = settings
        self._clock = clock
        self._main = main
        self._prefs_load = prefs_load
        self._prefs_save = prefs_save
        self._screens_fn = screens or _real_screens
        self._cursor = cursor or win32.cursor_pos
        self._hide_at: float | None = None
        self._order_out_at: float | None = None   # window hidden once the CSS fade is done
        # The monitor layout is polled from tick(); a change is acted on
        # once it has been quiet for SCREEN_CHANGE_SETTLE_S.
        self._reposition_at: float | None = None
        self._screen_poll_at = clock() + SCREEN_POLL_S
        self._layout: tuple | None = None
        self._closed = False
        self._visible = False
        self.available = False
        self._loaded = False
        self._pending_js: list[str] = []
        self._drag_origin: Rect | None = None
        self.api = _HudApi(self)
        # Set by the tray app (`hud.on_menu = self._popup_menu_at`); called
        # with (x, y) screen coordinates when the orb is clicked.
        self.on_menu: Callable[[float, float], None] | None = None

        saved: dict = {}
        try:
            saved = self._prefs_load() or {}
        except Exception:
            log.warning("failed to load HUD prefs", exc_info=True)
        mode = saved.get("hud_mode") or settings.hud_mode
        self._mode = mode if mode in MODES else "full"
        # Dragged position persists per mode (full vs mini have very
        # different default locations, so a drag in one shouldn't move the
        # other).
        self._pos: dict[str, tuple[float, float] | None] = {"full": None, "mini": None}
        for m in MODES:
            pos = saved.get(f"hud_pos_{m}")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                self._pos[m] = (float(pos[0]), float(pos[1]))

        is_real_webview = webview_factory is None
        try:
            self._web = (webview_factory or _real_webview)(settings, self.api)
            self._panel = (panel_factory or _real_panel)(settings, self._web)
            self.available = True
        except Exception:
            log.warning("HUD unavailable; continuing without it", exc_info=True)
            self._web = self._panel = None

        if self.available and is_real_webview:
            try:
                # pywebview fires `loaded` on one of its own threads.
                self._web.events.loaded += lambda: self._main(self._on_loaded)
            except Exception:
                log.warning("failed to hook HUD page load", exc_info=True)

        if self.available:
            self._layout = self._layout_key(self._screens())
            # Size and place it (saved spot, or the default one for the
            # mode) before it's ever shown.
            self._apply_geometry()

    # -- mode / geometry --------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Switch between the full card layout and the Siri-style mini orb,
        resizing/repositioning the window and persisting the preference."""
        if mode not in MODES:
            return
        self._mode = mode
        if self.available:
            self._apply_geometry()
        try:
            self._prefs_save({"hud_mode": self._mode})
        except Exception:
            log.warning("failed to save HUD mode pref", exc_info=True)

    # -- JS dispatch / page load -------------------------------------------------
    def _run_js(self, js: str) -> None:
        try:
            self._web.run_js(js)
        except Exception:
            log.debug("HUD JS failed", exc_info=True)

    def _js(self, js: str) -> None:
        """Route a JS call through the load gate: queued until the page has
        actually finished loading (index.html), otherwise it's lost — e.g. a
        setMode('mini')/push() fired right after construction, before
        WebView2 finishes navigation. UI thread only."""
        if not self._loaded:
            self._pending_js.append(js)
            return
        self._run_js(js)

    def mark_loaded(self) -> None:
        """Public hook for the page-load event (and for tests, whose fake
        windows never fire one) to signal that index.html has loaded."""
        self._on_loaded()

    def _on_loaded(self) -> None:
        self._loaded = True
        if self._closed:
            # close() already discarded the queue; nothing left to do.
            self._pending_js = []
            return
        # Re-apply the current mode first (whatever was queued during
        # construction may be stale/duplicated after this), then flush
        # everything else queued while the page was still loading, in order.
        self._run_js("window.hud.setMode(" + json.dumps(self._mode) + ")")
        # Initial orb config (particle count / intensity) from settings,
        # before anything queued so a queued configure() from a live
        # settings change still wins.
        self._run_js(self._configure_js(self._initial_config()))
        pending, self._pending_js = self._pending_js, []
        for js in pending:
            self._run_js(js)

    def _scale(self) -> float:
        try:
            return float(self._panel.scale()) or 1.0
        except Exception:
            return 1.0

    def _geometry(self) -> tuple[int, int]:
        """The window size for the current mode, in physical pixels (the
        settings are CSS pixels)."""
        k = self._scale()
        if self._mode == "mini":
            return round(self.s.hud_mini_width * k), round(self.s.hud_mini_height * k)
        return round(self.s.hud_width * k), round(self.s.hud_height * k)

    def _screens(self) -> list:
        """Work area of every attached display ([] if unknown)."""
        try:
            return list(self._screens_fn() or [])
        except Exception:
            return []

    @staticmethod
    def _layout_key(screens: list) -> tuple:
        return tuple((r.x, r.y, r.w, r.h) for r in screens)

    def _top_right_origin(self, w: int, h: int) -> tuple[float, float]:
        screens = self._screens()
        if not screens:
            return 0.0, 0.0
        work, margin = screens[0], self.s.hud_margin * self._scale()
        return work.x + work.w - w - margin, work.y + margin

    def _top_center_origin(self, w: int, h: int) -> tuple[float, float]:
        """Mini mode's default spot: a slim bar centered at the top of the
        primary display, rather than the full card's top-right corner."""
        screens = self._screens()
        if not screens:
            return 0.0, 0.0
        work = screens[0]
        return work.x + (work.w - w) / 2, work.y + 8 * self._scale()

    def _default_origin(self, w: int, h: int) -> tuple[float, float]:
        if self._mode == "mini":
            return self._top_center_origin(w, h)
        return self._top_right_origin(w, h)

    @staticmethod
    def _clamp_to_frame(x: float, y: float, w: int, h: int, screen: Rect) -> tuple[float, float]:
        max_x = screen.x + screen.w - w
        max_y = screen.y + screen.h - h
        return min(max(x, screen.x), max_x), min(max(y, screen.y), max_y)

    @staticmethod
    def _intersection_area(x: float, y: float, w: int, h: int, screen: Rect) -> float:
        ix = min(x + w, screen.x + screen.w) - max(x, screen.x)
        iy = min(y + h, screen.y + screen.h) - max(y, screen.y)
        return max(ix, 0.0) * max(iy, 0.0)

    def _forget_position(self, mode: str) -> None:
        self._pos[mode] = None
        try:
            self._prefs_save({f"hud_pos_{mode}": None})
        except Exception:
            log.warning("failed to clear HUD position pref", exc_info=True)

    def _place(self, pos: tuple[float, float], w: int, h: int) -> tuple[float, float]:
        """Validate a saved/dragged origin against the displays that exist
        right now: pick the screen the rect (pos, w, h) mostly lands on and
        clamp it fully inside; if it's (almost) entirely off every screen
        — the display it was on is gone — forget it and use the default
        spot on the primary screen. With no screen info, pass it through."""
        screens = self._screens()
        if not screens:
            return pos
        x, y = pos
        best = max(screens, key=lambda scr: self._intersection_area(x, y, w, h, scr))
        if self._intersection_area(x, y, w, h, best) < MIN_VISIBLE_FRACTION * w * h:
            log.info("hud position (%.0f,%.0f) is off every screen; using the default spot", x, y)
            self._forget_position(self._mode)
            return self._default_origin(w, h)
        return self._clamp_to_frame(x, y, w, h, best)

    def _origin(self, w: int, h: int) -> tuple[float, float]:
        pos = self._pos[self._mode]
        if pos is not None:
            return self._place(pos, w, h)
        return self._default_origin(w, h)

    def _apply_geometry(self) -> None:
        mode = self._mode

        def _do():
            w, h = self._geometry()
            x, y = self._origin(w, h)
            try:
                self._panel.set_frame(x, y, w, h)
            except Exception:
                log.warning("failed to resize/reposition HUD window", exc_info=True)
            self._js("window.hud.setMode(" + json.dumps(mode) + ")")
        self._main(_do)

    def _save_position(self) -> None:
        """Read the window's current on-screen origin (UI thread) and
        persist it, so the HUD reopens where it was dragged."""
        if not self.available:
            return
        try:
            f = self._panel.frame()
            x, y, w, h = float(f.x), float(f.y), int(f.w), int(f.h)
        except Exception:
            return
        # Never persist a half-off-screen origin (the window can be dragged
        # partly past an edge): keep the clamped spot instead.
        x, y = self._place((x, y), w, h)
        self._pos[self._mode] = (x, y)
        try:
            self._prefs_save({f"hud_pos_{self._mode}": [x, y]})
        except Exception:
            log.warning("failed to save HUD position pref", exc_info=True)

    def reset_position(self) -> None:
        """The "where are you?" intent: forget both saved positions, put the
        window back at its default spot on the primary screen and show it."""
        for m in MODES:
            self._forget_position(m)
        if not self.available or self._closed:
            return
        self._apply_geometry()
        self.show()

    # -- display changes ------------------------------------------------------
    def _poll_screens(self, now: float) -> None:
        self._screen_poll_at = now + SCREEN_POLL_S
        layout = self._layout_key(self._screens())
        if not layout or layout == self._layout:
            return
        self._layout = layout
        self._on_screens_changed()

    def _on_screens_changed(self) -> None:
        """Display layout changed (monitor plugged/unplugged, resolution or
        arrangement changed): re-validate the window's frame, once things
        have settled, so it lands on a screen that exists."""
        if self._closed or not self.available:
            return
        self._reposition_at = self._clock() + SCREEN_CHANGE_SETTLE_S

    # -- drag / menu (js_api, any thread) ------------------------------------------
    def _drag_start(self) -> None:
        if not self.available or self._closed:
            return
        try:
            self._drag_origin = self._panel.frame()
        except Exception:
            self._drag_origin = None

    def _drag_to(self, dx: float, dy: float) -> None:
        """hud.js reports the pointer's offset (physical px) from where the
        drag started; move the window by the same amount. Called straight
        from the js_api thread — SetWindowPos is thread-safe, and going
        through the UI queue would make the window lag the pointer."""
        origin = self._drag_origin
        if origin is None:
            return
        try:
            self._panel.move(origin.x + dx, origin.y + dy)
        except Exception:
            log.debug("HUD drag move failed", exc_info=True)

    def _drag_end(self) -> None:
        if self._drag_origin is None:
            return
        self._drag_origin = None
        self._main(self._save_position)

    def _menu(self) -> None:
        """A plain click or right-click on the page: pop the menu up at the
        cursor."""
        try:
            x, y = self._cursor()
        except Exception:
            log.warning("failed to read the cursor position for the HUD menu", exc_info=True)
            return
        self._main(lambda: self._popup_menu(float(x), float(y)))

    def _popup_menu(self, x: float, y: float) -> None:
        if self.on_menu is not None:
            self.on_menu(x, y)

    # -- events ---------------------------------------------------------------
    def push(self, event: dict) -> None:
        if not self.available or self._closed:
            return
        js = "window.hud.push(" + json.dumps(event, ensure_ascii=False) + ")"
        self._main(lambda: self._js(js))

    def _initial_config(self) -> dict:
        return {"particles": int(self.s.hud_particles), "intensity": float(self.s.hud_intensity)}

    @staticmethod
    def _configure_js(cfg: dict) -> str:
        return "window.hud.configure(" + json.dumps(cfg) + ")"

    def configure(self, cfg: dict) -> None:
        """Live orb config (`{"particles": int, "intensity": float}`),
        routed through the load gate like push()."""
        if not self.available or self._closed:
            return
        js = self._configure_js(dict(cfg or {}))
        self._main(lambda: self._js(js))

    def on_state(self, state: str) -> None:
        if not self.available or self._closed:
            return
        if state in NON_IDLE:
            self._hide_at = None
            self.show()
        elif state == "idle":
            self._hide_at = self._clock() + self.s.hud_hide_after_s

    def tick(self) -> None:
        """Called ~30 times a second on the UI thread (the tray's drain)."""
        now = self._clock()
        if self._hide_at is not None and now >= self._hide_at:
            self._hide_at = None
            self.hide()
        if self._order_out_at is not None and now >= self._order_out_at:
            self._order_out_at = None
            self._order_out()
        if self.available and not self._closed and now >= self._screen_poll_at:
            self._poll_screens(now)
        if self._reposition_at is not None and now >= self._reposition_at:
            self._reposition_at = None
            if not self._closed and self.available:
                log.info("display change: repositioning HUD")
                self._apply_geometry()

    # -- visibility -----------------------------------------------------------
    def _order_out(self) -> None:
        try:
            self._panel.hide()
        except Exception:
            log.warning("failed to hide HUD window", exc_info=True)

    def show(self) -> None:
        if not self.available:
            return
        # A newer show supersedes a hide whose fade is still running.
        self._order_out_at = None

        def _do():
            self._js("window.hud.setVisible(true)")
            try:
                self._panel.show()
            except Exception:
                log.warning("failed to show HUD window", exc_info=True)
                return
            self._visible = True
            # One line per show so a "HUD disappeared" report can be
            # matched against where the window actually was and whether the
            # page had loaded.
            try:
                f = self._panel.frame()
                log.info("hud show mode=%s frame=(%.0f,%.0f %.0fx%.0f) loaded=%s",
                         self._mode, f.x, f.y, f.w, f.h, self._loaded)
            except Exception:
                log.info("hud show mode=%s loaded=%s", self._mode, self._loaded)
        self._main(_do)

    def hide(self) -> None:
        if not self.available:
            return

        def _do():
            if not self._closed:
                self._js("window.hud.setVisible(false)")
            if self._visible:
                self._save_position()
            self._visible = False
            log.info("hud hide mode=%s", self._mode)
            # hud.css fades the page out; hide the window once that's done
            # (tick() does it), unless a show() comes in first.
            self._order_out_at = self._clock() + FADE_S
        self._main(_do)

    def close(self) -> None:
        self._closed = True
        # Discard anything still queued for a page that may never finish
        # loading now (or already has, in which case this is a no-op).
        self._pending_js = []
        if not self.available:
            return

        def _do():
            if self._visible:
                self._save_position()
            self._visible = False
            self._order_out_at = None
            self._order_out()
        self._main(_do)
