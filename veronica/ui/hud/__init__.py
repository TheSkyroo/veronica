"""Floating HUD panel hosting the orb web view (PyObjC)."""
import json
import logging
import time
from collections.abc import Callable
from importlib.resources import files

from veronica import prefs
from veronica.config import Settings

log = logging.getLogger("veronica.ui.hud")

NON_IDLE = frozenset({"listening", "thinking", "speaking", "followup", "confirming", "error", "warming"})
MODES = frozenset({"full", "mini"})


def _real_webview(s: Settings):
    import AppKit
    import Foundation
    import WebKit

    cfg = WebKit.WKWebViewConfiguration.alloc().init()
    web = WebKit.WKWebView.alloc().initWithFrame_configuration_(
        Foundation.NSMakeRect(0, 0, s.hud_width, s.hud_height), cfg)
    web.setValue_forKey_(False, "drawsBackground")
    # Fill whatever size the panel's content view ends up being (mini <->
    # full resizes happen by resizing the panel; the webview tracks it).
    web.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    html = files("veronica.ui.hud") / "index.html"
    url = Foundation.NSURL.fileURLWithPath_(str(html))
    web.loadFileURL_allowingReadAccessToURL_(url, url.URLByDeletingLastPathComponent())
    return web


def _real_panel(s: Settings, web):
    import AppKit
    import Foundation

    screen = AppKit.NSScreen.mainScreen().visibleFrame()
    x = screen.origin.x + screen.size.width - s.hud_width - s.hud_margin
    y = screen.origin.y + screen.size.height - s.hud_height - s.hud_margin
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        Foundation.NSMakeRect(x, y, s.hud_width, s.hud_height), style, AppKit.NSBackingStoreBuffered, False)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setLevel_(AppKit.NSFloatingWindowLevel)
    panel.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces | AppKit.NSWindowCollectionBehaviorStationary)
    # Draggable by clicking anywhere on the (background of the) panel, while
    # staying a non-activating panel (it never steals key focus/Space).
    panel.setIgnoresMouseEvents_(False)
    panel.setMovableByWindowBackground_(True)
    panel.setHasShadow_(False)
    panel.setAlphaValue_(0.0)
    panel.setContentView_(web)
    return panel


def _main_thread(fn: Callable[[], None]) -> None:
    import Foundation
    from PyObjCTools import AppHelper

    if Foundation.NSThread.isMainThread():
        fn()
    else:
        AppHelper.callAfter(fn)


class HudWindow:
    def __init__(self, settings: Settings, *, webview_factory=None, panel_factory=None,
                 clock: Callable[[], float] = time.monotonic, main: Callable[[Callable[[], None]], None] = _main_thread,
                 prefs_load: Callable[[], dict] = prefs.load,
                 prefs_save: Callable[[dict], None] = prefs.save) -> None:
        self.s = settings
        self._clock = clock
        self._main = main
        self._prefs_load = prefs_load
        self._prefs_save = prefs_save
        self._hide_at: float | None = None
        self._closed = False
        self._fade_gen = 0
        self.available = False

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

        try:
            self._web = (webview_factory or _real_webview)(settings)
            self._panel = (panel_factory or _real_panel)(settings, self._web)
            self.available = True
        except Exception:
            log.warning("HUD unavailable; continuing without it", exc_info=True)
            self._web = self._panel = None

        # Only reposition/resize on construction if the saved state actually
        # differs from what the factories already built (full size, top
        # right) — keeps a fresh install's first launch untouched.
        if self.available and (self._mode == "mini" or self._pos[self._mode] is not None):
            self._apply_geometry()

    # -- mode / geometry --------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Switch between the full card layout and the Siri-style mini orb,
        resizing/repositioning the panel and persisting the preference."""
        if mode not in MODES:
            return
        self._mode = mode
        if self.available:
            self._apply_geometry()
        try:
            self._prefs_save({"hud_mode": self._mode})
        except Exception:
            log.warning("failed to save HUD mode pref", exc_info=True)

    def _geometry(self) -> tuple[int, int]:
        if self._mode == "mini":
            return self.s.hud_mini_width, self.s.hud_mini_height
        return self.s.hud_width, self.s.hud_height

    def _visible_frame(self):
        import AppKit
        return AppKit.NSScreen.mainScreen().visibleFrame()

    def _top_right_origin(self, w: int, h: int) -> tuple[float, float]:
        try:
            screen = self._visible_frame()
            return (
                screen.origin.x + screen.size.width - w - self.s.hud_margin,
                screen.origin.y + screen.size.height - h - self.s.hud_margin,
            )
        except Exception:
            return 0.0, 0.0

    def _top_center_origin(self, w: int, h: int) -> tuple[float, float]:
        """Mini mode's default spot: a notch/Dynamic-Island-style bar
        centered under the menu bar, rather than the full card's top-right
        corner."""
        try:
            screen = self._visible_frame()
            return (
                screen.origin.x + (screen.size.width - w) / 2,
                screen.origin.y + screen.size.height - h - 8,
            )
        except Exception:
            return 0.0, 0.0

    def _default_origin(self, w: int, h: int) -> tuple[float, float]:
        if self._mode == "mini":
            return self._top_center_origin(w, h)
        return self._top_right_origin(w, h)

    def _clamp_to_screen(self, x: float, y: float, w: int, h: int) -> tuple[float, float]:
        try:
            screen = self._visible_frame()
            max_x = screen.origin.x + screen.size.width - w
            max_y = screen.origin.y + screen.size.height - h
            x = min(max(x, screen.origin.x), max_x)
            y = min(max(y, screen.origin.y), max_y)
        except Exception:
            pass
        return x, y

    def _origin(self, w: int, h: int) -> tuple[float, float]:
        pos = self._pos[self._mode]
        if pos is not None:
            return self._clamp_to_screen(pos[0], pos[1], w, h)
        return self._default_origin(w, h)

    def _apply_geometry(self) -> None:
        w, h = self._geometry()
        x, y = self._origin(w, h)
        mode = self._mode

        def _do():
            try:
                import Foundation
                frame = Foundation.NSMakeRect(x, y, w, h)
                self._panel.setFrame_display_animate_(frame, True, True)
            except Exception:
                log.warning("failed to resize/reposition HUD panel", exc_info=True)
            self._web.evaluateJavaScript_completionHandler_(
                "window.hud.setMode(" + json.dumps(mode) + ")", None)
        self._main(_do)

    def _save_position(self) -> None:
        """Read the panel's current on-screen origin (must run on the main
        thread) and persist it, so the HUD reopens where it was dragged."""
        if not self.available:
            return
        try:
            origin = self._panel.frame().origin
            x, y = float(origin.x), float(origin.y)
        except Exception:
            return
        self._pos[self._mode] = (x, y)
        try:
            self._prefs_save({f"hud_pos_{self._mode}": [x, y]})
        except Exception:
            log.warning("failed to save HUD position pref", exc_info=True)

    # -- events ---------------------------------------------------------------
    def push(self, event: dict) -> None:
        if not self.available or self._closed:
            return
        js = "window.hud.push(" + json.dumps(event, ensure_ascii=False) + ")"
        self._main(lambda: self._web.evaluateJavaScript_completionHandler_(js, None))

    def on_state(self, state: str) -> None:
        if not self.available or self._closed:
            return
        if state in NON_IDLE:
            self._hide_at = None
            self.show()
        elif state == "idle":
            self._hide_at = self._clock() + self.s.hud_hide_after_s

    def tick(self) -> None:
        if self._hide_at is not None and self._clock() >= self._hide_at:
            self._hide_at = None
            self.hide()

    # -- visibility -----------------------------------------------------------
    def _fade(self, alpha: float, then=None) -> None:
        def _do():
            try:
                import AppKit
                AppKit.NSAnimationContext.beginGrouping()
                try:
                    AppKit.NSAnimationContext.currentContext().setDuration_(0.15)
                    if then:
                        AppKit.NSAnimationContext.currentContext().setCompletionHandler_(then)
                    self._panel.animator().setAlphaValue_(alpha)
                finally:
                    AppKit.NSAnimationContext.endGrouping()
            except Exception:
                self._panel.setAlphaValue_(alpha)
                if then:
                    then()
        self._main(_do)

    def show(self) -> None:
        self._fade_gen += 1

        def _do():
            self._web.evaluateJavaScript_completionHandler_("window.hud.setVisible(true)", None)
            self._panel.orderFrontRegardless()
            self._fade(1.0)
        self._main(_do)

    def hide(self) -> None:
        self._fade_gen += 1
        gen = self._fade_gen

        def _on_faded():
            if gen == self._fade_gen:
                self._panel.orderOut_(None)

        def _do():
            if not self._closed:
                self._web.evaluateJavaScript_completionHandler_("window.hud.setVisible(false)", None)
            self._save_position()
            self._fade(0.0, then=_on_faded)
        self._main(_do)

    def close(self) -> None:
        self._closed = True
        if self.available:
            self.hide()
