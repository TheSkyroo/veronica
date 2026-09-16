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


def _webview_class():
    """Lazily build the draggable WKWebView subclass (imports WebKit here,
    not at module scope, so this file still imports — and its unit tests
    still run — on machines without PyObjC/WebKit installed).

    WKWebView returns NO for mouseDownCanMoveWindow by default, so the
    panel's setMovableByWindowBackground_(True) never fires for a click
    that lands on the web view — i.e. almost the entire panel. Override it
    (and acceptsFirstMouse_, so the very first click on this non-activating
    panel starts a drag instead of just activating/focusing it) to make the
    HUD draggable through the web view."""
    import objc
    import WebKit

    class _DraggableWebView(WebKit.WKWebView):
        # Set by HudWindow after construction: Callable[[object], None] | None,
        # called with the triggering NSEvent on a plain click (mouseUp with no
        # drag in between) or a right-click anywhere on the panel.
        on_menu = None

        def mouseDownCanMoveWindow(self):
            return True

        def acceptsFirstMouse_(self, event):
            return True

        def mouseDown_(self, event):
            # Remember the panel's on-screen origin at mouseDown so mouseUp_
            # can tell a plain click (origin unchanged) from the end of a
            # window drag (setMovableByWindowBackground_ moved it).
            try:
                origin = self.window().frame().origin
                self._menu_click_origin = (origin.x, origin.y)
            except Exception:
                self._menu_click_origin = None
            objc.super(_DraggableWebView, self).mouseDown_(event)

        def mouseUp_(self, event):
            objc.super(_DraggableWebView, self).mouseUp_(event)
            origin_before = getattr(self, "_menu_click_origin", None)
            if origin_before is None or self.on_menu is None:
                return
            try:
                origin = self.window().frame().origin
                dragged = (origin.x, origin.y) != origin_before
            except Exception:
                dragged = True
            if not dragged:
                self.on_menu(event)

        def rightMouseDown_(self, event):
            objc.super(_DraggableWebView, self).rightMouseDown_(event)
            if self.on_menu is not None:
                self.on_menu(event)

    return _DraggableWebView


def _real_webview(s: Settings):
    import AppKit
    import Foundation
    import WebKit

    cfg = WebKit.WKWebViewConfiguration.alloc().init()
    web = _webview_class().alloc().initWithFrame_configuration_(
        Foundation.NSMakeRect(0, 0, s.hud_width, s.hud_height), cfg)
    web.setValue_forKey_(False, "drawsBackground")
    # Fill whatever size the panel's content view ends up being (mini <->
    # full resizes happen by resizing the panel; the webview tracks it).
    web.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    html = files("veronica.ui.hud") / "index.html"
    url = Foundation.NSURL.fileURLWithPath_(str(html))
    web.loadFileURL_allowingReadAccessToURL_(url, url.URLByDeletingLastPathComponent())
    return web


def _make_nav_delegate_class():
    """Lazily build the WKNavigationDelegate PyObjC class. Defined here (in
    the module) rather than at import time so this file can still be
    imported — and its tests run — on machines without PyObjC installed."""
    import objc
    from Foundation import NSObject

    class _HudNavDelegate(NSObject):
        def initWithHudWindow_(self, hud_window):
            self = objc.super(_HudNavDelegate, self).init()
            if self is None:
                return None
            self._hud_window = hud_window
            return self

        def webView_didFinishNavigation_(self, webView, nav):
            self._hud_window._on_loaded()

    return _HudNavDelegate


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
        self._loaded = False
        self._pending_js: list[str] = []
        self._nav_delegate = None
        # Set by the menu bar app (e.g. `hud.on_menu = self._popup_menu_at`);
        # called with (x, y) screen coordinates when the orb is clicked.
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
            self._web = (webview_factory or _real_webview)(settings)
            self._panel = (panel_factory or _real_panel)(settings, self._web)
            self.available = True
        except Exception:
            log.warning("HUD unavailable; continuing without it", exc_info=True)
            self._web = self._panel = None

        if self.available and is_real_webview:
            try:
                delegate_cls = _make_nav_delegate_class()
                self._nav_delegate = delegate_cls.alloc().initWithHudWindow_(self)
                self._web.setNavigationDelegate_(self._nav_delegate)
            except Exception:
                log.warning("failed to set HUD navigation delegate", exc_info=True)

        if self.available:
            try:
                self._web.on_menu = self._on_webview_menu
            except Exception:
                log.warning("failed to wire HUD menu click handler", exc_info=True)

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

    # -- JS dispatch / page load -------------------------------------------------
    def _js(self, js: str) -> None:
        """Route a JS call through the load gate: queued until the page has
        actually finished loading (index.html), otherwise it's lost — e.g. a
        setMode('mini')/push() fired right after construction, before WebKit
        finishes navigation."""
        if not self._loaded:
            self._pending_js.append(js)
            return
        self._web.evaluateJavaScript_completionHandler_(js, None)

    def mark_loaded(self) -> None:
        """Public hook for the navigation delegate (and for tests, whose fake
        web views never fire a real navigation callback) to signal that
        index.html has finished loading."""
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
        self._web.evaluateJavaScript_completionHandler_(
            "window.hud.setMode(" + json.dumps(self._mode) + ")", None)
        pending, self._pending_js = self._pending_js, []
        for js in pending:
            self._web.evaluateJavaScript_completionHandler_(js, None)

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
            self._js("window.hud.setMode(" + json.dumps(mode) + ")")
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

    # -- menu -------------------------------------------------------------------
    def _on_webview_menu(self, event) -> None:
        """Called by the draggable web view (already on the main thread —
        AppKit event handlers always are) on a plain click or a right-click
        anywhere on the panel: resolve the screen point and hand off to
        _popup_menu."""
        try:
            import AppKit
            point = AppKit.NSEvent.mouseLocation()
            x, y = float(point.x), float(point.y)
        except Exception:
            log.warning("failed to resolve HUD menu click location", exc_info=True)
            return
        self._popup_menu(x, y)

    def _popup_menu(self, x: float, y: float) -> None:
        if self.on_menu is not None:
            self.on_menu(x, y)

    # -- events ---------------------------------------------------------------
    def push(self, event: dict) -> None:
        if not self.available or self._closed:
            return
        js = "window.hud.push(" + json.dumps(event, ensure_ascii=False) + ")"
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
            self._js("window.hud.setVisible(true)")
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
                self._js("window.hud.setVisible(false)")
            self._save_position()
            self._fade(0.0, then=_on_faded)
        self._main(_do)

    def close(self) -> None:
        self._closed = True
        # Discard anything still queued for a page that may never finish
        # loading now (or already has, in which case this is a no-op).
        self._pending_js = []
        if self.available:
            self.hide()
