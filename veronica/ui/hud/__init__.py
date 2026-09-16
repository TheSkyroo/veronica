"""Floating HUD panel hosting the orb web view (PyObjC)."""
import json
import logging
import time
from collections.abc import Callable
from importlib.resources import files

from veronica.config import Settings

log = logging.getLogger("veronica.ui.hud")

NON_IDLE = frozenset({"listening", "thinking", "speaking", "followup", "confirming", "error", "warming"})


def _real_webview(s: Settings):
    import AppKit  # noqa: F401
    import Foundation
    import WebKit

    cfg = WebKit.WKWebViewConfiguration.alloc().init()
    web = WebKit.WKWebView.alloc().initWithFrame_configuration_(
        Foundation.NSMakeRect(0, 0, s.hud_width, s.hud_height), cfg)
    web.setValue_forKey_(False, "drawsBackground")
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
    panel.setIgnoresMouseEvents_(True)
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
                 clock: Callable[[], float] = time.monotonic, main: Callable[[Callable[[], None]], None] = _main_thread) -> None:
        self.s = settings
        self._clock = clock
        self._main = main
        self._hide_at: float | None = None
        self._closed = False
        self.available = False
        try:
            self._web = (webview_factory or _real_webview)(settings)
            self._panel = (panel_factory or _real_panel)(settings, self._web)
            self.available = True
        except Exception:
            log.warning("HUD unavailable; continuing without it", exc_info=True)
            self._web = self._panel = None

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
                AppKit.NSAnimationContext.currentContext().setDuration_(0.15)
                self._panel.animator().setAlphaValue_(alpha)
                AppKit.NSAnimationContext.endGrouping()
            except Exception:
                self._panel.setAlphaValue_(alpha)
            if then:
                then()
        self._main(_do)

    def show(self) -> None:
        def _do():
            self._web.evaluateJavaScript_completionHandler_("window.hud.setVisible(true)", None)
            self._panel.orderFrontRegardless()
            self._fade(1.0)
        self._main(_do)

    def hide(self) -> None:
        def _do():
            if not self._closed:
                self._web.evaluateJavaScript_completionHandler_("window.hud.setVisible(false)", None)
            self._fade(0.0, then=lambda: self._panel.orderOut_(None))
        self._main(_do)

    def close(self) -> None:
        self._closed = True
        if self.available:
            self.hide()
