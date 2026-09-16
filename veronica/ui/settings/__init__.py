"""The Settings window: a normal titled NSWindow hosting a WKWebView that
loads `index.html`, wired to `SettingsBridge` with a script-message bridge.

JS → Python: `window.webkit.messageHandlers.veronica.postMessage({id, cmd,
args})` lands in `_on_message` (main thread), which calls `bridge.handle`
and answers with `window.settings.reply(id, result)`. Long commands
(`LONG_COMMANDS`) run on a worker via `bridge.run_thread` and reply when
done. Python → JS: `push_state(state)` → `window.settings.state(json)`.

Same lazy-PyObjC/factory shape as `HudWindow`: the module imports without
AppKit/WebKit, `webview_factory`/`window_factory` let tests pass fakes, and
JS is queued until the page has finished loading (`_pending_js`).
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from importlib.resources import files

from veronica.config import Settings
from veronica.ui.hud import _main_thread

log = logging.getLogger("veronica.ui.settings")

TITLE = "Veronica Settings"
WIDTH, HEIGHT = 720, 520
MESSAGE_HANDLER = "veronica"
#: Commands that may block (git fetch/pull, build): run off the main thread.
LONG_COMMANDS = frozenset({"check_update", "update_now"})
TABS = ("general", "voice", "listening", "briefings", "brain", "history", "about")


def _thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="veronica-settings-cmd", daemon=True).start()


def _plain(obj):
    """NSDictionary/NSArray/NSString from a WKScriptMessage → plain Python
    (json-able, and what the bridge's coercion expects)."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, str):
        return str(obj)   # objc.pyobjc_unicode → plain str
    if isinstance(obj, Sequence):
        return [_plain(v) for v in obj]
    return obj


def _make_message_handler_class():
    """Lazily build the WKScriptMessageHandler PyObjC class (module-level
    import of Foundation would break importing this file without PyObjC)."""
    import objc
    from Foundation import NSObject

    class _SettingsMessageHandler(NSObject):
        def initWithWindow_(self, window):
            self = objc.super(_SettingsMessageHandler, self).init()
            if self is None:
                return None
            self._window = window
            return self

        def userContentController_didReceiveScriptMessage_(self, ucc, message):
            try:
                body = message.body()
            except Exception:
                log.warning("bad script message", exc_info=True)
                return
            self._window._on_message(body)

    return _SettingsMessageHandler


def _make_nav_delegate_class():
    import objc
    from Foundation import NSObject

    class _SettingsNavDelegate(NSObject):
        def initWithWindow_(self, window):
            self = objc.super(_SettingsNavDelegate, self).init()
            if self is None:
                return None
            self._window = window
            return self

        def webView_didFinishNavigation_(self, webView, nav):
            self._window._on_loaded()

    return _SettingsNavDelegate


def _make_window_delegate_class():
    import objc
    from Foundation import NSObject

    class _SettingsWindowDelegate(NSObject):
        def initWithWindow_(self, window):
            self = objc.super(_SettingsWindowDelegate, self).init()
            if self is None:
                return None
            self._window = window
            return self

        def windowShouldClose_(self, sender):
            # Hide, never destroy: the window is reused for the app's life.
            self._window.hide()
            return False

    return _SettingsWindowDelegate


def _real_webview(s: Settings, owner: SettingsWindow):
    import AppKit
    import Foundation
    import WebKit

    cfg = WebKit.WKWebViewConfiguration.alloc().init()
    handler_cls = _make_message_handler_class()
    owner._handler = handler_cls.alloc().initWithWindow_(owner)
    cfg.userContentController().addScriptMessageHandler_name_(owner._handler, MESSAGE_HANDLER)
    web = WebKit.WKWebView.alloc().initWithFrame_configuration_(
        Foundation.NSMakeRect(0, 0, WIDTH, HEIGHT), cfg)
    web.setValue_forKey_(False, "drawsBackground")
    web.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
    nav_cls = _make_nav_delegate_class()
    owner._nav_delegate = nav_cls.alloc().initWithWindow_(owner)
    web.setNavigationDelegate_(owner._nav_delegate)
    html = files("veronica.ui.settings") / "index.html"
    url = Foundation.NSURL.fileURLWithPath_(str(html))
    web.loadFileURL_allowingReadAccessToURL_(url, url.URLByDeletingLastPathComponent())
    return web


def _real_window(s: Settings, web, owner: SettingsWindow):
    import AppKit
    import Foundation

    style = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
             | AppKit.NSWindowStyleMaskMiniaturizable | AppKit.NSWindowStyleMaskResizable)
    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        Foundation.NSMakeRect(0, 0, WIDTH, HEIGHT), style, AppKit.NSBackingStoreBuffered, False)
    win.setTitle_(TITLE)
    win.setReleasedWhenClosed_(False)
    win.setLevel_(AppKit.NSNormalWindowLevel)
    win.setMinSize_(Foundation.NSMakeSize(560, 400))
    win.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.03, 0.04, 0.06, 1.0))
    try:
        win.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
    except Exception:  # cosmetic: the page is dark regardless
        log.debug("dark appearance unavailable", exc_info=True)
    win.setContentView_(web)
    delegate_cls = _make_window_delegate_class()
    owner._win_delegate = delegate_cls.alloc().initWithWindow_(owner)   # NSWindow.delegate is weak
    win.setDelegate_(owner._win_delegate)
    win.center()
    return win


def _activate_app() -> None:
    import AppKit

    app = AppKit.NSApp
    if app is not None:
        app.activateIgnoringOtherApps_(True)


class SettingsWindow:
    def __init__(self, settings: Settings, bridge, *, webview_factory=None, window_factory=None,
                 main: Callable[[Callable[[], None]], None] = _main_thread,
                 activate: Callable[[], None] = _activate_app) -> None:
        self.s = settings
        self._bridge = bridge
        self._main = main
        self._activate = activate
        self._run_thread = getattr(bridge, "run_thread", None) or _thread
        self.available = False
        self._loaded = False
        self._pending_js: list[str] = []
        # Strong refs for the PyObjC delegates/handlers (AppKit holds them weakly).
        self._handler = self._nav_delegate = self._win_delegate = None
        self._web = self._window = None
        try:
            self._web = (webview_factory or _real_webview)(settings, self)
            self._window = (window_factory or _real_window)(settings, self._web, self)
            self.available = True
        except Exception:
            log.warning("settings window unavailable", exc_info=True)
            self._web = self._window = None
        try:
            bridge.on_state_changed = self.push_state
        except Exception:
            log.warning("failed to attach state listener", exc_info=True)

    # -- JS dispatch / page load --------------------------------------------------
    def _js(self, js: str) -> None:
        """Main thread only. Queued until index.html has finished loading."""
        if not self.available:
            return
        if not self._loaded:
            self._pending_js.append(js)
            return
        try:
            self._web.evaluateJavaScript_completionHandler_(js, None)
        except Exception:
            log.warning("settings JS eval failed", exc_info=True)

    def mark_loaded(self) -> None:
        self._on_loaded()

    def _on_loaded(self) -> None:
        self._loaded = True
        pending, self._pending_js = self._pending_js, []
        for js in pending:
            self._js(js)

    def _hop(self, fn: Callable[[], None]) -> None:
        """Run `fn` on the main thread; never raises (callers include the
        bridge's worker threads, which must not die on a UI hiccup)."""
        try:
            self._main(fn)
        except Exception:
            log.warning("settings main-thread hop failed", exc_info=True)

    # -- Python → JS ------------------------------------------------------------------
    def push_state(self, state: dict) -> None:
        if not self.available:
            return
        try:
            js = "window.settings.state(" + json.dumps(state, ensure_ascii=False) + ")"
        except (TypeError, ValueError):
            log.warning("settings state not serialisable", exc_info=True)
            return
        self._hop(lambda: self._js(js))

    def _reply(self, mid, result: dict) -> None:
        try:
            payload = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            log.warning("settings reply not serialisable", exc_info=True)
            payload = json.dumps({"ok": False, "message": "internal error"})
        self._js("window.settings.reply(" + json.dumps(mid) + ", " + payload + ")")

    # -- JS → Python ------------------------------------------------------------------
    def _on_message(self, body) -> None:
        """Called by the script-message handler on the main thread with the
        posted `{id, cmd, args}`."""
        if not isinstance(body, Mapping):
            log.warning("ignoring non-dict settings message: %r", body)
            return
        body = _plain(body)
        mid, cmd = body.get("id"), body.get("cmd")
        if mid is None or not isinstance(cmd, str) or not cmd:
            log.warning("ignoring malformed settings message: %r", body)
            return
        args = body.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        def run() -> dict:
            try:
                return self._bridge.handle(cmd, args)
            except Exception as e:  # the bridge already catches; belt and braces
                log.exception("settings command %s failed", cmd)
                return {"ok": False, "message": str(e) or e.__class__.__name__}

        if cmd in LONG_COMMANDS:
            def work() -> None:
                result = run()
                self._hop(lambda: self._reply(mid, result))
            self._run_thread(work)
            return
        self._reply(mid, run())

    # -- visibility ---------------------------------------------------------------------
    def show(self, tab: str = "general") -> None:
        if not self.available:
            return
        tab = tab if tab in TABS else "general"

        def _do() -> None:
            try:
                self._window.makeKeyAndOrderFront_(None)
                self._activate()
            except Exception:
                log.warning("failed to show settings window", exc_info=True)
            self._js("window.settings.select(" + json.dumps(tab) + ")")
            try:
                state = self._bridge.get_state()
            except Exception:
                log.exception("get_state failed")
                return
            self._js("window.settings.state(" + json.dumps(state, ensure_ascii=False) + ")")
        self._hop(_do)

    def hide(self) -> None:
        if not self.available:
            return

        def _do() -> None:
            try:
                self._window.orderOut_(None)
            except Exception:
                log.warning("failed to hide settings window", exc_info=True)
        self._hop(_do)
