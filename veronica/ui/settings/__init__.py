"""The Settings window: a normal titled pywebview (WebView2) window loading
`index.html`, wired to `SettingsBridge` through a js_api object.

JS → Python: settings.js calls `window.pywebview.api.handle(id, cmd, args)`;
pywebview runs that on one of its own threads, which hops onto the UI thread
(`_on_message`) so commands run in order, one at a time. `bridge.handle` answers and the reply goes back as
`window.settings.reply(id, result)`. Long commands (`LONG_COMMANDS`) run on a
worker via `bridge.run_thread` and reply when done. Python → JS:
`push_state(state)` → `window.settings.state(json)`.

The window is created on first `show()` (webview.create_window is allowed
once webview.start() is running) — or up front, hidden, with
`create_hidden()` when the app needs it as the GUI loop's first window.
Closing it only hides it, so it's reused for the app's life; `close()` (on
quit) really destroys it.

Same factory shape as `HudWindow`: the module imports without pywebview,
`window_factory` lets tests pass a fake, and JS is queued until the page has
finished loading (`_pending_js`).
"""
from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from importlib.resources import files
from pathlib import Path

from veronica.config import Settings
from veronica.ui import win32
from veronica.ui.dispatch import on_ui_thread

log = logging.getLogger("veronica.ui.settings")

TITLE = "Veronica Settings"
WIDTH, HEIGHT = 720, 520
MIN_SIZE = (560, 400)
BACKGROUND = "#080a0f"
#: Commands that may block (git fetch/pull, build): run off the UI thread.
LONG_COMMANDS = frozenset({"check_update", "update_now"})
TABS = ("general", "voice", "listening", "briefings", "brain", "history", "about")

_main_thread = on_ui_thread


def _thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="veronica-settings-cmd", daemon=True).start()


def _plain(obj):
    """A message body as the bridge's coercion expects it: plain, json-able
    dicts/lists/strs (pywebview already hands us those; mapping/sequence
    look-alikes are normalised too)."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, Sequence):
        return [_plain(v) for v in obj]
    return obj


class _SettingsApi:
    """The js_api object settings.js talks to. Only public methods are
    exposed to the page."""

    def __init__(self, owner: SettingsWindow) -> None:
        self._owner = owner

    def handle(self, id, cmd, args=None) -> None:  # noqa: A002 — the page's own field name
        body = {"id": id, "cmd": cmd, "args": args}
        self._owner._hop(lambda: self._owner._on_message(body))


def _real_window(s: Settings, owner: SettingsWindow, hidden: bool):
    import webview

    html = files("veronica.ui.settings") / "index.html"
    window = webview.create_window(
        TITLE, url=Path(str(html)).as_uri(), js_api=owner.api, width=WIDTH, height=HEIGHT,
        min_size=MIN_SIZE, background_color=BACKGROUND, hidden=hidden, text_select=True,
    )
    # pywebview fires these on its own threads (closing on the GUI thread,
    # where the handler's return value decides whether the close goes ahead).
    window.events.loaded += lambda: owner._hop(owner._on_loaded)
    window.events.closing += owner._on_closing
    window.events.closed += owner._on_closed
    return window


def _activate_window(window) -> None:
    """Bring the (just shown) window to the front, best effort."""
    hwnd = win32.hwnd_of(window, timeout=5.0)
    if hwnd is not None:
        win32.bring_to_front(hwnd)


class SettingsWindow:
    def __init__(self, settings: Settings, bridge, *, window_factory=None,
                 main: Callable[[Callable[[], None]], None] = _main_thread,
                 activate: Callable[[object], None] = _activate_window) -> None:
        self.s = settings
        self._bridge = bridge
        self._main = main
        self._activate = activate
        self._factory = window_factory or _real_window
        self._run_thread = getattr(bridge, "run_thread", None) or _thread
        #: False once creating the window has failed (no pywebview/WebView2).
        self.available = True
        self._loaded = False
        self._quitting = False
        self._pending_js: list[str] = []
        self._window = None
        self.api = _SettingsApi(self)
        try:
            bridge.on_state_changed = self.push_state
        except Exception:
            log.warning("failed to attach state listener", exc_info=True)

    # -- window lifecycle -----------------------------------------------------------
    def _ensure_window(self, hidden: bool) -> bool:
        """Create the window if it doesn't exist yet. True if it was created
        just now (shown already unless `hidden`)."""
        if self._window is not None or not self.available:
            return False
        self._loaded = False
        try:
            self._window = self._factory(self.s, self, hidden)
        except Exception:
            log.warning("settings window unavailable", exc_info=True)
            self._window = None
            self.available = False
            self._pending_js = []
            return False
        return True

    def create_hidden(self) -> None:
        """Create the window now, hidden (before webview.start(): the GUI
        loop needs at least one window to start with)."""
        self._ensure_window(hidden=True)

    def _on_closing(self):
        """The user closed the window: hide it instead (returning False
        cancels the close), unless the app is quitting."""
        if self._quitting:
            return True
        self._hop(self.hide)
        return False

    def _on_closed(self) -> None:
        def _do() -> None:
            self._window = None
            self._loaded = False
            self._pending_js = []
        self._hop(_do)

    def close(self) -> None:
        """Quit: destroy the window for good."""
        self._quitting = True

        def _do() -> None:
            window, self._window = self._window, None
            self._pending_js = []
            if window is not None:
                try:
                    window.destroy()
                except Exception:
                    log.warning("failed to destroy settings window", exc_info=True)
        self._hop(_do)

    # -- JS dispatch / page load --------------------------------------------------
    def _js(self, js: str) -> None:
        """UI thread only. Queued until index.html has finished loading."""
        if self._window is None:
            return
        if not self._loaded:
            self._pending_js.append(js)
            return
        try:
            self._window.run_js(js)
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
        """Run `fn` on the UI thread; never raises (callers include the
        bridge's worker threads, which must not die on a UI hiccup)."""
        try:
            self._main(fn)
        except Exception:
            log.warning("settings UI-thread hop failed", exc_info=True)

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
        """Runs on the UI thread with the page's `{id, cmd, args}`."""
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
            created = self._ensure_window(hidden=False)
            if self._window is None:
                return
            try:
                if not created:
                    self._window.show()
                self._activate(self._window)
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
        def _do() -> None:
            if self._window is None:
                return
            try:
                self._window.hide()
            except Exception:
                log.warning("failed to hide settings window", exc_info=True)
        self._hop(_do)
