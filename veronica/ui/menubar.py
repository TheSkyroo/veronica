import asyncio
import contextlib
import logging
import queue
import threading

import rumps

from veronica.__main__ import build_orchestrator
from veronica.config import settings
from veronica.ui import login_item
from veronica.ui.hud import HudWindow

ICONS = {"idle": "◯", "listening": "◉", "thinking": "…", "speaking": "♪", "followup": "◎", "error": "✕", "warming": "…", "confirming": "?"}


def _make_menu_handler_class():
    """Lazily build the tiny NSObject subclass used as the target for the
    fallback popup menu's items (built fresh when the rumps menu's own
    live NSMenu isn't available, e.g. under a faked rumps in tests). Each
    action method just forwards to the same Python callback the
    corresponding rumps menu bar item already uses.

    The Objective-C runtime's class registry is process-global (unlike a
    Python module namespace), so redefining a same-named class — e.g. this
    module getting reloaded, as the menu bar test fixture does per test —
    would normally raise `objc.error: ... is overriding existing
    Objective-C class`. Look the class up first and reuse it if it's
    already registered, rather than caching in Python (a plain
    functools.lru_cache wouldn't survive a module reload anyway)."""
    import objc
    from Foundation import NSObject

    with contextlib.suppress(Exception):
        return objc.lookUpClass("_VeronicaPopupMenuHandler")

    class _VeronicaPopupMenuHandler(NSObject):
        def initWithApp_(self, app):
            self = objc.super(_VeronicaPopupMenuHandler, self).init()
            if self is None:
                return None
            self._app = app
            return self

        def onMute_(self, _sender):
            self._app.toggle_mute(self._app.menu[0])

        def onToggleHud_(self, _sender):
            self._app.toggle_hud_mode(self._app._hud_mode_item)

        def onToggleLogin_(self, _sender):
            item = self._app._login_item_item
            if item.callback is not None:
                self._app.toggle_login_item(item)

        def onQuit_(self, _sender):
            self._app.quit(None)

    return _VeronicaPopupMenuHandler


class _NoopHud:
    """Stand-in for HudWindow when the HUD is disabled or unavailable, so
    _drain/quit never need to branch on whether a real HUD exists."""

    _mode = "full"

    def push(self, event: dict) -> None:
        pass

    def on_state(self, state: str) -> None:
        pass

    def tick(self) -> None:
        pass

    def show(self) -> None:
        pass

    def hide(self) -> None:
        pass

    def close(self) -> None:
        pass

    def set_mode(self, mode: str) -> None:
        pass


class VeronicaApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("V ◯", quit_button=None)
        self._state = "idle"
        self._muted = False
        self._quitting = False
        hud_mode_item = rumps.MenuItem("HUD: Full", callback=self.toggle_hud_mode)
        login_item_item = self._make_login_item()
        self.menu = [
            rumps.MenuItem("Mute", callback=self.toggle_mute), hud_mode_item, login_item_item, None,
            rumps.MenuItem("Quit", callback=self.quit),
        ]
        self._hud_mode_item = hud_mode_item
        self._login_item_item = login_item_item
        hud = HudWindow(settings) if settings.hud_enabled else None
        self._hud = hud if (hud is not None and hud.available) else _NoopHud()
        self._hud.on_menu = self._popup_menu_at
        self._popup_menu_handler = None  # strong ref for the fallback menu's target
        self._refresh_hud_mode_item()
        self._events: queue.Queue = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._timer = rumps.Timer(self._refresh, 0.25)
        self._timer.start()
        self._hud_timer = rumps.Timer(self._drain, 1 / 30)
        self._hud_timer.start()

    # asyncio side (background thread)
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            # model construction (and any first-run whisper download) happens
            # inside build_orchestrator, so surface "warming" before calling
            # it rather than only once warmup() starts.
            self._state = "warming"
            self._orch = build_orchestrator(
                settings, on_state=self._on_state, on_event=lambda k, p: self._events.put((k, p)),
                on_quit=self._schedule_quit,
            )
            self._loop.run_until_complete(self._orch.warmup())
            self._loop.run_until_complete(self._orch.run_forever())
        except Exception as e:  # surface startup failures (no mic, not logged in)
            if self._quitting:
                # loop.stop() makes run_until_complete raise "Event loop stopped
                # before Future completed" — an expected part of a clean quit,
                # not a real failure.
                return
            self._state = "error"
            self._error = str(e)
            logging.getLogger("veronica.ui").exception("menu bar background loop failed")

    def _on_state(self, state: str) -> None:
        self._state = state

    def _schedule_quit(self) -> None:
        # Called from the orchestrator's background asyncio thread (the
        # "quit" voice intent) after confirmation; quit() tears down AppKit
        # state and must run on the main thread.
        from PyObjCTools import AppHelper
        AppHelper.callAfter(lambda: self.quit(None))

    # AppKit side (main thread)
    def _refresh(self, _timer) -> None:
        if self._muted:
            self.title = "V zz"
        elif self._state == "error":
            self.title = "V ✕"
        else:
            self.title = f"V {ICONS.get(self._state, '?')}"

    def _drain(self, _timer) -> None:
        # If the backlog has grown past 1000 (the HUD/UI thread falling
        # behind the producer), drop this batch's mic level rather than
        # push a stale one: mic is a continuously-refreshed level meter,
        # so the freshest reading is always about to replace it anyway,
        # and skipping a push here is strictly cheaper than catching up.
        overflow = self._events.qsize() > 1000
        popped = []
        for _ in range(64):
            try:
                popped.append(self._events.get_nowait())
            except queue.Empty:
                break
        last_mic = None
        for kind, payload in popped:
            if kind == "mic":
                last_mic = payload
                continue
            if kind == "hud":
                mode = payload.get("mode") if isinstance(payload, dict) else None
                if mode == "hide":
                    self._hud.hide()
                elif mode in ("mini", "full"):
                    self._hud.set_mode(mode)
                    self._refresh_hud_mode_item()
                continue
            if kind == "state":
                self._hud.on_state(payload)
            self._hud.push({"kind": kind, "payload": payload})
        if last_mic is not None and not overflow:
            self._hud.push({"kind": "mic", "payload": last_mic})
        self._hud.tick()

    def _refresh_hud_mode_item(self) -> None:
        mode = getattr(self._hud, "_mode", "full")
        self._hud_mode_item.title = f"HUD: {'Mini' if mode == 'mini' else 'Full'}"

    def toggle_hud_mode(self, _item: rumps.MenuItem) -> None:
        mode = getattr(self._hud, "_mode", "full")
        self._hud.set_mode("full" if mode == "mini" else "mini")
        self._refresh_hud_mode_item()

    def _make_login_item(self) -> rumps.MenuItem:
        app_path = login_item.bundle_app_path()
        if app_path is None:
            item = rumps.MenuItem("Start at Login (build the app first)", callback=None)
            return item
        item = rumps.MenuItem("Start at Login", callback=self.toggle_login_item)
        item.state = login_item.is_enabled()
        return item

    def toggle_login_item(self, item: rumps.MenuItem) -> None:
        app_path = login_item.bundle_app_path()
        if app_path is None:
            return
        if login_item.is_enabled():
            login_item.disable()
        else:
            login_item.enable(app_path)
        item.state = login_item.is_enabled()

    def toggle_mute(self, item: rumps.MenuItem) -> None:
        self._muted = not self._muted
        item.state = self._muted
        # stop any speech; the wake loop keeps running but muting is honoured in _refresh only.
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.muted = self._muted
            if self._muted:
                orch.player.stop()

    # -- HUD orb click -> menu ---------------------------------------------
    def _build_popup_menu(self):
        """Return an NSMenu mirroring the 4 menu bar items (Mute, HUD
        Mini/Full, Start at Login, Quit). Prefers rumps' own live NSMenu
        (`self.menu._menu`, already wired and kept in sync by rumps) so the
        popup always matches the real menu bar exactly; falls back to
        building a fresh one (with its own tiny target/action handler) when
        that's unavailable."""
        live_menu = getattr(self.menu, "_menu", None)
        if live_menu is not None:
            return live_menu

        import AppKit

        handler = _make_menu_handler_class().alloc().initWithApp_(self)
        self._popup_menu_handler = handler  # AppKit doesn't retain the target

        menu = AppKit.NSMenu.alloc().init()

        mute_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Mute", "onMute:", "")
        mute_item.setTarget_(handler)
        mute_item.setState_(1 if self._muted else 0)
        menu.addItem_(mute_item)

        hud_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            self._hud_mode_item.title, "onToggleHud:", "")
        hud_item.setTarget_(handler)
        menu.addItem_(hud_item)

        login_item_ = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            self._login_item_item.title, "onToggleLogin:", "")
        login_item_.setTarget_(handler)
        login_item_.setEnabled_(self._login_item_item.callback is not None)
        login_item_.setState_(1 if getattr(self._login_item_item, "state", False) else 0)
        menu.addItem_(login_item_)

        quit_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit", "onQuit:", "")
        quit_item.setTarget_(handler)
        menu.addItem_(quit_item)

        return menu

    def _popup_menu_at(self, x: float, y: float) -> None:
        """hud.on_menu callback: show the menu at the given screen point.
        Called from the HUD panel's AppKit click handler, which — like all
        AppKit event handling — already runs on the main thread."""
        import Foundation

        menu = self._build_popup_menu()
        menu.popUpMenuPositioningItem_atLocation_inView_(None, Foundation.NSMakePoint(x, y), None)

    def quit(self, _item) -> None:
        self._quitting = True
        self._hud.close()
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.player.close()
            store = getattr(orch, "store", None)
            if store is not None:
                store.close()
        self._loop.call_soon_threadsafe(self._loop.stop)
        rumps.quit_application()


def run_app() -> None:
    VeronicaApp().run()
