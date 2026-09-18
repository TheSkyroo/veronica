import asyncio
import contextlib
import logging
import queue
import subprocess
import threading

import rumps

from veronica import updater, version
from veronica.__main__ import build_orchestrator
from veronica.audio.hotkey import HotkeyMonitor
from veronica.config import settings
from veronica.speech import voices
from veronica.ui import login_item
from veronica.ui.hud import HudWindow
from veronica.ui.relaunch import relaunch
from veronica.ui.settings import SettingsWindow, _main_thread
from veronica.ui.settings.bridge import UPDATE_FAILED, SettingsBridge
from veronica.updater import UpdateInProgress

log = logging.getLogger("veronica.ui")

ACCESSIBILITY_PANE_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"

# Self-update (D3): the "Check for Updates…" item runs a check on a thread;
# the item below it reflects the last result and, when something newer
# exists, installs it. The same check runs silently once an hour.
UPDATE_CHECK_INTERVAL_S = 3600
UPDATE_UNCHECKED_TITLE = "Updates: not checked yet"
UPDATE_LATEST_TITLE = "Up to date"
UPDATE_AVAILABLE_TITLE = "Update available — Restart to update"
UPDATE_UPDATING_TITLE = "Updating…"
UPDATE_FAILED_TITLE = "Update failed — check the log"
UPDATE_CHECK_FAILED_TITLE = "Couldn't check for updates"
UPDATE_READY_SPOKEN = "An update is ready. Say update yourself, or use the Settings window."
UPDATE_LATEST_SPOKEN = "You're already on the latest."
UPDATE_CHECK_FAILED_SPOKEN = "Couldn't check for updates, check the log."

ICONS = {"idle": "◯", "listening": "◉", "thinking": "…", "speaking": "♪", "followup": "◎", "error": "✕", "warming": "…", "confirming": "?"}

# Voice submenu speed entries: menu title -> Orchestrator._voice_turn("speed", arg).
SPEED_TITLES = {"Faster": "faster", "Slower": "slower", "Normal speed": "normal"}


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
            self._app.toggle_mute(self._app._mute_item)

        def onSettings_(self, _sender):
            self._app.open_settings(None)

        def onToggleHud_(self, _sender):
            self._app.toggle_hud_mode(self._app._hud_mode_item)

        def onToggleLogin_(self, _sender):
            item = self._app._login_item_item
            if item.callback is not None:
                self._app.toggle_login_item(item)

        # The popup's voice/speed items carry the rumps item's title as
        # their representedObject, so the handler forwards to the exact
        # same rumps MenuItem (and callback) the menu bar uses.
        def onPickVoice_(self, sender):
            self._app._pick_voice(self._app._voice_items[sender.representedObject()])

        def onSpeed_(self, sender):
            self._app._speed(self._app._speed_items[sender.representedObject()])

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

    def reset_position(self) -> None:
        pass


class VeronicaApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("V ◯", quit_button=None)
        self._state = "idle"
        self._muted = False
        self._quitting = False
        self._orch = None
        # -- settings window + self-update (Batch D) --------------------------
        # The bridge is pure Python and needs the orchestrator only at call
        # time (get_orch, and the store callable), so both it and the window
        # can be built now, before the background thread has built the
        # orchestrator (and with it the MemoryStore).
        self._bundle_path = login_item.bundle_app_path()
        self._repo = version.REPO
        self._build_info = version.build_info()
        self._bridge = SettingsBridge(
            settings=settings, get_orch=lambda: self._orch, store=lambda: getattr(self._orch, "store", None),
            run_on_loop=self._schedule, relaunch=self._relaunch, bundle_path=self._bundle_path,
            repo=self._repo, marshal=_main_thread,
        )
        self._settings = SettingsWindow(settings, self._bridge)
        # The window installs its own state listener; chain ours in front so
        # the update item also tracks checks/updates started from the window.
        self._window_on_state = getattr(self._bridge, "on_state_changed", None)
        self._bridge.on_state_changed = self._on_bridge_state
        self._checking_update = False
        self._about_title = f"About Veronica — {version.describe(self._build_info)}"
        about_item = rumps.MenuItem(self._about_title, callback=None)
        settings_item = rumps.MenuItem("Settings…", callback=self.open_settings)
        check_item = rumps.MenuItem("Check for Updates…", callback=self.check_for_updates)
        self._update_item = rumps.MenuItem(UPDATE_UNCHECKED_TITLE, callback=None)
        self._mute_item = rumps.MenuItem("Mute", callback=self.toggle_mute)
        hud_mode_item = rumps.MenuItem("HUD: Full", callback=self.toggle_hud_mode)
        login_item_item = self._make_login_item()
        self._voice_items: dict[str, rumps.MenuItem] = {}
        self._speed_items: dict[str, rumps.MenuItem] = {}
        voice_menu = rumps.MenuItem("Voice")
        # English voices, a separator, then the Hindi voices (picking one
        # sets the voice Hindi replies use; the English voice is untouched).
        for vid in voices.VOICE_IDS:
            name = voices.display_name(vid)
            item = rumps.MenuItem(name, callback=self._pick_voice)
            self._voice_items[name] = item
            voice_menu.add(item)
        voice_menu.add(None)
        for vid in voices.HINDI_VOICE_IDS:
            name = voices.display_name(vid)
            item = rumps.MenuItem(name, callback=self._pick_voice)
            self._voice_items[name] = item
            voice_menu.add(item)
        voice_menu.add(None)
        for title in SPEED_TITLES:
            item = rumps.MenuItem(title, callback=self._speed)
            self._speed_items[title] = item
            voice_menu.add(item)
        self._voice_menu = voice_menu
        menu_items = [
            about_item, settings_item, check_item, self._update_item, None,
            self._mute_item, hud_mode_item, voice_menu, login_item_item, None,
        ]
        self._hud_mode_item = hud_mode_item
        self._login_item_item = login_item_item
        hud = HudWindow(settings) if settings.hud_enabled else None
        self._hud = hud if (hud is not None and hud.available) else _NoopHud()
        self._hud.on_menu = self._popup_menu_at
        self._popup_menu_handler = None  # strong ref for the fallback menu's target
        self._events: queue.Queue = queue.Queue()
        self._loop = asyncio.new_event_loop()

        # push-to-talk (A2): the monitor needs `self._loop` (the background
        # orchestrator loop, not yet running) to marshal its callbacks onto,
        # since it's started here on the AppKit main thread.
        self._hotkey: HotkeyMonitor | None = None
        self._ptt_item: rumps.MenuItem | None = None
        if settings.ptt_enabled:
            self._hotkey = HotkeyMonitor(self._on_ptt_press, self._on_ptt_release, keycode=settings.ptt_keycode)
            self._hotkey.start(loop=self._loop)
            if not self._hotkey.available:
                self._ptt_item = rumps.MenuItem(
                    "Enable Push-to-talk… (Input Monitoring)", callback=self.open_accessibility_settings
                )
                menu_items.append(self._ptt_item)

        menu_items.append(rumps.MenuItem("Quit", callback=self.quit))
        self.menu = menu_items
        self._refresh_hud_mode_item()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._timer = rumps.Timer(self._refresh, 0.25)
        self._timer.start()
        self._hud_timer = rumps.Timer(self._drain, 1 / 30)
        self._hud_timer.start()
        self._update_timer = rumps.Timer(self._hourly_update_check, UPDATE_CHECK_INTERVAL_S)
        self._update_timer.start()

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
                updater_check=lambda: updater.check(self._repo, info=self._build_info),
                updater_update=self._voice_update,
                relaunch=self._relaunch,
                can_relaunch=lambda: self._bundle_path is not None,
                version_describe=lambda: version.describe(self._build_info),
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
        self._refresh_voice_menu()

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
                elif mode == "reset":
                    self._hud.reset_position()
                continue
            if kind == "settings":
                # "open settings" / "show history" voice intents: the window
                # is AppKit, and _drain already runs on the main thread.
                tab = payload.get("tab") if isinstance(payload, dict) else None
                self._settings.show(tab if isinstance(tab, str) else "general")
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

    # -- settings window / self-update (Batch D) -----------------------------------
    def open_settings(self, _item=None) -> None:
        self._settings.show("general")

    def _relaunch(self) -> bool:
        """Restart the app after an update (bridge "Update & restart" /
        "Restart", or the orchestrator's "update yourself" turn). May be
        called from any thread: quitting is marshalled to the main thread."""
        return relaunch(self._bundle_path, self._schedule_quit)

    def _voice_update(self, status) -> str:
        """The orchestrator's `updater_update` hook ("update yourself"): the
        bridge owns the single update slot, so claim it first — a second
        spoken update, or one while the window/menu is already updating,
        raises UpdateInProgress instead of running two builds on dist/."""
        if not self._bridge.begin_update():
            raise UpdateInProgress("Updating already.")
        try:
            out = updater.update(self._repo, status)
        except Exception:
            self._bridge.end_update(UPDATE_FAILED)
            raise
        self._bridge.end_update()
        return out

    def _notify(self, subtitle: str, text: str, *, spoken: str | None = None) -> None:
        """Post a notification; when the notification center isn't
        available (rumps raises RuntimeError without a CFBundleIdentifier —
        the launcher execs the venv python, so NSBundle.mainBundle() is
        .venv/bin), log it and have Veronica say it when she's next idle."""
        try:
            rumps.notification("Veronica", subtitle, text)
            return
        except RuntimeError as e:
            log.info("notification unavailable (%s): %s — %s", e, subtitle, text)
        orch = getattr(self, "_orch", None)
        if orch is not None:
            self._schedule(orch.announce(spoken or text))

    def _set_update_item(self, title: str, installable: bool = False) -> None:
        self._update_item.title = title
        self._update_item.set_callback(self.update_now if installable else None)

    def _run_update_check(self, *, notify: bool) -> None:
        """Check for updates on the bridge's worker thread (a git fetch can
        take seconds) and reflect the result in the update item; with
        `notify`, also post a notification with the outcome."""
        if self._checking_update:
            return
        self._checking_update = True

        def work() -> None:
            try:
                res = self._bridge.check_update()
            except Exception as e:  # noqa: BLE001 — check_update already catches; belt and braces
                res = {"ok": False, "message": str(e)}

            def apply() -> None:
                self._checking_update = False
                if not res.get("ok"):
                    self._set_update_item(UPDATE_CHECK_FAILED_TITLE)
                    if notify:
                        self._notify("Couldn't check for updates", res.get("message") or "",
                                     spoken=UPDATE_CHECK_FAILED_SPOKEN)
                    return
                available = bool(res.get("available"))
                self._set_update_item(UPDATE_AVAILABLE_TITLE if available else UPDATE_LATEST_TITLE, available)
                if notify:
                    self._notify("Update available" if available else "Up to date", res.get("detail") or "",
                                 spoken=UPDATE_READY_SPOKEN if available else UPDATE_LATEST_SPOKEN)

            _main_thread(apply)

        self._bridge.run_thread(work)

    def check_for_updates(self, _item=None) -> None:
        self._run_update_check(notify=True)

    def _hourly_update_check(self, _timer) -> None:
        self._run_update_check(notify=False)

    def update_now(self, _item=None) -> None:
        """The "Update available — Restart to update" item: pull/build on a
        thread via the bridge, which relaunches on success (or reports the
        failure through the state push handled in _on_bridge_state)."""
        res = self._bridge.update_now()
        if not res.get("ok"):
            self._notify("", res.get("message") or "")
            return
        self._set_update_item(UPDATE_UPDATING_TITLE)

    def _on_bridge_state(self, state: dict) -> None:
        """Bridge state listener (already marshalled to the main thread):
        keep the update item in step with checks/updates started anywhere —
        the menu, the hourly timer, or the settings window's About tab —
        then hand the state to the window."""
        about = state.get("about") if isinstance(state, dict) else None
        about = about if isinstance(about, dict) else {}
        update = about.get("update")
        if about.get("updating"):
            self._set_update_item(UPDATE_UPDATING_TITLE)
        elif isinstance(update, dict) and (update.get("detail") or update.get("available")):
            if update.get("detail") == UPDATE_FAILED:
                self._set_update_item(UPDATE_FAILED_TITLE)
            elif update.get("available"):
                self._set_update_item(UPDATE_AVAILABLE_TITLE, True)
            else:
                self._set_update_item(UPDATE_LATEST_TITLE)
        if self._window_on_state is not None:
            self._window_on_state(state)

    # -- push-to-talk (A2) --------------------------------------------------------
    def _on_ptt_press(self) -> None:
        # Called via HotkeyMonitor's call_soon_threadsafe, so this already
        # runs on the orchestrator's background loop thread. ptt_start()/
        # ptt_end() are synchronous signals (they set an asyncio.Event /
        # finish the in-flight capture); the push-to-talk turn itself is
        # run by the orchestrator's own run_forever loop, so there's no
        # fire-and-forget task to keep a reference to here.
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.ptt_start()

    def _on_ptt_release(self) -> None:
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.ptt_end()

    def open_accessibility_settings(self, _item: rumps.MenuItem) -> None:
        subprocess.run(["open", ACCESSIBILITY_PANE_URL])

    def toggle_mute(self, item: rumps.MenuItem) -> None:
        self._muted = not self._muted
        item.state = self._muted
        # stop any speech; the wake loop keeps running but muting is honoured in _refresh only.
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.muted = self._muted
            if self._muted:
                orch.player.stop()

    # -- Voice submenu ------------------------------------------------------------
    def _schedule(self, coro) -> None:
        """Run `coro` on the orchestrator's background loop from the AppKit
        main thread (or, under test with a loop that isn't running yet,
        queue it as a task for the next run_until_complete)."""
        loop = getattr(self, "_loop", None)
        if loop is None:
            coro.close()
            return
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            loop.create_task(coro)

    def _voice_action(self, action: tuple[str, str]) -> None:
        orch = getattr(self, "_orch", None)
        if orch is None:
            return

        async def _turn():
            # The orchestrator's own dispatch resets the player before
            # _voice_turn (it doesn't do so itself), so mirror that here:
            # cut off any in-flight speech and confirm in the new voice.
            orch.player.reset()
            await orch._voice_turn(action)

        self._schedule(_turn())

    def _pick_voice(self, item: rumps.MenuItem) -> None:
        self._voice_action(("voice", item.title.lower()))
        self._refresh_voice_menu()

    def _speed(self, item: rumps.MenuItem) -> None:
        self._voice_action(("speed", SPEED_TITLES[item.title]))

    def _refresh_voice_menu(self) -> None:
        # Runs on the 0.25 s _refresh timer too, so it must stay cheap and
        # tolerate no orchestrator (startup) or no tts on it.
        orch = getattr(self, "_orch", None)
        tts = getattr(orch, "tts", None)
        checked = {
            voices.display_name(v)
            for v in (getattr(tts, "voice", None), getattr(tts, "hindi_voice", None)) if v
        }
        for name, item in self._voice_items.items():
            item.state = 1 if name in checked else 0

    # -- HUD orb click -> menu ---------------------------------------------
    def _build_popup_menu(self):
        """Return an NSMenu mirroring the menu bar items (About, Settings…,
        Mute, HUD Mini/Full, Voice submenu, Start at Login, Quit). Prefers rumps' own live NSMenu
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

        about_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(self._about_title, None, "")
        about_item.setEnabled_(False)
        menu.addItem_(about_item)
        settings_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Settings…", "onSettings:", "")
        settings_item.setTarget_(handler)
        menu.addItem_(settings_item)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())

        mute_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Mute", "onMute:", "")
        mute_item.setTarget_(handler)
        mute_item.setState_(1 if self._muted else 0)
        menu.addItem_(mute_item)

        hud_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            self._hud_mode_item.title, "onToggleHud:", "")
        hud_item.setTarget_(handler)
        menu.addItem_(hud_item)

        self._refresh_voice_menu()
        voice_menu = AppKit.NSMenu.alloc().initWithTitle_("Voice")
        english = {voices.display_name(v) for v in voices.VOICE_IDS}
        hindi = {voices.display_name(v) for v in voices.HINDI_VOICE_IDS}
        for group in (english, hindi):
            for name, rumps_item in self._voice_items.items():
                if name not in group:
                    continue
                voice_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(name, "onPickVoice:", "")
                voice_item.setTarget_(handler)
                voice_item.setRepresentedObject_(name)
                voice_item.setState_(1 if rumps_item.state else 0)
                voice_menu.addItem_(voice_item)
            voice_menu.addItem_(AppKit.NSMenuItem.separatorItem())
        for title in self._speed_items:
            speed_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "onSpeed:", "")
            speed_item.setTarget_(handler)
            speed_item.setRepresentedObject_(title)
            voice_menu.addItem_(speed_item)
        voice_parent = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Voice", None, "")
        voice_parent.setSubmenu_(voice_menu)
        menu.addItem_(voice_parent)

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
        self._settings.hide()
        if self._hotkey is not None:
            self._hotkey.stop()
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
