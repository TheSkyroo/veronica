"""Veronica's desktop shell on Windows: a system tray icon (pystray) with the
app menu, the floating HUD and the Settings window (pywebview/WebView2), and
the orchestrator running on its own asyncio loop.

Threads:

- main: `webview.start()` — pywebview's GUI loop must own the main thread;
  it returns once every window has been destroyed (Quit).
- UI thread (veronica.ui.dispatch): all window/menu state changes, plus the
  timers — the 0.25 s refresh (tray icon/tooltip, submenus), the 30 Hz drain
  of orchestrator events into the HUD and the hourly update check.
  Everything below that touches UI state runs there, and callbacks from
  other threads hop onto it (`_main_thread`).
- tray: pystray's own message loop (`Icon.run` on a daemon thread). Menu
  clicks arrive there and are hopped onto the UI thread.
- orchestrator: a daemon thread running the asyncio loop (`_run_loop`).

Menu items are kept as small `MenuItem` records (title, callback, check
state); the pystray menu reads them through callables, and
the refresh timer calls `Icon.update_menu()` whenever one changes. A click
on the HUD orb pops up the same tray menu at the cursor (`_popup_menu_at`,
which asks pystray's own window to show it, so there's exactly one menu).
"""
import asyncio
import logging
import queue
import threading
import time

from veronica import updater, version
from veronica.__main__ import build_orchestrator
from veronica.audio.hotkey import HotkeyMonitor
from veronica.brain.backends import BACKENDS, check_backend
from veronica.config import settings
from veronica.speech import voices
from veronica.ui import dispatch, login_item, win32
from veronica.ui.hud import HudWindow
from veronica.ui.icon import state_image
from veronica.ui.relaunch import relaunch
from veronica.ui.settings import SettingsWindow
from veronica.ui.settings.bridge import UPDATE_FAILED, SettingsBridge
from veronica.updater import UpdateInProgress

log = logging.getLogger("veronica.ui")

# Seams for tests: marshal onto / schedule on the UI thread.
_main_thread = dispatch.on_ui_thread
_every = dispatch.every

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

PTT_UNAVAILABLE_TITLE = "Push-to-talk unavailable"
LOGIN_ITEM_TITLE = "Start at Login"
LOGIN_ITEM_BUILD_FIRST_TITLE = "Start at Login (build the app first)"

# Tray tooltip per state.
STATE_LABELS = {
    "idle": "Idle", "listening": "Listening", "thinking": "Thinking", "speaking": "Speaking",
    "followup": "Listening for a follow-up", "error": "Error", "warming": "Starting up",
    "confirming": "Waiting for your answer",
}

# Voice submenu speed entries: menu title -> Orchestrator._voice_turn("speed", arg).
SPEED_TITLES = {"Faster": "faster", "Slower": "slower", "Normal speed": "normal"}

# Brain submenu: how often the "(not installed)" / "(not logged in)" titles
# re-check the backends (PATH + marker files, cheap but not free on the
# 0.25 s refresh timer).
BRAIN_CHECK_INTERVAL_S = 60

# pystray's Win32 backend: the tray window's notification message, and the
# mouse message it carries for "show the menu" (see _popup_menu_at).
PYSTRAY_WM_NOTIFY = 0x0400 + 11   # pystray._util.win32.WM_NOTIFY (WM_USER + 11)
WM_RBUTTONUP = 0x0205


def _import_pystray():
    import pystray

    return pystray


def _import_webview():
    import webview

    return webview


class MenuItem:
    """One tray menu entry's live state. `callback(item)` runs on the UI
    thread when it's clicked; no callback = disabled. `state` is the check
    mark (only shown when `checkable`). `children` (with None for a
    separator) makes it a submenu."""

    def __init__(self, title: str, callback=None, *, checkable: bool = False) -> None:
        self.title = title
        self.callback = callback
        self.state = False
        self.checkable = checkable
        self.children: list = []

    def add(self, item) -> None:
        self.children.append(item)

    def set_callback(self, callback) -> None:
        self.callback = callback


class _NoopHud:
    """Stand-in for HudWindow when the HUD is disabled or unavailable, so
    _drain/quit never need to branch on whether a real HUD exists."""

    _mode = "full"
    on_menu = None

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

    def configure(self, cfg: dict) -> None:
        pass


class VeronicaApp:
    def __init__(self) -> None:
        self.title = "Veronica"
        self._state = "idle"
        self._muted = False
        self._quitting = False
        self._orch = None
        self._stopped = threading.Event()
        # -- settings window + self-update (Batch D) --------------------------
        # The bridge is pure Python and needs the orchestrator only at call
        # time (get_orch, and the store callable), so both it and the window
        # can be built now, before the background thread has built the
        # orchestrator (and with it the MemoryStore).
        self._exe_path = login_item.app_exe_path()
        self._repo = version.REPO
        self._build_info = version.build_info()
        self._bridge = SettingsBridge(
            settings=settings, get_orch=lambda: self._orch, store=lambda: getattr(self._orch, "store", None),
            run_on_loop=self._schedule, relaunch=self._relaunch, exe_path=self._exe_path,
            repo=self._repo, marshal=_main_thread,
        )
        self._settings = SettingsWindow(settings, self._bridge)
        # The window installs its own state listener; chain ours in front so
        # the update item also tracks checks/updates started from the window.
        self._window_on_state = getattr(self._bridge, "on_state_changed", None)
        self._bridge.on_state_changed = self._on_bridge_state
        self._checking_update = False
        self._about_title = f"About Veronica — {version.describe(self._build_info)}"
        self._about_item = MenuItem(self._about_title)
        self._settings_item = MenuItem("Settings…", self.open_settings)
        check_item = MenuItem("Check for Updates…", self.check_for_updates)
        self._update_item = MenuItem(UPDATE_UNCHECKED_TITLE)
        self._mute_item = MenuItem("Mute", self.toggle_mute, checkable=True)
        hud_mode_item = MenuItem("HUD: Full", self.toggle_hud_mode)
        login_item_item = self._make_login_item()
        self._voice_items: dict[str, MenuItem] = {}
        self._speed_items: dict[str, MenuItem] = {}
        voice_menu = MenuItem("Voice")
        # English voices, a separator, then the Hindi voices (picking one
        # sets the voice Hindi replies use; the English voice is untouched).
        for vid in voices.VOICE_IDS:
            name = voices.display_name(vid)
            item = MenuItem(name, self._pick_voice, checkable=True)
            self._voice_items[name] = item
            voice_menu.add(item)
        voice_menu.add(None)
        for vid in voices.HINDI_VOICE_IDS:
            name = voices.display_name(vid)
            item = MenuItem(name, self._pick_voice, checkable=True)
            self._voice_items[name] = item
            voice_menu.add(item)
        voice_menu.add(None)
        for title in SPEED_TITLES:
            item = MenuItem(title, self._speed)
            self._speed_items[title] = item
            voice_menu.add(item)
        self._voice_menu = voice_menu
        # Brain submenu: one item per backend; the parent's title doubles
        # as the "Brain: Codex" label (from the orchestrator's hud events).
        # Items for brains that aren't installed / logged in are disabled
        # with the reason in the title (_refresh_brain_menu).
        self._brain_items: dict[str, MenuItem] = {}
        self._brain_avail: dict[str, object] = {}
        self._brain_checked_at = -BRAIN_CHECK_INTERVAL_S
        brain_menu = MenuItem("Brain")
        for name, info in BACKENDS.items():
            item = MenuItem(info.label, self._pick_brain, checkable=True)
            self._brain_items[name] = item
            brain_menu.add(item)
        self._brain_menu = self._brain_item = brain_menu
        menu_items = [
            self._about_item, self._settings_item, check_item, self._update_item, None,
            self._mute_item, hud_mode_item, voice_menu, brain_menu, login_item_item, None,
        ]
        self._hud_mode_item = hud_mode_item
        self._login_item_item = login_item_item
        hud = HudWindow(settings) if settings.hud_enabled else None
        self._hud = hud if (hud is not None and hud.available) else _NoopHud()
        self._hud.on_menu = self._popup_menu_at
        self._events: queue.Queue = queue.Queue()
        self._loop = asyncio.new_event_loop()

        # push-to-talk (A2): the monitor needs `self._loop` (the background
        # orchestrator loop, not yet running) to marshal its callbacks onto.
        self._hotkey: HotkeyMonitor | None = None
        self._ptt_item: MenuItem | None = None
        if settings.ptt_enabled:
            self._hotkey = HotkeyMonitor(self._on_ptt_press, self._on_ptt_release, keycode=settings.ptt_keycode)
            self._hotkey.start(loop=self._loop)
            if not self._hotkey.available:
                # No permission to ask for on Windows: the keyboard hook
                # just couldn't be installed (see the log).
                self._ptt_item = MenuItem(PTT_UNAVAILABLE_TITLE)
                menu_items.append(self._ptt_item)

        menu_items.append(MenuItem("Quit", self.quit))
        self.menu = menu_items
        self._refresh_hud_mode_item()
        self._icon_key = None
        self._menu_sig = None
        self._icon = self._make_icon()
        self._thread = threading.Thread(target=self._run_loop, name="veronica-orchestrator", daemon=True)
        self._thread.start()
        self._timers = [
            _every(0.25, self._refresh),
            _every(1 / 30, self._drain),
            _every(UPDATE_CHECK_INTERVAL_S, self._hourly_update_check),
        ]

    # -- tray icon (pystray) -------------------------------------------------------
    def _make_icon(self):
        """Build the pystray Icon and run it on its own thread; None (and a
        logged warning) when pystray isn't available."""
        try:
            pystray = _import_pystray()
            icon = pystray.Icon("Veronica", icon=self._icon_image(), title=self.title,
                                menu=pystray.Menu(*self._pystray_items(pystray, self.menu)))
        except Exception:
            log.warning("system tray unavailable", exc_info=True)
            return None
        self._icon_thread = threading.Thread(target=self._run_icon, args=(icon,), name="veronica-tray", daemon=True)
        self._icon_thread.start()
        return icon

    @staticmethod
    def _run_icon(icon) -> None:
        try:
            icon.run()
        except Exception:
            log.exception("system tray loop failed")

    def _icon_image(self):
        return state_image(self._state, muted=self._muted)

    def _pystray_items(self, pystray, items) -> list:
        out = []
        for item in items:
            if item is None:
                out.append(pystray.Menu.SEPARATOR)
            elif item.children:
                out.append(pystray.MenuItem(self._text_of(item), pystray.Menu(*self._pystray_items(pystray, item.children))))
            else:
                out.append(pystray.MenuItem(
                    self._text_of(item), self._action_of(item),
                    checked=self._checked_of(item), enabled=self._enabled_of(item),
                    default=item is self._settings_item,   # left-click on the icon opens Settings
                ))
        return out

    # pystray calls these with the pystray MenuItem; each closes over ours.
    @staticmethod
    def _text_of(item: MenuItem):
        def text(_menu_item):
            return item.title
        return text

    @staticmethod
    def _checked_of(item: MenuItem):
        def checked(_menu_item):
            return bool(item.state) if item.checkable else None
        return checked

    @staticmethod
    def _enabled_of(item: MenuItem):
        def enabled(_menu_item):
            return item.callback is not None
        return enabled

    def _action_of(self, item: MenuItem):
        def action(_icon, _menu_item):
            self._main_activate(item)
        return action

    def _main_activate(self, item: MenuItem) -> None:
        """A click, on pystray's thread: run the item's callback on the UI thread."""
        def _do() -> None:
            if item.callback is not None:
                item.callback(item)
        _main_thread(_do)

    def _menu_signature(self) -> tuple:
        def walk(items):
            for item in items:
                if item is None:
                    continue
                yield (id(item), item.title, bool(item.state), item.callback is not None)
                yield from walk(item.children)
        return tuple(walk(self.menu))

    def _sync_tray(self) -> None:
        """Push title/icon/menu changes to the tray (UI thread)."""
        icon = self._icon
        if icon is None:
            return
        key = (self._state, self._muted)
        try:
            if key != self._icon_key:
                self._icon_key = key
                icon.icon = self._icon_image()
            if getattr(icon, "title", None) != self.title:
                icon.title = self.title
            sig = self._menu_signature()
            if sig != self._menu_sig:
                self._menu_sig = sig
                icon.update_menu()
        except Exception:
            log.debug("tray update failed", exc_info=True)

    def _notify(self, subtitle: str, text: str, *, spoken: str | None = None) -> None:
        """Show a tray notification; when that isn't possible (no tray, or
        notifications unsupported), log it and have Veronica say it when
        she's next idle."""
        icon = self._icon
        if icon is not None and getattr(icon, "HAS_NOTIFICATION", True):
            try:
                icon.notify(text or subtitle, f"Veronica — {subtitle}" if subtitle else "Veronica")
                return
            except Exception as e:
                log.info("notification unavailable (%s): %s — %s", e, subtitle, text)
        else:
            log.info("notification unavailable: %s — %s", subtitle, text)
        orch = getattr(self, "_orch", None)
        if orch is not None:
            self._schedule(orch.announce(spoken or text))

    # -- asyncio side (background thread) ----------------------------------------------
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
                can_relaunch=lambda: self._exe_path is not None,
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
            logging.getLogger("veronica.ui").exception("tray app background loop failed")

    def _on_state(self, state: str) -> None:
        self._state = state

    def _schedule_quit(self) -> None:
        # Called from the orchestrator's background asyncio thread (the
        # "quit" voice intent) after confirmation; quit() tears down window
        # state and must run on the UI thread.
        _main_thread(lambda: self.quit(None))

    # -- UI thread ------------------------------------------------------------------------
    def _refresh(self, _timer=None) -> None:
        if self._muted:
            self.title = "Veronica — Muted"
        elif self._state == "error":
            err = getattr(self, "_error", "")
            self.title = f"Veronica — Error: {err}"[:120] if err else "Veronica — Error"
        else:
            self.title = f"Veronica — {STATE_LABELS.get(self._state, self._state)}"
        self._refresh_voice_menu()
        self._refresh_brain_menu()
        self._sync_tray()

    def _drain(self, _timer=None) -> None:
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
                elif isinstance(payload, dict) and isinstance(payload.get("config"), dict):
                    self._hud.configure(payload["config"])
                elif isinstance(payload, dict) and isinstance(payload.get("backend"), str):
                    # Which brain is answering: the submenu title and the
                    # HUD's own "Brain: …" line (the other hud payloads are
                    # window-level and never reach the page).
                    self._brain_item.title = f"Brain: {payload['backend']}"
                    self._hud.push({"kind": kind, "payload": payload})
                continue
            if kind == "settings":
                # "open settings" / "show history" voice intents.
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

    def toggle_hud_mode(self, _item=None) -> None:
        mode = getattr(self._hud, "_mode", "full")
        self._hud.set_mode("full" if mode == "mini" else "mini")
        self._refresh_hud_mode_item()

    def _make_login_item(self) -> MenuItem:
        if login_item.app_exe_path() is None:
            return MenuItem(LOGIN_ITEM_BUILD_FIRST_TITLE, checkable=True)
        item = MenuItem(LOGIN_ITEM_TITLE, self.toggle_login_item, checkable=True)
        item.state = login_item.is_enabled()
        return item

    def toggle_login_item(self, item: MenuItem) -> None:
        exe = login_item.app_exe_path()
        if exe is None:
            return
        if login_item.is_enabled():
            login_item.disable()
        else:
            login_item.enable(exe)
        item.state = login_item.is_enabled()

    # -- settings window / self-update (Batch D) -----------------------------------
    def open_settings(self, _item=None) -> None:
        self._settings.show("general")

    def _relaunch(self) -> bool:
        """Restart the app after an update (bridge "Update & restart" /
        "Restart", or the orchestrator's "update yourself" turn). May be
        called from any thread: quitting is marshalled to the UI thread."""
        return relaunch(self._exe_path, self._schedule_quit)

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

    def _set_update_item(self, title: str, installable: bool = False) -> None:
        self._update_item.title = title
        self._update_item.set_callback(self.update_now if installable else None)

    def _run_update_check(self, *, notify: bool) -> None:
        """Check for updates on the bridge's worker thread (a git fetch can
        take seconds) and reflect the result in the update item; with
        `notify`, also show a notification with the outcome."""
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

    def _hourly_update_check(self, _timer=None) -> None:
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
        """Bridge state listener (already marshalled to the UI thread):
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

    def toggle_mute(self, item: MenuItem) -> None:
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
        """Run `coro` on the orchestrator's background loop from any thread
        (or, under test with a loop that isn't running yet, queue it as a
        task for the next run_until_complete)."""
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

    def _pick_voice(self, item: MenuItem) -> None:
        self._voice_action(("voice", item.title.lower()))
        self._refresh_voice_menu()

    def _speed(self, item: MenuItem) -> None:
        self._voice_action(("speed", SPEED_TITLES[item.title]))

    def _pick_brain(self, item: MenuItem) -> None:
        orch = getattr(self, "_orch", None)
        name = next((n for n, i in self._brain_items.items() if i is item), None)
        if orch is None or name is None:
            return
        # The orchestrator runs it as the "switch to codex" turn on its own
        # loop (thread-safe from here); it says "Switched to Codex." or why not.
        orch.request_brain_switch(name)
        self._refresh_brain_menu()

    def _refresh_brain_menu(self) -> None:
        # On the 0.25 s _refresh timer: cheap, and fine before the
        # orchestrator (or its switcher) exists. The availability check
        # itself runs at most every BRAIN_CHECK_INTERVAL_S.
        now = time.monotonic()
        if now - self._brain_checked_at >= BRAIN_CHECK_INTERVAL_S:
            self._brain_checked_at = now
            self._brain_avail = {name: check_backend(name) for name in BACKENDS}
        switcher = getattr(getattr(self, "_orch", None), "switcher", None)
        active = getattr(getattr(switcher, "brain", None), "name", None)
        preferred = getattr(switcher, "preferred", None)
        standing_in = bool(getattr(switcher, "standing_in", False)) and preferred in BACKENDS
        for name, item in self._brain_items.items():
            label = BACKENDS[name].label
            avail = self._brain_avail.get(name)
            if avail is not None and not avail.ok:
                title, callback = f"{label} ({avail.reason})", None
            elif standing_in and name == active and name != preferred:
                title, callback = f"{label} — standing in for {BACKENDS[preferred].label}", self._pick_brain
            else:
                title, callback = label, self._pick_brain
            if item.title != title:
                item.title = title
            if item.callback != callback:
                item.set_callback(callback)
            item.state = name == active

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
            item.state = name in checked

    # -- HUD orb click -> menu ---------------------------------------------
    def _popup_menu_at(self, x: float, y: float) -> None:
        """hud.on_menu callback (UI thread): show the tray menu at the
        cursor. pystray's Win32 backend shows its menu when its hidden
        window gets a right-button-up tray notification, at GetCursorPos —
        i.e. where the orb was just clicked — so post it exactly that; the
        menu is the tray's own, refreshed first. (x, y) is the click point,
        which that cursor position already is.)"""
        icon = self._icon
        hwnd = getattr(icon, "_hwnd", None)
        if icon is None or not hwnd:
            log.info("HUD menu unavailable (no system tray window)")
            return
        self._refresh_voice_menu()
        self._refresh_brain_menu()
        try:
            icon.update_menu()
            self._menu_sig = self._menu_signature()
            win32.post_message(hwnd, PYSTRAY_WM_NOTIFY, 0, WM_RBUTTONUP)
        except Exception:
            log.warning("failed to pop up the menu at the HUD", exc_info=True)

    # -- lifecycle ---------------------------------------------------------------
    def quit(self, _item=None) -> None:
        if self._quitting:
            return
        self._quitting = True
        for timer in getattr(self, "_timers", []):
            timer.cancel()
        self._hud.close()
        self._settings.close()
        if self._hotkey is not None:
            self._hotkey.stop()
        orch = getattr(self, "_orch", None)
        if orch is not None:
            orch.player.close()
            store = getattr(orch, "store", None)
            if store is not None:
                store.close()
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                log.debug("tray stop failed", exc_info=True)
        # Last: with every window destroyed webview.start() returns on the
        # main thread and the process exits.
        _main_thread(self._close_windows)
        self._stopped.set()

    @staticmethod
    def _close_windows() -> None:
        try:
            webview = _import_webview()
        except ImportError:
            return
        for window in list(getattr(webview, "windows", [])):
            try:
                window.destroy()
            except Exception:
                log.debug("window destroy failed", exc_info=True)

    def run(self) -> None:
        """Run the GUI loop on this (the main) thread until Quit."""
        try:
            webview = _import_webview()
        except ImportError:
            log.error("pywebview isn't installed: running without the HUD and Settings windows")
            self._stopped.wait()
            return
        if isinstance(self._hud, _NoopHud):
            # webview.start() needs a window to start with; without a HUD,
            # create the Settings window up front (hidden) instead.
            self._settings.create_hidden()
        try:
            webview.start(gui="edgechromium", private_mode=True)
        except Exception:
            log.exception("GUI loop failed")
        if not self._quitting:
            # Every window went away without a Quit (e.g. WebView2 crashed):
            # stop the rest too rather than linger as a headless process.
            self.quit(None)


def run_app() -> None:
    VeronicaApp().run()
