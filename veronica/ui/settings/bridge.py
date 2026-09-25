"""The settings window's command surface, kept free of AppKit/WebKit so it
can be driven end-to-end in tests.

`SettingsWindow` only marshals: JS posts `{id, cmd, args}`, the window calls
`bridge.handle(cmd, args)` and replies with the dict. Everything that touches
the orchestrator goes through `run_on_loop(coro)` (the menubar's `_schedule`)
because the window runs on the AppKit main thread while the orchestrator lives
on its own asyncio loop. Long-running work (the update) runs via `run_thread`.

Two classes of setting:

- *live*: applied to the running process right away (assigned onto `orch.s`,
  or dispatched as an orchestrator turn) and persisted.
- *restart*: persisted only (`prefs.save_settings_override`); the process
  keeps its current value and `restart_required` latches until relaunch.

While the orchestrator is still warming up (`get_orch()` is None) live changes
are refused with a friendly message; restart-class ones still save.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from veronica import config, proactive
from veronica import prefs as _prefs
from veronica import updater as _updater
from veronica import version as _version
from veronica.brain.backends import check_backend as _check_backend
from veronica.speech import voices
from veronica.ui import login_item as _login_item

log = logging.getLogger("veronica.ui.settings")

STARTING_UP = "Still starting up, try again in a moment."
BUSY = "Busy, try again in a moment."
UPDATING = "Updating, back in a moment."
UPDATE_FAILED = "The update failed, check the log."
LATEST = "You're already on the latest."
BUILD_FIRST = "Build the app first."
RESTART_FROM_TERMINAL = "Restart me from the terminal."
VOICE_TEST = {"en": "This is how I sound now.", "hi": "Main aise bolti hoon."}
LANGUAGE_MODES = ("en", "hi", "auto")
HUD_MODES = ("full", "mini")
LOGIN_ITEMS_URL = "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
_PUSH_AFTER_TURN = "_push_after_turn"   # internal reply marker, stripped before the window sees it

# Which Settings fields each section exposes. Anything in EDITABLE_SETTINGS
# not listed here is unreachable from the window (deliberately).
SETTING_SECTIONS: dict[str, tuple[str, ...]] = {
    "general": ("ptt_enabled", "hud_hide_after_s", "hud_particles", "hud_intensity"),
    "listening": ("followup_window_s", "confirm_listen_s", "ack_after_s", "vad_silence_ms", "max_utterance_s",
                  "wake_min_rms", "wake_window_s", "wake_hop_s", "wake_phrases", "input_volume_floor"),
    "brain": ("effort", "memory_enabled", "memory_facts_max", "brain_cwd", "brain_session_max_age_h", "computer_trust_s", "preapprove_by_wording", "shortcut_allowlist",
              "brain_backend", "brain_failover", "brain_failover_order", "brain_limit_cooldown_min",
              "codex_native_tools", "antigravity_native_tools", "copilot_native_tools",
              "brain_offline_fallback", "local_server_bin", "local_model", "local_ctx", "local_port"),
}
HUD_CONFIG_KEYS = ("hud_particles", "hud_intensity")
BRIEFING_KEYS = ("briefing_enabled", "briefing_time", "nudges_enabled", "nudge_minutes",
                 "quiet_enabled", "quiet_from", "quiet_to", "battery_enabled",
                 "unread_enabled", "unread_time")
BRIEFING_TIME_KEYS = ("briefing_time", "quiet_from", "quiet_to", "unread_time")


def _inline(fn: Callable[[], None]) -> None:
    fn()


def _thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="veronica-settings", daemon=True).start()


def _open_path(target: Path | str) -> None:
    subprocess.Popen(["open", str(target)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ok(**extra) -> dict:
    return {"ok": True, "message": "", **extra}


def _fail(message: str, **extra) -> dict:
    return {"ok": False, "message": message, **extra}


class SettingsBridge:
    def __init__(
        self,
        *,
        settings: config.Settings,
        get_orch: Callable[[], Any | None],
        store,
        run_on_loop: Callable[[Coroutine], None],
        prefs=_prefs,
        login_item=_login_item,
        version=_version,
        updater=_updater,
        relaunch: Callable[[], bool],
        bundle_path: Path | None,
        repo: Path,
        run_thread: Callable[[Callable[[], None]], None] = _thread,
        open_path: Callable[[Path | str], None] = _open_path,
        marshal: Callable[[Callable[[], None]], None] = _inline,
        check_backend: Callable[[str], Any] = _check_backend,
    ) -> None:
        self._settings = settings
        self._check_backend = check_backend
        self._get_orch = get_orch
        #: A store, None, or a zero-arg callable returning either (the menu
        #: bar passes a callable: its orchestrator — and so the MemoryStore
        #: — only exists once the background thread has built it).
        self._store = store
        self._run_on_loop = run_on_loop
        self._prefs = prefs
        self._login_item = login_item
        self._version = version
        self._updater = updater
        self._relaunch = relaunch
        self._bundle_path = bundle_path
        self._repo = repo
        self._run_thread = run_thread
        #: Public for the window: long commands it dispatches run here too.
        self.run_thread = run_thread
        self._open_path = open_path
        self._marshal = marshal
        self.restart_required = False
        #: Set by the window to push a fresh `get_state()` to JS after any
        #: change. It may fire from the update worker thread (a failed
        #: update) or from whichever thread ran `check_update`, so every call
        #: is routed through `marshal` — the window passes something that
        #: hops to the AppKit main thread; the default runs inline.
        self.on_state_changed: Callable[[dict], None] | None = None
        self._update_status = None          # last UpdateStatus from check_update
        self._update_error: str | None = None
        self._updating = False
        self._update_lock = threading.Lock()   # begin_update/end_update from any thread
        self._build_info: dict | None = None   # what's running never changes; computed once

    # -- dispatch ----------------------------------------------------------------
    def handle(self, cmd: str, args: dict | None = None) -> dict:
        args = dict(args or {})
        handlers: dict[str, Callable[..., dict]] = {
            "get_state": self.get_state,
            "set": self.set,
            "test_voice": self.test_voice,
            "history": self.history,
            "forget_turn": self.forget_turn,
            "clear_history": self.clear_history,
            "check_update": self.check_update,
            "update_now": self.update_now,
            "restart": self.restart,
            "open_logs": self.open_logs,
            "open_login_items": self.open_login_items,
        }
        fn = handlers.get(cmd)
        if fn is None:
            return _fail("unknown command")
        try:
            return fn(**args)
        except Exception as e:  # the window must always get a reply
            log.exception("settings command %s failed", cmd)
            return _fail(str(e) or e.__class__.__name__)

    def _push(self) -> None:
        cb = self.on_state_changed
        if cb is None:
            return
        try:
            state = self.get_state()
        except Exception:
            log.exception("get_state failed")
            return

        def deliver() -> None:
            try:
                cb(state)
            except Exception:
                log.exception("on_state_changed failed")

        self._marshal(deliver)

    # -- state -------------------------------------------------------------------
    def _setting(self, name: str, overrides: dict, s) -> Any:
        """The value the window should show: the persisted override if there
        is one (a restart-class change the process hasn't picked up yet),
        else what the live Settings holds."""
        if name in overrides:
            return overrides[name]
        return getattr(s, name)

    def _schedule(self, orch) -> proactive.Schedule:
        pro = getattr(orch, "proactive", None) if orch is not None else None
        if pro is not None and getattr(pro, "schedule", None) is not None:
            return pro.schedule
        return proactive.load_schedule(self._prefs.load)

    def _cached_build_info(self) -> dict:
        if self._build_info is None:
            self._build_info = dict(self._version.build_info())
        return self._build_info

    def get_state(self) -> dict:
        orch = self._get_orch()
        saved = self._prefs.load() or {}
        overrides = saved.get("settings") or {}
        s = orch.s if orch is not None else self._settings

        def setting(name):
            v = self._setting(name, overrides, s)
            return str(v) if isinstance(v, Path) else v

        if orch is not None:
            language = getattr(orch, "language", None) or s.language
            tts = getattr(orch, "tts", None)
            voice = getattr(tts, "voice", None) or voices.DEFAULT_VOICE
            hindi_voice = getattr(tts, "hindi_voice", None) or voices.DEFAULT_HINDI_VOICE
            speed = float(getattr(tts, "speed", None) or voices.DEFAULT_SPEED)
        else:
            language = saved.get("language") or s.language
            voice = saved.get("tts_voice") or voices.DEFAULT_VOICE
            hindi_voice = saved.get("tts_hindi_voice") or voices.DEFAULT_HINDI_VOICE
            speed = voices.clamp_speed(saved.get("tts_speed", voices.DEFAULT_SPEED))
        sched = self._schedule(orch)
        info = self._cached_build_info()
        if self._update_error:
            update = {"available": True, "detail": self._update_error}
        elif self._update_status is not None:
            update = {"available": bool(self._update_status.available), "detail": self._update_status.detail}
        else:
            update = {"available": False, "detail": ""}

        return {
            "general": {
                "language": language,
                "start_at_login": bool(self._login_item.is_enabled()),
                "ptt_enabled": setting("ptt_enabled"),
                "hud_mode": saved.get("hud_mode") or s.hud_mode,
                "hud_hide_after_s": setting("hud_hide_after_s"),
                "hud_particles": setting("hud_particles"),
                "hud_intensity": setting("hud_intensity"),
                "can_start_at_login": self._bundle_path is not None,
            },
            "voice": {
                "voice": voice,
                "hindi_voice": hindi_voice,
                "speed": speed,
                "voices": [
                    {"id": vid, "name": voices.display_name(vid), "hindi": voices.is_hindi_voice(vid)}
                    for vid in voices.ALL_VOICE_IDS
                ],
            },
            "listening": {name: setting(name) for name in SETTING_SECTIONS["listening"]},
            "briefings": sched.to_prefs(),
            "brain": {
                **{name: setting(name) for name in SETTING_SECTIONS["brain"]},
                # What's actually answering right now ("Codex", "Claude (for
                # Codex)" while standing in); "" until the switcher exists.
                "brain_label": self._brain_label(orch),
            },
            "about": {
                "version": self._version.APP_VERSION,
                "build": info.get("sha", ""),
                "built_at": info.get("built_at", ""),
                "dirty": bool(info.get("dirty", False)),
                "describe": self._version.describe(info),
                "update": update,
                "updating": self._updating,
                "log_path": str(self._settings.log_file),
                "can_restart": self._bundle_path is not None,
            },
            "meta": {
                "restart_required": self.restart_required,
                "fields": {
                    name: {
                        "kind": f.kind, "label": f.label, "help": f.help,
                        "choices": list(f.choices) if f.choices else None,
                        "min": f.min, "max": f.max, "restart": f.restart,
                    }
                    for name, f in config.EDITABLE_SETTINGS.items()
                },
            },
        }

    # -- set ---------------------------------------------------------------------
    def set(self, section: str, key: str, value: Any = None) -> dict:
        try:
            if section == "general" and key == "language":
                result = self._set_language(value)
            elif section == "general" and key == "start_at_login":
                result = self._set_start_at_login(value)
            elif section == "general" and key == "hud_mode":
                result = self._set_hud_mode(value)
            elif section == "voice" and key in ("voice", "hindi_voice"):
                result = self._set_voice(key, value)
            elif section == "voice" and key == "speed":
                result = self._set_speed(value)
            elif section == "briefings" and key in BRIEFING_KEYS:
                result = self._set_briefing(key, value)
            elif key in SETTING_SECTIONS.get(section, ()):
                result = self._set_setting(key, value)
            else:
                result = _fail(f"unknown setting {section}.{key}")
        except (ValueError, TypeError) as e:
            result = _fail(str(e))
        result.setdefault("restart_required", self.restart_required)
        # Language/voice changes are orchestrator turns that only run later
        # on the asyncio loop: pushing now would show the *old* value. Those
        # handlers mark the reply and push themselves once the turn is done.
        deferred = result.pop(_PUSH_AFTER_TURN, False)
        if result["ok"] and not deferred:
            self._push()
        return result

    def _orch_or_none(self):
        return self._get_orch()

    @staticmethod
    def _brain_label(orch) -> str:
        switcher = getattr(orch, "switcher", None)
        label = getattr(switcher, "status_label", None)
        return str(label()) if callable(label) else ""

    def _set_brain(self, name: str) -> dict:
        """Brain choice: the switch runs as the same turn "switch to codex"
        does (interrupt, then "Switched to Codex."); the switcher persists
        the preference itself. Refused up front, with the spoken hint, when
        that brain isn't installed or logged in."""
        orch = self._orch_or_none()
        if orch is None or getattr(orch, "switcher", None) is None:
            return _fail(STARTING_UP)
        avail = self._check_backend(name)
        if not avail.ok:
            return _fail(avail.hint)
        self._run_on_loop(self._with_player_reset(orch, orch._brain_switch_turn(("switch", name)), push_after=True))
        return _ok(**{_PUSH_AFTER_TURN: True})

    async def _with_player_reset(self, orch, coro, *, push_after: bool = False) -> None:
        """The orchestrator's own dispatch resets the player before a turn
        (the turns don't do it themselves); mirror that, as the menubar's
        `_voice_action` does, so in-flight speech is cut before the reply.
        With `push_after`, push fresh state to the window once the turn has
        run (i.e. once orch.language / tts.voice actually changed)."""
        reset = getattr(getattr(orch, "player", None), "reset", None)
        if reset is not None:
            reset()
        try:
            await coro
        finally:
            if push_after:
                self._push()

    def _set_language(self, value) -> dict:
        mode = str(value or "").strip().lower()
        if mode not in LANGUAGE_MODES:
            return _fail(f"unknown language mode {value!r}")
        orch = self._orch_or_none()
        if orch is None:
            return _fail(STARTING_UP)
        if getattr(orch, "language", None) == mode:
            return _ok()   # already there: don't reload or re-announce
        self._run_on_loop(self._with_player_reset(orch, orch._language_turn(mode), push_after=True))
        return _ok(**{_PUSH_AFTER_TURN: True})

    def _set_start_at_login(self, value) -> dict:
        want = _to_bool(value)
        if self._bundle_path is None:
            return _fail(BUILD_FIRST)
        if want:
            self._login_item.enable(self._bundle_path)
        else:
            self._login_item.disable()
        return _ok()

    def _set_hud_mode(self, value) -> dict:
        mode = str(value or "").strip().lower()
        if mode not in HUD_MODES:
            return _fail(f"unknown HUD mode {value!r}")
        orch = self._orch_or_none()
        if orch is None:
            return _fail(STARTING_UP)
        # The menubar persists hud_mode when it handles the event, but that
        # happens on a later _drain tick — save first so the state we push
        # right after this already shows the new mode.
        self._prefs.save({"hud_mode": mode})
        orch._emit("hud", {"mode": mode})
        return _ok()

    def _set_voice(self, key: str, value) -> dict:
        raw = str(value or "").strip()
        name = voices.display_name(raw).lower() if raw in voices.ALL_VOICE_IDS else raw.lower()
        vid = voices.resolve_voice(name) if name else None
        if vid is None:
            return _fail(f"I don't have a voice called {raw!r}")
        want_hindi = key == "hindi_voice"
        if voices.is_hindi_voice(vid) != want_hindi:
            return _fail(f"{voices.display_name(vid)} is {'not ' if want_hindi else ''}a Hindi voice")
        orch = self._orch_or_none()
        if orch is None:
            return _fail(STARTING_UP)
        self._run_on_loop(self._with_player_reset(orch, orch._voice_turn(("voice", name)), push_after=True))
        return _ok(**{_PUSH_AFTER_TURN: True})

    def _set_speed(self, value) -> dict:
        speed = round(voices.clamp_speed(float(value)), 2)
        orch = self._orch_or_none()
        if orch is None:
            return _fail(STARTING_UP)
        orch.tts.speed = speed
        self._prefs.save({"tts_speed": speed})
        return _ok()

    def _set_briefing(self, key: str, value) -> dict:
        orch = self._orch_or_none()
        if orch is None:
            return _fail(STARTING_UP)
        pro = getattr(orch, "proactive", None)
        sched = getattr(pro, "schedule", None) if pro is not None else None
        if sched is None:
            return _fail("Briefings aren't available right now.")
        if key in BRIEFING_TIME_KEYS:
            text = str(value or "").strip()
            if not proactive._TIME_RE.match(text):
                return _fail("Time must be HH:MM (24-hour)")
            setattr(sched, key, text)
        elif key == "nudge_minutes":
            if isinstance(value, bool):
                raise TypeError("nudge_minutes must be a number")
            sched.nudge_minutes = max(1, min(60, int(value)))
        else:  # the on/off flags
            setattr(sched, key, _to_bool(value))
        proactive.save_schedule(sched, save=self._prefs.save)
        return _ok()

    def _set_setting(self, name: str, value) -> dict:
        field = config.EDITABLE_SETTINGS[name]
        coerced = config.coerce_setting(name, value)
        orch = self._orch_or_none()
        if field.restart:
            # The process keeps its current value, but run the new one
            # through Settings' validation first: load_settings drops invalid
            # overrides silently at startup, which would lose the change.
            probe = (orch.s if orch is not None else self._settings).model_copy()
            setattr(probe, name, coerced)   # ValidationError (a ValueError) on bad input
            self._prefs.save_settings_override(name, _jsonable(getattr(probe, name)))
            self.restart_required = True
            return _ok(restart_required=True)
        if name == "brain_backend":
            return self._set_brain(coerced)
        if orch is None:
            return _fail(STARTING_UP)
        setattr(orch.s, name, coerced)   # validate_assignment=True: raises ValueError on bad input
        self._prefs.save_settings_override(name, _jsonable(getattr(orch.s, name)))
        if name in HUD_CONFIG_KEYS:
            # The orb applies these live: the menubar's _drain maps a "hud"
            # event carrying "config" to HudWindow.configure().
            orch._emit("hud", {"config": {"particles": int(orch.s.hud_particles),
                                          "intensity": float(orch.s.hud_intensity)}})
        return _ok()

    # -- voice test -----------------------------------------------------------------
    def test_voice(self, lang: str | None = None) -> dict:
        orch = self._orch_or_none()
        if orch is None:
            return _fail(STARTING_UP)
        if lang is None:
            lang = "hi" if getattr(orch, "language", "en") == "hi" else "en"
        lang = "hi" if str(lang).lower() == "hi" else "en"
        self._run_on_loop(self._with_player_reset(orch, orch.say(VOICE_TEST[lang], lang="hi" if lang == "hi" else None)))
        return _ok()

    # -- history ----------------------------------------------------------------------
    def _history_store(self):
        store = self._store
        return store() if callable(store) else store

    def history(self, query: str = "", limit: int = 200, offset: int = 0) -> dict:
        store = self._history_store()
        if store is None:
            return _fail("Memory is off.", items=[])
        items = store.turns(limit=int(limit), offset=int(offset), query=str(query or ""))
        return _ok(items=list(items))

    def forget_turn(self, id: int) -> dict:
        store = self._history_store()
        if store is None:
            return _fail("Memory is off.")
        if not store.delete_turn(int(id)):
            return _fail("That one's already gone.")
        return _ok()

    def clear_history(self) -> dict:
        store = self._history_store()
        if store is None:
            return _fail("Memory is off.", count=0)
        return _ok(count=int(store.clear_turns()))

    # -- updates ------------------------------------------------------------------------
    def check_update(self) -> dict:
        try:
            status = self._updater.check(self._repo, info=self._cached_build_info())
        except Exception as e:  # noqa: BLE001 — network/git trouble becomes a message
            log.warning("update check failed: %s", e)
            return _fail(f"Couldn't check: {e}")
        self._update_status = status
        self._update_error = None
        self._push()
        return _ok(available=bool(status.available), kind=status.kind, detail=status.detail)

    # The bridge is the single owner of "an update is running": both its
    # own update_now and the orchestrator's spoken "update yourself" go
    # through begin_update/end_update, so two pulls/builds can never run on
    # dist/ at once, whichever way they were started.
    def begin_update(self) -> bool:
        """Claim the update slot. False if an update is already running."""
        with self._update_lock:
            if self._updating:
                return False
            self._updating = True
            self._update_error = None
        self._push()
        return True

    def end_update(self, error: str | None = None) -> None:
        """Release the slot; `error` (e.g. UPDATE_FAILED) is what the window
        and menu show for a failed one."""
        with self._update_lock:
            self._updating = False
            self._update_error = error
        self._push()

    def update_now(self) -> dict:
        orch = self._orch_or_none()
        if orch is None or getattr(orch, "state", None) != "idle" or self._updating:
            return _fail(BUSY)
        status = self._update_status
        if status is None:
            res = self.check_update()
            if not res["ok"]:
                return res
            status = self._update_status
        if not status.available:
            return _fail(LATEST)
        if not self.begin_update():
            return _fail(BUSY)

        def work() -> None:
            try:
                log.info("update: %s", self._updater.update(self._repo, status))
            except Exception:  # surfaced in state instead of raised
                log.exception("update failed")
                self.end_update(UPDATE_FAILED)
                return
            self.end_update()
            self._relaunch()

        self._run_thread(work)
        return {"ok": True, "message": UPDATING}

    # -- restart / logs --------------------------------------------------------------------
    def restart(self) -> dict:
        # Reply first, relaunch after: relaunch() schedules the quit, and the
        # window must get its answer before the app goes away.
        can = self._bundle_path is not None

        def go() -> None:
            if not self._relaunch() and can:
                log.warning("restart: relaunch could not be scheduled")

        self._run_thread(go)
        return _ok() if can else {"ok": True, "message": RESTART_FROM_TERMINAL}

    def open_logs(self) -> dict:
        self._open_path(self._settings.log_file)
        return _ok()

    def open_login_items(self) -> dict:
        self._open_path(LOGIN_ITEMS_URL)
        return _ok()


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _jsonable(value: Any) -> Any:
    """Settings values as prefs.json can hold them (Paths become strings)."""
    if isinstance(value, Path):
        return str(value)
    return value
