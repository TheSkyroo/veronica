"""SettingsBridge: every command driven with fakes — no AppKit, no prefs.json,
no orchestrator loop. `run_on_loop` runs the coroutine to completion inline
and `run_thread` runs its function inline so the update flow is synchronous."""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from veronica.config import EDITABLE_SETTINGS, Settings
from veronica.proactive import Schedule
from veronica.ui.settings.bridge import SettingsBridge


# -- fakes ---------------------------------------------------------------------
class FakeTTS:
    def __init__(self):
        self.voice = "af_sarah"
        self.hindi_voice = "hf_alpha"
        self.speed = 1.0


class FakeOrch:
    def __init__(self):
        self.state = "idle"
        self.s = Settings()
        self.tts = FakeTTS()
        self.language = "en"
        self.proactive = SimpleNamespace(schedule=Schedule())
        self.calls: list[tuple] = []
        self.emitted: list[tuple] = []

    async def _language_turn(self, mode):
        self.calls.append(("language", mode))

    async def _voice_turn(self, action):
        self.calls.append(("voice", action))

    async def say(self, text, *, lang=None):
        self.calls.append(("say", text, lang))

    def _emit(self, kind, payload):
        self.emitted.append((kind, payload))


class FakePrefs:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})
        self.saved: list[dict] = []
        self.overrides: list[tuple] = []

    def load(self):
        return dict(self.data)

    def save(self, d):
        self.saved.append(dict(d))
        self.data.update(d)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save_settings_override(self, field, value):
        self.overrides.append((field, value))
        cur = dict(self.data.get("settings", {}))
        cur[field] = value
        self.save({"settings": cur})

    def clear_settings_override(self, field):
        cur = dict(self.data.get("settings", {}))
        cur.pop(field, None)
        self.save({"settings": cur})


class FakeStore:
    def __init__(self):
        self.items = [
            {"id": 1, "ts": "2026-09-17T09:00:00", "heard": "hi", "reply": "hello"},
            {"id": 2, "ts": "2026-09-17T09:01:00", "heard": "weather", "reply": "sunny"},
        ]
        self.calls: list[tuple] = []

    def turns(self, limit=200, offset=0, query=""):
        self.calls.append(("turns", limit, offset, query))
        return list(self.items)

    def delete_turn(self, turn_id):
        self.calls.append(("delete", turn_id))
        before = len(self.items)
        self.items = [t for t in self.items if t["id"] != turn_id]
        return len(self.items) != before

    def clear_turns(self):
        self.calls.append(("clear",))
        n = len(self.items)
        self.items = []
        return n


class FakeLoginItem:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.calls: list[tuple] = []

    def is_enabled(self):
        return self.enabled

    def enable(self, app_path):
        self.calls.append(("enable", Path(app_path)))
        self.enabled = True

    def disable(self):
        self.calls.append(("disable",))
        self.enabled = False


class UpdateError(RuntimeError):
    pass


class FakeUpdater:
    UpdateError = UpdateError

    def __init__(self, status=None, fail: Exception | None = None):
        self.status = status or SimpleNamespace(
            available=False, kind="none", detail="You're on the latest (abc1234).",
            running_sha="abc1234", head_sha="abc1234", remote_sha=None,
        )
        self.fail = fail
        self.checks: list[Path] = []
        self.updates: list[tuple] = []

    def check(self, repo, run=None, info=None):
        self.checks.append(repo)
        if isinstance(self.status, Exception):
            raise self.status
        return self.status

    def update(self, repo, status, run=None, build=None, which=None):
        self.updates.append((repo, status))
        if self.fail is not None:
            raise self.fail
        return "ok"


FAKE_VERSION = SimpleNamespace(
    APP_VERSION="0.1.0",
    build_info=lambda: {"sha": "abc1234", "built_at": "2026-09-17T00:00:00+05:30", "dirty": False, "source": "git"},
    describe=lambda info=None: "Veronica 0.1.0 (abc1234, 17 Sep)",
)


def run_on_loop(coro):
    asyncio.run(coro)


def run_thread(fn):
    fn()


class Harness:
    def __init__(self, *, orch: FakeOrch | None = None, warming=False, bundle=Path("/tmp/Veronica.app"),
                 updater=None, prefs=None, login=None):
        self.orch = orch if orch is not None else (None if warming else FakeOrch())
        self.prefs = prefs or FakePrefs()
        self.store = FakeStore()
        self.login = login or FakeLoginItem()
        self.updater = updater or FakeUpdater()
        self.relaunches = 0
        self.opened: list = []
        self.states: list[dict] = []
        self.settings = Settings()

        def relaunch():
            self.relaunches += 1
            return bundle is not None

        self.bridge = SettingsBridge(
            settings=self.settings,
            get_orch=lambda: self.orch,
            store=self.store,
            run_on_loop=run_on_loop,
            prefs=self.prefs,
            login_item=self.login,
            version=FAKE_VERSION,
            updater=self.updater,
            relaunch=relaunch,
            bundle_path=bundle,
            repo=Path("/repo"),
            run_thread=run_thread,
            open_path=self.opened.append,
        )
        self.bridge.on_state_changed = self.states.append


@pytest.fixture
def h():
    return Harness()


# -- state shape ---------------------------------------------------------------
def test_state_has_every_section_and_key(h):
    st = h.bridge.get_state()
    assert set(st) == {"general", "voice", "listening", "briefings", "brain", "about", "meta"}
    assert {"language", "start_at_login", "ptt_enabled", "hud_mode", "hud_hide_after_s"} <= set(st["general"])
    assert {"voice", "hindi_voice", "speed", "voices"} == set(st["voice"])
    assert {"followup_window_s", "confirm_listen_s", "wake_min_rms", "wake_phrases",
            "vad_silence_ms", "max_utterance_s", "wake_window_s", "wake_hop_s"} <= set(st["listening"])
    assert set(st["briefings"]) == {"briefing_enabled", "briefing_time", "nudges_enabled", "nudge_minutes"}
    assert set(st["brain"]) == {"effort", "memory_enabled", "brain_cwd"}
    assert {"version", "build", "built_at", "dirty", "update", "log_path"} <= set(st["about"])
    assert st["about"]["version"] == "0.1.0"
    assert st["about"]["build"] == "abc1234"
    assert st["about"]["update"] == {"available": False, "detail": ""}
    assert st["about"]["log_path"].endswith("veronica.log")
    assert st["meta"]["restart_required"] is False
    assert set(st["meta"]["fields"]) == set(EDITABLE_SETTINGS)
    f = st["meta"]["fields"]["followup_window_s"]
    assert f == {"kind": "int", "label": EDITABLE_SETTINGS["followup_window_s"].label,
                 "help": EDITABLE_SETTINGS["followup_window_s"].help, "choices": None,
                 "min": 1, "max": 15, "restart": False}
    assert st["meta"]["fields"]["effort"]["choices"] == ["low", "medium", "high"]


def test_state_values_come_from_orchestrator(h):
    h.orch.language = "hi"
    h.orch.tts.voice = "bm_george"
    h.orch.tts.speed = 1.3
    h.orch.proactive.schedule.briefing_enabled = True
    st = h.bridge.get_state()
    assert st["general"]["language"] == "hi"
    assert st["voice"]["voice"] == "bm_george"
    assert st["voice"]["speed"] == 1.3
    assert st["briefings"]["briefing_enabled"] is True
    ids = [v["id"] for v in st["voice"]["voices"]]
    assert "af_sarah" in ids and "hf_alpha" in ids
    george = next(v for v in st["voice"]["voices"] if v["id"] == "bm_george")
    assert george == {"id": "bm_george", "name": "George", "hindi": False}
    assert next(v for v in st["voice"]["voices"] if v["id"] == "hf_alpha")["hindi"] is True
    assert isinstance(st["brain"]["brain_cwd"], str)
    assert st["general"]["start_at_login"] is False


def test_state_falls_back_to_prefs_while_warming():
    prefs = FakePrefs({"language": "auto", "tts_voice": "am_adam", "tts_hindi_voice": "hm_omega",
                       "tts_speed": 1.15, "hud_mode": "mini",
                       "proactive": {"briefing_enabled": True, "briefing_time": "07:30"}})
    h = Harness(warming=True, prefs=prefs)
    st = h.bridge.get_state()
    assert st["general"]["language"] == "auto"
    assert st["general"]["hud_mode"] == "mini"
    assert st["voice"] | {"voices": None} == {"voice": "am_adam", "hindi_voice": "hm_omega", "speed": 1.15, "voices": None}
    assert st["briefings"]["briefing_time"] == "07:30"


def test_state_shows_persisted_override_for_restart_fields(h):
    h.bridge.set("brain", "effort", "high")
    assert h.orch.s.effort == "low"          # restart-class: not applied live
    assert h.bridge.get_state()["brain"]["effort"] == "high"


def test_state_is_json_serialisable(h):
    import json
    json.dumps(h.bridge.get_state())


# -- set: general ---------------------------------------------------------------
def test_set_language_runs_language_turn(h):
    res = h.bridge.set("general", "language", "hi")
    assert res == {"ok": True, "restart_required": False, "message": ""}
    assert h.orch.calls == [("language", "hi")]
    assert len(h.states) == 1


def test_set_language_rejects_unknown_mode(h):
    res = h.bridge.set("general", "language", "fr")
    assert res["ok"] is False and "language" in res["message"].lower()
    assert h.orch.calls == [] and h.states == []


def test_set_start_at_login_enables_and_disables(h):
    assert h.bridge.set("general", "start_at_login", True)["ok"]
    assert h.login.calls == [("enable", Path("/tmp/Veronica.app"))]
    assert h.bridge.get_state()["general"]["start_at_login"] is True
    assert h.bridge.set("general", "start_at_login", False)["ok"]
    assert h.login.calls[-1] == ("disable",)


def test_set_start_at_login_needs_bundle():
    h = Harness(bundle=None)
    res = h.bridge.set("general", "start_at_login", True)
    assert res["ok"] is False and res["message"] == "Build the app first."
    assert h.login.calls == []


def test_set_hud_mode_emits_hud_event(h):
    assert h.bridge.set("general", "hud_mode", "mini")["ok"]
    assert h.orch.emitted == [("hud", {"mode": "mini"})]
    assert h.bridge.set("general", "hud_mode", "sideways")["ok"] is False


def test_set_ptt_enabled_is_restart_class(h):
    res = h.bridge.set("general", "ptt_enabled", False)
    assert res == {"ok": True, "restart_required": True, "message": ""}
    assert h.prefs.overrides == [("ptt_enabled", False)]
    assert h.orch.s.ptt_enabled is True
    assert h.bridge.restart_required is True
    assert h.bridge.get_state()["meta"]["restart_required"] is True


def test_set_hud_hide_after_is_live_and_clamped(h):
    res = h.bridge.set("general", "hud_hide_after_s", 99)
    assert res["ok"] and res["restart_required"] is False
    assert h.orch.s.hud_hide_after_s == 30.0
    assert h.prefs.overrides == [("hud_hide_after_s", 30.0)]


# -- set: voice ------------------------------------------------------------------
def test_set_voice_by_display_name(h):
    assert h.bridge.set("voice", "voice", "George")["ok"]
    assert h.orch.calls == [("voice", ("voice", "george"))]


def test_set_voice_by_id(h):
    assert h.bridge.set("voice", "voice", "am_adam")["ok"]
    assert h.orch.calls == [("voice", ("voice", "adam"))]


def test_set_voice_rejects_unknown_and_wrong_kind(h):
    assert h.bridge.set("voice", "voice", "Zorblax")["ok"] is False
    assert h.bridge.set("voice", "voice", "Alpha")["ok"] is False        # Hindi voice in English slot
    assert h.bridge.set("voice", "hindi_voice", "George")["ok"] is False  # English voice in Hindi slot
    assert h.orch.calls == []


def test_set_hindi_voice(h):
    assert h.bridge.set("voice", "hindi_voice", "hm_omega")["ok"]
    assert h.orch.calls == [("voice", ("voice", "omega"))]


def test_set_speed_clamps_persists_and_does_not_speak(h):
    res = h.bridge.set("voice", "speed", 9)
    assert res["ok"] and res["restart_required"] is False
    assert h.orch.tts.speed == 1.5
    assert {"tts_speed": 1.5} in h.prefs.saved
    assert h.orch.calls == []
    assert h.bridge.set("voice", "speed", "fast")["ok"] is False


def test_test_voice_copy_en_and_hi(h):
    assert h.bridge.test_voice()["ok"]
    assert h.orch.calls[-1] == ("say", "This is how I sound now.", None)
    assert h.bridge.test_voice(lang="hi")["ok"]
    assert h.orch.calls[-1] == ("say", "Main aise bolti hoon.", "hi")
    h.orch.language = "hi"
    h.bridge.test_voice()
    assert h.orch.calls[-1][2] == "hi"


# -- set: listening / brain (settings overrides) ---------------------------------
def test_set_live_setting_assigns_and_persists(h):
    res = h.bridge.set("listening", "followup_window_s", "7")
    assert res == {"ok": True, "restart_required": False, "message": ""}
    assert h.orch.s.followup_window_s == 7
    assert h.prefs.overrides == [("followup_window_s", 7)]
    assert h.bridge.restart_required is False


def test_set_live_setting_clamps(h):
    h.bridge.set("listening", "confirm_listen_s", 1000)
    assert h.orch.s.confirm_listen_s == 30


def test_set_restart_setting_latches(h):
    res = h.bridge.set("listening", "wake_min_rms", 0.02)
    assert res["restart_required"] is True
    assert h.orch.s.wake_min_rms != 0.02
    assert h.prefs.overrides == [("wake_min_rms", 0.02)]
    # a later live change still reports the latched flag
    res = h.bridge.set("listening", "followup_window_s", 5)
    assert res["restart_required"] is True


def test_set_wake_phrases_accepts_comma_string_or_list(h):
    h.bridge.set("listening", "wake_phrases", "veronica, hey veronica ,")
    assert h.prefs.overrides[-1] == ("wake_phrases", ["veronica", "hey veronica"])
    h.bridge.set("listening", "wake_phrases", ["nova"])
    assert h.prefs.overrides[-1] == ("wake_phrases", ["nova"])


def test_set_brain_fields(h):
    assert h.bridge.set("brain", "effort", "medium")["restart_required"] is True
    assert h.bridge.set("brain", "effort", "extreme")["ok"] is False
    assert h.bridge.set("brain", "memory_enabled", "off")["ok"]
    assert ("memory_enabled", False) in h.prefs.overrides
    assert h.bridge.set("brain", "brain_cwd", "/tmp")["ok"]
    assert ("brain_cwd", "/tmp") in h.prefs.overrides


def test_set_restart_class_saves_while_warming():
    h = Harness(warming=True)
    res = h.bridge.set("brain", "effort", "high")
    assert res["ok"] and res["restart_required"] is True
    assert h.prefs.overrides == [("effort", "high")]


def test_set_live_change_refused_while_warming():
    h = Harness(warming=True)
    for section, key, value in [("general", "language", "hi"), ("voice", "speed", 1.2),
                                ("listening", "followup_window_s", 5), ("general", "hud_mode", "mini"),
                                ("briefings", "briefing_enabled", True)]:
        res = h.bridge.set(section, key, value)
        assert res == {"ok": False, "restart_required": False,
                       "message": "Still starting up, try again in a moment."}, (section, key)
    assert h.prefs.overrides == [] and h.states == []


def test_set_unknown_section_or_key(h):
    res = h.bridge.set("nope", "x", 1)
    assert res["ok"] is False and res["message"]
    res = h.bridge.set("listening", "effort", "high")
    assert res["ok"] is False and res["message"]
    assert h.states == []


# -- set: briefings ------------------------------------------------------------------
def test_set_briefings_mutates_schedule_and_saves(h):
    assert h.bridge.set("briefings", "briefing_enabled", True)["ok"]
    assert h.bridge.set("briefings", "briefing_time", "07:45")["ok"]
    assert h.bridge.set("briefings", "nudges_enabled", True)["ok"]
    assert h.bridge.set("briefings", "nudge_minutes", 90)["ok"]
    sched = h.orch.proactive.schedule
    assert (sched.briefing_enabled, sched.briefing_time, sched.nudges_enabled, sched.nudge_minutes) == (
        True, "07:45", True, 60)
    assert h.prefs.saved[-1] == {"proactive": sched.to_prefs()}
    assert h.orch.calls == []  # silent


def test_set_briefing_time_validated(h):
    assert h.bridge.set("briefings", "briefing_time", "7:45")["ok"] is False
    assert h.bridge.set("briefings", "briefing_time", "25:00")["ok"] is False
    assert h.bridge.set("briefings", "nudge_minutes", "soon")["ok"] is False
    assert h.orch.proactive.schedule.briefing_time == "08:00"


def test_set_briefings_without_proactive(h):
    h.orch.proactive = None
    res = h.bridge.set("briefings", "briefing_enabled", True)
    assert res["ok"] is False and res["message"]


# -- history ---------------------------------------------------------------------
def test_history_passthrough(h):
    res = h.bridge.history(query="wea", limit=10, offset=5)
    assert res["ok"] and len(res["items"]) == 2
    assert h.store.calls == [("turns", 10, 5, "wea")]


def test_forget_and_clear(h):
    assert h.bridge.forget_turn(1) == {"ok": True, "message": ""}
    assert h.bridge.forget_turn(42)["ok"] is False
    assert h.bridge.clear_history() == {"ok": True, "count": 1, "message": ""}
    assert h.store.items == []


def test_history_without_store():
    h = Harness()
    h.bridge._store = None
    assert h.bridge.history()["ok"] is False
    assert h.bridge.forget_turn(1)["ok"] is False
    assert h.bridge.clear_history()["ok"] is False


# -- updates -----------------------------------------------------------------------
def test_check_update_none(h):
    res = h.bridge.check_update()
    assert res == {"ok": True, "available": False, "kind": "none",
                   "detail": "You're on the latest (abc1234).", "message": ""}
    assert h.updater.checks == [Path("/repo")]
    assert h.bridge.get_state()["about"]["update"] == {"available": False, "detail": "You're on the latest (abc1234)."}
    assert len(h.states) == 1


def _available():
    return SimpleNamespace(available=True, kind="remote", detail="A newer version is on origin (def5678).",
                           running_sha="abc1234", head_sha="abc1234", remote_sha="def5678")


def test_check_update_available():
    h = Harness(updater=FakeUpdater(status=_available()))
    res = h.bridge.check_update()
    assert res["available"] is True and res["kind"] == "remote"
    assert h.bridge.get_state()["about"]["update"]["available"] is True


def test_check_update_failure_is_a_message():
    h = Harness(updater=FakeUpdater(status=OSError("no git")))
    res = h.bridge.check_update()
    assert res["ok"] is False and "no git" in res["message"]


def test_update_now_runs_update_then_relaunches():
    h = Harness(updater=FakeUpdater(status=_available()))
    res = h.bridge.update_now()
    assert res == {"ok": True, "message": "Updating, back in a moment."}
    assert h.updater.updates[0][0] == Path("/repo") and h.updater.updates[0][1].kind == "remote"
    assert h.relaunches == 1


def test_update_now_none_available(h):
    res = h.bridge.update_now()
    assert res["ok"] is False and res["message"] == "You're already on the latest."
    assert h.updater.updates == [] and h.relaunches == 0


def test_update_now_busy():
    h = Harness(updater=FakeUpdater(status=_available()))
    h.orch.state = "thinking"
    assert h.bridge.update_now() == {"ok": False, "message": "Busy, try again in a moment."}
    h.orch = None
    assert h.bridge.update_now() == {"ok": False, "message": "Busy, try again in a moment."}
    assert h.updater.updates == []


def test_update_now_failure_reported_in_state():
    h = Harness(updater=FakeUpdater(status=_available(), fail=UpdateError("uv sync failed")))
    res = h.bridge.update_now()
    assert res["ok"] is True   # the thread hadn't failed when we replied
    assert h.relaunches == 0
    upd = h.bridge.get_state()["about"]["update"]
    assert upd["available"] is True
    assert "update failed" in upd["detail"].lower()
    assert h.states  # pushed so the window shows the failure


# -- restart / logs ------------------------------------------------------------------
def test_restart_relaunches(h):
    assert h.bridge.restart() == {"ok": True, "message": ""}
    assert h.relaunches == 1


def test_restart_without_bundle_says_so():
    h = Harness(bundle=None)
    res = h.bridge.restart()
    assert res["ok"] is True and res["message"] == "Restart me from the terminal."


def test_open_logs_and_login_items(h):
    assert h.bridge.open_logs()["ok"]
    assert h.opened == [h.settings.log_file]
    assert h.bridge.open_login_items()["ok"]
    assert h.opened[-1].startswith("x-apple.systempreferences:")


# -- handle ---------------------------------------------------------------------------
def test_handle_dispatches_and_reports_errors(h):
    assert h.bridge.handle("get_state", {})["general"]
    res = h.bridge.handle("set", {"section": "listening", "key": "followup_window_s", "value": 6})
    assert res["ok"] and h.orch.s.followup_window_s == 6
    assert h.bridge.handle("history", {"query": "x"})["ok"]
    assert h.bridge.handle("forget_turn", {"id": 2})["ok"]
    assert h.bridge.handle("test_voice", {"lang": "hi"})["ok"]
    assert h.bridge.handle("nope", {}) == {"ok": False, "message": "unknown command"}
    assert h.bridge.handle("set", {})["ok"] is False        # missing args → message, not raise
    h.store.turns = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    res = h.bridge.handle("history", {})
    assert res == {"ok": False, "message": "boom"}


def test_bridge_imports_no_appkit():
    import ast

    import veronica.ui.settings.bridge as mod

    tree = ast.parse(Path(mod.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert not {n for n in names if n.split(".")[0] in ("AppKit", "WebKit", "objc", "Foundation", "rumps")}
