# Veronica Batch D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Settings window (with a History tab) that edits everything a user should be able to tune, plus version display and a one-click/voice "update & restart".

**Architecture:** D0 adds persisted `Settings` overrides (`prefs.json["settings"]`) and a whitelist with UI metadata. D3 adds pure modules `version.py`, `updater.py`, `ui/relaunch.py` (all `subprocess.run`-injected). D2 adds history queries to `MemoryStore`. D1 is a pure `SettingsBridge` (commands ↔ orchestrator/prefs/store/updater) wrapped by a thin `SettingsWindow` (NSWindow + WKWebView + script-message handler) and a web page. Wiring: intents → orchestrator emits a `settings` event → menubar shows the window; menubar gets About/Settings/Check-for-Updates items + hourly check.

**Tech Stack:** Python 3.12, uv, pytest, PyObjC (AppKit/WebKit), rumps, Playwright (headless Chromium, `live` marker), SQLite FTS5, git CLI.

**Spec:** `docs/superpowers/specs/2026-09-17-veronica-batch-d-design.md`

## Global Constraints

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` is the only thing that may auto-allow a tool. Never set `allowed_tools`.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q`. No test touches mic/speakers/models/network/AppleScript/git remotes/real AppKit windows; every subprocess call goes through an injected `run` callable; Playwright page tests carry `@pytest.mark.live` (deselected by default; implementers run them explicitly once with `uv run pytest -q -m live tests/test_settings_web.py`).
- Follow existing module patterns (`veronica/ui/hud/__init__.py`, `veronica/ui/menubar.py`, `veronica/prefs.py`, `veronica/brain/intents.py`). Voice copy exactly as written.
- Branch `batch-d` (created; spec committed).

## File map

| File | Responsibility |
|---|---|
| `veronica/config.py` | `EditableField`, `EDITABLE_SETTINGS`, `load_settings`, `validate_assignment` |
| `veronica/prefs.py` | `get`, `save_settings_override`, `clear_settings_override` |
| `veronica/version.py` (new) | `APP_VERSION`, `build_info`, `describe` |
| `veronica/updater.py` (new) | `UpdateStatus`, `check`, `update` |
| `veronica/ui/relaunch.py` (new) | `relaunch` |
| `scripts/build_app.py` | writes `Contents/Resources/build.json`; launcher exports `VERONICA_BUNDLE_BUILD` |
| `veronica/memory/store.py` | `turns`, `delete_turn`, `clear_turns` |
| `veronica/ui/settings/bridge.py` (new) | `SettingsBridge` (pure) |
| `veronica/ui/settings/__init__.py` (new) | `SettingsWindow` (AppKit/WebKit, factories) |
| `veronica/ui/settings/{index.html,settings.css,settings.js}` (new) | the page |
| `veronica/brain/intents.py` | `match_settings_intent`, `match_version_intent`, `match_update_intent` |
| `veronica/orchestrator.py` | dispatch + `_version_turn`, `_update_turn`, `settings` event |
| `veronica/ui/menubar.py` | About/Settings…/Check for Updates… items, `settings` event → window, hourly check, popup mirror |
| `README.md` | Settings window, History, Updates |

---

### Task 1: D0 — settings overrides + editable-field whitelist

**Files:** modify `veronica/config.py`, `veronica/prefs.py`; tests `tests/test_config.py`, `tests/test_prefs.py`.

**Interfaces (produces):**
```python
@dataclass(frozen=True)
class EditableField:
    kind: Literal["bool", "int", "float", "str", "choice", "list"]
    label: str
    help: str = ""
    choices: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    restart: bool = True

EDITABLE_SETTINGS: dict[str, EditableField] = {
    "followup_window_s": EditableField("int", "Follow-up window (seconds)", "How long she keeps listening after answering.", min=1, max=15, restart=False),
    "confirm_listen_s": EditableField("int", "Confirmation timeout (seconds)", "How long she waits for yes/no.", min=3, max=30, restart=False),
    "hud_hide_after_s": EditableField("float", "Hide HUD after (seconds)", "", min=1, max=30, restart=False),
    "wake_min_rms": EditableField("float", "Wake sensitivity (min level)", "Lower = more sensitive; raise if she wakes on noise.", min=0.002, max=0.05),
    "wake_phrases": EditableField("list", "Wake phrases", "Comma-separated; 'veronica' is recommended."),
    "ptt_enabled": EditableField("bool", "Push-to-talk (hold Right Option)"),
    "effort": EditableField("choice", "Brain effort", "Higher is smarter and slower.", choices=("low", "medium", "high")),
    "memory_enabled": EditableField("bool", "Remember conversations"),
    "brain_cwd": EditableField("str", "Working folder", "Where shell commands run."),
}

def coerce_setting(name: str, value: Any) -> Any:  # clamps numbers, splits/strips lists, validates choices; raises ValueError
def load_settings(overrides: dict | None = None) -> Settings
```
`Settings.model_config` gains `validate_assignment=True`. `prefs.get(key, default)`, `prefs.save_settings_override(field, value)`, `prefs.clear_settings_override(field)`; `config.settings = load_settings(prefs.load().get("settings"))` — careful: `prefs` must not import `config` (it doesn't). `load_settings` drops unknown/invalid keys with `log.warning`, never raises.

- [ ] Tests first (`tests/test_config.py`): overrides applied (`load_settings({"effort": "high"}).effort == "high"`), unknown key ignored, invalid value ignored (`{"followup_window_s": "abc"}` → default), `coerce_setting` clamps (`followup_window_s`, 99 → 15; `wake_min_rms`, 0.0001 → 0.002), list from `"veronica, hey veronica"` → `["veronica", "hey veronica"]`, choice invalid → `ValueError`, every `EDITABLE_SETTINGS` key is a `Settings` field, `validate_assignment` rejects `s.followup_window_s = "x"`. (`tests/test_prefs.py`): override save merges under `"settings"`, clear removes only that key, `get` default.
- [ ] Implement; `uv run pytest -q`; commit `feat(config): persisted settings overrides with an editable-field whitelist`.

---

### Task 2: D3 — version, updater, relaunch, build.json

**Files:** create `veronica/version.py`, `veronica/updater.py`, `veronica/ui/relaunch.py`; modify `scripts/build_app.py`; tests `tests/test_version.py`, `tests/test_updater.py`, `tests/test_relaunch.py`, `tests/test_build_app.py`.

**Interfaces (produces):**
```python
# version.py
APP_VERSION: str                      # importlib.metadata.version("veronica") or "0.1.0"
def build_info(run=subprocess.run, env=os.environ, repo: Path = REPO) -> dict   # {"sha": "a517483", "built_at": "2026-09-17T00:00:48+05:30", "dirty": False, "source": "bundle"|"git"|"unknown"}
def describe(info: dict | None = None) -> str   # "Veronica 0.1.0 (a517483, 17 Sep)"; "(unknown build)" when sha missing
# updater.py
@dataclass
class UpdateStatus: available: bool; kind: Literal["remote","local","none"]; detail: str; running_sha: str; head_sha: str; remote_sha: str | None
def check(repo: Path, run=subprocess.run, info: dict | None = None) -> UpdateStatus
def update(repo: Path, status: UpdateStatus, run=subprocess.run, build=None) -> str   # returns a short log; raises UpdateError on failure
class UpdateError(RuntimeError): ...
# relaunch.py
def relaunch(bundle_path: Path | None, quit: Callable[[], None], popen=subprocess.Popen) -> bool   # True if a relaunch was scheduled
```
Semantics: `build_info` reads `env["VERONICA_BUNDLE_BUILD"]` JSON file if set and readable (`source="bundle"`), else git (`rev-parse --short HEAD`, `log -1 --format=%cI`, `status --porcelain`; any failure → `{"sha": "", "built_at": "", "dirty": False, "source": "unknown"}`). `check`: `git remote get-url origin` ok → `git fetch --quiet origin` (timeout 10) → `git rev-parse --short origin/<branch>` where branch = `git rev-parse --abbrev-ref HEAD`; `remote` if `remote_sha != head_sha`; else `local` if `info["sha"] and info["sha"] != head_sha`; else `none`. Detail strings: `"A newer version is on origin ({remote_sha})."`, `"Restart to run the latest code ({head_sha})."`, `"You're on the latest ({head_sha})."`; any git failure → `none` with `detail="Couldn't check: <stderr>"`. `update`: for `remote` run `git pull --ff-only --quiet`; then `uv sync --frozen` if `shutil.which("uv")` (injectable `which`); then `build()` (default `scripts.build_app.build_app` imported lazily); non-zero exit → `UpdateError(stderr)`. `relaunch`: if `bundle_path` → `popen(["/bin/sh", "-c", f'sleep 1; open -n "{bundle_path}"'], start_new_session=True)` then `quit()`, return True; else `quit()`, return False.
`build_app`: write `Contents/Resources/build.json` from `version.build_info(run)` (git source, computed at build time); launcher script exports `VERONICA_BUNDLE_BUILD="<app>/Contents/Resources/build.json"`.

- [ ] Tests first with a fake `run` that records argv and returns scripted `CompletedProcess`es (stdout/returncode); cover bundle vs git vs unknown, `describe` formatting (`"17 Sep"` from the ISO date), each `check` branch incl. no remote, fetch failure, `update` remote/local/failure and command order, `relaunch` both branches; `test_build_app.py`: build.json exists with a sha and the launcher contains `VERONICA_BUNDLE_BUILD`.
- [ ] Implement; full suite; commit `feat: version/build info, update checker and relaunch helper`.

---

### Task 3: D2 — history queries in `MemoryStore`

**Files:** modify `veronica/memory/store.py`; tests `tests/test_memory_store.py`.

**Interfaces:** `turns(limit: int = 200, offset: int = 0, query: str = "") -> list[dict]` (`{"id", "ts", "heard", "reply"}`, newest first; with `query`: FTS `MATCH` when `fts_enabled` else `LIKE '%q%'` on heard/reply; FTS query escaped the way `search()` does it), `delete_turn(turn_id: int) -> bool`, `clear_turns() -> int` (turns + turns_fts).

- [ ] Tests first: ordering/limit/offset, query hits both columns, FTS off path (construct store with `fts_enabled=False` the way existing tests do), delete returns True/False, clear count and `recent()` empty after.
- [ ] Implement; full suite; commit `feat(memory): history listing, delete and clear`.

---

### Task 4: D1 — `SettingsBridge` (pure)

**Files:** create `veronica/ui/settings/__init__.py` (empty for now) and `veronica/ui/settings/bridge.py`; tests `tests/test_settings_bridge.py`.

**Interfaces (produces):**
```python
class SettingsBridge:
    def __init__(self, *, settings, get_orch: Callable[[], Any | None], store, run_on_loop: Callable[[Coroutine], None],
                 prefs=prefs, login_item=login_item, version=version, updater=updater,
                 relaunch: Callable[[], bool], bundle_path: Path | None, repo: Path, run_thread: Callable[[Callable[[], None]], None] = _thread) -> None
    def handle(self, cmd: str, args: dict) -> dict          # dispatch; unknown cmd → {"ok": False, "message": "unknown command"}; exceptions → {"ok": False, "message": str(e)} + log
    def get_state(self) -> dict
    def set(self, section: str, key: str, value) -> dict     # {"ok", "restart_required", "message"}
    def test_voice(self) -> dict
    def history(self, query: str = "", limit: int = 200, offset: int = 0) -> dict   # {"ok", "items"}
    def forget_turn(self, id: int) -> dict
    def clear_history(self) -> dict
    def check_update(self) -> dict                            # synchronous check via updater.check; caches status for get_state
    def update_now(self) -> dict                              # refuses unless orch.state == "idle" ("Busy, try again in a moment."); runs updater.update in run_thread, then relaunch(); returns {"ok": True, "message": "Updating, back in a moment."}
    def restart(self) -> dict
    def open_logs(self) -> dict                               # subprocess open of settings.log_file (injectable)
    restart_required: bool
    on_state_changed: Callable[[dict], None] | None          # window sets this to push state to JS
```
Applying rules are in the spec (D1 "Applying"). Speed clamps via `voices.clamp_speed`; `voice.*` names resolve via `voices.resolve_voice` (bridge passes the display name lowercased); `briefings.briefing_time` validated `HH:MM`; `nudge_minutes` 1–60; `listening.wake_phrases` accepts a comma string or list. Every successful `set` calls `on_state_changed(get_state())`. When `get_orch()` is None (still warming) live changes return `{"ok": False, "message": "Still starting up, try again in a moment."}` but persistence-only changes (restart-class) still save.

- [ ] Tests first with fakes: orch (`state`, `s` = real `Settings()`, `tts` with voice/hindi_voice/speed, `proactive.schedule` real `Schedule`, async `_language_turn/_voice_turn/say` recorders, `_emit` recorder, `language`), prefs recorder (`save`, `save_settings_override`, `load`), store with `turns/delete_turn/clear_turns` recorders, login_item stub (`is_enabled/enable/disable`), version/updater stubs, `run_on_loop` that runs the coroutine to completion, `run_thread` that runs inline. Assert state shape (all sections/keys), each `set` path (live vs restart, clamps, invalid → message), voice test copy in en/hi, history passthrough, update flows (none/available/busy/failure), restart → relaunch called, `on_state_changed` invoked.
- [ ] Implement; full suite; commit `feat(settings): pure settings bridge`.

---

### Task 5: D1 — `SettingsWindow` + web page

**Files:** modify `veronica/ui/settings/__init__.py`; create `veronica/ui/settings/index.html`, `settings.css`, `settings.js`; tests `tests/test_settings_window.py`, `tests/test_settings_web.py` (live/Playwright). Add the three web files to `pyproject` package data if the HUD files needed that (check how `veronica/ui/hud/index.html` ships — `importlib.resources.files("veronica.ui.hud")`; mirror for `veronica.ui.settings`).

**Interfaces:** `SettingsWindow(settings, bridge, *, webview_factory=None, window_factory=None, main=_main_thread)` with `available`, `show(tab: str = "general")`, `hide()`, `push_state(state: dict)`, `_on_message(body: dict)` (called by the script handler on the main thread: `{"id", "cmd", "args"}` → `bridge.handle` → `_js(f"window.settings.reply({id}, {json})")`; long commands `check_update`/`update_now` run on a thread via `bridge.run_thread` and reply when done), JS deferred until page load (same `_pending_js` pattern as HUD), `bridge.on_state_changed = self.push_state`. Script message handler class built lazily (`_make_message_handler_class()` → `NSObject` subclass implementing `userContentController_didReceiveScriptMessage_`), registered on `WKWebViewConfiguration().userContentController()` with name `"veronica"`. Window: `NSWindow` titled "Veronica Settings", 720×520, `center()`, `setReleasedWhenClosed_(False)`, delegate `windowShouldClose_` → `orderOut_` + return False; `show()` → `makeKeyAndOrderFront_` + `NSApp.activateIgnoringOtherApps_(True)` then `_js("window.settings.select(<tab>)")` and `push_state`.

Web page contract (`settings.js`): `window.settings = { state(json), reply(id, json), select(tab) }`; sidebar tabs; renders sections from `state.meta.fields` + values; controls post `{id, cmd:"set", args:{section,key,value}}` via `window.webkit.messageHandlers.veronica.postMessage` when available, else (Playwright) into `window.__settings.sent` array and resolve replies via `window.settings.reply`; banner when `state.meta.restart_required`; History tab: search input (debounce 200 ms) → `history` cmd, rows with Forget (→ `forget_turn`), "Clear all" → in-page confirm bar (Yes/No) → `clear_history`; About: version/build/dirty, buttons Check now / Update & restart / Restart / Open log (cmds `check_update`, `update_now`, `restart`, `open_logs`), status line from replies. Test hooks: `window.__settings = {sent: [], state: () => model}`.

- [ ] Tests first: `tests/test_settings_window.py` mirrors `tests/test_hud_window.py` (fake webview records `evaluateJavaScript_completionHandler_` calls, fake window records `makeKeyAndOrderFront_`/`orderOut_`): show pushes state + select, JS queued until `_on_loaded()`, `_on_message` dispatches to a fake bridge and replies with the id, long commands go through `run_thread`, `hide`. `tests/test_settings_web.py` (`@pytest.mark.live`): load `index.html`, call `window.settings.state(<fixture state>)`, assert all seven tabs render, change a select → `__settings.sent[-1]` has the right `{cmd:"set", args}`, `window.settings.reply(id, {ok:true, restart_required:true})` shows the banner, History: `window.settings.reply` with items renders rows, Forget posts `forget_turn`, Clear shows in-page confirm and posts `clear_history` on Yes.
- [ ] Implement; full suite + `uv run pytest -q -m live tests/test_settings_web.py`; commit `feat(settings): settings window (WKWebView + bridge) and page`.

---

### Task 6: Wiring — intents, orchestrator, menubar, README

**Files:** modify `veronica/brain/intents.py`, `veronica/orchestrator.py`, `veronica/ui/menubar.py`, `veronica/__main__.py` (if the store/bundle path must be exposed), `README.md`; tests `tests/test_intents.py`, `tests/test_orchestrator.py`, `tests/test_menubar.py`.

**Interfaces:**
- `match_settings_intent(text) -> str | None` returning the tab: `open settings|show settings|settings|preferences|open preferences|settings kholo|setting kholo` → `"general"`; `show history|show my history|what did i ask you|what did i ask you earlier|history|history dikhao|conversation history` → `"history"`; whole-utterance candidates only; negatives `"open safari settings"`, `"history of rome"` → None.
- `match_version_intent(text) -> bool`: `what version are you|which version are you|what version|version|your version|kaunsa version hai`; negatives `"what version of python is installed"` → False.
- `match_update_intent(text) -> bool`: `update yourself|update now|check for updates|check for an update|apna update karo|update karo`; negatives `"update my calendar"`, `"update the note"` → False.
- Orchestrator: constructor kwargs `updater_check: Callable[[], UpdateStatus] | None = None`, `updater_update: Callable[[UpdateStatus], str] | None = None`, `relaunch: Callable[[], bool] | None = None` (all injected from `__main__`/menubar; None in tests/text mode). Dispatch order: after `lang_mode` and before `proactive_action`: `settings_tab` → `self._emit("settings", {"open": True, "tab": tab}); await self.say("Here you go.")` (Hindi utterance → "Yeh lijiye."); `version` → `say(version.describe())`; `update` → `_update_turn()`: if `updater_check is None` → "Updates aren't available in this mode."; `st = await asyncio.to_thread(updater_check)`; `none` → "You're already on the latest."; `local`/`remote` → say "Updating, back in a moment.", `_emit("tool", {"summary": "Update Veronica", "decision": "auto"})`, `await asyncio.to_thread(updater_update, st)` in try → on success `self.relaunch()` (which quits); on exception log + "The update failed, check the log." Note: the check runs while a turn is in progress by definition (this IS the turn), so the bridge's "idle" rule doesn't apply here; a barge cancels it like any turn.
- Menubar: items at top: disabled `About Veronica — <describe()>`, `Settings…` (→ `self._settings.show("general")`), `Check for Updates…` (→ thread: `updater.check`; result → `rumps.notification("Veronica", "", detail)` and, if available, `self._update_item.title = "Update available — Restart to update"` enabled (→ `bridge.update_now`)); hourly `rumps.Timer(3600)` check that only updates the item; `settings` event kind in `_drain` → `self._settings.show(tab)`; popup mirror gets Settings…/About. Build the bridge+window in `__init__` with `get_orch=lambda: self._orch`, `run_on_loop=self._schedule`, `store` from the orchestrator when available (`getattr(self._orch, "store", None)` at call time — pass a callable), `bundle_path=login_item.bundle_app_path()`, `repo=REPO` (from `scripts.build_app` or `Path(__file__).parents[2]`), `relaunch=lambda: relaunch(bundle_path, self._schedule_quit)`. Pass `updater_check=lambda: updater.check(repo, info=version.build_info())`, `updater_update=lambda st: updater.update(repo, st)`, `relaunch=...` into `build_orchestrator` → `Orchestrator`.
- README: "Settings window" (how to open, what's live vs restart), "History", "Version & updates" (what "update yourself" does; no remote = restart-to-latest-code).

- [ ] Tests first per the contracts above (menubar tests follow the file's `fake_env`/`_make_app` idiom; assert item titles/order, that the `settings` event shows the window with the tab, timer registered, popup has Settings…). Orchestrator tests: settings intent emits event + says copy and skips brain; version says describe (monkeypatch `version.describe`); update turn copy for none/available/failure/unavailable, relaunch called on success.
- [ ] Implement; full suite; commit `feat: settings/history/version/update wired into voice, menu and app`.

---

## Self-review
- Spec coverage: D0 → T1; D3 → T2 (+T6 voice/menu); D2 → T3 (+T5 UI); D1 → T4/T5 (+T6 opening); README → T6.
- Placeholders: T4–T6 are contract-style (signatures + copy + test expectations) by design; implementers read the spec sections named.
- Type consistency: `SettingsBridge.handle(cmd, args)` is what `SettingsWindow._on_message` calls; `on_state_changed` set by the window; `relaunch(bundle_path, quit)` signature used by both bridge (via lambda) and orchestrator; `UpdateStatus` shared by updater/bridge/orchestrator; `turns()` dict keys match the JS rows.
