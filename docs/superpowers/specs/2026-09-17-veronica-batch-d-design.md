# Veronica Batch D — Settings window, History, Version & update

Date: 2026-09-17. Follows Batch C. Branch `batch-d`, merged to `master` as one unit.

## Global constraints (apply to every task)

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` is the only thing that may auto-allow a tool. Never set `allowed_tools`.
- Speech stays local/zero-key.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q` (asyncio auto, `live` marker deselected). No test touches the mic, speakers, network, models, AppleScript, git remotes or a real AppKit window; the settings page's JS is tested with Playwright headless like `tests/test_hud_web.py`.
- Python 3.12, `uv`. Follow existing module patterns (`veronica/ui/hud/__init__.py` for the WKWebView window, `veronica/ui/menubar.py` for menu/popup items, `veronica/prefs.py` for persistence, `veronica/brain/intents.py` for local intents).
- Voice UX copy: short, friendly, spoken — no markdown.

---

## D0 — Persisted settings overrides (foundation)

- `prefs.json` gains a `"settings"` dict: `{<Settings field name>: value}` overrides applied on top of env/defaults when the process-wide `settings` is built. `veronica/config.py`: `def load_settings(overrides: dict | None = None) -> Settings` — builds `Settings(**{k: v for k, v in overrides.items() if k in EDITABLE_SETTINGS})`, logging and skipping invalid values (pydantic `ValidationError` → drop that key, retry). Module-level `settings = load_settings(prefs.load().get("settings"))` (prefs import is cheap; keep `Settings()` importable for tests).
- `EDITABLE_SETTINGS: dict[str, EditableField]` — the whitelist the settings window may write, with metadata for the UI: `EditableField(kind: "bool"|"int"|"float"|"str"|"choice"|"list", label: str, help: str, choices: list[str] | None, min: float | None, max: float | None, restart: bool)`. Fields (voice/speed/hindi voice/hud_mode/language/proactive already live in their own prefs keys and stay there — they are NOT settings overrides): `effort` (choice low/medium/high, restart), `memory_enabled` (bool, restart), `brain_cwd` (str path, restart), `followup_window_s` (int 1–15, live: orchestrator reads `self.s`, so assigning `orch.s.followup_window_s` applies immediately — Settings is a pydantic model; use `model_copy(update=...)` or `object.__setattr__`; ruling: `Settings` gets `model_config = SettingsConfigDict(validate_assignment=True, frozen=False)` and live fields are assigned on `orch.s`), `confirm_listen_s` (int 3–30, live), `wake_min_rms` (float 0.002–0.05, restart), `wake_phrases` (list of str, restart), `ptt_enabled` (bool, restart), `hud_hide_after_s` (float 1–30, live), `vad_silence_ms` (int 300–3000, live — Recorder recomputes per capture; default bumped 600→1200), `max_utterance_s` (int 5–60, live).
- `veronica/prefs.py`: `get(key, default)` helper and `save_settings_override(field, value)` (merges into `"settings"`); `clear_settings_override(field)`.

## D1 — Settings window (`veronica/ui/settings/`)

- **Window**: `SettingsWindow` in `veronica/ui/settings/__init__.py` — a normal titled, closable, resizable `NSWindow` (720×520, centered, `NSWindowStyleMaskTitled|Closable|Miniaturizable|Resizable`, level normal, `activateIgnoringOtherApps_` when shown, hidden on close not destroyed) hosting a `WKWebView` loading `veronica/ui/settings/index.html` (+ `settings.css`, `settings.js`). Same lazy-PyObjC/factory pattern as `HudWindow` (`webview_factory`, `window_factory` for tests; `available` flag). JS→Python bridge: `WKUserContentController.addScriptMessageHandler_name_(handler, "veronica")`; `window.webkit.messageHandlers.veronica.postMessage({id, cmd, args})`; Python replies with `window.settings.reply(id, result)` via `evaluateJavaScript`. Python→JS push: `window.settings.state(json)` on show and after any change.
- **Bridge commands** (`SettingsBridge` in `veronica/ui/settings/bridge.py`, pure Python, testable without AppKit; the window just marshals): `get_state()` → dict of all sections; `set(section, key, value)` → applies + persists, returns `{ok, restart_required, message}`; `test_voice()`; `history(query, limit, offset)`; `forget_turn(id)`; `clear_history()`; `check_update()`; `update_now()`; `restart()`; `open_logs()`; `open_login_items()` (System Settings). The bridge holds references to: `settings`, the orchestrator (may be None while warming), `prefs`, `MemoryStore`, `login_item`, `version`/`updater` modules, and a `run_on_loop(coro)` callable (the menubar's `_schedule`) for orchestrator calls.
- **Sections & state shape** (`get_state()`):
  - `general`: `language`, `start_at_login` (login_item.is_enabled), `ptt_enabled`, `hud_mode`.
  - `voice`: `voice`, `hindi_voice`, `speed`, `voices` (list of `{id, name, hindi}`).
  - `listening`: `followup_window_s`, `confirm_listen_s`, `wake_min_rms`, `wake_phrases`.
  - `briefings`: `briefing_enabled`, `briefing_time`, `nudges_enabled`, `nudge_minutes`.
  - `brain`: `effort`, `memory_enabled`, `brain_cwd`.
  - `about`: `version` (string), `build` (sha), `built_at`, `update` (`{available: bool, detail: str}` last known), `log_path`.
  - `meta`: `restart_required: bool` (any restart-class setting changed this session), `fields` (the `EDITABLE_SETTINGS` metadata for rendering).
- **Applying** (`set`): 
  - `general.language` → `run_on_loop(orch._language_turn(mode))` (speaks the confirmation; the window shows "Applying…" until `state` push).
  - `general.start_at_login` → `login_item.enable/disable` (reuse menubar logic; if no bundle path → message "Build the app first").
  - `general.hud_mode` → `orch._emit("hud", {"mode": ...})` equivalent (menubar's `toggle_hud_mode` path).
  - `voice.voice`/`hindi_voice` → `run_on_loop(orch._voice_turn(("voice", name)))`; `voice.speed` → set `orch.tts.speed = clamp`, `prefs.save({"tts_speed": …})` (no speech); `test_voice` → `run_on_loop(orch.say("This is how I sound now."))` (Hindi voice test says "Main aise bolti hoon.").
  - `briefings.*` → mutate `orch.proactive.schedule` + `save_schedule`.
  - `listening.followup_window_s`/`confirm_listen_s`, `brain.*`, `wake_*`, `ptt_enabled` → `prefs.save_settings_override`; live ones also assigned onto `orch.s`; restart-class set `restart_required`.
- **UI** (`settings.html/js/css`): left sidebar tabs (General, Voice, Listening, Briefings, Brain, History, About), right pane rendered from state; controls: toggles, selects, range slider with value, text inputs (commit on Enter/blur), a "Restart Veronica to apply" banner with a Restart button when `restart_required`. Dark holo theme reusing the HUD palette. Keyboard works (normal window). `window.settings = {state(json), reply(id, json)}`; `settings.js` exposes `window.__settings` test hooks like `hud.js` does for Playwright.
- **Opening**: orb popup + menu bar item "Settings…" (Cmd+, not needed); voice intents `open settings`, `show settings`, `settings`, `preferences`, `settings kholo`, `setting kholo` → `("open_settings", None)` handled locally: `_emit("settings", {"open": True, "tab": "general"})`; the menubar listens for the `settings` event kind and shows the window on the main thread. `show history` / `what did i ask you` / `history dikhao` → `("open_settings", "history")`.
- **Restart**: `restart()` = relaunch the app bundle if running from one (`open -n <bundle>` after quitting via `on_quit`), else just quit (spoken "Restart me from the terminal."). Implemented in `veronica/ui/relaunch.py`: `relaunch(bundle_path: Path | None, quit: Callable)` — spawns `/bin/sh -c 'sleep 1; open "<bundle>"'` detached then calls `quit()`.

## D2 — History tab

- `MemoryStore` additions: `turns(limit=200, offset=0, query="") -> list[dict(id, ts, heard, reply)]` (FTS when query and `fts_enabled`, else `LIKE`), `delete_turn(id) -> bool`, `clear_turns() -> int` (also FTS rows). Existing `recent()` untouched.
- Bridge: `history(query, limit, offset)`, `forget_turn(id)`, `clear_history()` (JS confirms first — an in-page confirm, NOT `window.confirm`).
- UI: search box (debounced 200 ms), list rows `HH:MM · date` / "You: …" / "Veronica: …", per-row "Forget" button, "Clear all" button, empty state "Nothing yet." Facts are NOT shown here (out of scope).

## D3 — Version & update

- `veronica/version.py`: `APP_VERSION` from `pyproject` (`importlib.metadata.version("veronica")` fallback `"0.1.0"`); `build_info() -> dict(sha, built_at, dirty)` read from `veronica/_build.json` if present (written by `build_app` — keep the file gitignored; when absent, computed live from `git` in the repo: `git rev-parse --short HEAD`, `git log -1 --format=%cI`, `git status --porcelain` non-empty → dirty). `describe() -> "Veronica 0.1.0 (a517483, 17 Sep)"`.
- `scripts/build_app.py`: writes `veronica/_build.json` `{sha, built_at, dirty}` into the bundle's copy of the package (the bundle runs the repo's venv/package directly today — ruling: write `dist/Veronica.app/Contents/Resources/build.json` and have `build_info()` look for `$VERONICA_BUNDLE_BUILD` env (exported by the launcher script pointing at that file) before falling back to git). Launcher exports `VERONICA_BUNDLE_BUILD`.
- `veronica/updater.py`: `check(repo: Path, run=subprocess.run) -> UpdateStatus(available: bool, kind: "remote"|"local"|"none", detail: str, running_sha: str, head_sha: str, remote_sha: str | None)`: if `git remote get-url origin` succeeds → `git fetch --quiet origin` (10 s timeout) and compare `origin/master` (or the tracking branch) vs HEAD → `remote` when ahead; then compare `build_info().sha` vs `HEAD` → `local` ("Restart to run the latest code") when they differ; else `none`. `update(repo, run) -> str` (log text): `git pull --ff-only` (only when kind remote), `uv sync --frozen` (skip if no `uv`), `build_app()`; then the caller relaunches. Never runs while a turn is in progress (bridge checks `orch.state == "idle"`, else message "Busy, try again in a moment.").
- Menubar: "About Veronica — 0.1.0 (a517483)" disabled item at the top; "Check for Updates…" item → runs check on a thread, shows result via `rumps.notification`/spoken announce ("An update is ready. Say update yourself, or use the Settings window."). Hourly background check (`rumps.Timer` 3600 s) sets a menu badge "Update available" item.
- Voice intents: `what version are you`, `which version`, `version` → say `describe()`; `update yourself`, `update now`, `check for updates`, `apna update karo` → `("update", None)`: check → if none: "You're already on the latest."; if available and idle: say "Updating, back in a moment.", run `update()` in a thread, then relaunch; on failure: "The update failed, check the log."
- About tab: version/build/dirty flag, `Check now`, `Update & restart`, `Restart`, `Open log`.
- README: "Settings window", "History", "Updates" sections.

## Tests
- `tests/test_config.py`: `load_settings` overrides/whitelist/invalid handling; `EDITABLE_SETTINGS` metadata sanity (every key is a real field, choices valid).
- `tests/test_prefs.py`: `save_settings_override`/`clear`.
- `tests/test_settings_bridge.py`: every command with fakes (orch with `_language_turn`/`_voice_turn`/`say`/`tts`/`proactive`/`s`, prefs recorder, store, login_item stub, updater stub, `run_on_loop` recorder) — state shape, live vs restart classes, clamping, error messages.
- `tests/test_settings_window.py`: factories like `test_hud_window.py` — show/hide, JS deferred until load, message handler dispatch to bridge, reply marshalling; `_main` stub.
- `tests/test_settings_web.py` (Playwright headless, marker like `test_hud_web.py`): renders every tab from a fixture state, edits post the right messages, banner appears on restart-class change, history search debounce + forget flow + clear confirmation.
- `tests/test_memory_store.py`: `turns` with query/offset, `delete_turn`, `clear_turns`.
- `tests/test_version.py`, `tests/test_updater.py` (fake `run` recording argv; remote/local/none outcomes; failure paths), `tests/test_relaunch.py`.
- `tests/test_intents.py`: settings/history/version/update phrases + negatives (`"open safari settings"` → None, `"what version of python"` → None).
- `tests/test_orchestrator.py`: settings intent emits the event; version turn speaks; update turn copy incl. busy/none/failure (updater + relaunch stubbed).
- `tests/test_menubar.py`: About item text, Settings… item, settings-event → show window, update check timer wiring, popup mirror.
- `tests/test_build_app.py`: build.json written; launcher exports `VERONICA_BUNDLE_BUILD`.

## Order & branching
`batch-d` off master: D0 → D3 (version/updater/relaunch, no UI) → D2 store → D1 bridge → D1 window+web → menubar/intents/orchestrator wiring → README. Full suite green per task; reviewer per task; whole-branch review; merge `--no-ff`; `make app`; relaunch.
