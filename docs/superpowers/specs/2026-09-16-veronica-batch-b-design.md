# Veronica Batch B — Voice choice & speed, Proactive briefings & nudges, Browser control

Date: 2026-09-16. Follows Batch A (`2026-09-16-veronica-batch-a-design.md`).
Implemented on branch `batch-b`, merged to `master` as one unit.

## Global constraints (apply to every task)

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` is the only thing that may auto-allow a tool. Never set `allowed_tools`.
- Speech stays local/zero-key (faster-whisper STT, Kokoro TTS).
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q` (asyncio auto, `live` marker deselected). No test may touch the real mic, speakers, network, AppleScript or a real browser — `osascript`/`subprocess` are monkeypatched.
- Python 3.12, `uv`. Follow existing module patterns (`veronica/tools/pim.py` for MCP tools, `veronica/brain/intents.py` for local intents).
- Voice UX copy: short, friendly, spoken — one sentence, no markdown.

---

## B1 — Voice choice & speed

### Goal
"Veronica, use a male voice" / "use a British voice" / "speak faster" / "speak slower" / "normal speed" / "default voice" change how she sounds, immediately and persistently.

### Design
- `veronica/speech/tts.py` `Synthesizer` gains mutable `voice: str` and `speed: float` (default 1.0) attributes; `synth()` passes `speed=self.speed` (was hardcoded `1.0`). Kokoro speed clamp: `0.7 ≤ speed ≤ 1.5`.
- New `veronica/speech/voices.py`:
  - `VOICES: dict[str, str]` — spoken name → Kokoro id, all present in `voices-v1.0.bin`:
    `"sarah": "af_sarah"`, `"bella": "af_bella"`, `"nicole": "af_nicole"`, `"sky": "af_sky"`,
    `"adam": "am_adam"`, `"michael": "am_michael"`,
    `"emma": "bf_emma"`, `"isabella": "bf_isabella"`, `"george": "bm_george"`, `"lewis": "bm_lewis"`.
  - `DEFAULT_VOICE = "af_sarah"`, `DEFAULT_SPEED = 1.0`, `SPEED_STEP = 0.15`, `SPEED_MIN = 0.7`, `SPEED_MAX = 1.5`.
  - `resolve_voice(request: str) -> str | None` — maps a request phrase to a Kokoro id: an exact spoken name (`"adam"`), or a descriptor combo: `male`/`man`/`female`/`woman` × `british`/`english`/`uk`/`american`/`us`. Descriptor picks the first matching id (prefix `am_`/`af_`/`bm_`/`bf_`); `"male"` alone → `"am_adam"`, `"british"` alone → `"bf_emma"`, `"british male"` → `"bm_george"`, `"default"` → `DEFAULT_VOICE`. Unknown → `None`.
  - `display_name(voice_id: str) -> str` — `"af_sarah"` → `"Sarah"`.
- Persistence: `prefs.json` keys `"tts_voice"` and `"tts_speed"`. `veronica/__main__.py` `build_orchestrator` reads them (falling back to `Settings.kokoro_voice` / 1.0) when constructing `Synthesizer`. Orchestrator saves via `prefs.save` on every change.
- Local intents in `veronica/brain/intents.py` (handled before the brain, like `mute`):
  - `match_voice_intent(text) -> VoiceAction | None` where `VoiceAction = ("voice", <request>) | ("speed", "faster"|"slower"|"normal")`.
  - Voice phrases (after the existing `normalize()` + wake-word strip): `^(?:use|switch to|change to|speak (?:in|with)) (?:a |the )?(.+?) voice$` → `("voice", <group>)`; also `"change your voice"`, `"different voice"` → `("voice", "next")` cycles to the next id in `VOICES` order.
  - Speed phrases: `speak faster|talk faster|faster please|speed up` → `faster`; `speak slower|talk slower|slow down` → `slower`; `normal speed|default speed|reset speed` → `normal`.
- Orchestrator `_voice_turn(action)`:
  - `voice`: `resolve_voice` (or cycle for `"next"`); unknown → say `"I don't have that voice. I have Sarah, Bella, Nicole, Sky, Adam, Michael, Emma, Isabella, George and Lewis."`; else set `tts.voice`, save pref, say `"Okay, this is <Name>."` in the new voice.
  - `speed`: `faster` → `min(SPEED_MAX, speed + SPEED_STEP)`, `slower` → `max(SPEED_MIN, speed - SPEED_STEP)`, `normal` → 1.0; if already at the limit say `"That's as fast as I go."` / `"That's as slow as I go."`; else say `"Like this?"` at the new speed. Save pref.
  - Emits `hud` event caption like other local intents; state goes thinking→speaking→idle (no follow-up window needed; reuse whatever `_mute`-style local turns do).
- Menu (menubar + orb popup, they share the item list): a `Voice` submenu listing the ten display names (checkmark on the current one) and `Faster` / `Slower` / `Normal speed`. Selecting calls the same `_voice_turn` on the orchestrator loop via `call_soon_threadsafe`.
- Text mode (`--text`) supports the same phrases.

### Tests
- `tests/test_voices.py`: resolve table (name, descriptors, combos, default, unknown, case/whitespace), `display_name`.
- `tests/test_intents.py` additions: every phrase above, negative cases (`"what's your voice like"` → None, `"faster internet please"` → None).
- `tests/test_orchestrator_voice.py`: fake Synthesizer records `voice`/`speed`; assert set+saved+spoken copy, clamps, unknown voice reply, cycling.
- `tests/test_tts.py`: `synth` passes `speed=self.speed`.

---

## B2 — Proactive briefings & meeting nudges

### Goal
Once enabled by voice, Veronica gives a morning briefing at a set time and warns before calendar events — spoken only when idle (via the existing `announce()` queue), never mid-turn, never while muted.

### Design
- New `veronica/proactive.py`:
  - `class Schedule(BaseModel-free dataclass)`: `briefing_enabled: bool = False`, `briefing_time: str = "08:00"` (`HH:MM`, 24h), `nudges_enabled: bool = False`, `nudge_minutes: int = 5`. Loaded/saved under `prefs.json` key `"proactive"` (dict). Helpers `load_schedule()`, `save_schedule(Schedule)`.
  - `class Proactive`: `__init__(schedule, announce: Callable[[str], Awaitable[None]], calendar_events: Callable[[str, int], Awaitable[str]], mail_unread_count: Callable[[], Awaitable[int]], reminders_due: Callable[[int], Awaitable[str]], now: Callable[[], datetime] = datetime.now)`. `start()` creates one asyncio task `_loop()`; `stop()` cancels it. `_loop()` ticks every 60 s (`TICK_S = 60`, overridable for tests):
    - Briefing: if enabled and local `now().strftime("%H:%M") == briefing_time` and not yet delivered today (`_last_briefing_date`) → `await announce(await build_briefing())`. Also `build_briefing()` is public so "brief me" can call it on demand.
    - Nudges: if enabled, every tick fetch today's events (cached per 5 min: `EVENTS_CACHE_S = 300`), and for each event whose start is within `[now, now + nudge_minutes]` and whose key `(title, start)` not in `_nudged` → `announce(f"Heads up, {title} starts in {n} minutes.")` (n = whole minutes, ≥1; `"in a minute"` for 1). Skip all-day events. `_nudged` pruned of entries older than today.
  - `build_briefing() -> str` composes from the three fetchers, each guarded (a failing fetcher contributes nothing, logged): `"Good morning, Manik. "` (use `"Good afternoon"` ≥ 12:00, `"Good evening"` ≥ 17:00) + events sentence (`"You have 3 events today: standup at 9:30, ... "` — max 4 titles, then `"and 2 more"`; `"Nothing on your calendar today."` if none) + `"You have N unread emails. "` (skip if 0) + reminders (`"Reminders due: X and Y."`, max 3; skip if none). Fetchers return the same text the pim tools return; parsing helpers in `proactive.py` must tolerate pim's `_format_events` output (`"HH:MM–HH:MM  title  [cal]  loc"` lines — read pim to match exactly) and its "no events" strings.
- Fetchers are thin adapters over the existing `pim` tool handlers (`calendar_events.handler({"day": "today", "days": 1})`, `mail_unread.handler({"limit": 50})` → count lines, `reminders_due.handler({"days": 1})`), wired in `veronica/__main__.py`.
- Local intents (`intents.py`): `match_proactive_intent(text) -> ProactiveAction | None`:
  - `"brief me"`, `"give me a briefing"`, `"morning briefing"`, `"what's my day look like"`, `"what does my day look like"` → `("brief_now", None)`.
  - `"(give me a|start the|turn on) (morning )?briefing (every day |every morning )?at <time>"` → `("briefing_on", "<HH:MM>")`; `<time>` parses `"8"`, `"8 am"`, `"8:30"`, `"7 30 am"`, `"6 pm"` → `HH:MM`. Without a time → `("briefing_on", None)` (keeps the stored time, default 08:00).
  - `"(stop|turn off|cancel) (the )?(morning )?briefing(s)?"` → `("briefing_off", None)`.
  - `"(remind me|warn me|nudge me|turn on nudges|tell me) before (my )?(meetings|events)"` optionally `"<n> minutes before"` → `("nudges_on", n or None)`.
  - `"(stop|turn off) (the )?(meeting )?(nudges|reminders before meetings)"` → `("nudges_off", None)`.
- Orchestrator `_proactive_turn(action)`: updates `Schedule`, saves, replies: `"Okay, I'll brief you every day at 8:00."`, `"Okay, no more morning briefings."`, `"Okay, I'll warn you 5 minutes before each event."`, `"Okay, no more meeting nudges."`; `brief_now` speaks `build_briefing()` directly (this is a normal turn, so it passes through `say`, and then the follow-up window as usual).
- Proactive announcements go through `announce()` — already idle-only and muted-aware. The `Proactive` instance is created in `__main__.build_orchestrator` and started when the orchestrator loop starts (`run_forever` calls `self.proactive.start()` if set); `--text` mode also runs it.
- HUD: announcements already show as speech; briefing additionally emits a `tool` card `"Morning briefing"`.

### Tests (`tests/test_proactive.py`)
- Fake clock + `TICK_S=0`-style manual `tick()` (expose `async def tick()` used by `_loop`): briefing fires once at the time, not twice same day, fires next day; greeting by hour; composition with each fetcher present/missing/failing; nudge fires once per event within window, skips all-day, "in a minute" copy; disabled → nothing.
- Intent tests: all phrase shapes and time parsing table.
- Orchestrator tests: `_proactive_turn` saves schedule and speaks copy; `brief_now` speaks the built text.

---

## B3 — Browser control

### Goal
"Read this page", "summarize this article", "open a new tab with github", "find 'pricing' on this page", "click the login button", "type my email in the search box", "scroll down" — against the browser the user is actually using, with their logged-in session.

### Approach decision
AppleScript + JavaScript injection into **Google Chrome** and **Safari** (both expose `execute javascript` / `do JavaScript`). No CDP/remote-debugging (Chrome ≥136 refuses remote debugging on the default profile, so CDP would lose the user's logins). One-time user setup, surfaced as a spoken hint when the tool fails with the corresponding AppleScript error: Chrome → View ▸ Developer ▸ **Allow JavaScript from Apple Events**; Safari → Develop ▸ **Allow JavaScript from Apple Events**. Playwright is used only in tests (not at all here — `osascript` is mocked).

### Design
- New `veronica/tools/browser.py`, MCP server name `browser` (registered in `Brain._options` `mcp_servers`), tools:
  - `browser_tabs({})` → `"<n>. <title> — <url>"` per tab of the front window, current tab marked `*`. Allow.
  - `browser_open({"url": str, "new_tab": bool})` → opens in the target browser (new tab default true). Allow (same risk as existing `mac.open_url`; `http(s)` only, else error).
  - `browser_read({"max_chars": int})` → readable text of the current tab: title, url, then `document.body.innerText` with whitespace collapsed, capped at `max_chars` (default 6000, max 20000) with `"…[truncated]"`. Allow.
  - `browser_find({"text": str})` → up to 10 lines of innerText containing `text` (case-insensitive) with 1-based line numbers, or `"not found"`. Allow.
  - `browser_click({"target": str})` → clicks the first visible element whose trimmed `innerText`/`aria-label`/`value`/`title`/`alt` equals `target` case-insensitively (buttons, links, inputs, `[role=button]`), then falls back to *contains*. Returns which element was clicked (`tag + text`) or `"no element matching …"`. **Confirm**.
  - `browser_type({"target": str, "text": str, "submit": bool})` → focuses the first input/textarea/contenteditable whose `placeholder`/`aria-label`/`name`/`id`/associated `<label>` matches `target` (same match rules), sets its value (dispatching `input` and `change` events), presses Enter if `submit`. **Confirm**.
  - `browser_scroll({"direction": "up"|"down"|"top"|"bottom"})` → scrolls by 80% viewport / to edge. Allow.
  - `browser_back({})` → history back. Allow.
- Target browser: `_target_browser()` → the frontmost app if it is `Google Chrome` or `Safari` (via `System Events` `name of first process whose frontmost is true`); else whichever of the two is running (Chrome preferred); else error `"No supported browser is open (Chrome or Safari)."`. Cached per call only.
- JS is sent as one self-contained IIFE string per tool, JSON-encoded results (`JSON.stringify`) parsed in Python; AppleScript strings escaped via the same `_q` helper style as pim. Chrome: `tell application "Google Chrome" to execute active tab of front window javascript "<js>"`; Safari: `tell application "Safari" to do JavaScript "<js>" in current tab of front window`.
- Error mapping: osascript stderr containing `"Allow JavaScript from Apple Events"` or error number `-1743`/`-10004` → `_err("JavaScript from Apple Events is off in <Browser>. Turn it on under <menu path> and try again.")`. Automation permission denial (`-1743`) → `_err("Veronica isn't allowed to control <Browser> yet; allow it in System Settings > Privacy & Security > Automation.")`.
- Policy: `MCP_TOOL_RISK["browser"]` as above (`click`, `type` confirm; rest allow).
- Summaries in `agent.summarize_detail`: `"List tabs"`, `"Open <url>"`, `"Read the page"`, `"Find '<text>' on the page"`, `"Click '<target>'"`, `"Type into '<target>'"`, `"Scroll <direction>"`, `"Go back"`.
- Prompt (`prompts.py`): one sentence — for anything about "this page", "this tab", the current article/site, or acting inside the browser, use the `browser` tools; summarise `browser_read` output in your own words, don't read it verbatim.
- Local intents: none — the brain drives these (they need judgment). Except `"read this page"`/`"summarize this page|article"` which are pure brain prompts anyway (no local shortcut).
- HUD: tool cards as usual.

### Tests (`tests/test_browser_tools.py`)
- Monkeypatch `run`/`_osascript` to capture the AppleScript and return canned JSON; assert per-tool script contains the right `tell application`, JS contains the selector logic markers, results parsed/capped/truncated; target-browser selection (frontmost Chrome, frontmost Safari, neither frontmost but Chrome running, none → error); error mapping for the two failure modes; `browser_open` rejects `file:`/`javascript:` URLs.
- `tests/test_policy.py`: `browser` risk table entries.
- `tests/test_agent.py`: summaries.

---

## Order & branching
`batch-b` branch off master: B1 → B2 → B3, each its own tasks; full suite green after each; reviewer per task; whole-branch review; merge `--no-ff`; `make app`; relaunch.
