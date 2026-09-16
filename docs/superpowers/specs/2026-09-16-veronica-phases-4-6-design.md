# Veronica Phases 4–6 — Calendar/Mail/Reminders/Timers, Memory, Mac App

Date: 2026-09-16 · Status: approved · Builds on master 1b10e77

## Phase 4 — Personal data via macOS apps (no OAuth)

New MCP server `pim` in `veronica/tools/pim.py` (same `@tool` + `_guard` + argv-only pattern as `mac.py`), all via `osascript` with the script passed as argv (never shell strings); JSON-ish output parsed in Python.

| tool | input | action | risk |
|---|---|---|---|
| `calendar_events` | `{"day": "today"|"tomorrow"|"YYYY-MM-DD", "days": int=1}` | Calendar.app events in range: title, start/end (local, HH:MM), calendar name, location | allow |
| `calendar_create` | `{"title", "start": "YYYY-MM-DD HH:MM", "minutes": int=60, "calendar": str=""}` | create event (default calendar) | confirm |
| `mail_unread` | `{"limit": int=5}` | Mail.app: unread messages in inbox: sender, subject, date, first 200 chars | allow |
| `mail_search` | `{"query", "limit": 5}` | subject/sender contains query (Mail.app) | allow |
| `mail_send` | `{"to", "subject", "body"}` | Mail.app compose+send | confirm |
| `reminder_create` | `{"title", "when": "YYYY-MM-DD HH:MM" or ""}` | Reminders.app, default list | confirm |
| `reminders_due` | `{"days": 1}` | incomplete reminders due within N days | allow |
| `timer_set` | `{"minutes": float, "label": str=""}` | in-process timer; when it fires Veronica speaks "Timer <label> done" (via a new `Orchestrator.announce(text)` that waits for idle, chimes, speaks) and shows a notification | allow |
| `timer_list` / `timer_cancel` | `{}` / `{"label"}` | | allow |

Timers live in `veronica/tools/timers.py` (`TimerService` with asyncio tasks; injected into the pim server at build time). `Orchestrator.announce()` = queue of announcements drained when state is idle (never interrupts a turn).

System prompt gains: "You can read the user's calendar, unread mail and reminders and set timers with your tools; prefer them over shell commands for these."

## Phase 5 — Memory

`veronica/memory/store.py`: SQLite at `~/.veronica/memory.db` — `turns(id, ts, heard, reply)` + FTS5 `turns_fts(heard, reply)`; `facts(id, ts, text)`.
- Orchestrator logs every completed turn (heard + full reply text) and every explicit fact.
- Local intent `remember`: utterance starting with "remember that" / "remember " → store fact, say "Got it." (no Claude).
- Local intent `forget`: "forget that …" → delete matching fact (FTS), say "Forgotten." / "I didn't have that."
- MCP server `memory` (`veronica/tools/memory_tools.py`): `recall(query, limit=5)` → matching past turns (allow); `facts_list()` (allow); `fact_add(text)` (allow — it's Veronica's own memory), `fact_delete(text)` (confirm).
- System prompt injection per turn (cheap, cached): "Facts about the user: …" (all facts, ≤ 2 KB) + "Recent: <last 6 turns, ≤ 1 KB>". Since the SDK session already carries context, this only matters after a fresh session; keep it small.
- Settings: `memory_enabled=True`, `memory_recent_turns=6`.

## Phase 6 — Mac app

`scripts/build_app.py` produces `dist/Veronica.app`:
- Bundle layout: `Contents/MacOS/Veronica` = shell launcher `exec "<repo>/.venv/bin/python" -m veronica` (paths baked at build time; `VERONICA_HOME` unchanged), `Contents/Info.plist` with `CFBundleIdentifier=io.manik.veronica`, `LSUIElement=1` (menu bar only), `NSMicrophoneUsageDescription`, `NSAppleEventsUsageDescription` (Calendar/Mail/Reminders automation), `CFBundleIconFile`; `Contents/Resources/Veronica.icns` generated from a rendered orb PNG (`scripts/make_icon.py` renders the HUD orb via Playwright at 1024 px → `iconutil`).
- `codesign --force --deep -s - dist/Veronica.app` (ad-hoc) so TCC permissions stick to the bundle.
- Menu bar gets "Start at Login" (checkbox) → writes/removes `~/Library/LaunchAgents/io.manik.veronica.plist` (`RunAtLoad`, `ProgramArguments=[<app>/Contents/MacOS/Veronica]`, `StandardOutPath` → `~/.veronica/logs/launchd.log`).
- `make app` target in a `Makefile`; README "Install as an app" section.
- Logging: when not attached to a TTY, the stream handler is dropped (file only).

## Testing
Unit with `subprocess.run` mocked for every pim tool (argv shape, output parsing from canned osascript output), timer service with a fake clock, announce queue, memory store (tmp db: insert/FTS search/facts), intents, prompt injection size caps, LaunchAgent plist write/remove (tmp HOME), build script structure (tmp dist; skips codesign/iconutil when unavailable). Live-marked: `calendar_events today` on this Mac; `timer_set 0.05` end to end; build + `open dist/Veronica.app` manual.
