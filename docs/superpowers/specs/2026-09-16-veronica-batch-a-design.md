# Veronica Batch A — Screen awareness, push-to-talk, music, notes & dictation

Date: 2026-09-16 · Status: approved · Builds on master (post hud-menu)

## A1 Screen awareness
- MCP server `screen` (`veronica/tools/screen.py`): `screenshot(region: "screen"|"window"|"selection" = "screen") -> image` — `screencapture -x -t png` (window: `-l <front window id>` via AppleScript/`CGWindowListCopyWindowInfo` of the frontmost app; selection: `-i` interactive). Returns MCP image content (base64 PNG, downscaled to ≤1568 px on the long edge via `sips`) so Claude can look at it. Risk: allow (read-only, local), but the HUD shows a "📸 Looked at your screen" action line and a chime.
- Local intents (fast path, no Claude round-trip ambiguity): "what's on my screen", "look at my screen", "summarize this page/screen", "what does this error say" → orchestrator takes the screenshot itself and sends the text + image to the brain in one turn (`Brain.ask(text, images=[png_bytes])` → SDK `query()` with an image content block).
- System prompt: "You can see the user's screen with the screenshot tool when they refer to what they're looking at."

## A2 Push-to-talk
- `veronica/audio/hotkey.py`: global modifier monitor using `Quartz` CGEventTap (`pyobjc-framework-Quartz` — add dependency) on `kCGEventFlagsChanged`; Right-Option keycode 61. Requires Accessibility permission (Input Monitoring); if not granted → log once + menu item "Enable push-to-talk (needs Accessibility)" opening the System Settings pane.
- Orchestrator: `ptt_start()` → if idle: stop wake listener, chime, start capture with `max_s=None` mode "hold": capture ends when `ptt_end()` is called (Recorder gets `stop_capture()` that ends the utterance gracefully — returns what was recorded, unlike `stop()` which discards) → normal turn. Works while she's speaking (acts like barge-in).
- Setting `ptt_enabled=True`, `ptt_key="right_option"`.

## A3 Music
- MCP server `music` (`veronica/tools/music.py`): `music_play(query: str = "")`, `music_pause()`, `music_next()`, `music_prev()`, `music_now_playing()`, `music_volume(level)`. Backend detection: if Spotify is running (`application "Spotify" is running`) use it, else Music.app; `music_play` with a query: Spotify → `play track "spotify:search:…"` isn't supported; use `tell application "Spotify" to play track (…)` via the `spotify:track:` URI from a search? Spotify AppleScript has no search; fallback: open `spotify:search:<query>` URL then `play`. Music.app: `play (first track of playlist "Library" whose name contains q or artist contains q)`. All allow-class. Local intents: "pause/resume/next/previous/what's playing" → direct, no Claude.

## A4 Notes & dictation
- `notes_create(title, body)` in the `pim` server → Notes.app (`make new note at folder "Notes" with properties {name, body}`); allow-class? It writes — confirm-class but with a lighter phrasing; ruling: **allow** for notes (append-only, harmless) — HUD shows it.
- Local intent "take a note: …" / "note that …" → direct `notes_create` with a timestamped title, say "Noted."
- Dictation: intent "dictate" / "start dictation" → listen (hold mode like PTT, ends on "stop dictation" or 3 s silence) and type the text into the frontmost app via `osascript -e 'tell application "System Events" to keystroke "…"'` (Accessibility permission). Confirm-class? Ruling: allow — the user explicitly asked to dictate, and text goes where they're focused.

## Permissions & app
- Info.plist: `NSAccessibilityUsageDescription`? (not a real key — Accessibility is granted in System Settings; add README instructions) ; add `NSScreenCaptureUsageDescription`? (screen recording permission prompt appears automatically for `screencapture`). README: grant Screen Recording + Accessibility to Veronica.app.

## Tests
Unit with mocked `subprocess.run`/`screencapture`/`osascript` argv; hotkey monitor with a fake event source; orchestrator PTT flow with fakes (start/end → capture returns buffered audio); intents; music backend detection; dictation typing argv escaping (quotes/newlines → `keystroke` with `return`). Live-marked: `screenshot("screen")` returns a PNG < 2 MB; `music_now_playing()` returns without error.
