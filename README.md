# Veronica

macOS voice assistant. Say "Hey Veronica" once you have trained the custom model (see scripts/train_wakeword.md); until then say "Hey Jarvis", ask, listen.
Brain = Claude via your Claude Code login. Speech = local (faster-whisper + Kokoro).

## Setup
    brew install uv portaudio
    uv venv --python 3.12 && uv pip install -e ".[dev]"
    uv run python scripts/download_models.py
    claude auth login        # if not already

Wake word: say "Veronica" or "hey Veronica" (whisper engine, default). To use the lighter openwakeword engine set VERONICA_WAKE_ENGINE=openwakeword (falls back to "hey jarvis" until you train a custom model — see scripts/train_wakeword.md).

Wake word not triggering? Run `uv run python scripts/wake_scores.py`, say the phrase, and (openwakeword engine) set VERONICA_WAKE_THRESHOLD in .env just below the scores you see.

Do not set `ANTHROPIC_API_KEY` — Veronica uses your Claude Code login (it is ignored if set).

- macOS will ask for Microphone access for your terminal app on first run (System Settings → Privacy & Security → Microphone).
- First run downloads the whisper `small.en` model (~470 MB).
- Say the wake word while Veronica is talking to interrupt her (barge-in).
- Risky actions (writing files, shell commands that change things, AppleScript, clipboard writes) ask "Run …?" — answer "yes" or "no".
- A floating HUD appears at the top-right when Veronica wakes (orb + transcript + tool activity) and fades after 3 s of idle. Disable with VERONICA_HUD_ENABLED=false.
- The HUD can be dragged anywhere on screen (click and drag its background) — it reopens wherever you left it.

## Using Veronica

Say "Veronica …" in one breath — e.g. "Veronica, what time is it" — rather than pausing after the wake word. She
buffers the tail of the wake-word audio and hands it straight to the recorder, so a command spoken in the same
breath as the wake word isn't lost and the wake chime is skipped when she can already hear you talking.

The HUD's status line under the orb shows what she's doing:

- **Warming up…** — models are loading (first run only).
- **Listening…** — she's capturing your voice (also shown during the follow-up window after a reply); the bar
  next to it tracks the live mic level.
- **Thinking…** — Claude is working on a reply.
- **Speaking** — she's talking.
- **Say yes or no** — she's asked for confirmation before a risky action and is listening for your answer; the
  question itself appears above, and the mic-level bar is still shown while she listens for it.
- **Error** — something went wrong; check the log.

While she's listening, the HUD also shows a live partial transcript of what you're saying (in italics), which is
replaced by the final transcript once you finish talking. This costs a bit of CPU; disable it with
`VERONICA_PARTIAL_STT=false` in `.env` to save power on slower Macs.

She stops listening rather than eavesdropping indefinitely: the follow-up window after a reply is short (4 s by
default — set `VERONICA_FOLLOWUP_WINDOW_S` in `.env` to change it), and if that follow-up capture comes back empty
(you didn't say anything more) she just goes idle rather than asking you to repeat yourself. You can also end the
conversation immediately by saying "thanks Veronica" / "thank you Veronica" (she replies "Okay.") or "that's all",
"stop", "goodbye", "never mind", "go idle", "turn yourself off", "go to sleep", "sleep", "go away", "bye", "dismiss"
(she just goes quiet, no follow-up window).

### HUD voice commands

A few phrases are handled locally (no round-trip to Claude) to control the HUD itself. Say them the same way as any
other command — with or without "Veronica"/"hey Veronica" first, optionally ending in "please":

- **Mini mode** — "shrink", "make yourself smaller", "minimize", "mini mode", "small mode", "go small": collapses
  the HUD to a compact bar (a small orb plus a single-line caption pill showing what's being heard or said), which
  defaults to a notch-style position centered under the menu bar rather than the full card's top-right corner.
- **Full mode** — "expand", "make yourself bigger", "full mode", "show details", "go big": returns to the full card
  layout (transcript, reply, tool activity).
- **Hide** — "hide", "hide yourself", "hide the hud", "hide the panel": hides the HUD immediately and goes idle.

The current mode (and the last dragged position) persists across restarts in `~/.veronica/prefs.json`. You can also
switch modes from the menu bar item ("HUD: Mini" / "HUD: Full" toggles it).

## Calendar, mail, reminders, timers

Veronica can read your Calendar events, unread Mail, and Reminders, and create events/reminders or send mail (all
via AppleScript/Apple Events — no OAuth, no cloud account of Veronica's own). She can also set simple in-process
timers ("set a timer for 5 minutes") that speak and show a notification when they fire, even while she's idle.

Requirements:

- The relevant account(s) (iCloud, Gmail, Exchange, …) need to be added in System Settings → Internet Accounts (or
  already configured in Calendar.app / Mail.app / Reminders.app) — Veronica reads/writes through those apps, not a
  separate login.
- The first time she touches Calendar, Mail, or Reminders, macOS shows an automation permission prompt ("Terminal"
  or the app running Veronica wants to control "Calendar"/"Mail"/"Reminders") — approve it once per app. You can
  review/reset these under System Settings → Privacy & Security → Automation.
- Reading (calendar events, unread mail, mail search, reminders due, timers) runs automatically; creating an event
  or reminder, sending mail, and any raw AppleScript still ask "Run …?" first, same as other risky actions.

## Memory

Veronica keeps a small local memory across sessions, in a SQLite database at `~/.veronica/memory.db` (FTS5 full-text
search when the local Python's sqlite3 build has it, otherwise a plain substring search — either way the same
`recall`/`fact_add`/`fact_delete` behavior).

- Every completed turn (what you said, what she replied) is logged, so she can look it up later or carry a little
  recent context into a fresh Claude session.
- **Remember a fact** — "remember that I take my coffee black" / "remember I'm allergic to peanuts": stored as a
  fact and said back as "Got it." This is a local intent (matched before the brain runs), so it works even offline
  and doesn't cost a Claude turn.
- **Forget a fact** — "forget that I take my coffee black" / "forget the peanut thing": removes any matching fact
  and says "Forgotten." (or "I didn't have that." if nothing matched).
- Claude can also manage memory itself mid-conversation via MCP tools: `recall` (search past turns) and `facts_list`
  run automatically; `fact_add` and `fact_delete` both ask "Run …?" first — a fact persists across every future
  session, so it gets the same confirmation as anything else that changes standing state.
- On every new Claude session, Veronica injects a short "Facts about the user" list and the last few turns
  ("Recent conversation") into the system prompt, capped small (2 KB / 1 KB) so it stays cheap — an existing session
  already carries its own context, so this only matters right after a fresh one starts.

Disable memory entirely (no DB, no injection, no remember/forget intents) with:

    VERONICA_MEMORY_ENABLED=false

## Run
    uv run python -m veronica                 # menu bar app
    uv run python -m veronica --text "hello"  # no audio, debug

## Install as an app

Build a real `dist/Veronica.app` menu-bar app bundle instead of running from a terminal:

    make app          # writes dist/Veronica.app, ad-hoc codesigned
    open dist/Veronica.app

(`make icon` re-renders `assets/Veronica.icns` from the HUD orb first, if you want a fresh icon — the built
one is already committed, so this is optional.)

On first launch macOS asks for **Microphone** access, and the first time Veronica touches Calendar, Mail, or
Reminders it asks for **Automation** access to that app — approve both (System Settings → Privacy & Security).
Because the bundle is ad-hoc codesigned, these approvals stick across rebuilds as long as the bundle identifier
(`io.manik.veronica`) doesn't change.

The bundle's launcher just `cd`s into this repo and execs `.venv/bin/python -m veronica`, so it needs the same
`.venv` (and `.env`, models, `claude auth login`) you set up for `uv run` — there's no separate install step.
`VERONICA_HOME` (default `~/.veronica`) is unchanged when running as a bundle.

**Start at Login** — the menu bar's "Start at Login" item writes a `LaunchAgent` at
`~/Library/LaunchAgents/io.manik.veronica.plist` that relaunches `dist/Veronica.app` at login. It's greyed out
("Start at Login (build the app first)") until you launch Veronica from the built `.app` at least once — it needs
a real bundle path to point the LaunchAgent at.

Logs: `~/.veronica/logs/veronica.log` (the app's own log; when running from the bundle, stdout isn't a TTY, so
only the file handler is attached — nothing is lost, it's just not duplicated to a terminal) and
`~/.veronica/logs/launchd.log` (stdout/stderr captured by launchd when started via "Start at Login").

## Test
    uv run pytest            # unit
    uv run pytest -m live    # needs mic/speaker/models/login
    uv run playwright install chromium   # once, for the live HUD tests
