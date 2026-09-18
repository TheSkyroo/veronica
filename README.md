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
- First run downloads the whisper `small.en` model (~470 MB). Hindi mode needs the multilingual `small`/`tiny`
  models too (~500 MB more) — fetched the first time you say "speak hindi", or ahead of time with
  `uv run python scripts/download_models.py --hindi`.
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

### Mute, unmute, quit

A few more phrases are also handled locally:

- **Mute** — "mute", "mute yourself", "be quiet", "silence": she says "Muted.", stops listening for commands, and
  hides the HUD. While muted she still wakes on the wake word, but only to check for the unmute phrase below — she
  won't chime, show the HUD, or respond to anything else (including calendar/timer announcements, which are held
  until you unmute her).
- **Unmute** — "unmute", "unmute yourself", "you can talk", "speak again": say this after the wake word while muted
  and she says "I'm back." and resumes normal listening. Anything else said after the wake word while muted is
  ignored silently.
- **Quit** — "quit", "quit veronica", "shut down", "shut yourself down", "exit", "turn off completely": asks for
  confirmation; say "yes" and she says "Goodbye." and quits the app.

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

## Screen awareness

Ask "what's on my screen", "look at my screen", "summarize this page/screen", or "what does this error say" and
Veronica takes a screenshot of the main display herself (downscaled to fit within 1568 px on the long edge) and
sends it to Claude along with your question in one turn — the HUD shows a "Look at screen" action line. Claude can
also decide to look at the screen on its own mid-conversation via the `screenshot` tool (allow-class, runs
automatically). Requires **Screen Recording** access — see Permissions below.

## Browser control

Veronica can read and act on the page you have open in **Chrome** or **Safari**. Say "read this page", "summarize
this article", "find pricing on this page", "click the login button", "type hello in the search box and press
enter", or "open a new tab with github" — Claude picks the right browser tool for the request.

One-time setup, per browser:

- **Chrome** — menu bar **View ▸ Developer ▸ Allow JavaScript from Apple Events**.
- **Safari** — enable the Develop menu in **Settings ▸ Advanced**, then **Develop ▸ Allow JavaScript from Apple
  Events**.

The first time a browser tool runs, macOS prompts for **Automation** access to the browser — allow it (see
Permissions below). Reading, listing tabs, opening a URL, finding text, scrolling and going back run automatically;
**clicking** and **typing** always ask for confirmation first, since they act inside your logged-in session.

## Push-to-talk

Hold **Right Option** (⌥, the key to the right of the spacebar) to talk to Veronica without saying the wake word —
release it when you're done. Works even while she's speaking (it interrupts her, like saying the wake word does).
Disable with `VERONICA_PTT_ENABLED=false`, or change the key with `VERONICA_PTT_KEYCODE` (macOS virtual keycode;
61 is Right Option).

Push-to-talk needs **Input Monitoring** access (prompted on first launch) (see Permissions below). If it isn't granted, the menu bar shows
"Enable Push-to-talk… (Input Monitoring)" — click it to jump straight to the right System Settings pane.

## Music

"Pause" / "pause music", "resume" / "play music", "next song" / "skip", "previous", and "what's playing" control
Spotify (if it's running) or Music.app (otherwise) directly, no round-trip to Claude. Claude can also control
playback and search for a track/artist mid-conversation via `music_play`, `music_pause`, `music_next`, `music_prev`,
`music_now_playing`, and `music_volume` (all allow-class).

## Notes & dictation

- **Take a note** — "take a note: buy milk" / "note that the wifi password is abc123": creates a note in Notes.app
  titled with the first 40 characters of what you said plus a timestamp, and says "Noted."
- **Dictate** — "dictate" / "start dictation": say "Go ahead.", then listen until you say "stop dictation" or pause
  for 3 seconds, and types everything you said into whichever app is currently focused (via System Events —
  requires **Accessibility** access, same as push-to-talk).

## Voice & speed

- **Pick a voice** — "use a british voice" / "switch to adam voice" / "speak with a female voice": ten Kokoro voices
  (Sarah, Bella, Nicole, Sky, Adam, Michael, Emma, Isabella, George, Lewis), picked by name or by descriptor
  (british/american, male/female), plus the four Hindi voices (see *Hindi & Hinglish*) — picking a Hindi voice while she's in English mode also switches her to
  "understand both" so she can hear Hindi; "use the default voice"
  goes back to the configured one. "Change your voice" /
  "different voice" cycles to the next one. She confirms in the new voice ("Okay, this is George.") so you hear it
  straight away; an unknown name gets the list back.
- **Speed** — "speak faster" / "speak slower" / "normal speed" nudge the speaking rate in 0.15x steps (0.7x–1.5x)
  and confirm with "Like this?".

Both are handled locally (no round-trip to Claude), persist across restarts in `~/.veronica/prefs.json`, and are
also in the menu bar / orb popup under **Voice** (the voice list plus Faster / Slower / Normal speed).

## Quick replies

Trivial questions are answered locally, instantly, without a round-trip to Claude — only when the whole
utterance is one of these (a longer request that merely contains "time" still goes to the brain):

- **Time, date, day** — "what time is it", "what's the date", "what day is it" (Hinglish: "kitne baje hain",
  "aaj kya tareekh hai", "aaj kya din hai").
- **Battery and volume** — "battery level", "how much battery do I have", "what's the volume" ("battery kitni
  hai", "volume kitna hai").
- **Arithmetic** — "what's 12 times 8", "144 divided by 12", "2 to the power of 10", "20 percent of 50"
  ("12 guna 8 kitna hota hai"), and the symbols whisper writes: "5 + 5", "10 - 3", "12 x 8", "100 / 8", "2^10",
  "15% of 80" (read back in words: "5 plus 5 is 10."). A tiny integer parser over numbers and operators, never
  `eval()`; decimals, clock times ("5:30 plus 10"), bare numbers and anything fancier ("5 plus 5 in binary") go
  to the brain.
- **Small talk** — hello / thanks / bye / how are you / who are you / what can you do / good morning / good night
  (and namaste, shukriya, alvida, kaise ho, tum kaun ho, shubh ratri).

Quick replies show up as a "Quick reply" tool card in the HUD and are logged to memory like any other turn.

## Hindi & Hinglish

- **Switch** — "speak hindi" / "hindi mein bolo" pins her to Hindi; "speak english" / "english mein bolo" goes
  back; "understand both" / "dono bhasha" lets whisper detect the language per utterance. She confirms in the new
  language ("अब हिंदी में बात करते हैं।" / "Okay, English it is." / "ठीक है, दोनों चलेगा।"), and the mode
  persists in `~/.veronica/prefs.json`. In pinned Hindi mode everything you say is treated as Hindi and spoken with
  the Hindi voice; say "understand both" / "dono bhasha" if you mix English and Hindi.
- **What to expect** — Hindi and auto mode swap the English-only whisper models for the multilingual ones; the
  first switch downloads them (~500 MB) after an "एक मिनट, हिंदी load कर रही हूँ।" Replies follow your
  language: Hindi or Hinglish in → Hindi out in Devanagari (Kokoro's Hindi voice needs Devanagari to sound
  natural — romanized Hinglish gets read like English); English in, English out. Hindi replies are spoken with a
  Hindi voice; timers, briefings and other announcements keep the English voice
  unless they contain Devanagari.
- **Hinglish commands** — the local intents understand romanized Hindi too: "bas karo" / "chup" ends the turn,
  "mute karo" / "awaaz band karo", "chhoti ho jao" / "badi ho jao" for the HUD, "haan" / "ji" / "nahi" answer a
  "Run …?" confirmation, plus the quick replies above. In pinned Hindi mode whisper writes Devanagari, so the
  common ones are understood in script as well ("बस करो", "हाँ" / "नहीं", "समय क्या है").
- **Hindi voices** — Alpha, Beta (female), Omega, Psi (male). "Use a hindi voice" / "use the omega voice" picks
  the voice Hindi replies use (the English voice is untouched, so both show a checkmark in the **Voice** menu),
  confirmed with "ठीक है, अब मैं ऐसे बोलूँगी।" Prefetch the models without switching:
  `uv run python scripts/download_models.py --hindi`.

## Briefings & nudges

- **Brief me** — "brief me" / "give me a briefing" / "what's my day look like": a spoken summary of today's
  calendar, unread mail count and reminders due, composed locally from Calendar.app, Mail.app and Reminders.app.
- **Daily briefing** — "give me a briefing every morning at 8" / "start the briefing every day at 6 pm" turns on a
  scheduled briefing at that time ("turn on the morning briefing" keeps the stored time, default 08:00); "stop the
  morning briefing" / "turn off briefings" turns it off. A briefing more than two hours late (the Mac was asleep) is
  skipped rather than read out mid-afternoon.
- **Meeting nudges** — "warn me 10 minutes before my meetings" / "remind me before my meetings" / "turn on nudges"
  announces "Heads up, <event> starts in 10 minutes." before each timed calendar event (1–60 minutes, default 5);
  "turn off nudges" / "stop the meeting nudges" turns them off.

Briefings and nudges are announcements: they're spoken only when Veronica is idle and not muted (anything that
fires mid-conversation or while muted waits, like a timer), and the schedule persists in `~/.veronica/prefs.json`.

## Settings window

A normal macOS window (tabs: General, Voice, Listening, Briefings, Brain, History, About) for everything that
used to need an environment variable or a voice command.

- **Open it** — say "open settings" / "settings" / "preferences" / "settings kholo", pick "Settings…" from the menu
  bar, or click the HUD orb and choose "Settings…". "Show history" / "what did I ask you" / "history dikhao" opens
  it straight on the History tab.
- **Live settings** apply to the running app right away and persist: language mode, voice, Hindi voice, speed (each
  spoken back so you hear the change), HUD mode, hide delay, follow-up window, confirm listen, silence and
  utterance limits, briefing/nudge schedule, start at login, push-to-talk.
- **Restart settings** are saved but only picked up on the next launch: wake sensitivity/window/hop, wake phrases,
  brain effort, memory on/off, working folder. Changing one shows a "Restart Veronica to apply" banner with a
  Restart button (from the built `.app` it quits and relaunches itself once the old process has exited; from a
  terminal it quits and says "Restart me from the terminal.").

Values you set here override the environment/`.env` defaults (they're stored in `~/.veronica/prefs.json`).

## History

The History tab lists past turns (what you said, what she replied) from the local memory database, with a search
box. Each row has a Forget button; "Clear all" (with a confirm step) removes them all. Facts you asked her to
remember are separate (see Memory) and aren't touched by clearing history. With memory disabled the tab just says
so.

## Version & updates

- **"What version are you"** / "version" / "kaunsa version hai" — says e.g. "Veronica 0.1.0 (a517483, 17 Sep)":
  the package version plus the commit that's actually running (from the bundle's `build.json` when launched as
  the app, else live from git). The same line sits at the top of the menu bar menu ("About Veronica — …") and on
  the About tab.
- **"Update yourself"** / "update now" / "check for updates" / "apna update karo" — checks the repo: if
  `origin` has newer commits it says "Updating, back in a moment.", runs `git pull --ff-only`, `uv sync`
  and rebuilds `dist/Veronica.app`, then relaunches. If there's no remote (or nothing new upstream) but the running
  build is behind the checked-out code, "update" just rebuilds and restarts you onto the latest local code. Already
  current: "You're already on the latest." Couldn't reach the remote: "Couldn't check for updates, check the log."
  Anything failing mid-update: "The update failed, check the log." Only one update runs at a time — a second
  "update yourself" (or the window/menu) while one is running gets "An update is already running." / "Busy, try
  again in a moment.". From a terminal run (no app bundle to reopen) it finishes with "Update installed. Restart me
  from the terminal."
- **Menu bar** — "Check for Updates…" runs the same check and posts a notification (when running from a terminal
  there's no notification center, so she says the result instead, next time she's idle); when something newer exists
  the item below it becomes "Update available — Restart to update" (click to install). Veronica also checks quietly
  once an hour and only flips that item, no notification. The About tab has "Check now", "Update & restart",
  "Restart" and "Open log".

Updates are refused while a conversation is in progress from the window/menu ("Busy, try again in a moment."); the
spoken "update yourself" is itself the turn, so it just runs. There are no API keys involved — updating is a git
pull plus rebuild of the local checkout.

## Permissions

> The bundle's executable is a small native launcher that embeds Python, so every macOS permission
> prompt and Privacy & Security entry says **Veronica** (not "python3.12"). Building it needs `clang`
> from the Xcode Command Line Tools (`xcode-select --install`). After upgrading from an older build,
> re-grant Microphone, Screen Recording, Input Monitoring and Automation to Veronica — the old grants
> belonged to the Python interpreter.


Grant these to Veronica (or your terminal, if running with `uv run` instead of the built app) under
**System Settings → Privacy & Security**:

- **Microphone** — wake word and voice commands (asked automatically on first run).
- **Automation** — Calendar/Mail/Reminders/Notes/Music/Spotify (asked automatically the first time each is used);
  **Google Chrome** and **Safari** for the browser tools; **System Events** for browser detection (which browser is
  in front) and for dictation's typing.
- **Screen Recording** — screenshots for screen awareness (asked automatically the first time `screenshot` runs).
- **Input Monitoring** — push-to-talk's global hotkey (asked for on first launch). **Accessibility** — dictation's typing into other apps. Not
  asked for automatically; grant it yourself, or use the menu bar's "Enable Push-to-talk… (Input Monitoring)" item if
  push-to-talk shows as unavailable.

## Run
    uv run python -m veronica                 # menu bar app
    uv run python -m veronica --text "hello"  # no audio, debug

## Install as an app

Build a real `dist/Veronica.app` menu-bar app bundle instead of running from a terminal:

    make app          # writes dist/Veronica.app, ad-hoc codesigned
    open dist/Veronica.app

(`make icon` re-renders `assets/Veronica.icns` from the HUD orb first, if you want a fresh icon — the built
one is already committed, so this is optional.)

On first launch macOS asks for **Microphone** access, and the first time Veronica touches Calendar, Mail,
Reminders, Notes, Music, Spotify, Chrome, Safari or System Events it asks for **Automation** access to that app; the first `screenshot` prompts
for **Screen Recording** — approve all of these (System Settings → Privacy & Security). **Accessibility** (for
push-to-talk and dictation) is not prompted for automatically — grant it yourself under System Settings → Privacy &
Security → Accessibility, or use the menu bar's "Enable Push-to-talk… (Input Monitoring)" item. Because the bundle is
ad-hoc codesigned, these approvals stick across rebuilds as long as the bundle identifier (`io.manik.veronica`)
doesn't change.

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

## Troubleshooting

**She only wakes when I lean into the mic.** The wake check ignores audio quieter than the "Wake sensitivity
(min level)" gate (`wake_min_rms`, default 0.003). Lower it in Settings to hear you from across the room (you
may get more false wakes; raise it if she wakes on noise). "Wake window" and "Wake hop" (`wake_window_s`,
`wake_hop_s`) control how much audio each check sees and how often it runs. Run with
`VERONICA_LOG_LEVEL=DEBUG` and watch `~/.veronica/logs/veronica.log` for `wake hop rms=... gate=...` lines
to see how loud your voice actually lands at the mic.

**AirPods / USB mic / headphones.** Mic switching is automatic: Veronica polls macOS's default input device
every couple of seconds and reopens the mic on the new device (`input device changed (...); reopening mic` in
the log), also re-reading the output device list so speech follows your headphones. The switch waits until any
in-flight recording finishes.

**She stopped hearing me after a call / after switching mics.** Call apps with auto-gain (Zoom, Meet,
FaceTime) and device switches quietly drop the Mac's input volume to ~30 %, which starves the wake check.
Veronica checks the input volume once a minute and right after every mic switch, and raises it back to the
"Input volume floor" setting (`input_volume_floor`, default 85; never lowers it). The first fix of a session
shows on the HUD; every fix is an `input volume 33 → 85 (...)` line in the log. Set the floor to 0 to turn it off.

## Test
    uv run pytest            # unit
    uv run pytest -m live    # needs mic/speaker/models/login
    uv run playwright install chromium   # once, for the live HUD tests
