# Veronica

A voice assistant for **Windows 10 and 11**. Say "Veronica, …", ask, and listen.

- **Speech stays on your PC.** faster-whisper understands you and Kokoro speaks the reply, both locally.
- **The thinking is done by a coding-agent CLI you already use:** Codex (default), Claude Code, GitHub Copilot or
  Google Antigravity, each through its own login (no API keys). There is also an offline local model.
- **She can act on your PC.** She can open apps and Windows places, see your screen, click and type, control
  Chrome/Edge and play music. With your Google account she handles Gmail, Calendar, Tasks and notes.
- **Anything risky asks first.** You answer "yes" or "no" out loud.

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [First run](#first-run)
- [Talking to Veronica](#talking-to-veronica)
- [What she can do](#what-she-can-do)
- [Brains](#brains)
- [Accounts: Google and Spotify](#accounts-google-and-spotify)
- [Browser extension](#browser-extension)
- [Safety and confirmations](#safety-and-confirmations)
- [Settings](#settings)
- [Build the app (Veronica.exe)](#build-the-app-veronicaexe)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Requirements

| What | Why |
| --- | --- |
| Windows 10 (21H2+) or Windows 11, 64-bit | the app targets Windows only |
| Python 3.12 via [uv](https://docs.astral.sh/uv/) | runtime and dependency manager |
| Git | install and self-update |
| Node.js LTS | the brain CLIs install with npm |
| Microsoft Edge WebView2 Runtime | the HUD and Settings windows (preinstalled on Windows 10/11) |
| A microphone and speakers | obviously |

Disk: about 1 GB for the speech models (another ~500 MB if you use Hindi).

## Install

In PowerShell:

```powershell
winget install astral-sh.uv Git.Git OpenJS.NodeJS.LTS
git clone https://github.com/TheSkyroo/veronica
cd veronica
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run python scripts/download_models.py          # add --hindi for the Hindi models too

npm i -g @openai/codex                            # the default brain …
codex login                                       # … and its login (others: see Brains)
```

Audio needs nothing extra, because PortAudio ships inside the `sounddevice` wheel.

## First run

```powershell
uv run python -m veronica                    # tray app with the HUD
uv run python -m veronica --text "hello"     # one question by text, no microphone (debugging)
```

1. Turn on microphone access: **Settings → Privacy & security → Microphone → Let desktop apps access your
   microphone**.
2. Look for the orb icon in the system tray (it may be under the **^** overflow arrow; drag it out to keep it
   visible).
3. Say **"Veronica, what time is it?"**. The HUD appears at the top-right of the screen, shows what she heard,
   and she answers.

Optional, but recommended:

- **Connect Google** for mail, calendar, tasks and notes. See [Accounts](#accounts-google-and-spotify).
- **Install the browser extension** so she can read and act on web pages. See [Browser extension](#browser-extension).
- **Say "Veronica, learn my voice"** so she ignores the TV and other people.
  See [Only my voice](#background-noise-and-only-my-voice).

---

## Talking to Veronica

- **Wake word:** say "Veronica" or "hey Veronica", then the request **in the same breath**: "Veronica, what's on
  my calendar today?". Anything you say right after the wake word is kept, so there's no need to pause.
- **Push-to-talk:** press the **Copilot key** (on laptops that have one) or **Ctrl+Alt+Space**. You can use it two
  ways:
  - **Hold** it while you speak, then let go.
  - **Tap** it, speak, and she stops listening when you go quiet. Some Copilot keys only ever send a tap, so this
    always works.

  The key doesn't do its usual job while you're using it (Copilot doesn't open, no space is typed), and letting go
  doesn't open the Start menu. To use other keys, set *Push-to-talk keys* in Settings → General. It takes a
  comma-separated list, for example `copilot, ctrl+alt+space`, `f9` or `right_ctrl`.
- **Listening mode:** choose how she listens. You can switch by voice ("push-to-talk mode" / "always listen"), from
  the tray's **Listening** menu, or in Settings → General.
  - **Always listening** (default): she listens for her name whenever she's idle, and push-to-talk works too.
  - **Push-to-talk only:** the microphone stays off until you press the push-to-talk key. There's no wake word and no follow-up
    window; she only opens the mic herself to hear your yes/no when she asks a question. While muted in this mode,
    press the push-to-talk key and say "unmute".
- **Follow-ups:** after a reply she listens for 4 more seconds, so you can keep going without the wake word
  (always-listening mode only).
- **Asking for more while she's busy:** say "Veronica, …" or press the push-to-talk key while she's working on something or
  answering. She holds her voice for a moment, hears you, and **adds the request to a queue**, while the first task
  keeps going. When it's finished she does the queued ones in order. The HUD shows what's **Up next**.
  - To **replace** what she's doing instead, say so: "no, open Spotify instead", "actually …", "cancel that".
  - **"Clear the queue"** drops the waiting requests but keeps the current task. "Stop" / "that's all" ends
    everything, including the queue.
  - To go back to the old behaviour, where a new request always replaces the current one, turn off *Queue requests
    made while she's busy* in Settings → General.
- **Interrupt:** say the wake word, or press push-to-talk, while she's talking, then "stop", "hold on" or a new
  request ending in "instead".
  - **Pause:** "hold on" / "wait" / "ruko" stops her but keeps the rest of the answer. "Continue" / "go on" / "aage
    bolo" picks up where she stopped.
  - **Stop:** "stop" / "that's all" / "never mind" cancels the answer.
- **End the conversation:** "thanks Veronica", "goodbye", "go to sleep" and so on. She goes quiet and stops
  listening.
- **Mute:** "mute" / "be quiet". While muted she only responds to "unmute" / "you can talk".
- **Quit:** "quit Veronica" (she asks you to confirm first).

### The HUD

A small floating card with an animated orb shows her state, which is also written under the orb:

| State | Meaning |
| --- | --- |
| Warming up… | first launch, models are loading |
| Listening… | she's recording you (a bar shows the live mic level and your words appear in italics) |
| Thinking… | the brain is working (the line below says which brain, e.g. "Brain: Codex") |
| Speaking | she's talking |
| Say yes or no | she's asking permission for an action |
| Error | something failed; see the log |

- **Moving it:** drag it anywhere and it stays where you left it.
- **Shrinking it:** "mini mode" / "make yourself smaller" makes it a one-line pill at the top centre. "Full mode"
  brings the full card back.
- **Hiding and finding it:** "hide" hides it. "Where are you?" resets its position.
- **Menu:** click the orb to open the same menu as the tray icon.

It never takes keyboard focus away from the app you're working in.

---

## What she can do

### Open apps, folders and Windows settings

Some examples: "open Spotify", "open This PC", "open Downloads", "open the Recycle Bin", "open Bluetooth settings",
"open Wi-Fi settings", "open Device Manager", "open Control Panel", "open Task Manager".

She can open:

- **Apps:** any app in the Start menu, including Store apps.
- **Places and tools:** about 80 Windows folders, Settings pages and system tools.
- **Web pages:** "open github.com" opens it in your default browser.

A plain "open …" / "launch …" / "start …" (or "… kholo") is handled by Veronica herself, instantly and without the
brain, whenever the name is a Windows place or an installed app. A slightly misheard name ("open this DC") still
finds This PC. Anything more involved ("open GitHub in Chrome") goes to the brain as usual.

### See your screen

"What's on my screen?", "summarise this page", "what does this error say?"

She takes a screenshot of the monitor your active window is on and looks at it. On several monitors she can look at
one (`display=2`) or all of them.

### Click, type and use any app (computer use)

"Click the Save button", "double-click Local Disk C", "type hello in the search box and press Enter", "scroll
down", "press ctrl S", "where's the Settings icon?"

- **How:** she screenshots, finds text on screen with Windows' built-in OCR, then moves the mouse, clicks, drags,
  scrolls, types and presses keys.
- **Permission:** the first action in an app asks "yes or no". After a yes she works freely **in that same app**
  for 90 seconds; switching apps asks again. You can change or turn off this window in Settings.
- **Always asks, every time:** pressing Enter, and anything in a terminal window.
- **Never:** typing into password fields, pressing Yes/Allow on security prompts (UAC, SmartScreen, Windows
  Security), or using keys that close, lock or sign you out (alt+f4, win+l, ctrl+alt+del…).
- **Admin windows:** Windows ignores simulated clicks in programs running as administrator unless Veronica also
  runs as administrator. She tells you when that's the problem.

### Browser (Chrome and Edge)

"Read this page", "summarise this article", "find pricing on this page", "click the login button", "type hello in
the search box and press Enter", "open a new tab with YouTube", "go back".

This needs the [browser extension](#browser-extension). Clicking and typing in a page always ask first.

### Mail, calendar, tasks and notes (Google)

| Say | What happens |
| --- | --- |
| "What's on my calendar today / tomorrow / this week?" | reads **every** Google calendar you have ticked |
| "Add lunch with Priya tomorrow at 1" | creates a calendar event (asks first) |
| "Any new email?" / "Search my mail for the invoice" | reads unread mail / searches Gmail |
| "Mail Priya that I'm running late" | finds Priya in your Google contacts and sends (always asks) |
| "Remind me to call mom on Friday" / "What's due?" | creates and reads Google Tasks |
| "Take a note: the wifi password is …" | saves a Google Doc in a "Veronica Notes" Drive folder |
| "Set a timer for 10 minutes" | spoken alert plus a Windows notification (works offline) |

This needs your [Google account connected](#accounts-google-and-spotify).

### Music

- **"Play Shape of You"** actually plays the song:
  - **In Spotify,** if you connected Spotify (Premium is required for remote control). She opens Spotify if it
    isn't running.
  - **On YouTube** otherwise: the top video for the song opens in your browser and starts playing.
- **Playback controls:** "pause", "resume", "next song", "previous", "what's playing?" and "music volume 40". These
  control whatever is playing (Spotify, a browser tab, any media app) through Windows' media controls.

### Quick answers (instant, no brain needed)

- **Time and date:** "what time is it", "what's the date".
- **Your PC:** "battery level", "what's the volume".
- **Arithmetic:** "12 times 8", "20 percent of 50".
- **Small talk:** hello, thanks, who are you.
- **Hinglish works too:** "kitne baje hain", "battery kitni hai".

### Dictation

"Dictate" → "Go ahead." Everything you say is typed into the focused app until you say "stop dictation" or pause
for 3 seconds. It types Unicode, so Hindi works too.

### Memory

- **Remember:** "remember that I take my coffee black". She answers "Got it."
- **Forget:** "forget that" or "forget everything about the office".
- **Recall:** she can also look up past conversations herself.

Memory is a local SQLite file (`%USERPROFILE%\.veronica\memory.db`). The History tab in Settings lists past turns and
can delete them. To turn memory off, set `VERONICA_MEMORY_ENABLED=false`.

### Briefings and nudges

- **On demand:** "brief me" reads today's calendar, unread mail count and tasks due.
- **Daily briefing:** "give me a briefing every morning at 8".
- **Meeting nudges:** "warn me 10 minutes before my meetings".
- **Snooze:** "snooze notifications for an hour".
- **More in Settings:** quiet hours, low-battery and unread-mail alerts are all on the Briefings tab.

### Voice, speed and language

- **Voice:** "use a British voice", "switch to the Adam voice", "change your voice". There are ten English and four
  Hindi voices.
- **Speed:** "speak faster", "speak slower" or "normal speed".
- **Language:** "speak Hindi" / "hindi mein bolo", "speak English", or "understand both" / "dono bhasha" to
  switch automatically.

### Background noise and only my voice

- **Noise reduction** is on by default, so keyboards, fans and music don't trigger or stretch a recording.
- **"Veronica, learn my voice"**: she has you repeat three short lines. After that, requests and yes/no answers in
  anyone else's voice (the TV, people in the room) are ignored. Push-to-talk is always trusted. "Forget my voice"
  undoes it.

---

## Brains

The "brain" is the coding-agent CLI that does the thinking. Speech, the HUD, the tools and the confirmations are the
same whichever brain is active, and each brain uses its own CLI's login.

| Brain | Install | Log in |
| --- | --- | --- |
| **Codex** (default): OpenAI, your ChatGPT plan | `npm i -g @openai/codex` | `codex login` |
| **Claude**: Claude Code, your Claude plan | `npm i -g @anthropic-ai/claude-code` (needs Git for Windows) | `claude` |
| **Copilot**: GitHub Copilot CLI, your Copilot plan | `npm i -g @github/copilot` | `copilot login` |
| **Antigravity**: Google `agy` (Gemini) | the Windows installer from antigravity.google | `agy` |
| **Local**: llama.cpp, offline, no account | `llama-server.exe` + a `.gguf` model | n/a |

- **Switching:** "switch to Claude", "use Copilot", "which brain are you on?". You can also use the tray's
  **Brain** menu or Settings → Brain. Only installed, logged-in brains are offered.
- **Usage limits:** when one brain hits its limit she says so and moves on to the next ready brain. She switches
  back by herself once the cooldown ends (60 minutes by default).
- **Offline:** with no internet she answers with the local model ("No internet — switching to the local model.").
  Put `llama-server.exe` in `%USERPROFILE%\.veronica\llama\` and a model in `%USERPROFILE%\.veronica\models\`, or
  set both paths in Settings → Brain. "Go offline" / "go online" switch by hand.
- **Shells:** Codex, Copilot and Antigravity run PowerShell. Claude Code runs Git Bash. Each brain's own shell can
  be turned off in Settings → Brain, leaving only Veronica's tools.
- **API keys:** do not set `ANTHROPIC_API_KEY` or any other vendor key. They are ignored.

---

## Accounts: Google and Spotify

Both are your own free developer apps, signed in once in your browser. Tokens stay in `%USERPROFILE%\.veronica\`.

### Google (Gmail, Calendar, Tasks, Docs)

1. Create a project at <https://console.cloud.google.com>.
2. Go to **APIs & Services → Library** and enable **Gmail API**, **Google Calendar API**, **Google Tasks API**,
   **People API** and **Google Drive API**.
3. On the **OAuth consent screen**, choose **External** with status **Testing**, and add your own Google account as a
   test user.
4. Under **Credentials → Create credentials → OAuth client ID**, choose type **Desktop app** and download the JSON.
5. Save the file as `%USERPROFILE%\.veronica\google_client.json`.
6. In Veronica, open **Settings → General → Accounts → Connect Google**, sign in, and allow every permission.

While the Google app is in *Testing*, the sign-in lasts 7 days, so you reconnect weekly. Publishing it (unverified,
personal use) avoids that. With the `drive.file` permission Veronica can only see the notes she created.

### Spotify (optional)

1. At <https://developer.spotify.com/dashboard>, choose **Create app** and pick **Web API**.
2. Add the redirect URI `http://127.0.0.1:8898/callback`. It must be exactly `127.0.0.1`, not `localhost`.
3. Copy the **Client ID** into **Settings → General → Accounts → Spotify Client ID**.
4. Click **Connect Spotify** and approve.

Spotify only allows playback control on **Premium**. On a free account, "play …" uses YouTube.

---

## Browser extension

The extension lets Veronica work in your normal Chrome or Edge profile, with your logins. It only talks to Veronica on
`127.0.0.1`.

1. Start Veronica once. It creates `%USERPROFILE%\.veronica\browser_token`.
2. Load the extension:
   - **Chrome:** open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose the
     `veronica\browser_extension` folder of this repo.
   - **Edge:** the same at `edge://extensions` (Developer mode is in the left sidebar).
3. On the options page that opens, paste the contents of `browser_token` and click **Save**. It should say
   *Connected to Veronica*.

You can install it in both browsers; she uses the one you focused last. The port is 8765 by default
(`VERONICA_BROWSER_PORT`; set the same port in the extension's options). Internal pages (`chrome://`, `edge://`,
web stores) can't be reached by any extension.

---

## Safety and confirmations

Every action goes through one gate. It decides whether the action runs on its own, asks "Run …?", or is refused. The
same gate applies to every brain, including commands the brain runs in its own shell.

- **Runs on its own:** reading things, such as screenshots, calendar, unread mail, page text, `dir` /
  `Get-ChildItem` and opening apps.
- **Asks first:** anything that changes something. That covers writing files, shell commands that modify things,
  PowerShell scripts, sending mail, creating events or tasks, clipboard writes, and clicking or typing on screen and
  in the browser.
- **Always asks, whatever you said:** sending mail, PowerShell scripts, `Remove-Item -Recurse` / `rd /s` / `rm -rf`,
  registry edits, shutdown/restart, `Start-Process -Verb RunAs`, disk tools and the like.

Ways to skip a question, and their limits:

- **"Just do it":** a request containing "go ahead", "without asking" or "haan kar do abhi" skips the question for
  that **one** action. It never covers the always-ask list.
- **"Always":** answer "always" (or "don't ask again") to stop being asked about a tool. Only seven harmless tools
  qualify: clipboard write, create event, create task, remember fact, forget fact, browser click and browser type. You
  can review them in Settings → Brain → Auto-allow tools.
- **Anything else you say** instead of yes or no ("no, open it in Edge instead") is taken as your next request.
  Silence counts as no.

---

## Settings

Open Settings by saying "open settings", left-clicking the tray icon, or choosing **Settings…** from the tray or orb
menu.

| Tab | What's there |
| --- | --- |
| General | listening mode, request queue, push-to-talk keys, HUD, start at login, **Accounts** (Google, Spotify), microphone privacy shortcut |
| Voice | voice, Hindi voice, speed, language |
| Listening | wake sensitivity, follow-up window, noise reduction, "learn my voice", input volume floor |
| Briefings | daily briefing, meeting nudges, quiet hours, battery and mail alerts |
| Brain | brain, failover, offline model, per-brain shell, auto-allowed tools, screen-control trust window |
| History | past conversations, with search and delete |
| About | version, check for updates, restart, open log |

Most settings apply immediately; the rest show a *Restart to apply* banner. Values are saved in
`%USERPROFILE%\.veronica\prefs.json` and override `.env`. Any setting can also be given as a `VERONICA_<NAME>`
environment variable or in a `.env` file in the repo, for example `VERONICA_HUD_ENABLED=false`,
`VERONICA_PTT_ENABLED=false`, `VERONICA_PTT_HOTKEY=f9`, `VERONICA_LISTEN_MODE=ptt`, `VERONICA_FOLLOWUP_WINDOW_S=6` or `VERONICA_LOG_LEVEL=DEBUG`.

---

## Build the app (Veronica.exe)

```powershell
uv run python scripts/build_app.py       # → dist\Veronica\Veronica.exe   (or: make app)
.\dist\Veronica\Veronica.exe
```

- **Icon:** `uv run python scripts/make_icon.py` regenerates `assets\Veronica.ico`. This is optional, because the
  icon is committed.
- **Needs this checkout:** the exe uses this repo's `.venv`, `.env`, models and CLI logins, so keep the repo where it
  is.
- **Start at Login:** the tray menu item adds Veronica to `HKCU\…\CurrentVersion\Run`, which you can also review in
  Settings → Apps → Startup. It is available once you've launched the built exe.
- **Updates:** say "update yourself", or use the tray's **Check for Updates…**. She runs `git pull`, `uv sync`,
  rebuilds and restarts. "What version are you?" tells you the running commit.
- **Signing:** this is optional (`signtool`). An unsigned exe may get one SmartScreen warning.

Logs are written to `%USERPROFILE%\.veronica\logs\veronica.log`.

---

## Troubleshooting

**She doesn't hear me.**
1. Check microphone privacy (see [First run](#first-run)).
2. Make sure the right mic is the *default* recording device (Settings → System → Sound).
3. Watch the HUD's level bar while you talk.

**She only wakes when I'm close to the mic.** Lower *Wake sensitivity* in Settings → Listening. With
`VERONICA_LOG_LEVEL=DEBUG`, the log shows how loud you actually are (`wake hop rms=…`). For the openwakeword engine,
`uv run python scripts/wake_scores.py` shows live scores.

**She stopped hearing me after a Teams/Zoom call.** Call apps lower the mic level. Veronica raises it back to the
*Input volume floor* (85 % by default) every minute and after every device change.

**Headset or USB mic switching.** Switching is automatic: she follows Windows' default devices within a couple of
seconds.

**Push-to-talk does nothing.** The tray shows "Push-to-talk unavailable" if the keyboard hook couldn't be installed.
Restart Veronica. Some games and anti-cheat tools block global hooks. If another app already uses the Copilot key or Ctrl+Alt+Space, pick
different keys in Settings → General → *Push-to-talk keys*.

**Clicks land in the wrong place.** Take a fresh screenshot ("look at my screen") first. Clicking into
administrator windows needs Veronica to run as administrator too.

**"No browser is connected."** Open the extension's options page; it should say *Connected to Veronica*. Re-paste
`browser_token` if needed.

**"Google isn't connected" / "Google isn't set up".** Check `google_client.json` exists, then use Settings → Accounts
→ Connect Google. In *Testing* mode, reconnect weekly.

**"Codex returned an error" on every request.** Check the log for the line after `turn ended early`. Codex prints a
warning when your own `%USERPROFILE%\.codex\config.toml` has a setting it doesn't recognise (for example one left by
another tool, such as `mcp_servers.<name>.type`). Veronica now treats that as a warning, but it's still worth removing
the line it names. Run `codex exec "say hi"` in a terminal to see whether Codex itself works.

**A brain is "not installed" or "not logged in".** Run its install and login commands from [Brains](#brains) in a
new terminal, so it picks up the updated `PATH`.

---

## Development

```powershell
uv run pytest                                    # unit tests (hermetic, no devices, no network)
uv run pytest -m live                            # needs mic, speakers, models, logins
uv run pytest -m live tests/test_brains_live.py  # real brain CLIs (each skips unless installed + logged in)
uv run playwright install chromium               # once, for the HUD/Settings page tests
uv run ruff check veronica tests
```

Project layout:

```
veronica/
  __main__.py          entry point: single-instance lock, DPI awareness, tray app
  orchestrator.py      the conversation loop: wake → record → brain → speak
  audio/               mic, wake word, recording, playback, noise reduction, speaker check, push-to-talk
  speech/              speech-to-text (faster-whisper) and text-to-speech (Kokoro)
  brain/               local intents, quick replies, the confirm gate and policy, brain backends
    backends/          codex, claude, copilot, antigravity, local (llama.cpp), Windows process helpers
  tools/               the tools brains can call: system, pim (Google), screen, computer, browser, music, memory
  ui/                  tray icon, HUD and Settings windows (pywebview), start-at-login, relaunch
  browser_extension/   Chrome/Edge extension for browser control
  google_account.py    Google sign-in
  spotify_account.py   Spotify sign-in
scripts/               model download, app build, icon, wake-word and voice tools
docs/                  design notes (written for the original macOS version)
```

The tests run on any OS. Windows APIs are imported lazily and replaced with fakes, so `uv run pytest` also works in
Linux CI. Behaviour that only a real Windows PC can show (the HUD window, the keyboard hook, OCR, the Google and
Spotify sign-ins, the brain CLIs' hooks) is covered by the `live` tests and manual checks.
