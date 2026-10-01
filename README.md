# Veronica

Voice assistant for **Windows 10/11**. Say "Veronica …" (or "Hey Jarvis" with the openwakeword engine until you
train a custom model — see scripts/train_wakeword.md), ask, listen.
Brain = Codex, Antigravity, Claude or Copilot, each through its own CLI login (no API keys; see Brains below).
Speech = local (faster-whisper + Kokoro).

## Setup

In PowerShell:

    winget install astral-sh.uv Git.Git OpenJS.NodeJS.LTS
    git clone https://github.com/TheSkyroo/veronica; cd veronica
    uv venv --python 3.12; uv pip install -e ".[dev]"
    uv run python scripts/download_models.py
    npm i -g @openai/codex; codex login    # the default brain; others under Brains below
    uv run python -m veronica              # tray app (see Run / Install as an app below)

PortAudio ships inside the `sounddevice` wheel, so there is nothing else to install for audio. The HUD and
Settings windows use Microsoft Edge WebView2, which Windows 10/11 already has.

Wake word: say "Veronica" or "hey Veronica" (whisper engine, default). To use the lighter openwakeword engine set VERONICA_WAKE_ENGINE=openwakeword (falls back to "hey jarvis" until you train a custom model — see scripts/train_wakeword.md).

Wake word not triggering? Run `uv run python scripts/wake_scores.py`, say the phrase, and (openwakeword engine) set VERONICA_WAKE_THRESHOLD in .env just below the scores you see.

Do not set `ANTHROPIC_API_KEY` (or any other vendor key) — every brain uses the login of its own CLI (`codex login`,
`agy`, `claude`, `copilot login`); keys are ignored if set.

- Windows must allow desktop apps to use the microphone: **Settings → Privacy & security → Microphone → Let desktop
  apps access your microphone** (Settings → General → "Microphone Privacy…" jumps there).
- First run downloads the whisper `small.en` model (~470 MB). Hindi mode needs the multilingual `small`/`tiny`
  models too (~500 MB more) — fetched the first time you say "speak hindi", or ahead of time with
  `uv run python scripts/download_models.py --hindi`.
- Say the wake word while Veronica is talking to interrupt her (barge-in).
- Interrupt with "hold on" / "wait" / "one sec" ("ruko", "ek minute") and she stops but keeps the rest of the
  answer: say "continue" / "carry on" / "go on" ("aage bolo") and she picks up at the next sentence. Anything
  else you say is treated as a new request and the remainder is dropped; "stop" / "that's all" still cancels.
- If a slow answer leaves her silent for more than 3.5 s she says "On it." once (Settings → Listening, or
  `VERONICA_ACK_AFTER_S`; 0 turns it off).
- Risky actions (writing files, shell commands that change things, PowerShell scripts, clipboard writes) ask "Run …?" — answer "yes" or "no".
- A floating HUD appears at the top-right when Veronica wakes (orb + transcript + tool activity) and fades after 3 s of idle. It never takes keyboard focus from the app you're in. Disable with VERONICA_HUD_ENABLED=false.
- The HUD can be dragged anywhere on screen (click and drag its background) — it reopens wherever you left it.

## Using Veronica

Say "Veronica …" in one breath — e.g. "Veronica, what time is it" — rather than pausing after the wake word. She
buffers the tail of the wake-word audio and hands it straight to the recorder, so a command spoken in the same
breath as the wake word isn't lost and the wake chime is skipped when she can already hear you talking.

The HUD's status line under the orb shows what she's doing:

- **Warming up…** — models are loading (first run only).
- **Listening…** — she's capturing your voice (also shown during the follow-up window after a reply); the bar
  next to it tracks the live mic level.
- **Thinking…** — the brain is working on a reply (the line under it says which one: "Brain: Codex").
- **Speaking** — she's talking.
- **Say yes or no** — she's asked for confirmation before a risky action and is listening for your answer; the
  question itself appears above, and the mic-level bar is still shown while she listens for it. During a
  confirmation, anything that isn't yes/no is taken as your next request ("no, open it in Edge instead",
  "what will that do?"): the action is skipped and she answers that instead. Silence skips it too.
- **Error** — something went wrong; check the log.

**Pre-approval by wording.** If the request itself already says go ahead — "copy this to the clipboard, just do
it", "open chrome and go ahead", "add the reminder without asking", "haan kar do abhi" — she skips the yes/no for
the **one** action that request produces (the HUD shows it with a gold "pre-approved" pill). It's one-shot and
short-lived (20 s, the first confirmable action of that request only); a second action in the same request is
asked as usual, and the pre-approval never covers sending mail, `rm -r` / `Remove-Item -Recurse`, force-pushes, shutdown /
restart / sleep, running as administrator, pressing Enter, typing into a terminal, or anything on a system dialog — those are
confirmed every time, however you phrase it. A bare "do it" or "yes" is an answer, not a request, and a question
("should I do it?") never pre-approves. Turn it off in **Settings → Brain → Pre-approve when I say "do it"**.

**Auto-allow tools.** If you've approved a tool once, it can stay approved. Answer a confirmation with "always"
— or "don't ask again", "stop asking", "mat pucho" — and that tool stops asking, for good: the call goes ahead
and the tool is added to **Settings → Brain → Auto-allow tools**, which survives a restart. She only offers the
option for tools that are allowed to be there; from then on those calls show a "auto" pill in the HUD, the same
as anything else that runs without asking. Seven tools are eligible, and **only** these can ever be added:

| Tool | What it does |
| --- | --- |
| `mcp__system__clipboard_write` | Copy to the clipboard — **ticked by default** |
| `mcp__pim__calendar_create` | Create a calendar event |
| `mcp__pim__reminder_create` | Create a reminder |
| `mcp__memory__fact_add` | Remember a fact |
| `mcp__memory__fact_delete` | Forget a fact |
| `mcp__browser__browser_click` | Click in the browser |
| `mcp__browser__browser_type` | Type in the browser (including a typed Enter) |

**Destructive tools can never be added**, whichever way you try. Sending mail, PowerShell, every
screen-control action (`mcp__computer__*`) and the shell are not on the list, so saying
"always" to one of them approves that single call and she answers "That one I'll always ask about." Typing one
into the free-form field by hand does nothing either: `policy.classify` only honours names in
`policy.AUTO_ALLOWABLE`, and `policy.always_confirm` is checked first, so a hand-typed `mcp__pim__mail_send` or
`mcp__computer__computer_click` still asks every single time.

The Settings section has a checkbox per eligible tool plus the full list as a text field, for review and for
revoking: untick one, or clear the field, and she starts asking again immediately — no restart.

While she's listening, the HUD also shows a live partial transcript of what you're saying (in italics), which is
replaced by the final transcript once you finish talking. This costs a bit of CPU; disable it with
`VERONICA_PARTIAL_STT=false` in `.env` to save power on slower PCs.

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
  defaults to the top centre of the screen rather than the full card's top-right corner.
- **Full mode** — "expand", "make yourself bigger", "full mode", "show details", "go big": returns to the full card
  layout (transcript, reply, tool activity).
- **Hide** — "hide", "hide yourself", "hide the hud", "hide the panel": hides the HUD immediately and goes idle.
- **HUD gone? say "where are you"** — also "show yourself", "come back", "reset the hud": forgets the saved
  position and shows the HUD at its default spot on the main screen (she answers "Here I am."). The HUD also
  re-checks its position on its own whenever a display is plugged in or unplugged, so it can't be stranded on a
  monitor that's no longer there.

A turn that runs more than one tool shows them in the full card as a checklist: ○ waiting on your yes, ▸ running,
✓ done, ✕ declined or failed — failed when the tool itself returned an error. (A Codex, Antigravity or Copilot
built-in tool, such as their own shell, reports no result to Veronica, so its step only ever shows done.)

The current mode (and the last dragged position) persists across restarts in `~/.veronica/prefs.json`. You can also
switch modes from the tray menu ("HUD: Mini" / "HUD: Full" toggles it).

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

## Brains

The "brain" is whichever coding-agent CLI does the thinking. Veronica runs it as a subprocess (or, for Claude, via
the Agent SDK) and hands it her own tools; the speech, the HUD and the confirmation gate are the same whichever
one is active. Every brain uses the vendor CLI's own login — there are no API keys anywhere.

| Brain | What it is | Install | Log in |
| --- | --- | --- | --- |
| **Codex** (default) | OpenAI's `codex` CLI, on your ChatGPT plan | `npm i -g @openai/codex` | `codex login` |
| **Antigravity** | Google's `agy` CLI (Gemini), on your Google account | the Windows installer from antigravity.google | `agy` |
| **Claude** | Anthropic's `claude` CLI (Claude Code), on your Claude plan | `npm i -g @anthropic-ai/claude-code` (needs Git for Windows, for Git Bash) | `claude` |
| **Copilot** | GitHub's `copilot` CLI, on your Copilot plan | `npm i -g @github/copilot` | `copilot login` |
| **Local** | llama.cpp on this PC — no account, no network | a `llama-server.exe` binary and a `.gguf` | — |

Only the brains that are installed *and* logged in are offered; the check is local and cheap (the binary on
`PATH` plus the file the login writes — `~/.codex/auth.json`, `~/.copilot/config.json`, Antigravity's
`~/.gemini/antigravity-cli/conversations` or its Windows Credential Manager entry; Claude reports a missing login itself). An
unavailable one is spoken as "Codex isn't installed — run npm i -g @openai/codex, then codex login." or "Codex
isn't logged in — run codex login in a terminal."

**Choosing one.** The preferred brain is **Settings → Brain → Brain** (`brain_backend`, default `codex`, live — no
restart). The tray menu has a **Brain: Codex** submenu with one radio item per brain (unavailable ones read
"Copilot (not installed)" / "(not logged in)" and are disabled). By voice, without a brain round-trip: "switch to
codex", "use antigravity", "use copilot", "back to claude" / "go back to claude", "switch brain to codex",
"codex pe switch karo", "copilot use karo"; she answers "Switched to Codex." or "Already on Codex." (or the
install/login hint above). "Which brain are you on" / "which model is this" / "who am I talking to" / "kaunsa
brain hai" answers "I'm on Codex." A switch interrupts whatever the current brain was doing and clears any
screen-control trust window; each brain keeps its own resumable session, so switching back picks up where it
left off. The HUD's status area shows "Brain: Codex" and the settings page's Brain tab shows the same label.

**The confirm gate applies to every brain.** Two paths, one gate:

- *Veronica's tools* (`system`, `pim` for calendar/mail/tasks/notes/timers, `memory`, `screen`, `music`, `browser`,
  `computer`) are served to an external brain as MCP servers over stdio (`python -m veronica.tools.serve <name>`,
  registered as `veronica-<name>`). That process is only a proxy: each call goes to the app's gate socket
  (a loopback port with a per-start token, published in `~/.veronica/run/gate.json`), which asks the same question the in-process gate asks Claude — policy, trust window,
  pre-approval, voice confirm — so "Run …?" sounds and behaves exactly the same, and then **runs the tool inside
  the app** and sends the result (text, or the screenshot's image) back. That last part matters: the app process is the one
  with the foreground rights, the account sign-ins and the audio session, which a helper the CLI spawned (often a
  hidden console process) doesn't have. Running it in the app is also
  why a timer set through an external brain announces like any other.
- *The CLI's own shell and file edits* (Codex's `Bash`/`apply_patch`, Antigravity's `run_command`, Copilot's `bash`)
  run with the vendor's own approvals turned off and Veronica's **pre-tool hook** as the only gate: the hook logs
  the call to `~/.veronica/backends/<brain>/hook.log`, asks the same gate socket, and a "no" blocks the command
  with her reason. Because a hook that silently stops firing would leave the shell ungated, every native call is
  a **canary**: a shell/edit call that finishes without a matching `hook.log` line kills the child, she says
  "Hooks aren't running on Codex, so I've turned off its shell. Tools still work.", the brain's shell is switched
  off (persisted) and the request is retried tools-off.
- Whether a brain may use its own shell at all is **Settings → Brain → Codex / Antigravity / Copilot: allow its
  own shell** (`codex_native_tools`, `antigravity_native_tools`, `copilot_native_tools`, default on, live). Off =
  only Veronica's tools: Codex runs in a read-only sandbox, headless Antigravity denies its own shell/file tools
  itself, and Copilot's shell/edit/agent tools are hidden from the model.

**Failover on usage limits.** When a turn ends with a limit error (usage limit, rate limit, 429, quota, out of
credits, overloaded …) and **Settings → Brain → Switch brains on usage limits** (`brain_failover`, default on) is
on, she says "Codex hit its usage limit — switching to Antigravity.", puts the failed brain in a cooldown of
**Limit cooldown (minutes)** (`brain_limit_cooldown_min`, default 60) and re-runs the same request once on the
next brain in **Failover order** (`brain_failover_order`, default `codex,antigravity,claude,copilot`; the
preferred brain is implicitly first) that is installed, logged in and not cooling down. The local model is never
picked automatically — a usage limit, or a brain that isn't logged in, is no reason to drop to the 3B weights;
only a dead wire is (see Offline). If none is: "Codex hit its usage limit and no other brain is ready." With
failover off she just says "Codex hit its usage limit." A stand-in that hits its own limit fails over again
down the order, each with its own cooldown, and a brain already cooling down is never retried in the same
chain. The preference is not changed by failover: the HUD
shows "Brain: Antigravity (for Codex)", the menu item reads "Antigravity — standing in for Codex", "which brain
are you on" answers "I'm on Antigravity — Codex hit its limit, I'll try it again in 42 minutes.", and once the
cooldown passes she returns to the preferred brain **silently** before the next turn (the label updates; the
next reply just comes from it). A manual "switch to …" clears that brain's cooldown. At startup, if the
preferred brain isn't installed or logged in, she starts on the first available one and says once "Codex isn't
logged in, so I'm on Claude for now." (re-checking the preferred one every minute); with nothing ready she
says "No brain is ready — log into Codex, Antigravity or Claude."

### Offline

The **Local** brain is a [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server.exe` running on this PC,
with a quantised model file. Nothing leaves the machine: no account, no login, no network call, not even to
check for one.

She starts the server herself on the first local turn (`llama-server --model <gguf> --ctx-size 8192 --host
127.0.0.1 --port 8749 --jinja --no-webui`), waits for `/health`, and leaves it running until she is closed or
ten minutes pass without a turn — a model load costs seconds, so it is worth keeping warm. A server already
listening on that port is used as-is rather than replaced.

**Settings → Brain → Offline:** *Use the local model when offline* (`brain_offline_fallback`, default on),
*Local model* (`local_model`, default `%USERPROFILE%\.veronica\models\granite-4.2-3b-q4_k_m.gguf` — small, fast and
instruction-tuned), *Local server* (`local_server_bin`, default `%USERPROFILE%\.veronica\llama\llama-server.exe`),
*Local context (tokens)* (`local_ctx`, default 8192) and *Local port* (`local_port`, default 8749). All live, no
restart. To think with different weights, pick another model from the *Local model* dropdown (the `.gguf` files
in the same folder as the current one, without `mmproj-*` projectors, embedding models such as `bge-*`, or the
later shards of a split model), or type any other path under *Model path* — a bigger one is slower to load and to
speak, a smaller one forgets more. The next local turn restarts the server on it (a change of context or port
does the same).

**When it takes over.** Before each turn she checks whether the active brain's vendor host is reachable (one TCP
connect, cached for 20 seconds). If it isn't, she says "No internet — switching to the local model." and answers
locally, exactly like a usage-limit stand-in: the preference doesn't change, "which brain are you on" answers
"I'm on the local model — there's no internet.", and she goes back to the preferred brain **silently** as soon
as the wire returns. Turn it off with `brain_offline_fallback`. By voice, any time: "go offline" / "offline
mode" / "use the local model" / "offline ho jao" switches to it, and "go online" / "back online" / "online ho
jao" hands the next turn back to the first ready vendor brain.

**What it can and cannot do.** Tools: **yes** — Veronica's own tools (system, calendar/mail/tasks/timers,
memory, music, browser, screen control) are offered to the model as function schemas and run in-process, each
call through the same confirm gate, with the same HUD cards and the same trust window. The web: **no** — there
is no search and no fetch, and she is told to say so rather than guess. Screenshots: **no** — the default model
is text-only, so she says she can't see. A small model that ignores the tool schema simply answers in words;
that is normal and not an error. Expect a short answer in a handful of seconds, and expect it to be less sharp
than the hosted brains — it is a three-billion-parameter model on a PC.

If the server won't start she moves to the next ready brain in the failover order and says so once ("The local
model wouldn't start — switching to Codex."), re-running the request there; the local model gets another try
after the usage-limit cooldown, and while it cools a dead wire doesn't put it back in. With nothing else ready
(or failover off) she says "The local model wouldn't start and no other brain is ready." The Local brain
is offered only when both the binary and the model file exist ("The local model server isn't there — set its
path in Settings.").

**Known limits.**

- **Antigravity edits user-level files.** Only the user-level `~/.gemini/config/hooks.json` is loaded by `agy`,
  so Veronica merges her PreToolUse entry into it, scoped to her own conversation id (your own interactive `agy`
  sessions are never gated). The first Antigravity turn also runs `agy mcp add` for each of her servers, which
  adds `veronica-*` entries to `~/.gemini/config/mcp_config.json`; they stay there (harmless outside Veronica —
  they need her gate socket to do anything). With its shell on, `agy` runs with `--dangerously-skip-permissions`;
  the hook and the canary are what keep it honest.
- **Codex** keeps its hook project-level (`~/.veronica/backends/codex/.codex/hooks.json`) and its MCP servers
  per-call, so nothing of yours is touched; with its shell off it falls back to a read-only sandbox, so it can
  read files but not write or run anything except through Veronica's tools.
- **Copilot** writes `~/.copilot/hooks/veronica.json` before every turn (scoped to her workspace) and removes it
  on close; each turn costs a **premium request** on your Copilot plan.
- Each brain's session id, hook log and workspace live under `~/.veronica/backends/<brain>/` (Claude keeps its
  session file where it always was).

## Calendar, mail, tasks, notes, timers

Veronica works with your **Google account**: Google Calendar (every calendar you have ticked in Google Calendar,
merged), Gmail (unread inbox, search across all mail, sending), Google Tasks (reminders) and Google Docs (notes,
kept in a "Veronica Notes" folder in your Drive). She can also set simple in-process timers ("set a timer for 5
minutes") that speak and show a Windows notification when they fire, even while she's idle.

One-time setup (about five minutes, free):

1. At https://console.cloud.google.com create a project.
2. **APIs & Services → Library**: enable Gmail API, Google Calendar API, Google Tasks API, People API and Google
   Drive API.
3. **OAuth consent screen**: External, publishing status "Testing", and add your own Google account as a test user.
4. **Credentials → Create credentials → OAuth client ID**, type **Desktop app**; download the JSON.
5. Save it as `%USERPROFILE%\.veronica\google_client.json`.
6. In Veronica: **Settings → General → Accounts → Connect Google**, sign in in the browser and allow every
   permission. The token is kept in `%USERPROFILE%\.veronica\google_token.json`; Disconnect revokes and deletes it.

While the app stays in "Testing", Google expires the sign-in after 7 days, so you reconnect weekly; publishing it
(unverified, for personal use) avoids that. Scopes asked for: calendar.readonly, calendar.events, gmail.readonly,
gmail.send, tasks, contacts.readonly, contacts.other.readonly (to find "mail Priya" in your contacts) and
drive.file (Veronica only ever sees the notes she created).

Notes:

- Google Tasks stores a due *date* only and never alerts at a time; a time you give ("remind me at 5") is written
  into the task as "Due at 17:00" and read back from there.
- Reading (calendar events, unread mail, mail search, tasks due, timers) runs automatically; creating an event or
  task and sending mail ask "Run …?" first, same as other risky actions. Mail by name confirms the resolved
  address ("Send mail to Priya Shah (priya@example.com)").
- Not connected yet? The tools say so ("Google isn't connected — …") instead of failing silently.

## Memory

Veronica keeps a small local memory across sessions, in a SQLite database at `~/.veronica/memory.db` (FTS5 full-text
search when the local Python's sqlite3 build has it, otherwise a plain substring search — either way the same
`recall`/`fact_add`/`fact_delete` behavior).

- Every completed turn (what you said, what she replied) is logged, so she can look it up later or carry a little
  recent context into a fresh Claude session.
- **Remember a fact** — "remember that I take my coffee black" / "remember I'm allergic to peanuts": stored as a
  fact and said back as "Got it." This is a local intent (matched before the brain runs), so it works even offline
  and doesn't cost a Claude turn. Say roughly the same thing again and the old wording is *replaced* rather than
  kept twice — she says "Updated — that replaces 'I take my coffee black'.", naming what went, because the match
  is fuzzy enough to get it wrong ("March 8" over "March 3") and that has to be audible.
- Every fact is filed under a kind — preference, person, place, routine or other — worked out from its wording when
  it is written (cue words, no model call). `facts_list` reads them back grouped under those headings.
- **Forget a fact** — "forget that I take my coffee black" / "forget the peanut thing": removes any matching fact
  and says "Forgotten." (or "I didn't have that." if nothing matched).
- **Forget a whole topic** — "forget everything about the office" / "forget anything related to Priya": removes
  every fact that topic turns up in and says how many went ("Forgot three things about the office."). Generic
  sweeps ("forget everything", "forget it") still delete nothing.
- Claude can also manage memory itself mid-conversation via MCP tools: `recall` (search past turns) and `facts_list`
  run automatically; `fact_add` and `fact_delete` both ask "Run …?" first — a fact persists across every future
  session, so it gets the same confirmation as anything else that changes standing state.
- On every new Claude session, Veronica injects a short "Facts about the user" list and the last few turns
  ("Recent conversation") into the system prompt, capped small (2 KB / 1 KB) so it stays cheap — an existing session
  already carries its own context, so this only matters right after a fresh one starts.
- The facts block holds at most `memory_facts_max` facts (40 by default, editable under Settings > Brain), most
  recently used first. "Used" is worked out from the words of her reply — after each turn, the facts whose
  distinctive words show up in what she just said are bumped to the front — so a long memory keeps the facts that
  actually come up and quietly drops the ones that never do (they stay in the database and her memory tools still
  find them).

Disable memory entirely (no DB, no injection, no remember/forget intents) with:

    VERONICA_MEMORY_ENABLED=false

## Screen awareness

Ask "what's on my screen", "look at my screen", "summarize this page/screen", or "what does this error say" and
Veronica takes a screenshot herself (downscaled to fit within 1568 px on the long edge) and
sends it to Claude along with your question in one turn — the HUD shows a "Look at screen" action line. Claude can
also decide to look at the screen on its own mid-conversation via the `screenshot` tool (allow-class, runs
automatically). With more than one monitor she captures the display your frontmost window is on (the app's real
window: helper strips such as Chrome's untitled 115 px-tall one, and small popups in front of the main window,
don't count); Claude can ask for
one by number (`display=2`, the primary display is 1) or for every screen at once (`display=all`), and clicks map back to the right monitor.
Windows needs no permission for this. Veronica runs per-monitor DPI aware, so screenshots and clicks are in
real pixels on scaled (125 %, 150 %) and mixed-DPI setups.

## Browser control

Veronica can read and act on the page you have open in **Google Chrome** or **Microsoft Edge**. Say "read this
page", "summarize this article", "find pricing on this page", "click the login button", "type hello in the search
box and press enter", or "open a new tab with github" — the brain picks the right browser tool for the request.

It works through a small extension that ships with Veronica (`veronica\browser_extension`). It keeps your normal
browser profile and logins, and talks only to Veronica on `127.0.0.1`. One-time setup:

1. Start Veronica once so it creates `%USERPROFILE%\.veronica\browser_token`.
2. Chrome: open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and pick the
   `veronica\browser_extension` folder. Edge: the same at `edge://extensions` (Developer mode is in the left
   sidebar).
3. On the options page that opens, paste the contents of `browser_token` and click Save — it should say
   "Connected to Veronica."

You can install it in both browsers; Veronica acts in whichever one was focused last. The default port is 8765;
change it with `VERONICA_BROWSER_PORT` and the same number in the extension's options. Reading, listing tabs,
opening a URL, finding text, scrolling and going back run automatically; **clicking** and **typing** always ask
for confirmation first, since they act inside your logged-in session. Browser pages (`chrome://`, `edge://`, the
web stores) are off limits to extensions, so she says she can't reach them.

## Computer use

Veronica can work the screen directly — any app, not just the browser. She takes a screenshot to see what's there,
then clicks, double-clicks, right-clicks, drags, scrolls, types and presses key combos. Try "click the Save button",
"type hello in that box and press enter", "scroll down", "press ctrl S", "double-click the file", or
"where's the Save button?" (find only — she points, she doesn't click).

**Opening things.** "Open This PC", "open Downloads", "open the Recycle Bin", "open Bluetooth settings", "open
Device Manager", "open Control Panel", "open Task Manager" and about eighty other Windows folders, Settings pages and
system tools open directly; any app in the Start menu (Chrome, Spotify, Calculator, Word …) opens by name.

**Permission.** Windows asks for none. One limit: Windows silently drops clicks and keys aimed at a program running
**as administrator** (Registry Editor, an elevated terminal, installers) unless Veronica itself runs as
administrator — she tells you when that's the case instead of clicking into nothing.

**Trust window.** Looking, moving the mouse, scrolling and finding text run automatically. Clicks, drags, typing
and key presses ask for confirmation — but only once per app: after you say yes, further actions in the **same app**
run without asking for the next **90 seconds** (the HUD shows them with the normal "auto" pill). Switching to a
different app asks again, the window expires on its own, and it's cleared when you barge in, say "that's all", or
answer no. Change the length (or set it to 0 to be asked every time — that also closes a window that's already
open) in **Settings → Brain → Screen-control trust window**. Two things the window never covers: **pressing Enter**
(a "type … and press enter" or "press return" is confirmed every time — it submits whatever is in front), and
**terminals** (Windows Terminal, PowerShell, Command Prompt, Git Bash, WSL and friends — no window opens there and none applies,
so every screen action in a terminal is confirmed on its own).

**Safety rules.** She never types into a password field; she refuses combos that close apps, lock, sign out or
open the security screen (alt+f4, win+l, ctrl+alt+delete, ctrl+shift+esc, win+x and friends). Windows security
prompts — User Account Control, credential prompts, SmartScreen, Windows Security — and **Windows Settings** are
handled as a last line: every action there needs its own confirmation regardless of the trust window, and even
once confirmed the tools refuse blind clicks, drags, typing and Enter/Space while such a window is in front — the
only buttons she will click there are No / Don't run / Cancel / Close and the like, so she can never press Yes,
Allow or Run on a security prompt (UAC's secure desktop can't receive synthetic input anyway).

## Push-to-talk

Hold **Right Ctrl** to talk to Veronica without saying the wake word — release it when you're done. Works even
while she's speaking (it interrupts her, like saying the wake word does). Disable with `VERONICA_PTT_ENABLED=false`,
or change the key with `VERONICA_PTT_KEYCODE` (a Windows virtual-key code; 163 / 0xA3 is Right Ctrl, 165 is Right
Alt — which is AltGr on many layouts). It's a global low-level keyboard hook, so no permission is needed; if the
hook can't be installed the tray shows "Push-to-talk unavailable".

## Music

"Pause" / "pause music", "resume" / "play music", "next song" / "skip", "previous", and "what's playing" control
whatever is playing through Windows' media controls — Spotify, a browser tab, any media app — with no round-trip to
the brain. The brain can also use `music_play`, `music_pause`, `music_next`, `music_prev`, `music_now_playing` and
`music_volume` (all allow-class).

**"Play <song>" actually plays it:**

- **On Spotify**, when it's connected: the top track (or the artist, for "play Coldplay", or "playlist <name>") starts
  in your Spotify app — she opens Spotify first if it isn't running. Needs **Spotify Premium** (Spotify only allows
  playback control on Premium). Setup: at https://developer.spotify.com/dashboard create an app, select "Web API",
  add the redirect URI `http://127.0.0.1:8898/callback` (exactly `127.0.0.1`, not `localhost`), copy its Client ID
  into **Settings → General → Accounts → Spotify Client ID**, then **Connect Spotify** and approve in the browser.
- **On YouTube** otherwise (or on a free Spotify account): she finds the top video for the song and opens it in your
  browser, where it starts playing. No account needed.

## Notes & dictation

- **Take a note** — "take a note: buy milk" / "note that the wifi password is abc123": creates a Google Doc in your
  "Veronica Notes" Drive folder titled with the first 40 characters of what you said plus a timestamp, and says
  "Noted." (needs Google connected — see Calendar, mail, tasks, notes, timers).
- **Dictate** — "dictate" / "start dictation": say "Go ahead.", then listen until you say "stop dictation" or pause
  for 3 seconds, and types everything you said into whichever app is currently focused (Unicode keyboard input, so
  Hindi works too; not into apps running as administrator).

## PowerShell

`system.powershell` runs a PowerShell script (Windows PowerShell 5.1, `-NoProfile`, 2-minute limit) and hands back
its output. It is **always confirmed** — it can do
anything you can, so it's never auto-allowed, never covered by "just do it", and never offered to the local model.

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
also in the tray menu (also shown when you click the orb) under **Voice** (the voice list plus Faster / Slower / Normal speed).

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
  "mute karo" / "awaaz band karo", "chhoti ho jao" / "badi ho jao" / "kahan ho" for the HUD, "haan" / "ji" / "nahi" answer a
  "Run …?" confirmation, plus the quick replies above. In pinned Hindi mode whisper writes Devanagari, so the
  common ones are understood in script as well ("बस करो", "हाँ" / "नहीं", "समय क्या है").
- **Hindi voices** — Alpha, Beta (female), Omega, Psi (male). "Use a hindi voice" / "use the omega voice" picks
  the voice Hindi replies use (the English voice is untouched, so both show a checkmark in the **Voice** menu),
  confirmed with "ठीक है, अब मैं ऐसे बोलूँगी।" Prefetch the models without switching:
  `uv run python scripts/download_models.py --hindi`.

## Briefings & nudges

- **Brief me** — "brief me" / "give me a briefing" / "what's my day look like": a spoken summary of today's
  calendar, unread mail count and reminders due, composed locally from Google Calendar, Gmail and Google Tasks.
- **Daily briefing** — "give me a briefing every morning at 8" / "start the briefing every day at 6 pm" turns on a
  scheduled briefing at that time ("turn on the morning briefing" keeps the stored time, default 08:00); "stop the
  morning briefing" / "turn off briefings" turns it off. A briefing more than two hours late (the PC was asleep) is
  skipped rather than read out mid-afternoon.
- **Meeting nudges** — "warn me 10 minutes before my meetings" / "remind me before my meetings" / "turn on nudges"
  announces "Heads up, <event> starts in 10 minutes." before each timed calendar event (1–60 minutes, default 5);
  "turn off nudges" / "stop the meeting nudges" turns them off.
- **Quiet hours** — off by default; once switched on in Settings (default 22:00–08:00), anything that comes due inside
  the window waits instead of being dropped and is spoken when it ends, the first one prefixed "While you were away:"
  if more than one waited. A nudge whose moment has passed by then (the meeting already happened) is dropped, and
  at most ten wait at once — past that she just adds "And 4 more I held back."
- **Snooze** — "snooze notifications for an hour" / "mute nudges until 5" / "notifications rok do" holds
  announcements until then ("Okay, quiet until 5 pm."); "resume notifications" / "unsnooze" releases them. A snooze
  lasts an hour by default and doesn't survive a restart.
- **Low battery** — off by default: below 15% on battery she says "Battery's at 12 percent.", once per discharge
  (plugging in arms it again).
- **Unread mail** — off by default: "You have 7 unread since this morning." once a day at the hour set in Settings
  (default 11:00), silent when the inbox is clear.

Briefings and nudges are announcements: they're spoken only when Veronica is idle and not muted (anything that
fires mid-conversation or while muted waits, like a timer), and the schedule persists in `~/.veronica/prefs.json`.

## Background noise & only my voice

- **Background noise** — on by default (Settings → Listening → "Reduce background noise"). A small offline
  noise-suppression network (GTCRN, 0.5 MB, fetched on first launch into `~/.veronica/models`) cleans what the
  speech detector hears, so keyboard clatter, dishes and music no longer open or stretch a recording, and a
  "Speech level floor" (`vad_min_rms`, default 0.001, only while suppression runs) keeps faint background talk
  out while you from across the room still get in. Whisper itself still transcribes the unprocessed audio — measured, it understands noisy
  speech better than cleaned-up speech — and the wake word check is untouched.
- **Only my voice** — say "Veronica, learn my voice" ("meri awaaz yaad rakho"), or press **Learn my voice** in
  Settings → Listening. She reads three short lines; repeat each after the beep. From then on a request,
  follow-up, dictated line or yes/no answer in someone else's voice (the TV, a person in the room) is ignored as
  if nothing was said, with an "Ignored another voice" card on the HUD: it never runs, and it never approves or
  redirects a confirm — she just listens once more, and if it's still not you the action is skipped.
  Push-to-talk is always trusted (the key is the proof), and "forget my voice" always gets through, so a
  profile that stops matching you can't lock you out. The wake word stays open to anyone unless you also turn
  on "Only wake for my voice" (off by default: a missed wake is worse than a stray one — which also means
  someone else saying "Veronica" can still interrupt her). If the voice model can't load she hears everyone
  and says so once on the HUD and in Settings.
- **Forget it** — "forget my voice" / "meri awaaz bhool jao", or **Forget my voice** in Settings. The profile is
  one small file, `~/.veronica/voice_profile.json` (readable only by you); the speaker model (CAM++, 29 MB,
  SHA256-pinned) is fetched the first time you enrol.
- **Tuning** — every check is logged: `speaker request: score=0.62 threshold=0.35 -> accepted (1.9 s voiced of
  3.1 s, 40 ms)` in `~/.veronica/logs/veronica.log`, and the last few show under the Learn/Forget buttons.
  In testing (synthetic voices) the enrolled voice scored 0.5–0.8 on whole sentences and other voices mostly
  under 0.3; real voices will differ, so watch your own scores. If she ignores you, lower "Voice match
  strictness" (`speaker_threshold`, default 0.35) a little below your low scores; if someone else gets through,
  raise it. Short answers ("yes", "haan") get a proportionally lower bar, but in a loud room say "yes, go
  ahead" or hold the push-to-talk key. Re-enrol ("learn my voice") after changing mics.

How the defaults were chosen, with the numbers:
`docs/superpowers/specs/2026-10-01-veronica-voice-isolation-design.md`. To re-measure on your machine (synthetic
voices, no microphone): `uv run python scripts/eval_voice_isolation.py --quick`.

## Settings window

A normal window (tabs: General, Voice, Listening, Briefings, Brain, History, About) for everything that
used to need an environment variable or a voice command.

- **Open it** — say "open settings" / "settings" / "preferences" / "settings kholo", pick "Settings…" from the tray menu (or
  left-click the tray icon), or click the HUD orb and choose "Settings…". "Show history" / "what did I ask you" / "history dikhao" opens
  it straight on the History tab.
- **Live settings** apply to the running app right away and persist: language mode, voice, Hindi voice, speed (each
  spoken back so you hear the change), HUD mode, hide delay, follow-up window, confirm listen, silence and
  utterance limits, briefing/nudge schedule, facts carried into a new conversation, start at login,
  push-to-talk, noise suppression, the speech level floor and the voice check.
- **Restart settings** are saved but only picked up on the next launch: wake sensitivity/window/hop, wake phrases,
  brain effort, memory on/off, working folder. Changing one shows a "Restart Veronica to apply" banner with a
  Restart button (from the built `Veronica.exe` it quits and relaunches itself once the old process has exited; from a
  terminal it quits and says "Restart me from the terminal.").

Values you set here override the environment/`.env` defaults (they're stored in `~/.veronica/prefs.json`).

## History

The History tab lists past turns (what you said, what she replied) from the local memory database, with a search
box. Each row has a Forget button; "Clear all" (with a confirm step) removes them all. Facts you asked her to
remember are separate (see Memory) and aren't touched by clearing history. With memory disabled the tab just says
so.

## Version & updates

- **"What version are you"** / "version" / "kaunsa version hai" — says e.g. "Veronica 0.1.0 (a517483, 17 Sep)":
  the package version plus the commit that's actually running (from the exe's `build.json` when launched as
  the app, else live from git). The same line sits at the top of the tray menu ("About Veronica — …") and on
  the About tab.
- **"Update yourself"** / "update now" / "check for updates" / "apna update karo" — checks the repo: if
  `origin` has newer commits it says "Updating, back in a moment.", runs `git pull --ff-only`, `uv sync`
  and rebuilds `dist\Veronica\Veronica.exe`, then relaunches. If there's no remote (or nothing new upstream) but the running
  build is behind the checked-out code, "update" just rebuilds and restarts you onto the latest local code. Already
  current: "You're already on the latest." Couldn't reach the remote: "Couldn't check for updates, check the log."
  Anything failing mid-update: "The update failed, check the log." Only one update runs at a time — a second
  "update yourself" (or the window/menu) while one is running gets "An update is already running." / "Busy, try
  again in a moment.". From a terminal run (no exe to reopen) it finishes with "Update installed. Restart me
  from the terminal."
- **Tray** — "Check for Updates…" runs the same check and posts a Windows notification (if it can't, she says the
  result instead, next time she's idle); when something newer exists
  the item below it becomes "Update available — Restart to update" (click to install). Veronica also checks quietly
  once an hour and only flips that item, no notification. The About tab has "Check now", "Update & restart",
  "Restart" and "Open log".

Updates are refused while a conversation is in progress from the window/menu ("Busy, try again in a moment."); the
spoken "update yourself" is itself the turn, so it just runs. There are no API keys involved — updating is a git
pull plus rebuild of the local checkout.

## Permissions

Windows asks for very little:

- **Microphone** — Settings → Privacy & security → Microphone → "Let desktop apps access your microphone" must be on.
- **Notifications** — timers and update notices use Windows toasts; Focus / Do not disturb hides them (she still
  speaks the timer).
- **Administrator** — not needed. Run Veronica normally; only run it as administrator if you want her to click
  into elevated windows (see Computer use).
- **Google / Spotify** — your own OAuth sign-ins, connected from Settings → General → Accounts.

## Run
    uv run python -m veronica                 # tray app
    uv run python -m veronica --text "hello"  # no audio, debug

## Install as an app

Build a real `Veronica.exe` (PyInstaller, one folder) instead of running from a terminal:

    uv run python scripts/build_app.py     # or: make app — writes dist\Veronica\Veronica.exe
    .\dist\Veronica\Veronica.exe

(`uv run python scripts/make_icon.py` re-renders `assets\Veronica.ico` from the HUD orb if you want a fresh icon —
the built one is committed.) Signing is optional (`signtool sign /fd sha256 …` with your own certificate); unsigned,
SmartScreen may warn once.

The exe uses this checkout: it changes into the repo (so `.env` applies), and the brains' helper processes run with
the checkout's `.venv\Scripts\python.exe`, so keep the `.venv`, models and CLI logins you set up for `uv run`.
`VERONICA_HOME` (default `%USERPROFILE%\.veronica`) is unchanged.

**Start at Login** — the tray's "Start at Login" item writes a `Veronica` value under
`HKCU\Software\Microsoft\Windows\CurrentVersion\Run` pointing at `Veronica.exe`. It's greyed out until you
launch Veronica from the built exe at least once. You can also review it under Settings → Apps → Startup.

Logs: `%USERPROFILE%\.veronica\logs\veronica.log`.

## Troubleshooting

**She only wakes when I lean into the mic.** The wake check ignores audio quieter than the "Wake sensitivity
(min level)" gate (`wake_min_rms`, default 0.003). Lower it in Settings to hear you from across the room (you
may get more false wakes; raise it if she wakes on noise). "Wake window" and "Wake hop" (`wake_window_s`,
`wake_hop_s`) control how much audio each check sees and how often it runs. Run with
`VERONICA_LOG_LEVEL=DEBUG` and watch `~/.veronica/logs/veronica.log` for `wake hop rms=... gate=...` lines
to see how loud your voice actually lands at the mic.

**Bluetooth headset / USB mic / headphones.** Mic switching is automatic: Veronica polls Windows' default
recording device every couple of seconds and reopens the mic on the new device (`input device changed (...);
reopening mic` in the log), also re-reading the output device list so speech follows your headphones. The switch
waits until any in-flight recording finishes. If the mic disappears mid-sentence (headset off, the PC sleeping),
the recording ends within about two seconds with what it already heard (`capture: no audio for 2.0s` in the log)
instead of holding up the turn. To pick a device, set it as the default in Settings → System → Sound.

**She stopped hearing me after a call / after switching mics.** Call apps with auto-gain (Zoom, Meet,
Teams) and device switches can quietly drop the microphone level to ~30 %, which starves the wake check.
Veronica checks the input volume once a minute and right after every mic switch, and raises it back to the
"Input volume floor" setting (`input_volume_floor`, default 85; never lowers it). The first fix of a session
shows on the HUD; every fix is an `input volume 33 → 85 (...)` line in the log. Set the floor to 0 to turn it off.

**Browser tools say no browser is connected.** Check the extension's options page says "Connected to Veronica";
re-paste `%USERPROFILE%\.veronica\browser_token` if Veronica's home folder was reset.

**"Google isn't connected".** Settings → General → Accounts → Connect Google. In "Testing" mode the sign-in lasts
7 days.

## Test
    uv run pytest            # unit
    uv run pytest -m live    # needs mic/speaker/models/login
    uv run pytest -m live tests/test_brains_live.py   # the real brain CLIs; each skips unless installed + logged in
    uv run playwright install chromium   # once, for the live HUD tests
