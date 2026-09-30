# Veronica

macOS voice assistant. Say "Hey Veronica" once you have trained the custom model (see scripts/train_wakeword.md); until then say "Hey Jarvis", ask, listen.
Brain = Codex, Antigravity, Claude or Copilot, each through its own CLI login (no API keys; see Brains below).
Speech = local (faster-whisper + Kokoro).

## Setup
    brew install uv portaudio
    uv venv --python 3.12 && uv pip install -e ".[dev]"
    uv run python scripts/download_models.py
    npm i -g @openai/codex && codex login    # the default brain; others under Brains below

Wake word: say "Veronica" or "hey Veronica" (whisper engine, default). To use the lighter openwakeword engine set VERONICA_WAKE_ENGINE=openwakeword (falls back to "hey jarvis" until you train a custom model — see scripts/train_wakeword.md).

Wake word not triggering? Run `uv run python scripts/wake_scores.py`, say the phrase, and (openwakeword engine) set VERONICA_WAKE_THRESHOLD in .env just below the scores you see.

Do not set `ANTHROPIC_API_KEY` (or any other vendor key) — every brain uses the login of its own CLI (`codex login`,
`agy`, `claude`, `copilot login`); keys are ignored if set.

- macOS will ask for Microphone access for your terminal app on first run (System Settings → Privacy & Security → Microphone).
- First run downloads the whisper `small.en` model (~470 MB). Hindi mode needs the multilingual `small`/`tiny`
  models too (~500 MB more) — fetched the first time you say "speak hindi", or ahead of time with
  `uv run python scripts/download_models.py --hindi`.
- Say the wake word while Veronica is talking to interrupt her (barge-in).
- Interrupt with "hold on" / "wait" / "one sec" ("ruko", "ek minute") and she stops but keeps the rest of the
  answer: say "continue" / "carry on" / "go on" ("aage bolo") and she picks up at the next sentence. Anything
  else you say is treated as a new request and the remainder is dropped; "stop" / "that's all" still cancels.
- If a slow answer leaves her silent for more than 3.5 s she says "On it." once (Settings → Listening, or
  `VERONICA_ACK_AFTER_S`; 0 turns it off).
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
- **Thinking…** — the brain is working on a reply (the line under it says which one: "Brain: Codex").
- **Speaking** — she's talking.
- **Say yes or no** — she's asked for confirmation before a risky action and is listening for your answer; the
  question itself appears above, and the mic-level bar is still shown while she listens for it. During a
  confirmation, anything that isn't yes/no is taken as your next request ("no, open it in Safari instead",
  "what will that do?"): the action is skipped and she answers that instead. Silence skips it too.
- **Error** — something went wrong; check the log.

**Pre-approval by wording.** If the request itself already says go ahead — "copy this to the clipboard, just do
it", "open chrome and go ahead", "add the reminder without asking", "haan kar do abhi" — she skips the yes/no for
the **one** action that request produces (the HUD shows it with a gold "pre-approved" pill). It's one-shot and
short-lived (20 s, the first confirmable action of that request only); a second action in the same request is
asked as usual, and the pre-approval never covers sending mail or messages, `rm -r`, force-pushes, shutdown /
restart / sleep, `sudo`, pressing Enter, typing into a terminal, or anything on a system dialog — those are
confirmed every time, however you phrase it. A bare "do it" or "yes" is an answer, not a request, and a question
("should I do it?") never pre-approves. Turn it off in **Settings → Brain → Pre-approve when I say "do it"**.

**Auto-allow tools.** If you've approved a tool once, it can stay approved. Answer a confirmation with "always"
— or "don't ask again", "stop asking", "mat pucho" — and that tool stops asking, for good: the call goes ahead
and the tool is added to **Settings → Brain → Auto-allow tools**, which survives a restart. She only offers the
option for tools that are allowed to be there; from then on those calls show a "auto" pill in the HUD, the same
as anything else that runs without asking. Seven tools are eligible, and **only** these can ever be added:

| Tool | What it does |
| --- | --- |
| `mcp__mac__clipboard_write` | Copy to the clipboard — **ticked by default** |
| `mcp__pim__calendar_create` | Create a calendar event |
| `mcp__pim__reminder_create` | Create a reminder |
| `mcp__memory__fact_add` | Remember a fact |
| `mcp__memory__fact_delete` | Forget a fact |
| `mcp__browser__browser_click` | Click in the browser |
| `mcp__browser__browser_type` | Type in the browser (including a typed Enter) |

**Destructive tools can never be added**, whichever way you try. Sending mail or messages, AppleScript, every
screen-control action (`mcp__computer__*`), running a Shortcut and the shell are not on the list, so saying
"always" to one of them approves that single call and she answers "That one I'll always ask about." Typing one
into the free-form field by hand does nothing either: `policy.classify` only honours names in
`policy.AUTO_ALLOWABLE`, and `policy.always_confirm` is checked first, so a hand-typed `mcp__pim__mail_send` or
`mcp__computer__computer_click` still asks every single time.

The Settings section has a checkbox per eligible tool plus the full list as a text field, for review and for
revoking: untick one, or clear the field, and she starts asking again immediately — no restart.

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
- **HUD gone? say "where are you"** — also "show yourself", "come back", "reset the hud": forgets the saved
  position and shows the HUD at its default spot on the main screen (she answers "Here I am."). The HUD also
  re-checks its position on its own whenever a display is plugged in or unplugged, so it can't be stranded on a
  monitor that's no longer there.

A turn that runs more than one tool shows them in the full card as a checklist: ○ waiting on your yes, ▸ running,
✓ done, ✕ declined or failed — failed when the tool itself returned an error. (A Codex, Antigravity or Copilot
built-in tool, such as their own shell, reports no result to Veronica, so its step only ever shows done.)

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

## Brains

The "brain" is whichever coding-agent CLI does the thinking. Veronica runs it as a subprocess (or, for Claude, via
the Agent SDK) and hands it her own tools; the speech, the HUD and the confirmation gate are the same whichever
one is active. Every brain uses the vendor CLI's own login — there are no API keys anywhere.

| Brain | What it is | Install | Log in |
| --- | --- | --- | --- |
| **Codex** (default) | OpenAI's `codex` CLI, on your ChatGPT plan | `npm i -g @openai/codex` | `codex login` |
| **Antigravity** | Google's `agy` CLI (Gemini), on your Google account | `curl -fsSL https://antigravity.google/cli/install.sh \| bash` | `agy` |
| **Claude** | Anthropic's `claude` CLI (Claude Code), on your Claude plan | `npm i -g @anthropic-ai/claude-code` | `claude` |
| **Copilot** | GitHub's `copilot` CLI, on your Copilot plan | `npm i -g @github/copilot` | `copilot login` |
| **Local** | llama.cpp on this Mac — no account, no network | a `llama-server` binary and a `.gguf` | — |

Only the brains that are installed *and* logged in are offered; the check is local and cheap (the binary on
`PATH` plus the file the login writes — `~/.codex/auth.json`, `~/.copilot/config.json`, Antigravity's
`~/.gemini/antigravity-cli/conversations` or its Keychain item; Claude reports a missing login itself). An
unavailable one is spoken as "Codex isn't installed — run npm i -g @openai/codex, then codex login." or "Codex
isn't logged in — run codex login in a terminal."

**Choosing one.** The preferred brain is **Settings → Brain → Brain** (`brain_backend`, default `codex`, live — no
restart). The menu bar has a **Brain: Codex** submenu with one radio item per brain (unavailable ones read
"Copilot (not installed)" / "(not logged in)" and are disabled). By voice, without a brain round-trip: "switch to
codex", "use antigravity", "use copilot", "back to claude" / "go back to claude", "switch brain to codex",
"codex pe switch karo", "copilot use karo"; she answers "Switched to Codex." or "Already on Codex." (or the
install/login hint above). "Which brain are you on" / "which model is this" / "who am I talking to" / "kaunsa
brain hai" answers "I'm on Codex." A switch interrupts whatever the current brain was doing and clears any
screen-control trust window; each brain keeps its own resumable session, so switching back picks up where it
left off. The HUD's status area shows "Brain: Codex" and the settings page's Brain tab shows the same label.

**The confirm gate applies to every brain.** Two paths, one gate:

- *Veronica's tools* (`mac`, `pim` for calendar/mail/reminders/timers, `memory`, `screen`, `music`, `browser`,
  `computer`) are served to an external brain as MCP servers over stdio (`python -m veronica.tools.serve <name>`,
  registered as `veronica-<name>`). That process is only a proxy: each call goes to the app's gate socket
  (`~/.veronica/gate.sock`), which asks the same question the in-process gate asks Claude — policy, trust window,
  pre-approval, voice confirm — so "Run …?" sounds and behaves exactly the same, and then **runs the tool inside
  the app** and sends the result (text, or the screenshot's image) back. That last part matters: macOS attributes
  what a helper process does to the CLI that spawned it, so a capture, a key press or an Apple Event issued from
  the stdio child would be checked against permissions the CLI was never granted. Running it in the app is also
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

The **Local** brain is a [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` running on this Mac,
with a quantised model file. Nothing leaves the machine: no account, no login, no network call, not even to
check for one.

She starts the server herself on the first local turn (`llama-server --model <gguf> --ctx-size 8192 --host
127.0.0.1 --port 8749 --jinja --no-webui`), waits for `/health`, and leaves it running until she is closed or
ten minutes pass without a turn — a model load costs seconds, so it is worth keeping warm. A server already
listening on that port is used as-is rather than replaced.

**Settings → Brain → Offline:** *Use the local model when offline* (`brain_offline_fallback`, default on),
*Local model* (`local_model`, default `~/Github/sih/manas/models/granite-4.2-3b-q4_k_m.gguf` — small, fast and
instruction-tuned), *Local server* (`local_server_bin`, default `~/Github/sih/manas/runtime/bin/llama-server`),
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

**What it can and cannot do.** Tools: **yes** — Veronica's own tools (mac, calendar/mail/reminders/timers,
memory, music, browser, screen control) are offered to the model as function schemas and run in-process, each
call through the same confirm gate, with the same HUD cards and the same trust window. The web: **no** — there
is no search and no fetch, and she is told to say so rather than guess. Screenshots: **no** — the default model
is text-only, so she says she can't see. A small model that ignores the tool schema simply answers in words;
that is normal and not an error. Expect a short answer in a handful of seconds, and expect it to be less sharp
than the hosted brains — it is a three-billion-parameter model on a laptop.

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
one by number (`display=2`) or for every screen at once (`display=all`), and clicks map back to the right monitor.
Requires **Screen Recording** access — see Permissions below.

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

## Computer use

Veronica can work the screen directly — any app, not just the browser. She takes a screenshot to see what's there,
then clicks, double-clicks, right-clicks, drags, scrolls, types and presses key combos. Try "click the Save button",
"type hello in that box and press enter", "scroll down", "press command S", "double-click the file", or
"where's the Save button?" (find only — she points, she doesn't click).

**Permission.** Screen control needs **Accessibility** access. The first action prompts for it; Veronica must then
be enabled under **System Settings → Privacy & Security → Accessibility** (see Permissions below). Until it's
granted, every action reports the missing permission instead of acting.

**Trust window.** Looking, moving the mouse, scrolling and finding text run automatically. Clicks, drags, typing
and key presses ask for confirmation — but only once per app: after you say yes, further actions in the **same app**
run without asking for the next **90 seconds** (the HUD shows them with the normal "auto" pill). Switching to a
different app asks again, the window expires on its own, and it's cleared when you barge in, say "that's all", or
answer no. Change the length (or set it to 0 to be asked every time — that also closes a window that's already
open) in **Settings → Brain → Screen-control trust window**. Two things the window never covers: **pressing Enter**
(a "type … and press enter" or "press return" is confirmed every time — it submits whatever is in front), and
**terminals** (Terminal, iTerm2, Warp, kitty, WezTerm, Alacritty, Ghostty — no window opens there and none applies,
so every screen action in a terminal is confirmed on its own).

**Safety rules.** She never types into a password field; she refuses combos that quit apps, force-quit, log out,
lock or restart the Mac (cmd+q, cmd+opt+esc, cmd+ctrl+q and friends). System permission dialogs and **System
Settings** (any pane) are handled as a last line: every action there needs its own confirmation regardless of
the trust window, and even once confirmed the tools themselves refuse blind clicks, drags, typing and Enter/Space
while such a window is frontmost — the only buttons she will click there are Don't Allow / Deny / Cancel / Not
Now / Close and the like, so she can never press Allow or OK (nor an OCR misread of them) on a permission prompt.

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

## Shortcuts & Messages

| Tool | Can | Cannot |
| --- | --- | --- |
| `mac.run_shortcut` | Run any shortcut installed in Shortcuts.app by name ("run the Morning shortcut"), case-insensitively, with optional text input handed over as a file (`shortcuts run <name> --input-path …`). A shortcut gets 2 minutes. | Create or edit shortcuts, or hand back what one returned — the CLI prints nothing on success, so she just says she ran it. A name that isn't installed is refused ("there's no shortcut called 'Morning' on this Mac") rather than guessed at. |
| `pim.message_send` | Send one iMessage/SMS through Messages.app to a phone number, an Apple ID, or a contact by name ("message Priya: on my way") — the name is looked up in Contacts. | Read your messages, send attachments, or guess: two Priyas ("Which Priya — Priya Shah or Priya Nair?"), a contact with several numbers, or no match comes back as a question instead of a send. |

**Which shortcuts run without asking.** Every shortcut is confirm-class by default — a shortcut is a program you
wrote, and Veronica can't see what's in it. Settings → Brain → "Shortcuts she may run without asking" is a
comma-separated list of names (empty out of the box); a shortcut whose name is on that list runs straight away,
everything else still asks "Run the shortcut 'X'?" first.

**Sending a message always asks.** `message_send` is in `policy.always_confirm` alongside sending mail: the
screen-control trust window never covers it, saying "just do it" in your request never pre-approves it, and no
setting turns the question off. The confirm reads the resolved contact, their handle and the first 40 characters
— "Message Priya Shah (+91 98765 43210): on my way" — and the send goes to that exact handle.

Messages.app needs the usual one-time automation permission the first time she sends (System Settings → Privacy &
Security → Automation). Messaging someone by name asks once for Contacts access (System Settings → Privacy &
Security → Contacts); without it she says so and asks for the number. Shortcuts must have been opened once for `shortcuts list` to report anything.

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
  "mute karo" / "awaaz band karo", "chhoti ho jao" / "badi ho jao" / "kahan ho" for the HUD, "haan" / "ji" / "nahi" answer a
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

A normal macOS window (tabs: General, Voice, Listening, Briefings, Brain, History, About) for everything that
used to need an environment variable or a voice command.

- **Open it** — say "open settings" / "settings" / "preferences" / "settings kholo", pick "Settings…" from the menu
  bar, or click the HUD orb and choose "Settings…". "Show history" / "what did I ask you" / "history dikhao" opens
  it straight on the History tab.
- **Live settings** apply to the running app right away and persist: language mode, voice, Hindi voice, speed (each
  spoken back so you hear the change), HUD mode, hide delay, follow-up window, confirm listen, silence and
  utterance limits, briefing/nudge schedule, facts carried into a new conversation, start at login,
  push-to-talk, noise suppression, the speech level floor and the voice check.
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
- **Input Monitoring** — push-to-talk's global hotkey (asked for on first launch). Not asked for automatically;
  grant it yourself, or use the menu bar's "Enable Push-to-talk… (Input Monitoring)" item if push-to-talk shows as
  unavailable.
- **Accessibility** — computer use (clicking/typing on screen; asked automatically the first time a screen action
  runs) and dictation's typing into other apps.

## Run
    uv run python -m veronica                 # menu bar app
    uv run python -m veronica --text "hello"  # no audio, debug

## Install as an app
### Permissions that stick

macOS ties Microphone, Screen Recording and Accessibility grants to an app's
code signature. Ad-hoc signing is just a hash of the bundle, so every rebuild
asks again. Run this once:

```bash
./scripts/make_signing_cert.sh     # asks for your login password once
make app
```

`make app` then signs with that certificate, the grant is keyed on the bundle
id plus the certificate, and rebuilds keep it. Grant Microphone and Screen
Recording one final time after the first certificate-signed build.


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
doesn't change. **Accessibility** is the exception to watch: macOS ties that grant to the launcher's code hash
(cdhash), which is stable as long as the launcher binary itself doesn't change — rebuilding the app around the
same launcher keeps the grant, but a rebuilt or updated launcher needs Accessibility re-granted (remove and re-add
Veronica in System Settings → Privacy & Security → Accessibility).

The bundle's launcher just `cd`s into this repo and execs `.venv/bin/python -m veronica`, so it needs the same
`.venv` (and `.env`, models, the brain CLI logins) you set up for `uv run` — there's no separate install step.
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
in-flight recording finishes. If the mic disappears mid-sentence (AirPods taken out, the Mac sleeping), the
recording ends within about two seconds with what it already heard (`capture: no audio for 2.0s` in the log)
instead of holding up the turn.

**She stopped hearing me after a call / after switching mics.** Call apps with auto-gain (Zoom, Meet,
FaceTime) and device switches quietly drop the Mac's input volume to ~30 %, which starves the wake check.
Veronica checks the input volume once a minute and right after every mic switch, and raises it back to the
"Input volume floor" setting (`input_volume_floor`, default 85; never lowers it). The first fix of a session
shows on the HUD; every fix is an `input volume 33 → 85 (...)` line in the log. Set the floor to 0 to turn it off.

## Test
    uv run pytest            # unit
    uv run pytest -m live    # needs mic/speaker/models/login
    uv run pytest -m live tests/test_brains_live.py   # the real brain CLIs; each skips unless installed + logged in
    uv run playwright install chromium   # once, for the live HUD tests
