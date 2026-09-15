# Veronica — Voice Agent Design

Date: 2026-09-15
Status: approved

## Goal

Veronica is a macOS menu-bar voice assistant. Say "Hey Veronica", ask a question or give a task, hear the answer. It answers questions, controls the Mac, looks things up on the web, and manages calendar/email/reminders/timers. Zero API keys: the brain runs on the user's Claude Code subscription via the Claude Agent SDK; speech runs locally.

## Decisions

| Area | Choice | Why |
|---|---|---|
| Platform | macOS, single Python process, `rumps` menu bar | Local system control; fastest to build |
| Activation | Wake word "Hey Veronica" (openwakeword, local) | Hands-free |
| STT | `faster-whisper` (`base.en`, CTranslate2) | Local, free, fast on Apple Silicon |
| TTS | Kokoro (`kokoro-onnx`), streamed per sentence | Local, free, good voice |
| Brain | `claude-agent-sdk` `query()` using Claude Code login | No API key; built-in Bash/Read/Write/WebSearch/WebFetch |
| Safety | Risky tools require spoken confirmation | Misheard command must not destroy things |
| Memory | SQLite + FTS5 in `~/.veronica/history.db` | Cross-session recall |

Constraints: personal use on the user's own machine only (subscription auth terms). Subscription rate limits are shared with Claude Code.

## Architecture

```
mic ──> WakeWord (openwakeword)
          ▼
        Recorder (sounddevice + silero VAD) → 16 kHz PCM
          ▼
        STT: faster-whisper
          ▼
        Brain: claude-agent-sdk query(), resumed session
          │  built-in tools: Bash, Read, Write, Edit, WebSearch, WebFetch
          │  custom in-process MCP tools: mac.*, google.*, timer.*, memory.recall
          │  can_use_tool hook: risky → voice confirm
          ▼
        TTS: Kokoro → speaker
          ▼
        8 s follow-up window (VAD only) → else back to WakeWord
```

### Package layout

```
veronica/
  audio/    wake.py      # openwakeword loop, emits "wake" events
            record.py    # VAD-segmented capture
            play.py      # PCM playback queue, cancellable
  speech/   stt.py       # faster-whisper adapter
            tts.py       # Kokoro adapter, sentence → PCM
  brain/    agent.py     # Agent SDK session, streaming, tool gate
            prompts.py   # system prompt + memory injection
            memory.py    # SQLite history + FTS5 recall
  tools/    mac.py       # open_app, clipboard, notify, volume
            google.py    # calendar read/create, gmail read/send (OAuth)
            timers.py    # set/list/cancel timers + reminders
  ui/       menubar.py   # rumps: state icon, mute, quit
  main.py               # asyncio orchestrator / state machine
  config.py             # pydantic-settings; .env only for Google OAuth
tests/
```

Each unit has one job and a small interface: `wake.wait()`, `record.capture() -> bytes`, `stt.transcribe(pcm) -> str`, `brain.ask(text) -> AsyncIterator[str]` (sentences), `tts.speak(sentence)`, `play.stop()`.

## Data flow / turn-taking

1. **Idle.** openwakeword scores 80 ms frames. Threshold hit → chime, state `listening`.
2. **Record** until silero VAD sees 700 ms silence, or 15 s cap. Under 300 ms of speech → ignore, back to idle.
3. **STT.** Empty result → speak "Sorry, didn't catch that", open follow-up window.
4. **Brain.** `query(prompt, options)` with `resume=<session_id>`. Stream `AssistantMessage` text → sentence splitter → TTS queue immediately. First audio target ≈1 s after brain starts.
5. **Tool gate.** `can_use_tool` callback:
   - Allow without asking: `Read`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, `mac.open_app`, `mac.clipboard_read`, `google.calendar_read`, `google.gmail_read`, `timer.*`, `memory.recall`.
   - Confirm: `Bash`, `Write`, `Edit`, `mac.clipboard_write`, `google.gmail_send`, `google.calendar_create`, anything else. Flow: pause TTS → speak "Run `<one-line summary>`?" → listen 5 s → `yes / do it / go / confirm` → allow; anything else → deny with reason "user declined".
6. **Follow-up.** After the reply, 8 s open-mic (VAD only). Speech → step 3. Silence → idle.
7. **Barge-in.** Wake word during TTS → `play.stop()`, go to step 2.

### System prompt

"You are Veronica, a voice assistant running on Mani's Mac. Reply in one to three spoken sentences. No markdown, no lists, no code unless asked. For long answers, give the short version and offer to say more. Today is {date}. Recent context: {last-20-turn summary}."

Options: `effort="low"`, `max_turns=8`, `permission_mode` default with `can_use_tool` callback, `resume` session id persisted in `~/.veronica/session`.

## Memory

- `~/.veronica/history.db`: table `turns(id, ts, user_text, reply_text)`, FTS5 virtual table over both.
- On start: summarise last 20 turns into the system prompt.
- `memory.recall(query)` MCP tool: FTS5 search, returns top 5 turns.

## Error handling

| Failure | Behaviour |
|---|---|
| STT empty / low confidence | "Say again?" + follow-up window |
| Brain > 60 s | cancel, "Taking too long, cancelled" |
| Claude usage limit | speak "Claude limit hit, try again in N minutes" (parse from error) |
| Claude Code not logged in | menu bar red, speak "Claude Code isn't logged in" |
| No audio device | menu bar red, log, retry every 10 s |
| Tool error | pass `is_error` result back to model; model explains in one sentence |

Logs: `~/.veronica/logs/veronica.log`, rotating 5 × 5 MB. Per-turn latency log per stage (wake→STT→first token→first audio).

## Phases

1. **Core loop** — wake word → record → STT → brain (no custom tools, all built-ins gated to confirm) → TTS. Menu bar. `--text` debug mode.
2. **Mac tools** — `mac.py` MCP tools, risk classifier, voice confirm.
3. **Web** — enable `WebSearch`/`WebFetch` in allowlist; prompt tuning for spoken summaries.
4. **Google + timers** — OAuth flow, calendar/gmail tools, timers with spoken alerts.
5. **Memory** — SQLite history, summary injection, `memory.recall`.

## Testing

- Unit: sentence splitter, VAD segmenter (fixture WAVs), risk classifier, every MCP tool with `subprocess`/Google clients mocked.
- Integration: `tests/e2e_text.py` — text in → text out through the Agent SDK against the real subscription (marked `@pytest.mark.live`).
- Manual smoke: `python -m veronica --text "what time is it"` bypasses audio.
- Latency target: wake → first audio < 3 s on M-series Mac.
