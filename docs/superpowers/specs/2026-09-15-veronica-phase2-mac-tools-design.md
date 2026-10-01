# Veronica Phase 2 — Mac Tools, Barge-in, Speech Quality

Date: 2026-09-15
Status: approved
Builds on: `2026-09-15-veronica-voice-agent-design.md` (Phase 1 merged at e57267e)

## Goal

Veronica can control the Mac by voice with a risk-aware confirmation gate, can be interrupted mid-sentence by the wake word, hears better, and answers faster.

## 1. Risk classifier (tool gate)

New module `veronica/brain/policy.py`:

```python
Decision = Literal["allow", "confirm"]
def classify(tool_name: str, input: dict) -> Decision
```

| tool | decision |
|---|---|
| `Read`, `Glob`, `Grep`, `WebSearch`, `WebFetch` | allow |
| `Write`, `Edit`, `NotebookEdit`, any unknown tool | confirm |
| `Bash` | allow iff **single simple command** (see below), else confirm |
| `mcp__mac__*` | per-tool table in §2 |

**Bash single simple command:** after `shlex.split`, the raw command string contains none of `| & ; > < $( ` (backtick) \n`, and the first token is in
`SAFE_BASH = {ls, cat, head, tail, pwd, date, cal, whoami, pbpaste, df, du, ps, which, echo, uptime, wc, file, stat}`,
or the first token is `open` and the argv matches exactly `open -a <App>` or `open https://… | http://…`. Anything else → confirm. `shlex` failure → confirm.

`Brain._can_use_tool` calls `classify`; "allow" → `PermissionResultAllow` (logged at INFO `auto-allow: <summary>`), "confirm" → existing voice `confirm(summary)` path; deny message stays `user declined`.

## 2. Typed Mac tools — in-process MCP server

`veronica/tools/mac.py` defines tools with `claude_agent_sdk.tool` and exposes `mac_server = create_sdk_mcp_server(name="mac", version="1.0.0", tools=[...])`. `Brain._options` adds `mcp_servers={"mac": mac_server}` and `allowed_tools=[f"mcp__mac__{n}" for n in MAC_TOOL_NAMES]` (visibility only; gate still applies).

| tool | input schema | action | risk |
|---|---|---|---|
| `open_app` | `{"name": str}` | `open -a <name>` | allow |
| `open_url` | `{"url": str}` (must start with `http://` or `https://`, else error) | `open <url>` | allow |
| `clipboard_read` | `{}` | `pbpaste` | allow |
| `clipboard_write` | `{"text": str}` | `pbcopy` (stdin) | confirm |
| `notify` | `{"title": str, "message": str}` | `osascript -e 'display notification …'` | allow |
| `volume_get` | `{}` | `osascript -e 'output volume of (get volume settings)'` | allow |
| `volume_set` | `{"level": int}` (0–100, clamped) | `osascript -e 'set volume output volume N'` | allow |
| `applescript` | `{"script": str}` | `osascript -e <script>` | confirm |

Every tool: `subprocess.run(..., capture_output=True, text=True, timeout=10)`; success → `{"content":[{"type":"text","text": stdout or "ok"}]}`; non-zero exit or exception → `{"content":[{"type":"text","text": f"error: {stderr or exc}"}], "is_error": True}`. AppleScript strings are passed as separate argv elements (never shell-interpolated).

`summarize_tool` extensions: `mcp__mac__open_app` → `Open <name>`; `open_url` → `Open <url>`; `clipboard_write` → `Copy to clipboard: <first 60 chars>`; `applescript` → `Run AppleScript: <first 60 chars>`; others → tool short name.

## 3. Barge-in

- While state ∈ {thinking, speaking, followup}, a background task runs `WakeWord.wait()` on its own input stream (PortAudio on macOS allows concurrent input streams; Task 1 of the plan verifies live and, if it fails, switches `WakeWord`/`Recorder` to a shared `AudioHub` that fans one `RawInputStream` out to subscribers — the interface of both classes is unchanged either way).
- On detection: `player.stop()`, `brain.interrupt()` (new method → `ClaudeSDKClient.interrupt()` then drain), cancel the current `handle_text`, state → `listening`, chime, `one_turn` restarts at recording. The follow-up window is skipped.
- Self-trigger guard: while `player.is_playing`, a detection requires score ≥ `settings.wake_threshold_while_speaking` (default 0.8).
- `Brain.interrupt()` also resets the splitter and is safe when no turn is active.

## 4. Speech quality and latency

- `Settings.whisper_model` default `"small.en"`; `vad_silence_ms` 700 → 800.
- Wake chime: `audio/chime.py` plays a 120 ms 880 Hz sine (float32, 24 kHz) via `Player` immediately after detection; also a lower 100 ms tone when the follow-up window opens.
- Pre-warm: `build_orchestrator` constructs `Transcriber` and `Synthesizer` eagerly; `Orchestrator.warmup()` runs one dummy `synth("ok")` and `transcribe(zeros)` before `run_forever`; menubar shows `V …` until warm.
- Pipelined TTS in `handle_text`: producer synthesizes sentence N+1 while N plays (bounded `asyncio.Queue(maxsize=2)`); ordering preserved; `stop()` drains the queue.

## 5. Backlog fixes

- After a successful wake retry, state returns to `idle`.
- `confirm()` when muted → return False without speaking.
- `--text` mode: "confirm"-class tools prompt `Run <summary>? [y/N] ` on stdin (via `asyncio.to_thread(input)`); "allow"-class run silently. Banner text updated accordingly.
- New state `confirming` (menubar icon `?`) while listening for yes/no.
- README: mic permission (System Settings → Privacy → Microphone for your terminal), first-run whisper download, `small.en` note, barge-in usage.

## 6. Testing

- `tests/test_policy.py`: table-driven ≥ 25 cases incl. `ls; rm -rf ~`, `ls $(rm x)`, `` ls `rm x` ``, `open -a Safari && rm x`, `open file:///etc/passwd`, `cat a | grep b`, `echo hi > f`, `sudo ls`, `pbpaste`, `open https://x.y`, unknown tool, each `mcp__mac__*` name.
- `tests/test_mac_tools.py`: each tool with `subprocess.run` mocked (argv asserted; no shell); url validation; volume clamping; error mapping to `is_error`. One `@pytest.mark.live` test: `open_app("Finder")`.
- `tests/test_orchestrator.py` additions: barge-in with a fake background wake (stops player, calls `brain.interrupt`, state sequence), pipelined TTS ordering under a slow fake synth, muted confirm, confirming state, wake-retry idle.
- `tests/test_agent.py` additions: classify wiring (allow path does not call confirm; confirm path does), `interrupt()`.
- `tests/test_main.py`: `--text` y/N prompt path with `input` faked.
- Live/manual: dual-input-stream check script; say "hey jarvis" mid-answer; "open Safari"; "what's on my clipboard"; "set volume to 30".
