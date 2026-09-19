# Veronica Brains — pluggable brain backends (Claude, Codex, Antigravity, Copilot, Qwen) + auto-failover

Date: 2026-09-19. Branch `brains` (worktree off master), merged to `master` as one unit. Approved by the user with the "native shell via socket confirm" gate model. Gemini CLI was dropped (retired for individual accounts on 2026-06-18; its Google-side replacement is Antigravity CLI) and Copilot CLI + automatic failover on usage limits were added at the user's request.

## Global constraints (apply to every task)

- Every backend uses the vendor CLI's own login. **No API key**, ever, for any backend. Claude stays on the Claude Code subscription via `claude-agent-sdk`.
- Confirm-gate stays strict: only `veronica/brain/policy.py` `classify()` and the designed exemptions (computer trust window) may auto-allow a tool. Never `allowed_tools`; never run an external CLI in an auto-approve mode without our hook as the gate **and** the canary (§3.4) watching that the hook fires.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q` hermetic — no real CLI spawned, no real socket outside a temp dir. Real-CLI tests carry the `live` marker.
- Python 3.12, `uv`. Follow existing patterns (`_ok`/`_err`/`_guard` in tools; `Settings` + `EDITABLE_SETTINGS`; local intents in `brain/intents.py`; settings rows hand-listed in `ui/settings/settings.js`).
- Verified CLI versions at design time: `codex-cli 0.155.1`, `agy 1.2.7` (Antigravity), `copilot 1.0.86`, `qwen 0.24.0`. Flags below were read from `--help` and the bundled sources; the plan re-verifies each against the installed binary in a `live` test.

## Goal

"Switch to Codex" → next turn runs on Codex CLI with the same tools, the same voice confirm, the same HUD cards. "Back to Claude" → resumes the saved Claude session. The brain is a swappable adapter; everything else (STT, TTS, orchestrator, policy, tools, memory) is unchanged.

## 1. Protocol and package layout

- `veronica/brain/base.py`
  ```python
  class Brain(Protocol):
      name: str                                   # "claude" | "codex" | "antigravity" | "copilot" | "qwen"
      def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]: ...   # sentences
      async def interrupt(self) -> None: ...
      async def close(self) -> None: ...
      def clear_trust(self) -> None: ...
  ```
  This is the interface the orchestrator already uses; no orchestrator call sites change.
- `veronica/brain/backends/claude.py`: today's `Brain` class from `agent.py`, renamed `ClaudeBrain`, behavior unchanged except the permission gate delegates to `ToolGate` (§2). `veronica/brain/agent.py` keeps `summarize_tool`, `summarize_detail`, `Confirm`, the `*_PREFIX` constants, `TRUST_EXCLUDED_BUNDLES`, `_always_confirms`, and re-exports `Brain = ClaudeBrain` so existing imports/tests keep working.
- `veronica/brain/backends/antigravity.py` (`AntigravityBrain`), `codex.py` (`CodexBrain`), `copilot.py` (`CopilotBrain`), `qwen.py` (`QwenBrain`). Shared subprocess plumbing in `veronica/brain/backends/cli.py` (§3.1).
- `veronica/brain/backends/__init__.py`:
  - `BACKENDS: dict[str, BackendInfo]` with `label` ("Claude", "Codex", "Antigravity", "Copilot", "Qwen"), `binary` (`claude` is not spawned by us — `claude_agent_sdk` finds it; listed for the availability check only), `install_cmd`, `login_cmd`, `login_marker` (path or env var):
    - claude: `claude`, `npm i -g @anthropic-ai/claude-code`, `claude`, marker `~/.claude/.credentials.json` **or** `claude auth status` exit 0 (checked lazily; never block startup).
    - codex: `codex`, `npm i -g @openai/codex`, `codex login`, `~/.codex/auth.json`.
    - antigravity: `agy`, `curl -fsSL https://antigravity.google/cli/install.sh | bash`, `agy` (browser Google login; credentials in the macOS Keychain), marker = `agy --version` ok **and** a keychain item exists (`security find-generic-password -s antigravity` — exact service name confirmed in T3) — fallback marker: `~/.gemini/antigravity-cli/conversations/` exists.
    - copilot: `copilot`, `npm i -g @github/copilot`, `copilot` then `/login` (GitHub Copilot subscription), `~/.copilot/` contains a token/config written by login (exact file confirmed in T5).
    - qwen: `qwen`, `npm i -g @qwen-code/qwen-code`, `qwen`, `~/.qwen/oauth_creds.json`.
  - `check_backend(name, *, which=shutil.which, exists=Path.exists) -> Availability(ok: bool, reason: str, hint: str)`. `reason` ∈ `"ok" | "not installed" | "not logged in"`; `hint` is the spoken sentence: "Codex isn't installed — run npm i dash g at openai slash codex, then codex login." (spoken form; the exact command also goes to the log and the HUD card).
  - `make_brain(name, settings, *, gate, on_tool, memory) -> Brain`. `BrainSwitcher` (in `veronica/brain/switch.py`): owns the current `Brain`, `current.name`, `async switch(name) -> Availability` (checks, closes the old brain — its session file stays so switching back resumes — instantiates the new one, writes `settings.brain_backend` through prefs). The orchestrator holds the switcher and reads `switcher.brain` per turn (so a change takes effect on the **next** turn; an in-flight turn finishes on the old backend).

## 2. The gate, shared

- `veronica/brain/gate.py` `ToolGate`: the logic of today's `Brain._can_use_tool` + `_gate_computer` + trust window, lifted out of the Claude backend: `async decide(tool_name, input) -> Decision(allow: bool, reason: str)`, `clear_trust()`, `on_tool` callback for HUD cards. Constructed once by the orchestrator with `confirm=self.confirm`, `frontmost`, `clock`. `ClaudeBrain` wraps it in `can_use_tool` (returning `PermissionResultAllow/Deny`). Bash redirect (`screencapture` → screenshot tool) stays in the gate.
- `GateServer` (same module): asyncio Unix-socket server at `settings.gate_socket` (`~/.veronica/gate.sock`, dir 0700, socket 0600, stale file unlinked on start). One JSON line per request: `{"v":1,"tool":"mcp__mac__open_app","input":{...},"origin":"mcp"|"hook","backend":"codex"}` → `{"allow":true}` or `{"allow":false,"reason":"user declined"}`. Requests are serialized (one confirm at a time — the orchestrator's `confirm()` is not reentrant). Started in `Orchestrator.run_forever` next to proactive; stopped on shutdown. Tool-name mapping for native tools happens in the hook (§3.3), not here — the gate only ever sees our canonical names (`Bash`, `Write`, `Edit`, `Read`, `mcp__<server>__<tool>`).
- `veronica/brain/gateclient.py`: sync `ask_gate(tool, input, *, origin, backend, sock=env VERONICA_GATE_SOCK, timeout=None) -> Decision`. Used by the two out-of-process entry points. **Fail closed**: socket missing/refused/malformed → `Decision(False, "Veronica's gate isn't reachable")`. No timeout while waiting for the answer other than the orchestrator's own confirm timeout (a voice confirm can take a while); a `VERONICA_GATE_TIMEOUT_S` env is honoured for tests.

## 3. External CLI backends

### 3.1 Common plumbing — `backends/cli.py`

- `CliBrain` base: `spawn(argv, *, cwd, env, stdin_text=None)` via `asyncio.create_subprocess_exec` (injectable `_spawn` for tests); reads stdout lines, hands each to the subclass's `parse(line) -> list[Event]` where `Event` is `Text(delta)`, `ToolStart(call_id, tool, input)`, `ToolEnd(call_id)`, `Session(id)`, `Done()`, `Error(msg)`; stderr captured to `~/.veronica/backends/<b>/last-stderr.log`.
- `ask()`: builds the prompt (system prompt is passed per §3.2; user text as the positional/`-p` prompt or stdin; images per backend), spawns, yields sentences from `SentenceSplitter` fed by `Text` events; on `ToolStart` emits `on_tool(summary, "auto")` **only** for native read-only tools the hook allows without consulting the gate (read_file/grep/web_search…); every gated call (MCP via `serve`, native shell/edit via the hook) gets its card from the gate itself ("ask"/"allowed"/"declined"/"auto"), so no tool is carded twice; on `Session` saves the per-backend session file; `Done` ends the turn; `Error` → yield "<Label> returned an error, check the log." and reset the session if the error text matches the same overflow markers as the Claude backend.
- Whole-reply fallback: if the stream produced no `Text` deltas but the final message is present (a backend's non-streaming `json` mode, codex `item.completed` without deltas), feed the final text through the splitter at the end — TTS still works, just not overlapped.
- `interrupt()`: if a child is running, `SIGINT`, wait `interrupt_drain_s`, then `kill()`. Mark the turn ended. `close()` = interrupt + drop handles.
- `brain_timeout_s` applies per line-read as today; on timeout kill the child and speak "Taking too long, cancelled."
- Workspace: `settings.backend_dir(name)` = `~/.veronica/backends/<name>/` (0700). Regenerated on every spawn (idempotent writes): the CLI's project-local config (§3.2) and a `hook.log` (truncated per turn). `cwd` of the child = this workspace **not** `brain_cwd` — the workspace is where the CLI looks for project config; shell commands the model runs are told (in the system prompt) that the user's working folder is `brain_cwd`, and the hook accepts `cd`-less commands as today.
- Native tool summaries for HUD cards: shell → `Run: <command[:60]>`, file write/edit → `Write file <basename>` / `Edit file <basename>`, MCP → `summarize_detail(canonical_name, input)`. Canonical MCP name = `mcp__<server>__<tool>` rebuilt from the CLI's naming (qwen/antigravity: `<tool>` with server in the event's `serverName` or prefixed `<server>__<tool>`; copilot: `veronica-<server>__<tool>`; codex: `mcp_tool_call` item carries `server` + `tool`).

### 3.2 Per-backend invocation

| | Antigravity (`agy`) | Copilot (`copilot`) | Codex (`codex`) | Qwen (`qwen`) |
|---|---|---|---|---|
| headless | `-p <text> --output-format stream-json --print-timeout 0` | `-p <text> --output-format json --silent --no-ask-user` | `exec --json --skip-git-repo-check -C <backend_dir> -c approval_policy="never" -c sandbox_mode="workspace-write" -c 'sandbox_workspace_write.writable_roots=["<brain_cwd>"]' --dangerously-bypass-hook-trust <text>` | `<text> -o stream-json --approval-mode yolo` |
| system prompt | `<backend_dir>/AGENTS.md` rewritten per turn (agy reads workspace instructions); if T3 finds a system-prompt/agent-file flag, use that instead | `<backend_dir>/.github/copilot-instructions.md` rewritten per turn — T5 confirms it is read in `-p` mode; the prompt text is also prefixed to the user message as a fallback if not | `-c developer_instructions=<toml string>` (+ `-c include_permissions_instructions=false`) | `--system-prompt <text>` |
| session | first turn plain; later `--conversation <id>` (id from the `init` event) | `--session-id <uuid4>` on the first turn, `--resume <id>` after | first turn `codex exec …`, later `codex exec resume <thread_id> …` (`thread.started`) | `--session-id <uuid4>` then `--resume <id>` |
| images | write to `<backend_dir>/img-N.jpg\|png`, prepend `@<path> ` to the prompt (T3 verifies agy honours file references; else `_image_fallback_text`) | `--attachment <path>` per image | `-i <path>` per image | `@<path>` references |
| permissions | `permissions.allow` in `<backend_dir>/.agy/settings.json` (T3 confirms project scope; else the user-level `~/.gemini/antigravity-cli/settings.json` `projects.<dir>` scope) = our MCP tools + `run_command`/edit tools; the hook is the gate | `--allow-all-tools` with the hook as the gate (Copilot preToolUse fails closed on non-zero exit — the safest of the four) | `approval_policy="never"` with the hook as the gate | `--approval-mode yolo` with the hook as the gate |
| MCP config | `agy mcp add --env VERONICA_GATE_SOCK=… veronica-<name> <venv python> -m veronica.tools.serve <name>` run once per launch (idempotent "add or update") — user-level config, names prefixed `veronica-` so they never collide with the user's own servers | `--additional-mcp-config '<json>'` per invocation: `{"mcpServers":{"veronica-<name>":{"type":"local","command":"<venv python>","args":["-m","veronica.tools.serve","<name>"],"env":{"VERONICA_GATE_SOCK":"…"},"tools":["*"]}}}` | `-c 'mcp_servers.<name>.command="<venv python>"' -c 'mcp_servers.<name>.args=[…]' -c 'mcp_servers.<name>.env.VERONICA_GATE_SOCK="…"'` | `<backend_dir>/.qwen/settings.json` `mcpServers.<name> = {command, args, env, trust: true, timeout: 120000}` |
| hooks | `<backend_dir>/.agy/hooks.json` (T3 confirms path; else user-level) `{"hooks":{"PreToolUse":[{"matcher":"*","hooks":[{"type":"command","command":"<venv python> -m veronica.brain.hook antigravity"}]}]}}` | `<backend_dir>/.github/hooks/veronica.json` `{"version":1,"hooks":{"preToolUse":[{"type":"command","bash":"<venv python> -m veronica.brain.hook copilot"}]}}` | `<backend_dir>/.codex/hooks.json` PreToolUse + `-c features.hooks=true` | `.qwen/settings.json` `hooks.BeforeTool` |
| native tools disabled (fallback, §3.4) | `permissions.deny` for `run_command`/edit tools; MCP stays allowed | `--allow-tool 'veronica-*'` only (no `--allow-all-tools`) — Copilot soft-denies the rest | `-c sandbox_mode="read-only"` | `--approval-mode plan` + `tools.exclude` |
| stream format | NDJSON `init`, `step_update` (text deltas + tool calls), `result` | JSONL: assistant message deltas, tool call/result events, final result with session id | JSONL `thread.started`, `item.started/updated/completed{item.type: agent_message\|command_execution\|mcp_tool_call\|file_change}`, `turn.completed`, `turn.failed` | `init`, `message{delta}`, `tool_use`, `tool_result`, `result` |

The exact field names in the "stream format" row are the ones each backend task captures as fixtures from the real CLIs (`tests/fixtures/brains/<backend>-*.jsonl`) — parsers are written against captured output, not this table. Every "T3/T5 confirms" note above is a verification step in the plan, not an open design question: the fallback is stated inline.

`effort` maps: codex `-c model_reasoning_effort="…"`, antigravity `--effort <low|medium|high>`; copilot/qwen: ignored. Model: CLI default (no setting — YAGNI).

### 3.3 Hook — `python -m veronica.brain.hook <backend>`

- Reads one JSON object from stdin. Antigravity/Codex/Copilot payloads are Claude-style (`tool_name`, `tool_input`, `session_id`; Copilot uses camelCase `toolName`/`toolArgs` — normalised); Qwen `BeforeTool` (`tool_name`, `tool_input`).
- Maps to canonical: shell tools (`run_command`, `run_shell_command`, `Bash`, `shell`, `local_shell`, `bash`) → `Bash{command}`; write/edit tools (`write_file`, `write_to_file`, `Write`, `replace`, `edit`, `edit_file`, `apply_patch`, `str_replace_editor`) → `Write`/`Edit`; read-only tools (`read_file`, `view`, `glob`, `grep`, `list_directory`, `find`, `web_fetch`, `web_search`, `google_web_search`, `fetch`) → allow without asking (same class as today's Read/Grep for Claude). Tools of our MCP servers (`mcp__veronica-<server>__<tool>`, `veronica-<server>__<tool>`, `<server>__<tool>` — normalised to `mcp__<server>__<tool>`) → allow immediately, **no second prompt** (already gated inside `veronica.tools.serve`). Any other native tool → confirm-class with summary `"<tool_name>"`.
- Appends one line to `<backend_dir>/hook.log`: `{"ts","call":<tool_name>,"key":<command|file_path|tool>,"decision"}` **before** contacting the gate (the canary needs the line even if the gate call hangs).
- Asks `ask_gate(canonical, input, origin="hook", backend=…)`.
- Output — allow: Claude-style backends (antigravity, codex, copilot) `{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"allow"}}` exit 0 (Copilot: `{"permissionDecision":"allow"}` — T5 confirms the exact key against the hooks reference; the module has one `emit(backend, decision, reason)` table); Qwen: exit 0, no output. Deny: Claude-style `permissionDecision: "deny"` + `permissionDecisionReason`; Qwen `{"decision":"deny","reason":…}`.
- Any exception → deny (fail closed) with reason `"gate error"`.

### 3.4 Canary (the hook is the only gate under auto-approve)

- The adapter watches the event stream for **native** tool starts (any tool call event whose tool is not one of ours). For each, after the tool completes (or 3 s), it checks `hook.log` for a line with the same `key` written since the turn began. Missing → `log.error`, `kill()` the child, speak "Hooks aren't running on <Label>, so I've turned off its shell. Tools still work.", set `settings.<backend>_native_tools = False` via prefs (persisted), and re-run the same user text once in fallback mode (§3.2 "native tools disabled" row).
- `<backend>_native_tools` (bool, default True, editable under Brain) is also the user's manual switch to "MCP-only".

### 3.5 `python -m veronica.tools.serve <name>`

- `veronica/tools/serve.py`: looks up `<name>_server` in the existing modules, takes `.instance` (`mcp.server.Server`), wraps its `call_tool` handler: `d = ask_gate(f"mcp__{name}__{tool}", args, origin="mcp", backend=env VERONICA_BRAIN)`; deny → `{"content":[{"type":"text","text":"Not allowed: <reason>"}],"isError":true}`; allow → original handler. Serves over `mcp.server.stdio.stdio_server`. Logging to stderr only (stdout is the protocol).
- The gate's `on_tool` still fires the HUD "ask/allowed/declined/auto" cards, so cards look identical across backends.

## 4. Settings, UI, voice

- `Settings.brain_backend: str = "codex"` (choice `codex|antigravity|claude|copilot|qwen`, editable "Brain", live) — the **preferred** brain (user's default order: Codex, then Antigravity, then Claude). At startup, if the preferred brain is unavailable (not installed / not logged in) the switcher starts on the first available one in `brain_failover_order` and says once "Codex isn't logged in, so I'm on Claude for now."; it returns to the preferred brain automatically when the next availability check (every 60 s while standing in) passes. `Settings.gate_socket: Path`, `Settings.backend_dir(name)`, `Settings.session_file_for(name)` (`session_file` stays the Claude one). `codex_native_tools`, `antigravity_native_tools`, `copilot_native_tools`, `qwen_native_tools: bool = True` (editable, live). Failover settings in §4a. Rows added to `SETTING_SECTIONS` **and** hand-listed in `settings.js` under Brain.
- HUD: status label shows `Brain: <Label>` (new `hud` event field `backend`; emitted on start and on switch). Menu bar: submenu "Brain: <Label>" with one radio item per backend; unavailable ones rendered "Copilot (not installed)" / "(not logged in)" and disabled; the item active by failover (§4a) reads "Codex — standing in for Claude"; selecting one calls `switcher.switch`.
- Local intents (`intents.py`, no brain round-trip): `switch_brain(name)` from "switch to codex", "use antigravity", "use copilot", "use qwen", "go back to claude", "back to claude", "switch brain to …", Hinglish "codex pe switch karo"/"copilot use karo"; `which_brain` from "which brain are you on", "which model/brain is this", "who am i talking to". Replies: "Switched to Copilot." / "Already on Copilot." / availability hint / "I'm on Claude." / while failed over: "I'm on Codex — Claude hit its limit at 2:10, I'll try it again at 3:10."
- Orchestrator: on switch, current turn (if any) is interrupted first; `_trust` cleared; `hud("backend")` emitted.

### 4a. Automatic failover on usage limits

- Trigger: a turn ends with a **limit error** — the backend reports an error whose text matches `LIMIT_MARKERS` (`backends/cli.py`): `usage limit`, `rate limit`, `rate_limit`, `429`, `quota`, `resource exhausted`, `too many requests`, `limit reached`, `out of credits`, `insufficient_quota`, `overloaded` (per-backend extras captured with the fixtures). The Claude backend checks the same list on its `ResultMessage` error path. Context-overflow markers keep their existing "fresh conversation" handling and never trigger failover.
- Action (when `brain_failover: bool = True`, editable under Brain): `BrainSwitcher.failover(reason)` picks the next backend in `brain_failover_order: str = "codex,antigravity,claude,copilot,qwen"` (editable; the preferred one is implicitly first) that `check_backend` reports available and that hasn't hit a limit within `brain_limit_cooldown_min: int = 60`; marks the failing backend `limited_until = now + cooldown`; speaks "Claude hit its usage limit — switching to Codex."; re-runs the **same user text once** on the new backend (sentences already spoken by the failed turn are not repeated — the re-run is a fresh turn on the stand-in's own session). No available stand-in → speak "Claude hit its usage limit and no other brain is ready." and stop.
- The preference (`brain_backend`) is **not** changed by failover; the active brain is a runtime override. Return: before each turn, if the active brain is a stand-in and the preferred one's `limited_until` has passed, switch back silently (HUD label updates; the next reply comes from the preferred brain). A manual "switch to …" clears the override and any cooldown for that backend.
- A stand-in that also hits its limit fails over again down the order (each with its own cooldown); at most one re-run per user turn per hop, and a backend already in cooldown is never retried in the same chain, so it cannot loop.
- HUD label while failed over: `Brain: Codex (for Claude)`; a HUD tool card "Claude: usage limit — on Codex until 3:10".
- Tests: fake backends raising limit errors; order/availability/cooldown selection; single re-run; cooldown expiry switches back before the next turn; `brain_failover=False` speaks the error as today; manual switch clears cooldown; chain of two limits ends after the second hop.

## 5. Docs

README "Brains" section: what each backend is, install + login commands, how the confirm gate applies (MCP servers via stdio + hook for the CLI's own shell; the canary), the `*_native_tools` switch, automatic failover and its settings, known limits (codex read-only fallback; Antigravity MCP entries are user-level, prefixed `veronica-`), voice phrases.

## 6. Tests

- `tests/test_gate.py`: `ToolGate.decide` reproduces today's `test_agent.py` gate cases (moved, not duplicated); `GateServer` round-trip over a temp socket with a fake confirm; serialized confirms; malformed request → deny; `ask_gate` fail-closed when the socket is missing.
- `tests/test_hook.py`: each mapping row; log line written before the gate call; Claude-style vs qwen output shapes; exception → deny.
- `tests/test_tools_serve.py`: wrapped `call_tool` denies/allows via a fake gate; stdio server starts and lists tools (in-memory transport from the `mcp` package).
- `tests/test_backend_antigravity.py`, `test_backend_codex.py`, `test_backend_copilot.py`, `test_backend_qwen.py`: fake `_spawn` replaying `tests/fixtures/brains/*.jsonl` → sentences in order, tool events with expected summaries, session id saved and reused in argv, images → files/flags, error → spoken error + session reset on overflow markers, timeout kills child, `interrupt()` sends SIGINT then kill, canary kill + fallback rerun + setting persisted, whole-reply fallback.
- `tests/test_backends_registry.py`: `check_backend` matrix (missing binary / missing marker / ok), `make_brain`, `BrainSwitcher.switch` (closes old, keeps its session file, writes prefs, refuses when unavailable).
- `tests/test_intents.py` / `test_orchestrator.py`: switch/which intents, unavailable → hint spoken and no switch, in-flight turn interrupted on switch, HUD backend event.
- `tests/test_config.py` / `test_settings_bridge.py` / `test_settings_web.py`: new fields and rows.
- `live` (skipped by default): `tests/test_brains_live.py` — for each installed+logged-in CLI (agy, codex, copilot, qwen): capture fixtures (`--capture-fixtures` option rewrites `tests/fixtures/brains/`), one real turn "say the word pineapple" yields a sentence containing it, one MCP call (`read_battery`) goes through the gate, one native shell call (`echo hi`) hits the hook (log line present).

## 7. Order & branching

Worktree `brains` off master (after `particle-orb`, confirm-redirect and pre-approval merge): T1 gate extraction + `ClaudeBrain` (all existing tests green) → T2 `serve` + `hook` + `gateclient` → T3 `cli.py` + `AntigravityBrain` (+ fixture capture) → T4 `CodexBrain` → T5 `CopilotBrain` → T6 `QwenBrain` → T7 registry/switcher/failover/settings/intents/HUD/menubar → T8 README. Reviewer per task, whole-branch review, merge `--no-ff`, `make app`, relaunch. The user logs into `agy`, `codex`, `copilot` (and `qwen` if wanted) before T3–T6's live tests.
