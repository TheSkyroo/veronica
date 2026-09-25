# Veronica Batch F — offline brain, interruption, plan card, proactive v2, memory v2, Shortcuts

Date: 2026-09-26. Branch `batch-f` (worktree), merged to `master` as one unit. Continues the roadmap agreed after Batch E and the brains work.

## Global constraints (apply to every task)

- Brains stay on the user's own logins (Codex default, then Antigravity, Claude, Copilot). **No API keys**, ever. The new local brain is a process on this Mac — no network at all.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` plus the designed exemptions (computer trust window, one-shot pre-approval by wording) are the only things that may auto-allow. Everything an external or local brain does reaches `ToolGate.decide` — over the gate socket for out-of-process callers. Never set `allowed_tools`.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests hermetic: `uv run pytest -q`; anything that needs a real model, device or CLI carries the `live` marker.
- Python 3.12, `uv`. Follow existing patterns (`Settings` + `EDITABLE_SETTINGS` + `SETTING_SECTIONS` + hand-listed `settingRow` in `settings.js`; local intents in `brain/intents.py`; short spoken copy; dense "why" comments in `orchestrator.py`).

## F1 — Offline brain (local llama.cpp), and automatic use when the network is gone

The user already has a local stack on this Mac: `~/Github/sih/manas/runtime/bin/llama-server` (llama.cpp build 10700) and GGUF weights in `~/Github/sih/manas/models` (`qwen2.5-coder-3b-instruct-q4_k_m`, `granite-4.2-3b`, `minicpm5-1b`, `lfm2.5-8b-a1b`, `gpt-oss-20b-mxfp4`, …).

- New backend `local` (`veronica/brain/backends/local.py`, `LocalBrain`), registered in `BACKENDS` like the others: label "Local", no install/login command, available when both the server binary and the chosen model file exist.
- Settings: `local_server_bin: Path` (default `~/Github/sih/manas/runtime/bin/llama-server`), `local_model: Path` (default the `granite-4.2-3b-q4_k_m.gguf` — small, fast, instruction-tuned), `local_ctx: int = 8192`, `local_port: int = 8749` (fixed, ours), all editable under Brain. A Settings pane row lists the `.gguf` files found next to `local_model` so the user can switch model without typing a path.
- Lifecycle: the server is started lazily on the first local turn (`--model <gguf> --ctx-size <n> --port <p> --host 127.0.0.1 --jinja`), health-checked on `/health`, and left running (a model load costs seconds) until `close()` or 10 idle minutes. Startup failure speaks "The local model wouldn't start — check the Local settings." and the switcher falls back.
- Turns: POST `/v1/chat/completions` with `stream: true`; the system prompt is `prompts.system_prompt(...)` as usual; the conversation is kept in memory (a rolling window bounded by `local_ctx`, oldest turns dropped) since there is no vendor session to resume. Streamed deltas feed `SentenceSplitter`, so TTS overlaps exactly as with the other brains.
- Tools: v1 exposes Veronica's own tools through llama.cpp's OpenAI-style function calling (`tools=[...]` built from the same MCP servers, `--jinja` so the model's template supports it). Every call goes through `ToolGate.decide` in-process — same confirms, same HUD cards, same trust window. A model that ignores the tool schema simply answers in words; that is acceptable and must not error.
- Automatic offline use: a lightweight reachability check (`veronica/net.py` `online(timeout=1.5)` — a TCP connect to the vendor host the active brain needs, cached for 20 s) runs before a turn. When the active brain needs the network and it is gone, `BrainSwitcher` activates `local` exactly like a failover: "No internet — switching to the local model." and back silently when connectivity returns (same cooldown/return machinery as the usage-limit path; the reason is recorded so "which brain are you on" says why). `brain_offline_fallback: bool = True` turns it off.
- Voice: "go offline" / "use the local model" / "offline mode" → switch to `local`; "go online" / "back online" → return to the preferred brain. `which_brain` already reports it.

## F2 — Interruption that pauses instead of cancelling

- "hold on", "wait", "one sec", "ruko", "ek minute" while she is speaking: stop playback, keep the turn's remaining sentences, speak nothing, and enter a `paused` state (HUD shows "Paused — say continue"). "continue" / "carry on" / "go on" / "aage bolo" resumes from the next unspoken sentence; anything else is a new request and the paused remainder is dropped.
- Implemented in the orchestrator's speaking loop: the queue of (sentence, synth future) is retained on a pause barge instead of being reset; `_paused_tail` holds it, cleared on the next turn.
- Long turns get an acknowledgement: if the brain has produced nothing after `ack_after_s: float = 3.5` (editable, 0 = off), she says a short "On it." / "एक सेकंड।" once per turn, before the first real sentence.
- Barge semantics otherwise unchanged (the wake word still interrupts; "stop" still cancels).

## F3 — Plan card on the HUD

- When a turn makes more than one tool call, the HUD shows them as a checklist instead of one card at a time: `{"kind": "plan", "payload": {"steps": [{"summary", "state"}]}}` where `state` is `pending|running|done|failed|declined`. The gate emits a plan step per decision it already reports; the orchestrator resets the plan at the start of each turn.
- `hud.js`: a compact list under the action card, max 5 visible steps (older ones collapse to "+N more"), same palette as the tool card; mini mode shows only the running step.

## F4 — Proactive v2

- Quiet hours (`proactive_quiet_from`/`proactive_quiet_to`, default 22:00–08:00): nudges and briefings are held, not dropped; the first announcement after quiet hours says "While you were away: …" when more than one was held.
- "snooze notifications for an hour" / "mute nudges until 5" → a local intent that sets a hold-until timestamp; "resume notifications" clears it.
- Triggers beyond the calendar: a battery threshold (below 15%, once per charge cycle) and a "you have N unread since morning" nudge at a configurable hour, both off by default.

## F5 — Memory v2

- Facts get a kind (`preference | person | place | routine | other`) inferred at write time and stored alongside the text; `fact_list` groups by kind when she recites them.
- Writing a fact that is a near-duplicate (ratio ≥ 0.9) of an existing one replaces it instead of adding; "forget everything about X" deletes by fuzzy topic match and says how many went.
- The system prompt's facts block is capped (`memory_facts_max: int = 40`) and ordered most-recently-used first; using a fact in a reply bumps it (the brain reports nothing, so this is a substring check over the spoken text against each fact — cheap and good enough).

## F6 — Shortcuts and Messages

- `mcp__mac__run_shortcut(name, input?)` via the `shortcuts` CLI: `allow` for a list the user has marked safe in Settings (empty by default), `confirm` otherwise; the tool refuses a name that isn't installed and says so.
- `mcp__pim__message_send(to, body)` via Messages' AppleScript bridge: **always-confirm** (added to `policy.always_confirm`), never pre-approvable, never covered by the trust window; the confirm reads the recipient and the first 40 characters.
- Both are listed in the README's tool table with what they can and cannot do.

## Order

F1 (offline brain) → F2 (pause/continue + ack) → F3 (plan card) → F4 (proactive v2) → F5 (memory v2) → F6 (Shortcuts + Messages). Each task: implementer, reviewer, fix round; whole-branch review before the merge, then `make app` and relaunch.
