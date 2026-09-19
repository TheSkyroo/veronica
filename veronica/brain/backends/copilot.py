"""GitHub Copilot CLI (`copilot` 1.0.86) as a Veronica brain.

How it runs (verified on this Mac, 2026-09-19):
- One `copilot -p <prompt> --output-format json --silent --no-ask-user`
  child per turn (`per_turn`), stdin closed. The stream is JSONL with
  many `ephemeral` events: `assistant.message_delta{data.deltaContent}`
  carries the text, `tool.execution_start{data:{toolCallId,toolName,
  arguments}}` / `tool.execution_complete` bracket a tool call, and the
  LAST line is `result{sessionId,exitCode}`. The first turn sets its own
  id with `--session-id <uuid4>` (so the id is known even if the stream
  dies); later turns pass `--resume <id>`.
- System prompt: `.github/copilot-instructions.md` in the workspace IS
  read in `-p` mode (the "secret word" probe answered it; the prompt is
  not prefixed onto the user text).
- Native tools ON: `--allow-all-tools --allow-all-paths`, with our hook
  as the gate. Only the USER-level `~/.copilot/hooks/veronica.json` fires
  in `-p` mode (repository `.github/hooks` needs folder trust), so it is
  written before every turn as `{"version":1,"hooks":{"preToolUse":[{
  "type":"command","bash":<cmd>,"timeoutSec":600}]}}` and scoped with
  `--scope-cwd <workspace>` (the payload's `cwd` is the realpath). The
  timeout matters: a hook that times out FAILS OPEN, and 600 s covers a
  slow voice confirm. The payload is `{sessionId,timestamp,cwd,toolName,
  toolArgs}`; the decision is `{permissionDecision,permissionDecisionReason}`
  and a deny fails the call with "Denied by preToolUse hook: <reason>".
  Native tools OFF: the shell/edit/agent tools are hidden from the model
  with `--excluded-tools` (Copilot runs "safe" commands like `echo`
  without asking even when they are not allowed, so hiding is the only
  real off switch) and each of our servers is allowed by name
  (`--allow-tool veronica-<name>`; the `kind(arg)` rule syntax has no
  glob).
- Copilot's shell tool is `bash` (`toolArgs.command`); its edit tool is
  `apply_patch`, whose `toolArgs`/`arguments` is the PATCH TEXT itself
  (a string, not a dict) on both the hook and stream sides — it is
  wrapped as `{"command": <patch>}` so the hook confirms it as an Edit
  and the canary key (the patch) matches by equality.
- Our MCP servers are passed per call as `--additional-mcp-config
  '{"mcpServers":{"veronica-<name>":{"type":"local","command","args",
  "env","tools":["*"]}}}'` (never written to `~/.copilot/mcp-config.json`);
  their calls show up as `toolName: "veronica-<server>-<tool>"` (one
  hyphen) with `mcpServerName`/`mcpToolName` alongside, and are gated in
  tools.serve. `--disable-builtin-mcps` drops the GitHub MCP server.
- `effort` is not passed: `--reasoning-effort` is an error with the
  default model "auto" ('Model "auto" does not support reasoning effort').
"""
import json
import sys
import uuid
from pathlib import Path

from veronica.brain import hook
from veronica.brain.backends.cli import (
    CliBrain,
    Done,
    Error,
    Event,
    Session,
    Text,
    ToolEnd,
    ToolStart,
)

# Copilot's own tools that act (run, edit, spawn); hidden with
# --excluded-tools when native tools are off. What remains (view, rg,
# glob, web_fetch, ...) is read-only.
NATIVE_ACTION_TOOLS = ("bash", "read_bash", "stop_bash", "list_bash", "apply_patch", "task",
                       "write_agent", "sql", "session_store_sql", "skill")
HOOK_TIMEOUT_S = 600


class CopilotBrain(CliBrain):
    name, label, binary = "copilot", "Copilot", "copilot"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        # User-level hook file; tests point this at tmp_path.
        self.hooks_file: Path = Path.home() / ".copilot" / "hooks" / "veronica.json"
        self._new_session_id: str | None = None
        self._last_content = ""

    # -- spawn ------------------------------------------------------------------
    def mcp_config(self) -> dict:
        env = {"VERONICA_GATE_SOCK": str(self.s.gate_socket), "VERONICA_BRAIN": "copilot",
               "VERONICA_HOOK_LOG": str(self.hook_log)}
        return {"mcpServers": {
            f"veronica-{server}": {"type": "local", "command": sys.executable,
                                   "args": ["-m", "veronica.tools.serve", server],
                                   "env": dict(env), "tools": ["*"]}
            for server in hook.OUR_SERVERS}}

    def argv(self, text: str, session_id: str | None, image_paths: list[Path], native: bool) -> list[str]:
        a = ["copilot", "-p", text, "--output-format", "json", "--silent", "--no-ask-user",
             "--disable-builtin-mcps"]
        if session_id:
            a += ["--resume", session_id]
        else:
            self._new_session_id = str(uuid.uuid4())
            a += ["--session-id", self._new_session_id]
        for p in image_paths:
            a += ["--attachment", str(p)]
        if native:
            a += ["--allow-all-tools", "--allow-all-paths"]
        else:
            a += ["--allow-tool"] + [f"veronica-{server}" for server in hook.OUR_SERVERS]
            a += ["--excluded-tools"] + list(NATIVE_ACTION_TOOLS)
        return a + ["--additional-mcp-config", json.dumps(self.mcp_config())]

    def env(self) -> dict[str, str]:
        return {}

    # -- workspace ----------------------------------------------------------------
    def hook_command(self) -> str:
        return (f"{sys.executable} -m veronica.brain.hook copilot --sock {self.s.gate_socket} "
                f"--log {self.hook_log} --scope-cwd {self.workspace}")

    def prepare_workspace(self, prompt_text: str, native: bool) -> None:
        if not native:
            prompt_text += ("\n\nDo not run shell commands or edit files yourself in this session; "
                            "use Veronica's tools only.")
        d = self.workspace / ".github"
        d.mkdir(exist_ok=True)
        (d / "copilot-instructions.md").write_text(prompt_text)
        hooks = {"version": 1, "hooks": {"preToolUse": [
            {"type": "command", "bash": self.hook_command(), "timeoutSec": HOOK_TIMEOUT_S}]}}
        self.hooks_file.parent.mkdir(parents=True, exist_ok=True)
        self.hooks_file.write_text(json.dumps(hooks, indent=2) + "\n")

    async def close(self) -> None:
        await super().close()
        # Our own file, scoped to our workspace anyway; drop it so the
        # user's copilot sessions don't spawn the hook for nothing.
        try:
            self.hooks_file.unlink()
        except OSError:
            pass

    # -- stream -------------------------------------------------------------------
    def readonly_summary(self, tool: str, input: dict) -> str:
        what = input.get("path") or input.get("url") or input.get("pattern") or input.get("query") or ""
        return f"{tool} {what}".strip()

    def parse(self, line: str) -> list[Event]:  # shapes: tests/fixtures/brains/copilot-*.jsonl
        e = json.loads(line)
        kind = e.get("type")
        data = e.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if kind == "assistant.message_delta":
            delta = data.get("deltaContent")
            return [Text(delta)] if delta else []
        if kind == "assistant.message":
            content = data.get("content")
            if isinstance(content, str) and content:
                self._last_content = content
            return []
        if kind == "tool.execution_start":
            call_id = str(data.get("toolCallId") or "")
            tool = str(data.get("toolName") or "")
            args = data.get("arguments")
            if isinstance(args, str):                 # apply_patch: the patch text
                args = {"command": args}
            elif not isinstance(args, dict):
                args = {}
            ours = hook._ours(tool)
            if ours is None and data.get("mcpServerName"):
                ours = hook._ours(f"{data['mcpServerName']}__{data.get('mcpToolName') or ''}")
            if ours:
                return [ToolStart(call_id, ours, dict(args), native=False)]
            # Copilot's own tools — and any other MCP server's, which the hook
            # confirms by name — go through the hook, so the canary watches them.
            return [ToolStart(call_id, tool, dict(args), native=True)]
        if kind == "tool.execution_complete":
            return [ToolEnd(str(data.get("toolCallId") or ""))]
        if kind == "result":
            sid = str(e.get("sessionId") or self._new_session_id or "")
            code = e.get("exitCode", 0)
            final, self._last_content = self._last_content, ""
            if code not in (0, None):
                return [Error(str(e.get("error") or e.get("message") or f"exit {code}"))]
            return ([Session(sid)] if sid else []) + [Done(final)]
        if kind == "error":
            return [Error(str(data.get("message") or e.get("message") or "error"))]
        return []
