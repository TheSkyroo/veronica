"""Codex (`codex` 0.155.1) as a Veronica brain.

How it runs (verified against codex 0.155 on macOS, 2026-09-19 — recheck on Windows; the
Windows specifics below follow the same contract and are noted where they
differ):
- One `codex exec --json` child per turn (`per_turn`), the prompt last
  on the command line, stdin closed (a stdin pipe makes it wait on
  "Reading additional input from stdin..."). The stream is JSONL:
  `thread.started{thread_id}`, `turn.started`, `item.started/completed
  {item:{id,type,...}}` and `turn.completed` / `turn.failed{error}`.
  The next turn resumes with `codex exec resume <thread_id>` (which
  takes no `-C`: it reuses the session's recorded cwd, our workspace).
- The system prompt goes in as `-c developer_instructions=<TOML basic
  string>`; there is no prompt file to write. The whole prompt rides on
  the command line, well inside Windows' 32 767-character limit. `include_permissions_
  instructions=false` keeps Codex from explaining its own approvals.
- Approvals are off (`approval_policy="never"`) and the sandbox is
  `workspace-write` with `brain_cwd` as a writable root, so our hook is
  the gate for its shell/patch tools: project-level
  `<workspace>/.codex/hooks.json` PreToolUse fires in exec mode with
  `-c features.hooks=true --dangerously-bypass-hook-trust` (two "bypass
  is enabled" notices arrive as `error` items — ignored). The payload
  is Claude-shaped (`tool_name:"Bash"`, `tool_input.command`) and so is
  the decision JSON `hook.emit` prints; a deny blocks the command with
  our reason. With native tools OFF the sandbox is `read-only`.
- Canary: the hook sees `tool_input.command = "echo x"` while the
  stream's `command_execution.command` wraps it in the shell Codex used
  (`powershell.exe -Command 'echo x'` on Windows, `bash -lc 'echo x'` in
  Git Bash), so `canary_matches` unwraps `<shell> -Command|-c|/c <cmd>`
  (falling back to substring) instead of the base's equality. File edits: the hook
  gets `tool_name:"apply_patch"` with the whole patch as
  `tool_input.command` (logged as the key, confirmed as an Edit) while
  the stream shows a `file_change` item with `changes:[{path,kind}]`;
  its key is the first changed path, matched as a substring of the
  logged patch text.
- Our MCP servers are passed per call as `-c mcp_servers.veronica-<name>.*`
  overrides (command/args/env, plus `default_tools_approval_mode=
  "approve"` — without it Codex fails every MCP call with "requires
  approval, but approval policy is never"), never written to
  `~/.codex/config.toml`; their calls arrive as `mcp_tool_call` items
  (`server`, `tool`, `arguments`) and are gated inside tools.serve.
"""
import json
import re
from pathlib import Path

from veronica.brain import hook
from veronica.brain.backends import winproc
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
from veronica.brain.gateclient import HOOK_TIMEOUT_S

BYPASS_NOTICE = "--dangerously-bypass-hook-trust` is enabled"


def toml_str(s: str) -> str:
    """`s` as a TOML basic string. JSON's escaping is a subset of TOML's
    (\\" \\\\ \\n \\t \\uXXXX); non-ASCII goes through raw (JSON would emit
    surrogate pairs for non-BMP characters, which TOML rejects) and DEL,
    which JSON leaves alone, is escaped because TOML forbids it."""
    return json.dumps(s, ensure_ascii=False).replace("\x7f", "\\u007F")


# `<shell> [-NoProfile ...] -Command <cmd>`, `cmd.exe /c <cmd>`, `bash -lc <cmd>`:
# the shell (bare or quoted path), any switches, then the one that takes the
# command, then the command itself.
_SHELL_WRAP = re.compile(
    r"""^\s*(?:"(?P<qshell>[^"]*)"|'(?P<sshell>[^']*)'|(?P<shell>\S+))"""
    r"(?:\s+-(?!command\b|c\b|lc\b|lic\b|ic\b)\w+)*"
    r"\s+(?:-command|-c|-lc|-lic|-ic|/c)\s+(?P<cmd>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_SHELLS = ("powershell", "pwsh", "cmd", "bash", "sh", "zsh")   # zsh: the recorded fixtures


def unwrap_shell(command: str) -> str:
    """`powershell.exe -NoProfile -Command 'echo hi'` -> `echo hi` (and the
    same for pwsh, `cmd.exe /c`, `bash -lc`); anything else unchanged.
    One layer of matching quotes around the command is dropped (a
    PowerShell '...' also gives back its doubled quotes)."""
    m = _SHELL_WRAP.match(command)
    if not m:
        return command
    shell = m.group("qshell") or m.group("sshell") or m.group("shell")
    name = shell.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".exe")
    if name not in _SHELLS:
        return command
    inner = m.group("cmd")
    if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "'\"":
        quote, inner = inner[0], inner[1:-1]
        if quote == "'" and name in ("powershell", "pwsh"):
            inner = inner.replace("''", "'")
    return inner


class CodexBrain(CliBrain):
    name, label, binary = "codex", "Codex", "codex"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self._prompt = ""

    # -- spawn ------------------------------------------------------------------
    def _mcp_overrides(self) -> list[str]:
        out: list[str] = []
        for server in hook.OUR_SERVERS:
            key = f"mcp_servers.veronica-{server}"
            out += ["-c", f"{key}.command={toml_str(winproc.console_python())}",
                    "-c", f'{key}.args=["-m","veronica.tools.serve","{server}"]',
                    "-c", f"{key}.env.VERONICA_GATE={toml_str(str(self.s.gate_endpoint))}",
                    "-c", f'{key}.env.VERONICA_BRAIN="codex"',
                    "-c", f"{key}.env.VERONICA_HOOK_LOG={toml_str(str(self.hook_log))}",
                    # Codex 0.155 wants its own approval per MCP tool call, and with
                    # approval_policy="never" that is a hard failure ("MCP tool call
                    # requires approval"). Ours are gated inside tools.serve, so let
                    # Codex approve them.
                    "-c", f'{key}.default_tools_approval_mode="approve"']
        return out

    def argv(self, text: str, session_id: str | None, image_paths: list[Path], native: bool) -> list[str]:
        a = ["codex", "exec"]
        if session_id:
            a += ["resume", session_id]      # no -C here: resume keeps the session's cwd
        a += ["--json", "--skip-git-repo-check"]
        if not session_id:
            a += ["-C", str(self.workspace)]
        a += [
            "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="workspace-write"' if native else 'sandbox_mode="read-only"',
            "-c", f"sandbox_workspace_write.writable_roots=[{toml_str(str(self.s.brain_cwd))}]",
            "-c", "features.hooks=true", "--dangerously-bypass-hook-trust",
            "-c", f"developer_instructions={toml_str(self._prompt)}",
            "-c", "include_permissions_instructions=false",
            "-c", f'model_reasoning_effort="{self.s.effort}"',
        ] + self._mcp_overrides()
        for p in image_paths:
            a += ["-i", str(p)]
        return a + [text]

    def env(self) -> dict[str, str]:
        return {}

    # -- workspace ----------------------------------------------------------------
    def hook_command(self) -> str:
        # The child inherits VERONICA_* from us, but the flags make the hook
        # independent of that. Codex hands the string to a shell, so it is
        # written to read the same in cmd.exe and PowerShell.
        return winproc.neutral_command([winproc.console_python(), "-m", "veronica.brain.hook", "codex",
                                        "--gate", str(self.s.gate_endpoint), "--log", str(self.hook_log)])

    def prepare_workspace(self, prompt_text: str, native: bool) -> None:
        if not native:
            prompt_text += ("\n\nDo not run shell commands or edit files yourself in this session; "
                            "use Veronica's tools only.")
        self._prompt = prompt_text
        d = self.workspace / ".codex"
        d.mkdir(exist_ok=True)
        # codex 0.155 reads a per-hook timeout as `timeout` (seconds) on the
        # handler and `timeoutSec` on the matcher group; both are set so the
        # gate's answer can never outlive it.
        hooks = {"hooks": {"PreToolUse": [{"matcher": "*", "timeoutSec": HOOK_TIMEOUT_S, "hooks": [
            {"type": "command", "command": self.hook_command(), "timeout": HOOK_TIMEOUT_S}]}]}}
        (d / "hooks.json").write_text(json.dumps(hooks, indent=2) + "\n", encoding="utf-8")

    # -- stream -------------------------------------------------------------------
    def native_key(self, tool: str, input: dict) -> str:
        if tool == "apply_patch":
            changes = input.get("changes") or []
            paths = [str(c.get("path") or "") for c in changes if isinstance(c, dict)]
            return next((p for p in paths if p), tool)
        return hook.canary_key(tool, input)

    def canary_matches(self, logged_key: str, stream_key: str) -> bool:
        if logged_key.startswith("*** Begin Patch"):        # apply_patch: the path is in the patch text
            return stream_key in logged_key or Path(stream_key).name in logged_key
        return logged_key == stream_key or logged_key == unwrap_shell(stream_key) or logged_key in stream_key

    def parse(self, line: str) -> list[Event]:  # shapes: tests/fixtures/brains/codex-*.jsonl
        e = json.loads(line)
        kind = e.get("type")
        if kind == "thread.started":
            return [Session(str(e.get("thread_id") or ""))]
        if kind == "turn.completed":
            return [Done()]
        if kind == "turn.failed":
            err = e.get("error") or {}
            return [Error(str(err.get("message") if isinstance(err, dict) else err) or "turn failed")]
        if kind == "error":
            return [Error(str(e.get("message") or "error"))]
        if kind not in ("item.started", "item.completed"):
            return []
        item = e.get("item") or {}
        itype, item_id = item.get("type"), str(item.get("id") or "")
        done = kind == "item.completed"
        if itype == "agent_message":
            text = item.get("text")
            return [Text(text)] if done and text else []
        if itype == "error":
            msg = str(item.get("message") or "")
            return [] if BYPASS_NOTICE in msg else [Error(msg or "error")]
        if itype == "command_execution":
            if done:
                return [ToolEnd(item_id)]
            return [ToolStart(item_id, "shell", {"command": str(item.get("command") or "")}, native=True)]
        if itype == "file_change":
            if done:
                return [ToolEnd(item_id)]
            changes = item.get("changes") or []
            return [ToolStart(item_id, "apply_patch", {"changes": list(changes) if isinstance(changes, list) else []},
                              native=True)]
        if itype == "mcp_tool_call":
            if done:
                return [ToolEnd(item_id)]
            server, tool = str(item.get("server") or ""), str(item.get("tool") or "")
            ours = hook._ours(f"{server}__{tool}")
            args = item.get("arguments") or {}
            # Ours are gated in tools.serve; any other server's tool is gated by
            # the hook, so the canary has to watch it like a native call.
            return [ToolStart(item_id, ours or f"mcp__{server}__{tool}",
                              dict(args) if isinstance(args, dict) else {}, native=ours is None)]
        return []
