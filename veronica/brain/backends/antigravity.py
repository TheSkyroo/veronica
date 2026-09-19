"""Antigravity (`agy` 1.2.7) as a Veronica brain.

How it runs (verified on this Mac, 2026-09-19):
- One long-lived child in persistent stream mode: `agy --output-format
  stream-json --input-format stream-json --print= ...` (`--print=` with
  an EMPTY value; a bare `-p` swallows the next flag). It emits
  `init{conversation_id}` at once, then runs one turn per stdin line
  `{"event":"user","message":{"role":"user","content":[{"type":"text",...}]}}`
  and ends each with `result`. After a kill it is respawned with
  `--conversation <id>`.
- Headless `agy` cannot prompt, ignores `permissions.allow` in
  settings.json and treats a hook's `allow` as inert, so with native
  tools ON it runs with `--dangerously-skip-permissions` and our hook is
  the gate (the base's canary checks the hook really fires). With native
  tools OFF the flag is dropped: headless agy then auto-denies every
  shell/file tool itself.
- Only the USER-level `~/.gemini/config/hooks.json` is loaded (workspace
  `.agy/` and `.agents/` hooks are not), so our PreToolUse entry is merged
  into that file and scoped with `--scope-file <backend_dir>/active-conversation`,
  written right after `init` and before the first prompt. The user's own
  interactive `agy` sessions are never gated.
- No system-prompt flag: the prompt goes in `AGENTS.md` in the workspace
  (cwd). Images: only "text" content blocks exist on stdin, so screenshots
  are referenced as `@/path/img-1.png` in the text (whether agy attaches
  them this way is unverified; the paths are real files it can view_file).
- MCP servers are registered once per process with `agy mcp add` (into
  `~/.gemini/config/mcp_config.json`) and called through `call_mcp_tool`.
"""
import json
import sys
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


class AntigravityBrain(CliBrain):
    name, label, binary = "antigravity", "Antigravity", "agy"
    mode = "persistent"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        # User-level hook file; tests point this at tmp_path.
        self.hooks_file: Path = Path.home() / ".gemini" / "config" / "hooks.json"
        self.scope_file: Path = self.workspace / "active-conversation"
        self._mcp_registered = False
        self._conversation_id: str | None = None

    # -- spawn ------------------------------------------------------------------
    def argv(self, text: str, session_id: str | None, image_paths: list[Path], native: bool) -> list[str]:
        a = ["agy", "--output-format", "stream-json", "--input-format", "stream-json", "--print="]
        if native:
            a.append("--dangerously-skip-permissions")
        if session_id:
            a += ["--conversation", session_id]
        if self.s.effort in ("low", "medium", "high"):
            a += ["--effort", self.s.effort]
        return a

    def env(self) -> dict[str, str]:
        return {}

    def turn_message(self, text: str, image_paths: list[Path]) -> str:
        refs = " ".join(f"@{p}" for p in image_paths)
        body = f"{refs} {text}" if refs else text
        return json.dumps({"event": "user", "message": {"role": "user", "content": [{"type": "text", "text": body}]}})

    def session_started(self, session_id: str) -> None:
        self._conversation_id = session_id
        self.scope_file.write_text(session_id)

    # -- workspace ----------------------------------------------------------------
    def hook_command(self) -> str:
        return (f"{sys.executable} -m veronica.brain.hook antigravity --sock {self.s.gate_socket} "
                f"--log {self.hook_log} --scope-file {self.scope_file}")

    def prepare_workspace(self, prompt_text: str, native: bool) -> None:
        if not native:
            prompt_text += ("\n\nDo not run shell commands or edit files yourself in this session; "
                            "use Veronica's tools only.")
        (self.workspace / "AGENTS.md").write_text(prompt_text)
        self._merge_hook()
        self._register_mcp()

    def _merge_hook(self) -> None:
        """Ensure our PreToolUse entry is in the user-level hooks file,
        keeping everything else in it. Idempotent."""
        try:
            data = json.loads(self.hooks_file.read_text()) if self.hooks_file.exists() else {}
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        hooks = data.setdefault("hooks", {})
        pre = hooks.setdefault("PreToolUse", [])
        cmd = self.hook_command()
        for entry in pre:
            if any(h.get("command") == cmd for h in entry.get("hooks", []) if isinstance(h, dict)):
                return
        pre.append({"matcher": "*", "hooks": [{"type": "command", "command": cmd}]})
        self.hooks_file.parent.mkdir(parents=True, exist_ok=True)
        self.hooks_file.write_text(json.dumps(data, indent=2) + "\n")

    def _register_mcp(self) -> None:
        """`agy mcp add` is add-or-update; run once per process."""
        if self._mcp_registered:
            return
        for server in hook.OUR_SERVERS:
            self._run(["agy", "mcp", "add",
                       "--env", f"VERONICA_GATE_SOCK={self.s.gate_socket}",
                       "--env", "VERONICA_BRAIN=antigravity",
                       "--env", f"VERONICA_HOOK_LOG={self.hook_log}",
                       f"veronica-{server}", sys.executable, "-m", "veronica.tools.serve", server],
                      capture_output=True, timeout=30)
        self._mcp_registered = True

    # -- stream -------------------------------------------------------------------
    def native_key(self, tool: str, input: dict) -> str:
        return hook.canary_key(tool, input)

    def parse(self, line: str) -> list[Event]:  # shapes: tests/fixtures/brains/antigravity-*.jsonl
        e = json.loads(line)
        kind = e.get("event")
        if kind == "init":
            return [Session(str(e.get("conversation_id") or ""))]
        if kind == "step_update":
            su = e.get("step_update") or {}
            step, state = su.get("step_type"), su.get("state")
            if step == "agent_response":
                delta = su.get("text_delta")
                return [Text(delta)] if delta else []
            if step == "tool":
                call_id = f"{su.get('conversation_id') or self._conversation_id}:{su.get('step_index')}"
                if state == "ACTIVE":
                    info = su.get("tool_info") or {}
                    tool = str(su.get("tool_name") or info.get("name") or "")
                    params = info.get("parameters") or {}
                    return [ToolStart(call_id=call_id, tool=tool, input=dict(params) if isinstance(params, dict) else {},
                                      native=tool != "call_mcp_tool")]
                if state in ("DONE", "ERROR"):
                    return [ToolEnd(call_id)]
            return []
        if kind == "result":
            r = e.get("result") or {}
            if r.get("status") == "SUCCESS":
                return [Done(str(r.get("response") or ""))]
            return [Error(str(r.get("error") or r.get("status") or "error"))]
        return []

