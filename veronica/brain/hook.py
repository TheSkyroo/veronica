"""`python -m veronica.brain.hook <backend>`: the pre-tool hook Veronica
installs in each external CLI's workspace. Reads the hook payload on
stdin, maps the CLI's tool to our canonical name, logs it to
$VERONICA_HOOK_LOG (the canary in backends/cli.py checks this log), asks
the gate socket, and prints the CLI's decision shape. Anything
unexpected -> deny. Our own MCP tools are gated inside tools.serve, so
the hook just lets them through; read-only native tools skip the gate
entirely."""
import json
import os
import shlex
import sys
import time
from pathlib import Path

from veronica.brain.gateclient import ask_gate

SHELL_TOOLS = {"run_command", "run_shell_command", "Bash", "shell", "local_shell", "bash"}
WRITE_TOOLS = {"write_file", "write_to_file", "Write"}
EDIT_TOOLS = {"replace", "edit", "edit_file", "apply_patch", "str_replace_editor", "Edit"}
READONLY_TOOLS = {"read_file", "view", "glob", "grep", "list_directory", "find", "web_fetch",
                  "web_search", "google_web_search", "fetch", "Read", "Glob", "Grep", "ls"}
OUR_SERVERS = ("mac", "pim", "memory", "screen", "music", "browser", "computer")

# Vendor tool name -> what the gate (policy.classify) knows how to judge.
CANONICAL: dict[str, str] = (
    {t: "Bash" for t in SHELL_TOOLS}
    | {t: "Write" for t in WRITE_TOOLS}
    | {t: "Edit" for t in EDIT_TOOLS}
    | {t: "read" for t in READONLY_TOOLS}
)


def _ours(tool_name: str) -> str | None:
    """'mcp__veronica-mac__open_app' / 'veronica-mac__open_app' / 'mac__open_app' -> 'mcp__mac__open_app'."""
    t = tool_name.removeprefix("mcp__").removeprefix("veronica-")
    server, sep, short = t.partition("__")
    return f"mcp__{server}__{short}" if sep and server in OUR_SERVERS else None


def canonical_tool(backend: str, tool_name: str, tool_input: dict) -> tuple[str, dict] | None:
    """None = read-only native tool, allow without the gate. Unknown tools
    pass through by name so the gate confirms them."""
    ours = _ours(tool_name)
    if ours:
        return ours, dict(tool_input)
    kind = CANONICAL.get(tool_name)
    if kind == "read":
        return None
    if kind == "Bash":
        cmd = tool_input.get("command", "")
        if isinstance(cmd, list):    # codex: argv
            cmd = " ".join(shlex.quote(str(c)) if " " in str(c) else str(c) for c in cmd)
        return "Bash", {"command": str(cmd)}
    if kind in ("Write", "Edit"):
        return kind, dict(tool_input)
    return tool_name, dict(tool_input)


def emit(backend: str, allow: bool, reason: str = "") -> str:
    """The per-CLI decision JSON. Qwen: silence means allow. Copilot's key
    is confirmed against its hooks reference in Task 5."""
    if backend == "qwen":
        return "" if allow else json.dumps({"decision": "deny", "reason": reason})
    if backend == "copilot":
        return json.dumps({"permissionDecision": "allow" if allow else "deny", "permissionDecisionReason": reason})
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                              "permissionDecision": "allow" if allow else "deny",
                                              "permissionDecisionReason": reason}})


def _key(canon: str, inp: dict) -> str:
    return inp.get("command") or inp.get("file_path") or inp.get("path") or canon


def run(backend: str, stdin_text: str, *, ask=ask_gate, log_path: Path | None = None) -> tuple[str, int]:
    """(stdout, exit code). The log line is written *before* the gate is
    asked so the canary sees the call even if the answer never comes."""
    try:
        payload = json.loads(stdin_text or "{}")
        call = payload.get("toolCall") or {}          # agy: {"toolCall": {"name", "args"}}
        tool = payload.get("tool_name") or payload.get("toolName") or call.get("name") or ""
        inp = (payload.get("tool_input") or payload.get("toolArgs") or payload.get("toolInput")
               or call.get("args") or {})
        canon = canonical_tool(backend, str(tool), dict(inp))
        if canon is None:
            return "", 0
        name, cinp = canon
        if name.startswith("mcp__"):
            return emit(backend, True), 0     # gated inside tools.serve already
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "call": tool, "key": _key(name, cinp), "decision": "pending"}) + "\n")
        d = ask(name, cinp, origin="hook", backend=backend)
        return emit(backend, d.allow, d.message), 0
    except Exception:
        return emit(backend, False, "gate error"), 0


if __name__ == "__main__":
    backend = sys.argv[1]
    log_env = os.environ.get("VERONICA_HOOK_LOG", "")
    out, code = run(backend, sys.stdin.read(), log_path=Path(log_env) if log_env else None)
    sys.stdout.write(out)
    sys.exit(code)
