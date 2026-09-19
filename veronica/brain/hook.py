"""`python -m veronica.brain.hook <backend>`: the pre-tool hook Veronica
installs in each external CLI's workspace. Reads the hook payload on
stdin, maps the CLI's tool to our canonical name, logs it to
$VERONICA_HOOK_LOG (the canary in backends/cli.py checks this log), asks
the gate socket, and prints the CLI's decision shape. Anything
unexpected -> deny. Our own MCP tools are gated inside tools.serve, so
the hook just lets them through; read-only native tools skip the gate
entirely.

Flags (Antigravity's hook is user-level, so it gets no env from us):
`--sock <path>` / `--log <path>` override $VERONICA_GATE_SOCK /
$VERONICA_HOOK_LOG; `--scope-file <path>` makes the hook a no-op unless
the payload's `conversationId` equals that file's content, so only the
conversation Veronica is driving is gated and the user's own `agy` is
untouched."""
import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

from veronica.brain.gateclient import ask_gate

SHELL_TOOLS = {"run_command", "run_shell_command", "Bash", "shell", "local_shell", "bash"}
WRITE_TOOLS = {"write_file", "write_to_file", "Write"}
EDIT_TOOLS = {"replace", "edit", "edit_file", "apply_patch", "str_replace_editor", "Edit",
              "replace_file_content", "multi_replace_file_content", "sed_file"}     # agy
READONLY_TOOLS = {"read_file", "view", "glob", "grep", "list_directory", "find", "web_fetch",
                  "web_search", "google_web_search", "fetch", "Read", "Glob", "Grep", "ls",
                  "view_file", "list_dir", "grep_search", "find_by_name", "search_web", "read_url_content"}  # agy
# Where each CLI puts the one string that identifies a native call: the
# command line, else the file. Same order on both sides of the canary.
_KEY_FIELDS = ("command", "CommandLine", "file_path", "TargetFile", "AbsolutePath", "path")
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


def _agy_mcp(tool_input: dict) -> tuple[str, dict] | None:
    """agy calls every MCP tool through `call_mcp_tool`; unwrap ours so the
    hook lets them through (tools.serve gates them). Any other server's
    tool keeps the wrapper name and gets confirmed."""
    server = str(tool_input.get("ServerName") or tool_input.get("server_name") or "")
    tool = str(tool_input.get("ToolName") or tool_input.get("tool_name") or "")
    args = tool_input.get("Arguments") or tool_input.get("arguments") or {}
    ours = _ours(f"{server}__{tool}") if server and tool else None
    return (ours, dict(args) if isinstance(args, dict) else {}) if ours else None


def canonical_tool(backend: str, tool_name: str, tool_input: dict) -> tuple[str, dict] | None:
    """None = read-only native tool, allow without the gate. Unknown tools
    pass through by name so the gate confirms them."""
    if tool_name == "call_mcp_tool":
        unwrapped = _agy_mcp(tool_input)
        if unwrapped:
            return unwrapped
    ours = _ours(tool_name)
    if ours:
        return ours, dict(tool_input)
    kind = CANONICAL.get(tool_name)
    if kind == "read":
        return None
    if kind == "Bash":
        cmd = tool_input.get("command") or tool_input.get("CommandLine") or ""   # CommandLine: agy
        if isinstance(cmd, list):    # codex: argv
            cmd = " ".join(shlex.quote(str(c)) if " " in str(c) else str(c) for c in cmd)
        return "Bash", {"command": str(cmd)}
    if kind in ("Write", "Edit"):
        inp = dict(tool_input)
        if "file_path" not in inp:   # agy: TargetFile / AbsolutePath
            target = inp.get("TargetFile") or inp.get("AbsolutePath") or inp.get("path")
            if target:
                inp["file_path"] = target
        return kind, inp
    return tool_name, dict(tool_input)


def emit(backend: str, allow: bool, reason: str = "") -> str:
    """The per-CLI decision JSON. Qwen: silence means allow. Antigravity's
    PreToolHookResult is {decision, reason} (verified against agy 1.2.7:
    the Claude-style hookSpecificOutput shape is rejected as an unknown
    field). Copilot's key is confirmed against its hooks reference in Task 5."""
    if backend == "qwen":
        return "" if allow else json.dumps({"decision": "deny", "reason": reason})
    if backend == "antigravity":
        return json.dumps({"decision": "allow" if allow else "deny", "reason": reason})
    if backend == "copilot":
        return json.dumps({"permissionDecision": "allow" if allow else "deny", "permissionDecisionReason": reason})
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                              "permissionDecision": "allow" if allow else "deny",
                                              "permissionDecisionReason": reason}})


def canary_key(tool: str, inp: dict) -> str:
    """The string the hook logs for a native call and the adapter's canary
    looks for: the command line, else the target file, else the first
    string argument (by key name), else the tool name."""
    for k in _KEY_FIELDS:
        v = inp.get(k)
        if isinstance(v, str) and v:
            return v
    for k in sorted(inp):
        if isinstance(inp[k], str) and inp[k]:
            return inp[k]
    return tool



def in_scope(payload: dict, scope_file: Path | None) -> bool:
    """Without a scope file every call is ours. With one, only the
    conversation whose id it holds (unreadable/empty file -> nothing is)."""
    if scope_file is None:
        return True
    try:
        wanted = scope_file.read_text().strip()
    except OSError:
        return False
    return bool(wanted) and str(payload.get("conversationId") or payload.get("session_id") or "") == wanted


def run(backend: str, stdin_text: str, *, ask=ask_gate, log_path: Path | None = None,
        scope_file: Path | None = None) -> tuple[str, int]:
    """(stdout, exit code). The log line is written *before* the gate is
    asked so the canary sees the call even if the answer never comes."""
    try:
        payload = json.loads(stdin_text or "{}")
        if not in_scope(payload, scope_file):
            return "", 0                               # not our conversation: the CLI's own flow applies
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
                f.write(json.dumps({"ts": time.time(), "call": tool, "key": canary_key(str(tool), cinp), "decision": "pending"}) + "\n")
        d = ask(name, cinp, origin="hook", backend=backend)
        return emit(backend, d.allow, d.message), 0
    except Exception:
        return emit(backend, False, "gate error"), 0


def main(argv: list[str], stdin_text: str, *, ask=ask_gate) -> tuple[str, int]:
    ap = argparse.ArgumentParser(prog="veronica.brain.hook", add_help=False)
    ap.add_argument("backend")
    ap.add_argument("--sock", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--scope-file", default=None)
    a = ap.parse_args(argv)
    if a.sock:
        os.environ["VERONICA_GATE_SOCK"] = a.sock   # gateclient reads it; flags win over env
    log_path = a.log or os.environ.get("VERONICA_HOOK_LOG", "")
    return run(a.backend, stdin_text, ask=ask, log_path=Path(log_path) if log_path else None,
               scope_file=Path(a.scope_file) if a.scope_file else None)


if __name__ == "__main__":
    out, code = main(sys.argv[1:], sys.stdin.read())
    sys.stdout.write(out)
    sys.exit(code)
