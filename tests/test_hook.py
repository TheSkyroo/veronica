import json

import pytest

from veronica.brain import hook
from veronica.brain.base import Decision


@pytest.mark.parametrize("backend,tool,inp,expected", [
    ("antigravity", "run_command", {"command": "ls -la"}, ("Bash", {"command": "ls -la"})),
    ("qwen", "run_shell_command", {"command": "echo hi"}, ("Bash", {"command": "echo hi"})),
    ("codex", "shell", {"command": ["bash", "-lc", "ls"]}, ("Bash", {"command": "bash -lc ls"})),
    ("codex", "shell", {"command": ["bash", "-lc", "echo hi there"]}, ("Bash", {"command": "bash -lc 'echo hi there'"})),
    ("copilot", "bash", {"command": "pwd"}, ("Bash", {"command": "pwd"})),
    ("antigravity", "write_file", {"file_path": "/tmp/x", "content": "y"}, ("Write", {"file_path": "/tmp/x", "content": "y"})),
    ("codex", "apply_patch", {"patch": "*** Begin Patch"}, ("Edit", {"patch": "*** Begin Patch"})),
    ("antigravity", "read_file", {"file_path": "/tmp/x"}, None),
    ("copilot", "grep", {"pattern": "x"}, None),
    ("qwen", "google_web_search", {"query": "x"}, None),
    ("codex", "mcp__veronica-mac__clipboard_write", {"text": "x"}, ("mcp__mac__clipboard_write", {"text": "x"})),
    # verified live: codex sanitises the server name in the hook payload
    ("codex", "mcp__veronica_mac__volume_get", {}, ("mcp__mac__volume_get", {})),
    ("copilot", "veronica-mac__clipboard_write", {"text": "x"}, ("mcp__mac__clipboard_write", {"text": "x"})),
    ("copilot", "veronica-mac-clipboard_write", {"text": "x"}, ("mcp__mac__clipboard_write", {"text": "x"})),  # verified
    ("copilot", "veronica-pim-mail_send", {"to": "x"}, ("mcp__pim__mail_send", {"to": "x"})),
    ("copilot", "veronica-mac-", {}, ("veronica-mac-", {})),
    ("copilot", "github-mcp-server-search_code", {"q": "x"}, ("github-mcp-server-search_code", {"q": "x"})),
    ("copilot", "rg", {"pattern": "x"}, None),
    ("copilot", "read_bash", {"id": "0"}, None),
    # bare <server>__<tool> is NOT ours: every backend registers our servers as
    # veronica-<name>, so that form can only be somebody else's MCP server.
    ("qwen", "mac__clipboard_write", {"text": "x"}, ("mac__clipboard_write", {"text": "x"})),
    ("copilot", "memory__create_entities", {"e": []}, ("memory__create_entities", {"e": []})),
    ("codex", "mcp__memory__create_entities", {"e": []}, ("mcp__memory__create_entities", {"e": []})),
    ("codex", "mcp__github__create_issue", {"title": "x"}, ("mcp__github__create_issue", {"title": "x"})),
    ("antigravity", "some_new_tool", {"a": 1}, ("some_new_tool", {"a": 1})),
])
def test_canonical_tool(backend, tool, inp, expected):
    assert hook.canonical_tool(backend, tool, inp) == expected


def test_run_logs_before_asking_and_allows(tmp_path):
    seen = []

    def ask(tool, input, **kw):
        seen.append((tool, input, kw, (tmp_path / "hook.log").read_text()))
        return Decision(True, "approved")

    out, code = hook.run("codex", json.dumps({"tool_name": "shell", "tool_input": {"command": "ls"}}),
                         ask=ask, log_path=tmp_path / "hook.log")
    assert code == 0
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert seen[0][0] == "Bash" and seen[0][1] == {"command": "ls"}
    assert seen[0][2] == {"origin": "hook", "backend": "codex"}
    entry = json.loads(seen[0][3])            # the log line existed before the gate answered
    assert entry["call"] == "shell" and entry["key"] == "ls" and entry["decision"] == "pending"


def test_run_deny_shapes(tmp_path):
    deny = lambda *a, **k: Decision(False, "denied", "user declined")
    out, _ = hook.run("antigravity", json.dumps({"tool_name": "run_command", "tool_input": {"command": "rm x"}}),
                      ask=deny, log_path=tmp_path / "l")
    assert json.loads(out) == {"decision": "deny", "reason": "user declined"}
    out, _ = hook.run("antigravity", json.dumps({"conversationId": "c1", "toolCall": {"name": "run_command", "args": {"command": "rm x"}}}),
                      ask=deny, log_path=tmp_path / "l")
    assert json.loads(out) == {"decision": "deny", "reason": "user declined"}
    out, _ = hook.run("qwen", json.dumps({"tool_name": "run_shell_command", "tool_input": {"command": "rm x"}}),
                      ask=deny, log_path=tmp_path / "l")
    assert json.loads(out) == {"decision": "deny", "reason": "user declined"}
    out, _ = hook.run("copilot", json.dumps({"toolName": "bash", "toolArgs": {"command": "rm x"}}),
                      ask=deny, log_path=tmp_path / "l")
    assert json.loads(out)["permissionDecision"] == "deny"


def test_run_mcp_and_readonly_skip_gate(tmp_path):
    def boom(*a, **k):
        raise AssertionError("gate must not be asked")

    out, _ = hook.run("codex", json.dumps({"tool_name": "mcp__veronica-mac__clipboard_write", "tool_input": {"text": "x"}}),
                      ask=boom, log_path=tmp_path / "l")
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    out, _ = hook.run("qwen", json.dumps({"tool_name": "read_file", "tool_input": {}}), ask=boom, log_path=tmp_path / "l")
    assert out == ""
    assert not (tmp_path / "l").exists()


def test_run_gates_other_servers_mcp_tools(tmp_path):
    """Only our own MCP tools skip the gate (tools.serve gates those). A
    third-party MCP tool the user configured in the CLI — including one on a
    server whose name happens to match ours — is confirm-class."""
    asked = []

    def ask(tool, input, **kw):
        asked.append(tool)
        return Decision(False, "denied", "user declined")

    for name in ("mcp__github__create_issue", "memory__create_entities", "mcp__memory__create_entities"):
        out, _ = hook.run("codex", json.dumps({"tool_name": name, "tool_input": {"x": "y"}}),
                          ask=ask, log_path=tmp_path / "l")
        assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert asked == ["mcp__github__create_issue", "memory__create_entities", "mcp__memory__create_entities"]
    # agy wraps them; a stranger's server is confirmed under the wrapper name
    out, _ = hook.run("antigravity", json.dumps({"toolCall": {"name": "call_mcp_tool",
                                                              "args": {"ServerName": "memory", "ToolName": "create_entities"}}}),
                      ask=ask, log_path=tmp_path / "l")
    assert json.loads(out)["decision"] == "deny" and asked[-1] == "call_mcp_tool"


def test_run_allows_every_form_of_our_own_mcp_tools(tmp_path):
    def boom(*a, **k):
        raise AssertionError("gate must not be asked")

    for name in ("mcp__veronica-mac__clipboard_write", "veronica-mac__clipboard_write",
                 "veronica-mac-clipboard_write", "mcp__veronica_mac__clipboard_write"):
        out, _ = hook.run("copilot", json.dumps({"toolName": name, "toolArgs": {"text": "x"}}),
                          ask=boom, log_path=tmp_path / "l")
        assert json.loads(out)["permissionDecision"] == "allow"
    out, _ = hook.run("antigravity", json.dumps({"toolCall": {"name": "call_mcp_tool",
                                                              "args": {"ServerName": "veronica-mac", "ToolName": "clipboard_write"}}}),
                      ask=boom, log_path=tmp_path / "l")
    assert json.loads(out)["decision"] == "allow"


def test_run_exception_fails_closed(tmp_path):
    def boom(*a, **k):
        raise RuntimeError("x")

    out, code = hook.run("codex", json.dumps({"tool_name": "shell", "tool_input": {"command": "ls"}}),
                         ask=boom, log_path=tmp_path / "l")
    assert code == 0 and json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"] == "gate error"


def test_run_bad_payload_fails_closed(tmp_path):
    out, code = hook.run("qwen", "not json", ask=lambda *a, **k: Decision(True, "approved"), log_path=None)
    assert code == 0 and json.loads(out) == {"decision": "deny", "reason": "gate error"}


def test_agy_shapes_map_to_canonical():
    assert hook.canonical_tool("antigravity", "run_command", {"CommandLine": "echo hi", "Cwd": "/"}) == ("Bash", {"command": "echo hi"})
    name, inp = hook.canonical_tool("antigravity", "write_to_file", {"TargetFile": "/tmp/x", "CodeContent": "y"})
    assert name == "Write" and inp["file_path"] == "/tmp/x" and inp["TargetFile"] == "/tmp/x"
    name, inp = hook.canonical_tool("antigravity", "replace_file_content", {"TargetFile": "/tmp/x", "TargetContent": "a"})
    assert name == "Edit" and inp["file_path"] == "/tmp/x"
    for t in ("view_file", "list_dir", "grep_search", "find_by_name", "search_web", "read_url_content"):
        assert hook.canonical_tool("antigravity", t, {"AbsolutePath": "/x"}) is None
    # agy wraps MCP calls: ours are unwrapped (allowed here, gated in tools.serve), others confirmed by name
    assert hook.canonical_tool("antigravity", "call_mcp_tool", {"ServerName": "veronica-mac", "ToolName": "read_battery", "Arguments": {}}) == ("mcp__mac__read_battery", {})
    assert hook.canonical_tool("antigravity", "call_mcp_tool", {"ServerName": "github", "ToolName": "create_issue", "Arguments": {}})[0] == "call_mcp_tool"


def test_canary_key_same_on_both_sides():
    """The adapter keys a native call from the raw stream parameters; the
    hook keys it from the canonicalised payload. They must agree."""
    raw = {"CommandLine": "echo canary-ok", "Cwd": "/tmp"}
    _, canon = hook.canonical_tool("antigravity", "run_command", raw)
    assert hook.canary_key("run_command", raw) == hook.canary_key("run_command", canon) == "echo canary-ok"
    raw = {"TargetFile": "/tmp/a.txt", "CodeContent": "hi"}
    _, canon = hook.canonical_tool("antigravity", "write_to_file", raw)
    assert hook.canary_key("write_to_file", raw) == hook.canary_key("write_to_file", canon) == "/tmp/a.txt"
    assert hook.canary_key("browser_click_element", {"Zeta": "z", "Alpha": "a"}) == "a"
    assert hook.canary_key("wait", {}) == "wait"


def test_run_logs_agy_key(tmp_path):
    out, _ = hook.run("antigravity", json.dumps({"toolCall": {"name": "run_command", "args": {"CommandLine": "echo canary-ok"}}}),
                      ask=lambda *a, **k: Decision(True, "approved"), log_path=tmp_path / "l")
    assert json.loads(out) == {"decision": "allow", "reason": ""}
    assert json.loads((tmp_path / "l").read_text())["key"] == "echo canary-ok"


def test_scope_file_limits_hook_to_our_conversation(tmp_path):
    scope = tmp_path / "active-conversation"
    payload = lambda cid: json.dumps({"conversationId": cid, "toolCall": {"name": "run_command", "args": {"CommandLine": "ls"}}})
    deny = lambda *a, **k: Decision(False, "denied", "user declined")
    # no scope file at all -> everything is ours
    out, _ = hook.run("antigravity", payload("c1"), ask=deny, log_path=tmp_path / "l")
    assert json.loads(out)["decision"] == "deny"
    # scope file names another conversation -> silent no-op, no log line
    scope.write_text("c2\n")
    out, _ = hook.run("antigravity", payload("c1"), ask=deny, log_path=tmp_path / "l2", scope_file=scope)
    assert out == "" and not (tmp_path / "l2").exists()
    # matching conversation -> gated
    out, _ = hook.run("antigravity", payload("c2"), ask=deny, log_path=tmp_path / "l2", scope_file=scope)
    assert json.loads(out)["decision"] == "deny" and (tmp_path / "l2").exists()
    # missing/empty scope file -> nothing is ours (fail quiet, agy's own flow applies)
    scope.unlink()
    out, _ = hook.run("antigravity", payload("c2"), ask=deny, log_path=tmp_path / "l3", scope_file=scope)
    assert out == "" and not (tmp_path / "l3").exists()


def test_scope_cwd_limits_hook_to_our_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    link = tmp_path / "link"
    link.symlink_to(ws)
    payload = lambda cwd: json.dumps({"cwd": cwd, "toolName": "bash", "toolArgs": {"command": "ls"}})
    deny = lambda *a, **k: Decision(False, "denied", "user declined")
    # our workspace, by realpath (copilot reports /private/tmp/... for /tmp/...)
    out, _ = hook.run("copilot", payload(str(link)), ask=deny, log_path=tmp_path / "l", scope_cwd=ws)
    assert json.loads(out)["permissionDecision"] == "deny" and (tmp_path / "l").exists()
    out, _ = hook.run("copilot", payload(str(ws)), ask=deny, log_path=tmp_path / "l", scope_cwd=link)
    assert json.loads(out)["permissionDecision"] == "deny"
    # another directory, or no cwd at all -> silent no-op, no log line
    for cwd in (str(tmp_path), "", None):
        out, _ = hook.run("copilot", payload(cwd), ask=deny, log_path=tmp_path / "l2", scope_cwd=ws)
        assert out == "" and not (tmp_path / "l2").exists()
    # the flag
    out, code = hook.main(["copilot", "--log", str(tmp_path / "l3"), "--scope-cwd", str(ws)], payload(str(ws)), ask=deny)
    assert code == 0 and json.loads(out)["permissionDecision"] == "deny"
    out, _ = hook.main(["copilot", "--log", str(tmp_path / "l4"), "--scope-cwd", str(ws)], payload(str(tmp_path)), ask=deny)
    assert out == "" and not (tmp_path / "l4").exists()


def test_main_flags_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE_SOCK", "/env/sock")
    monkeypatch.setenv("VERONICA_HOOK_LOG", str(tmp_path / "env.log"))
    seen = {}

    def ask(tool, input, **kw):
        seen["sock"] = hook.os.environ.get("VERONICA_GATE_SOCK")
        return Decision(True, "approved")

    scope = tmp_path / "scope"
    scope.write_text("c9")
    stdin = json.dumps({"conversationId": "c9", "toolCall": {"name": "run_command", "args": {"CommandLine": "ls"}}})
    out, code = hook.main(["antigravity", "--sock", "/flag/sock", "--log", str(tmp_path / "flag.log"),
                           "--scope-file", str(scope)], stdin, ask=ask)
    assert code == 0 and json.loads(out)["decision"] == "allow"
    assert seen["sock"] == "/flag/sock"
    assert (tmp_path / "flag.log").exists() and not (tmp_path / "env.log").exists()
    # env alone still works
    out, _ = hook.main(["antigravity"], stdin, ask=ask)
    assert json.loads(out)["decision"] == "allow" and (tmp_path / "env.log").exists()
