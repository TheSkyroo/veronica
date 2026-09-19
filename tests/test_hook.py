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
    ("copilot", "veronica-mac__clipboard_write", {"text": "x"}, ("mcp__mac__clipboard_write", {"text": "x"})),
    ("qwen", "mac__clipboard_write", {"text": "x"}, ("mcp__mac__clipboard_write", {"text": "x"})),
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
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"] == "user declined"
    out, _ = hook.run("antigravity", json.dumps({"conversationId": "c1", "toolCall": {"name": "run_command", "args": {"command": "rm x"}}}),
                      ask=deny, log_path=tmp_path / "l")
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
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
