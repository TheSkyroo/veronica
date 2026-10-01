import json
import sys
import tomllib
from pathlib import Path

import pytest

from tests.brains_fakes import FakeProc
from veronica.brain import gateclient, hook
from veronica.brain.backends import cli
from veronica.brain.backends.codex import CodexBrain, toml_str, unwrap_shell
from veronica.brain.gate import ToolGate
from veronica.config import EDITABLE_SETTINGS, Settings, coerce_setting

FIX = Path(__file__).parent / "fixtures" / "brains"


def make(tmp_path, spawn=None, **settings):
    s = Settings(home=tmp_path, **settings)
    cards = []
    b = CodexBrain(s, ToolGate(s, None, on_tool=lambda su, d: cards.append((su, d))),
                   on_tool=lambda su, d: cards.append((su, d)), spawn=spawn)
    return b, cards


def lines(name):
    return [l for l in (FIX / name).read_text().splitlines() if l.strip()]


def thread_id(name):
    """The fixtures are refreshed by the live tests, so ids are read, not hardcoded."""
    return json.loads(lines(name)[0])["thread_id"]


def shell_command(name):
    """The (zsh-wrapped) command of the fixture's one command_execution item."""
    for l in lines(name):
        e = json.loads(l)
        if e["type"] == "item.started" and e["item"]["type"] == "command_execution":
            return e["item"]["command"]
    raise AssertionError("no command_execution in " + name)


def cfg(argv):
    """The `-c key=value` overrides as a dict, values parsed as TOML like codex does."""
    out = {}
    for i, x in enumerate(argv):
        if x == "-c":
            k, _, v = argv[i + 1].partition("=")
            out[k] = tomllib.loads(f"v = {v}")["v"]
    return out


def test_is_per_turn_and_argv_first_turn(tmp_path):
    b, _ = make(tmp_path)
    assert b.mode == "per_turn" and b.binary == "codex"
    b.prepare_workspace("SYS", native=True)
    a = b.argv("hi", None, [], native=True)
    assert a[:5] == ["codex", "exec", "--json", "--skip-git-repo-check", "-C"] and a[5] == str(b.workspace)
    assert a[-1] == "hi" and "--dangerously-bypass-hook-trust" in a and "-i" not in a
    c = cfg(a)
    assert c["approval_policy"] == "never" and c["sandbox_mode"] == "workspace-write"
    assert c["sandbox_workspace_write.writable_roots"] == [str(b.s.brain_cwd)]
    assert c["features.hooks"] is True and c["include_permissions_instructions"] is False
    assert c["developer_instructions"] == "SYS" and c["model_reasoning_effort"] == "low"
    # our MCP servers, with the gate env, never written to ~/.codex/config.toml
    assert c["mcp_servers.veronica-mac.command"] == sys.executable
    assert c["mcp_servers.veronica-mac.args"] == ["-m", "veronica.tools.serve", "mac"]
    assert c["mcp_servers.veronica-mac.env.VERONICA_GATE_SOCK"] == str(b.s.gate_socket)
    assert c["mcp_servers.veronica-mac.env.VERONICA_BRAIN"] == "codex"
    assert c["mcp_servers.veronica-mac.env.VERONICA_HOOK_LOG"] == str(b.hook_log)
    assert c["mcp_servers.veronica-mac.default_tools_approval_mode"] == "approve"   # else codex refuses every MCP call
    assert {k.split(".")[1] for k in c if k.startswith("mcp_servers.")} == {f"veronica-{s}" for s in hook.OUR_SERVERS}


def test_argv_resumed_native_off_images_effort(tmp_path):
    b, _ = make(tmp_path, effort="high")
    b.prepare_workspace("SYS", native=False)
    a = b.argv("look", "thr-1", [tmp_path / "img-1.png", tmp_path / "img-2.jpg"], native=False)
    assert a[:6] == ["codex", "exec", "resume", "thr-1", "--json", "--skip-git-repo-check"]
    assert "-C" not in a                     # `exec resume` rejects -C; it keeps the session's cwd
    assert a[-5:] == ["-i", str(tmp_path / "img-1.png"), "-i", str(tmp_path / "img-2.jpg"), "look"]
    c = cfg(a)
    assert c["sandbox_mode"] == "read-only" and c["model_reasoning_effort"] == "high"
    assert c["developer_instructions"].startswith("SYS\n\nDo not run shell commands")


def test_workspace_hooks_json(tmp_path):
    b, _ = make(tmp_path)
    b.prepare_workspace("SYS", native=True)
    hooks = json.loads((b.workspace / ".codex" / "hooks.json").read_text())
    [entry] = hooks["hooks"]["PreToolUse"]
    assert entry["matcher"] == "*" and entry["hooks"][0]["type"] == "command"
    cmd = entry["hooks"][0]["command"]
    assert cmd == b.hook_command()
    assert cmd == f"{sys.executable} -m veronica.brain.hook codex --sock {b.s.gate_socket} --log {b.hook_log}"
    # An explicit hook timeout, so the gate answer can't outlive it (codex 0.155
    # reads `timeout` on the handler and `timeoutSec` on the matcher group).
    assert entry["timeoutSec"] == entry["hooks"][0]["timeout"] == gateclient.HOOK_TIMEOUT_S
    assert gateclient.GATE_ANSWER_BUDGET_S < gateclient.HOOK_TIMEOUT_S
    b.prepare_workspace("SYS2", native=False)     # rewritten every turn, still one entry
    assert len(json.loads((b.workspace / ".codex" / "hooks.json").read_text())["hooks"]["PreToolUse"]) == 1
    assert b._prompt.startswith("SYS2")


@pytest.mark.parametrize("s", [
    "plain", "", 'quote " and back \\ slash', "multi\nline\ttabs\r\n", "unicode: café 日本 🍍",
    "ctrl \x01\x1f and del \x7f", "single 'quotes' and #hash = equals [brackets]",
])
def test_toml_str_roundtrips(s):
    assert tomllib.loads(f"x = {toml_str(s)}")["x"] == s


def test_unwrap_shell():
    assert unwrap_shell("/bin/zsh -lc 'echo canary-ok'") == "echo canary-ok"
    assert unwrap_shell("/bin/bash -lc \"ls -la 'My Dir'\"") == "ls -la 'My Dir'"
    assert unwrap_shell("echo hi") == "echo hi"
    assert unwrap_shell("bad 'quote") == "bad 'quote"


def test_canary_matches_hook_key_against_wrapped_stream_command(tmp_path):
    b, _ = make(tmp_path)
    stream = "/bin/zsh -lc 'echo canary-ok'"
    assert b.canary_matches("echo canary-ok", stream)          # what the hook logs (verified)
    assert b.canary_matches(stream, stream)                    # or the same string
    assert b.canary_matches("/bin/zsh -lc 'echo canary-ok'", "echo canary-ok") is False
    assert b.canary_matches("echo other", stream) is False
    assert b.canary_matches("ls", "/bin/zsh -lc 'ls -la'")     # substring fallback
    # the base stays strict
    assert cli.CliBrain.canary_matches(b, "echo canary-ok", stream) is False


def test_parse_shell_fixture(tmp_path):
    b, _ = make(tmp_path)
    events = [e for line in lines("codex-shell.jsonl") for e in b.parse(line)]
    assert isinstance(events[0], cli.Session) and events[0].id == thread_id("codex-shell.jsonl")
    assert not any(isinstance(e, cli.Error) for e in events)     # the two bypass notices are ignored
    starts = [e for e in events if isinstance(e, cli.ToolStart)]
    ends = [e for e in events if isinstance(e, cli.ToolEnd)]
    assert len(starts) == 1 and len(ends) == 1 and starts[0].call_id == ends[0].call_id
    cmd = shell_command("codex-shell.jsonl")
    assert starts[0].tool == "shell" and starts[0].native and starts[0].input == {"command": cmd}
    assert cmd.startswith("/bin/zsh -lc '") and unwrap_shell(cmd).endswith("echo canary-ok")
    assert b.native_key(starts[0].tool, starts[0].input) == cmd
    text = "".join(e.delta for e in events if isinstance(e, cli.Text))
    assert text.endswith("canary-ok")
    assert isinstance(events[-1], cli.Done)


def test_parse_plain_fixture(tmp_path):
    b, _ = make(tmp_path)
    events = [e for line in lines("codex-plain.jsonl") for e in b.parse(line)]
    assert isinstance(events[0], cli.Session) and events[0].id == thread_id("codex-plain.jsonl")
    text = "".join(e.delta for e in events if isinstance(e, cli.Text))
    assert "pineapple" in text.lower()
    assert isinstance(events[-1], cli.Done) and not any(isinstance(e, (cli.ToolStart, cli.Error)) for e in events)


def test_parse_file_change_items_and_patch_canary(tmp_path):
    """Verified: the hook gets tool_name "apply_patch" with the whole patch as
    tool_input.command; the stream shows a file_change item with the paths."""
    b, _ = make(tmp_path)
    path = str(tmp_path / "hello.txt")
    item = {"id": "item_3", "type": "file_change", "changes": [{"path": path, "kind": "add"}], "status": "in_progress"}
    [start] = b.parse(json.dumps({"type": "item.started", "item": item}))
    assert start == cli.ToolStart("item_3", "apply_patch", {"changes": [{"path": path, "kind": "add"}]}, native=True)
    assert b.parse(json.dumps({"type": "item.completed", "item": {**item, "status": "completed"}})) == [cli.ToolEnd("item_3")]
    assert b.native_key(start.tool, start.input) == path
    assert b.native_key("apply_patch", {"changes": []}) == "apply_patch"
    patch = f"*** Begin Patch\n*** Add File: {path}\n+hi\n*** End Patch"
    assert hook.canonical_tool("codex", "apply_patch", {"command": patch}) == ("Edit", {"command": patch})
    assert hook.canary_key("apply_patch", {"command": patch}) == patch          # what the hook logs
    assert b.canary_matches(patch, path)
    assert b.canary_matches(patch.replace(path, "hello.txt"), path)             # relative path in the patch
    assert b.canary_matches(patch, str(tmp_path / "other.txt")) is False


def test_parse_mcp_fixture(tmp_path):
    b, _ = make(tmp_path)
    events = [e for line in lines("codex-mcp.jsonl") for e in b.parse(line)]
    [start] = [e for e in events if isinstance(e, cli.ToolStart)]
    [end] = [e for e in events if isinstance(e, cli.ToolEnd)]
    assert start.tool == "mcp__mac__volume_get" and start.native is False and start.input == {}
    assert start.call_id == end.call_id and isinstance(events[-1], cli.Done)
    assert not any(isinstance(e, cli.Error) for e in events)


def test_parse_mcp_items(tmp_path):
    b, _ = make(tmp_path)
    item = {"id": "item_2", "type": "mcp_tool_call", "server": "veronica-mac", "tool": "volume_get",
            "arguments": {}, "status": "in_progress"}
    [start] = b.parse(json.dumps({"type": "item.started", "item": item}))
    assert start == cli.ToolStart("item_2", "mcp__mac__volume_get", {}, native=False)
    [end] = b.parse(json.dumps({"type": "item.completed", "item": {**item, "status": "completed", "result": {}}}))
    assert end == cli.ToolEnd("item_2")
    # another server's tool is not gated in tools.serve, so it must go through
    # the hook — native=True puts it under the canary too.
    [other] = b.parse(json.dumps({"type": "item.started", "item": {**item, "server": "github", "tool": "search",
                                                                   "arguments": {"q": "x"}}}))
    assert other.tool == "mcp__github__search" and other.input == {"q": "x"} and other.native is True
    [named] = b.parse(json.dumps({"type": "item.started", "item": {**item, "server": "memory", "tool": "create_entities"}}))
    assert named.tool == "mcp__memory__create_entities" and named.native is True


def test_parse_errors_and_limits(tmp_path):
    b, _ = make(tmp_path)
    [err] = b.parse(json.dumps({"type": "turn.failed", "error": {"message": "usage limit reached"}}))
    assert isinstance(err, cli.Error) and err.message == "usage limit reached"
    [err] = b.parse(json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "error", "message": "boom"}}))
    assert isinstance(err, cli.Error) and err.message == "boom"
    [err] = b.parse(json.dumps({"type": "error", "message": "stream disconnected"}))
    assert isinstance(err, cli.Error) and err.message == "stream disconnected"
    assert b.parse(json.dumps({"type": "turn.started"})) == []
    assert b.parse(json.dumps({"type": "item.started", "item": {"id": "i", "type": "agent_message", "text": "partial"}})) == []
    assert b.parse(json.dumps({"type": "item.completed", "item": {"id": "i", "type": "reasoning", "text": "hmm"}})) == []
    assert b.parse(json.dumps({"type": "something.new"})) == []


def test_hook_side_maps_codex_bash_payload(tmp_path):
    """The verified PreToolUse payload: tool_name "Bash" with a string command."""
    assert hook.canonical_tool("codex", "Bash", {"command": "echo canary-ok"}) == ("Bash", {"command": "echo canary-ok"})
    seen = []
    log = tmp_path / "hook.log"

    def ask(name, inp, **kw):
        seen.append((name, inp, kw))
        return type("D", (), {"allow": False, "message": "no"})()
    out, code = hook.main(["codex", "--log", str(log)],
                          json.dumps({"tool_name": "Bash", "tool_input": {"command": "echo canary-ok"}}), ask=ask)
    assert code == 0 and json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert seen[0][2] == {"origin": "hook", "backend": "codex"}
    [entry] = [json.loads(l) for l in log.read_text().splitlines()]
    assert entry["key"] == "echo canary-ok"


async def test_end_to_end_with_fake_spawn(tmp_path):
    procs = []

    async def spawn(argv, cwd, env):
        procs.append((argv, cwd, env, FakeProc(lines("codex-plain.jsonl"))))
        return procs[-1][3]

    b, _ = make(tmp_path, spawn=spawn)
    out = [x async for x in b.ask("say pineapple")]
    assert out == ["pineapple"]
    argv, cwd, env, proc = procs[0]
    assert argv[0] == "codex" and "resume" not in argv and argv[-1] == "say pineapple"
    assert cwd == str(b.workspace) and env["VERONICA_BRAIN"] == "codex"
    assert (b.workspace / ".codex" / "hooks.json").exists()
    assert b.s.session_file_for("codex").read_text() == thread_id("codex-plain.jsonl")
    assert b._proc is None and proc.returncode == 0
    # the next turn resumes the thread
    [x async for x in b.ask("again")]
    assert procs[1][0][1:4] == ["exec", "resume", thread_id("codex-plain.jsonl")]


async def test_shell_turn_passes_canary_when_hook_logged(tmp_path):
    async def spawn(argv, cwd, env):
        # "the hook fired": its key is the bare command, not the zsh-wrapped stream one
        key = unwrap_shell(shell_command("codex-shell.jsonl"))
        b.hook_log.write_text(json.dumps({"ts": 9e12, "call": "Bash", "key": key, "decision": "pending"}) + "\n")
        return FakeProc(lines("codex-shell.jsonl"))

    b, _ = make(tmp_path, spawn=spawn)
    out = [x async for x in b.ask("run it")]
    assert out[-1] == "canary-ok" and b.s.codex_native_tools is True


async def test_shell_turn_trips_canary_without_hook_line(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(cli.prefs, "save_settings_override", lambda k, v: saved.update({k: v}))
    runs = []

    async def spawn(argv, cwd, env):
        runs.append(argv)
        return FakeProc(lines("codex-shell.jsonl") if len(runs) == 1 else lines("codex-plain.jsonl"))

    b, _ = make(tmp_path, spawn=spawn)
    b.canary_grace_s = 0.1
    out = [x async for x in b.ask("run it")]
    assert "Hooks aren't running on Codex, so I've turned off its shell. Tools still work." in out
    assert out[-1] == "pineapple" and "canary-ok" not in out      # killed before the reply, retried tools-off
    assert saved == {"codex_native_tools": False} and b.s.codex_native_tools is False
    assert 'sandbox_mode="read-only"' in runs[1] and 'sandbox_mode="workspace-write"' in runs[0]


async def test_limit_raises_limit_error(tmp_path):
    async def spawn(argv, cwd, env):
        return FakeProc([json.dumps({"type": "thread.started", "thread_id": "t"}),
                         json.dumps({"type": "turn.failed", "error": {"message": "You've hit your usage limit"}})])

    b, _ = make(tmp_path, spawn=spawn)
    with pytest.raises(cli.LimitError):
        [x async for x in b.ask("hi")]
    assert b._proc is None


def test_codex_native_tools_is_a_live_bool():
    assert Settings().codex_native_tools is True
    f = EDITABLE_SETTINGS["codex_native_tools"]
    assert f.kind == "bool" and f.restart is False and f.label == "Codex: allow its own shell"
    assert coerce_setting("codex_native_tools", "off") is False
