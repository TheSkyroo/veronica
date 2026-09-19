import json
import sys
import uuid
from pathlib import Path

import pytest

from tests.brains_fakes import FakeProc
from veronica.brain import hook
from veronica.brain.backends import cli
from veronica.brain.backends.copilot import (
    HOOK_TIMEOUT_S,
    NATIVE_ACTION_TOOLS,
    CopilotBrain,
)
from veronica.brain.gate import ToolGate
from veronica.config import EDITABLE_SETTINGS, Settings, coerce_setting

FIX = Path(__file__).parent / "fixtures" / "brains"


def make(tmp_path, spawn=None, **settings):
    s = Settings(home=tmp_path, **settings)
    cards = []
    b = CopilotBrain(s, ToolGate(s, None, on_tool=lambda su, d: cards.append((su, d))),
                     on_tool=lambda su, d: cards.append((su, d)), spawn=spawn)
    b.hooks_file = tmp_path / "dot-copilot" / "hooks" / "veronica.json"   # never the real ~/.copilot
    return b, cards


def lines(name):
    return [l for l in (FIX / name).read_text().splitlines() if l.strip()]


def session_id(name):
    """The fixtures are refreshed by the live tests, so ids are read, not hardcoded."""
    return json.loads(lines(name)[-1])["sessionId"]


def shell_command(name):
    """The command of the fixture's one bash call."""
    for l in lines(name):
        e = json.loads(l)
        if e["type"] == "tool.execution_start" and e["data"]["toolName"] == "bash":
            return e["data"]["arguments"]["command"]
    raise AssertionError("no bash call in " + name)


def flag(argv, name):
    """Values following `--name` up to the next flag (variadic options)."""
    i = argv.index(name)
    out = []
    for x in argv[i + 1:]:
        if x.startswith("--"):
            break
        out.append(x)
    return out


def test_is_per_turn_and_argv_first_turn(tmp_path):
    b, _ = make(tmp_path)
    assert b.mode == "per_turn" and b.binary == "copilot"
    a = b.argv("hi", None, [], native=True)
    assert a[:8] == ["copilot", "-p", "hi", "--output-format", "json", "--silent", "--no-ask-user",
                     "--disable-builtin-mcps"]
    assert "--resume" not in a and "--attachment" not in a
    [sid] = flag(a, "--session-id")
    assert uuid.UUID(sid).version == 4 and sid == b._new_session_id
    assert "--reasoning-effort" not in a           # an error with the default model "auto"
    assert "--allow-all-tools" in a and "--allow-all-paths" in a
    assert "--allow-tool" not in a and "--excluded-tools" not in a
    # our MCP servers, with the gate env, never written to ~/.copilot/mcp-config.json
    [cfg] = flag(a, "--additional-mcp-config")
    servers = json.loads(cfg)["mcpServers"]
    assert set(servers) == {f"veronica-{s}" for s in hook.OUR_SERVERS}
    mac = servers["veronica-mac"]
    assert mac["type"] == "local" and mac["command"] == sys.executable and mac["tools"] == ["*"]
    assert mac["args"] == ["-m", "veronica.tools.serve", "mac"]
    assert mac["env"] == {"VERONICA_GATE_SOCK": str(b.s.gate_socket), "VERONICA_BRAIN": "copilot",
                          "VERONICA_HOOK_LOG": str(b.hook_log)}
    # a fresh id per first-turn spawn (a failed first turn never reuses one)
    assert flag(b.argv("hi", None, [], native=True), "--session-id") != [sid]


def test_argv_resumed_native_off_images(tmp_path):
    b, _ = make(tmp_path, effort="high")
    a = b.argv("look", "sess-1", [tmp_path / "img-1.png", tmp_path / "img-2.jpg"], native=False)
    assert flag(a, "--resume") == ["sess-1"] and "--session-id" not in a
    assert "--reasoning-effort" not in a
    assert a.count("--attachment") == 2
    assert flag(a, "--attachment") == [str(tmp_path / "img-1.png")]
    assert a[a.index("--attachment", a.index("--attachment") + 1) + 1] == str(tmp_path / "img-2.jpg")
    # tools off: the shell/edit/agent tools are hidden and only our servers are allowed (no glob syntax)
    assert "--allow-all-tools" not in a and "--allow-all-paths" not in a
    assert flag(a, "--allow-tool") == [f"veronica-{s}" for s in hook.OUR_SERVERS]
    assert flag(a, "--excluded-tools") == list(NATIVE_ACTION_TOOLS)
    assert {"bash", "apply_patch", "task"} <= set(NATIVE_ACTION_TOOLS)
    assert a[-2] == "--additional-mcp-config"      # variadic flags never swallow the config


def test_workspace_instructions_and_user_level_hook(tmp_path):
    b, _ = make(tmp_path)
    b.prepare_workspace("SYS", native=True)
    assert (b.workspace / ".github" / "copilot-instructions.md").read_text() == "SYS"
    hooks = json.loads(b.hooks_file.read_text())
    assert hooks["version"] == 1 and list(hooks["hooks"]) == ["preToolUse"]
    [entry] = hooks["hooks"]["preToolUse"]
    assert entry["type"] == "command" and entry["timeoutSec"] == HOOK_TIMEOUT_S == 600   # a timeout fails OPEN
    assert entry["bash"] == b.hook_command()
    assert entry["bash"] == (f"{sys.executable} -m veronica.brain.hook copilot --sock {b.s.gate_socket} "
                             f"--log {b.hook_log} --scope-cwd {b.workspace}")
    b.prepare_workspace("SYS2", native=False)     # rewritten every turn, still one entry
    assert len(json.loads(b.hooks_file.read_text())["hooks"]["preToolUse"]) == 1
    text = (b.workspace / ".github" / "copilot-instructions.md").read_text()
    assert text.startswith("SYS2\n\nDo not run shell commands")


async def test_close_removes_hook_file(tmp_path):
    b, _ = make(tmp_path)
    b.prepare_workspace("SYS", native=True)
    assert b.hooks_file.exists()
    await b.close()
    assert not b.hooks_file.exists()
    await b.close()                                # idempotent


def test_parse_plain_fixture(tmp_path):
    b, _ = make(tmp_path)
    events = [e for line in lines("copilot-plain.jsonl") for e in b.parse(line)]
    text = "".join(e.delta for e in events if isinstance(e, cli.Text))
    assert "pineapple" in text.lower()
    assert isinstance(events[-1], cli.Done) and isinstance(events[-2], cli.Session)
    assert events[-2].id == session_id("copilot-plain.jsonl")
    assert events[-1].final_text == "pineapple"           # assistant.message content, for a stream without deltas
    assert not any(isinstance(e, (cli.ToolStart, cli.Error)) for e in events)


def test_parse_shell_fixture(tmp_path):
    b, _ = make(tmp_path)
    events = [e for line in lines("copilot-shell.jsonl") for e in b.parse(line)]
    starts = [e for e in events if isinstance(e, cli.ToolStart)]
    ends = [e for e in events if isinstance(e, cli.ToolEnd)]
    assert starts and len(starts) == len(ends) and {s.call_id for s in starts} == {e.call_id for e in ends}
    [bash] = [s for s in starts if s.tool == "bash"]
    cmd = shell_command("copilot-shell.jsonl")
    assert bash.native and bash.input["command"] == cmd and "canary-ok" in cmd
    assert b.native_key(bash.tool, bash.input) == cmd == hook.canary_key("bash", {"command": cmd})
    text = "".join(e.delta for e in events if isinstance(e, cli.Text))
    assert "canary-ok" in text
    assert isinstance(events[-1], cli.Done)


def test_parse_mcp_fixture(tmp_path):
    b, _ = make(tmp_path)
    events = [e for line in lines("copilot-mcp.jsonl") for e in b.parse(line)]
    starts = [e for e in events if isinstance(e, cli.ToolStart)]
    [vol] = [s for s in starts if s.tool == "mcp__mac__volume_get"]
    assert vol.native is False and vol.input == {}
    assert any(isinstance(e, cli.ToolEnd) and e.call_id == vol.call_id for e in events)
    assert isinstance(events[-1], cli.Done) and not any(isinstance(e, cli.Error) for e in events)


def test_parse_tool_events(tmp_path):
    """Verified shapes: bash with a dict, apply_patch with the patch as a bare
    string, our MCP tools as veronica-<server>-<tool>."""
    b, _ = make(tmp_path)
    start = {"type": "tool.execution_start", "data": {"toolCallId": "c1", "toolName": "bash",
                                                       "arguments": {"command": "echo hi", "description": "d"}}}
    assert b.parse(json.dumps(start)) == [cli.ToolStart("c1", "bash", {"command": "echo hi", "description": "d"}, native=True)]
    assert b.parse(json.dumps({"type": "tool.execution_complete", "data": {"toolCallId": "c1", "success": True}})) == [cli.ToolEnd("c1")]
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch\n"
    [s] = b.parse(json.dumps({"type": "tool.execution_start", "data": {"toolCallId": "c2", "toolName": "apply_patch",
                                                                      "arguments": patch}}))
    assert s == cli.ToolStart("c2", "apply_patch", {"command": patch}, native=True)
    assert b.native_key(s.tool, s.input) == patch                          # equals what the hook logs
    assert hook.canary_key("apply_patch", {"command": patch}) == patch
    assert b.canary_matches(patch, patch) and not b.canary_matches(patch, patch + "x")
    [m] = b.parse(json.dumps({"type": "tool.execution_start", "data": {
        "toolCallId": "c3", "toolName": "veronica-mac-volume_get", "arguments": {},
        "mcpServerName": "veronica-mac", "mcpToolName": "volume_get"}}))
    assert m == cli.ToolStart("c3", "mcp__mac__volume_get", {}, native=False)
    [other] = b.parse(json.dumps({"type": "tool.execution_start", "data": {
        "toolCallId": "c4", "toolName": "github-mcp-server-search_code", "arguments": {"q": "x"},
        "mcpServerName": "github-mcp-server", "mcpToolName": "search_code"}}))
    assert other.tool == "github-mcp-server-search_code" and other.native is True   # hook confirms it by name
    [view] = b.parse(json.dumps({"type": "tool.execution_start", "data": {"toolCallId": "c5", "toolName": "view",
                                                                         "arguments": {"path": "/tmp/x"}}}))
    assert view.tool in hook.READONLY_TOOLS and b.readonly_summary(view.tool, view.input) == "view /tmp/x"
    # the deny by our hook is an ordinary end, not an error
    assert b.parse(json.dumps({"type": "tool.execution_complete", "data": {
        "toolCallId": "c2", "success": False, "error": {"message": "Denied by preToolUse hook: user declined"}}})) == [cli.ToolEnd("c2")]


def test_parse_result_errors_and_noise(tmp_path):
    b, _ = make(tmp_path)
    assert b.parse(json.dumps({"type": "result", "sessionId": "s1", "exitCode": 0})) == [cli.Session("s1"), cli.Done("")]
    [err] = b.parse(json.dumps({"type": "result", "sessionId": "s1", "exitCode": 1}))
    assert isinstance(err, cli.Error) and err.message == "exit 1"
    [err] = b.parse(json.dumps({"type": "error", "data": {"message": "rate limit exceeded"}}))
    assert isinstance(err, cli.Error) and err.message == "rate limit exceeded"
    # no sessionId in the result: the pre-generated id stands in
    b.argv("hi", None, [], native=True)
    assert b.parse(json.dumps({"type": "result", "exitCode": 0}))[0] == cli.Session(b._new_session_id)
    for noise in ({"type": "session.tools_updated", "data": {}, "ephemeral": True},
                  {"type": "assistant.turn_start", "data": {"turnId": "0"}},
                  {"type": "tool.execution_partial_result", "data": {"toolCallId": "c1", "partialOutput": "x"}},
                  {"type": "assistant.message_delta", "data": {"deltaContent": ""}},
                  {"type": "assistant.idle", "data": {}}, {"type": "something.new"}):
        assert b.parse(json.dumps(noise)) == []


def test_hook_side_maps_copilot_payloads(tmp_path):
    """The verified preToolUse payload: camelCase, cwd as realpath, apply_patch args as a string."""
    seen = []
    log = tmp_path / "hook.log"
    ws = tmp_path / "ws"
    ws.mkdir()

    def ask(name, inp, **kw):
        seen.append((name, inp, kw))
        return type("D", (), {"allow": False, "message": "no"})()
    base = {"sessionId": "s", "timestamp": 1, "cwd": str(ws.resolve())}
    out, code = hook.main(["copilot", "--log", str(log), "--scope-cwd", str(ws)],
                          json.dumps({**base, "toolName": "bash", "toolArgs": {"command": "echo canary-ok", "description": "d"}}), ask=ask)
    assert code == 0 and json.loads(out) == {"permissionDecision": "deny", "permissionDecisionReason": "no"}
    assert seen[0][:2] == ("Bash", {"command": "echo canary-ok"}) and seen[0][2] == {"origin": "hook", "backend": "copilot"}
    patch = "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch\n"
    out, _ = hook.main(["copilot", "--log", str(log), "--scope-cwd", str(ws)],
                       json.dumps({**base, "toolName": "apply_patch", "toolArgs": patch}), ask=ask)
    assert json.loads(out)["permissionDecision"] == "deny" and seen[1][:2] == ("Edit", {"command": patch})
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert [e["key"] for e in entries] == ["echo canary-ok", patch]
    # ours pass the hook untouched (tools.serve gates them) and are not logged
    out, _ = hook.main(["copilot", "--log", str(log), "--scope-cwd", str(ws)],
                       json.dumps({**base, "toolName": "veronica-mac-volume_get", "toolArgs": {}}), ask=ask)
    assert json.loads(out)["permissionDecision"] == "allow" and len(seen) == 2
    assert len(log.read_text().splitlines()) == 2
    # another cwd (the user's own copilot session): silent no-op
    out, _ = hook.main(["copilot", "--log", str(log), "--scope-cwd", str(ws)],
                       json.dumps({**base, "cwd": str(tmp_path), "toolName": "bash", "toolArgs": {"command": "rm -rf x"}}), ask=ask)
    assert out == "" and len(seen) == 2


async def test_end_to_end_with_fake_spawn(tmp_path):
    procs = []

    async def spawn(argv, cwd, env):
        procs.append((argv, cwd, env, FakeProc(lines("copilot-plain.jsonl"))))
        return procs[-1][3]

    b, _ = make(tmp_path, spawn=spawn)
    out = [x async for x in b.ask("say pineapple")]
    assert out == ["pineapple"]
    argv, cwd, env, proc = procs[0]
    assert argv[:3] == ["copilot", "-p", "say pineapple"] and "--resume" not in argv
    assert cwd == str(b.workspace) and env["VERONICA_BRAIN"] == "copilot"
    assert (b.workspace / ".github" / "copilot-instructions.md").exists() and b.hooks_file.exists()
    assert b.s.session_file_for("copilot").read_text() == session_id("copilot-plain.jsonl")
    assert b._proc is None and proc.returncode == 0
    # the next turn resumes the session
    [x async for x in b.ask("again")]
    assert flag(procs[1][0], "--resume") == [session_id("copilot-plain.jsonl")]


async def test_stream_dying_before_result_saves_no_session(tmp_path):
    async def spawn(argv, cwd, env):
        return FakeProc(lines("copilot-plain.jsonl")[:-1], exit_code=1)   # no result line

    b, _ = make(tmp_path, spawn=spawn)
    out = [x async for x in b.ask("hi")]
    assert out == ["Copilot returned an error, check the log."]
    assert not b.s.session_file_for("copilot").exists()


async def test_shell_turn_passes_canary_when_hook_logged(tmp_path):
    async def spawn(argv, cwd, env):
        # "the hook fired": the key is the bare command on both sides
        b.hook_log.write_text(json.dumps({"ts": 9e12, "call": "bash", "key": shell_command("copilot-shell.jsonl"),
                                          "decision": "pending"}) + "\n")
        return FakeProc(lines("copilot-shell.jsonl"))

    b, _ = make(tmp_path, spawn=spawn)
    out = [x async for x in b.ask("run it")]
    assert any("canary-ok" in x for x in out) and b.s.copilot_native_tools is True


async def test_shell_turn_trips_canary_without_hook_line(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(cli.prefs, "save_settings_override", lambda k, v: saved.update({k: v}))
    runs = []

    async def spawn(argv, cwd, env):
        runs.append(argv)
        return FakeProc(lines("copilot-shell.jsonl") if len(runs) == 1 else lines("copilot-plain.jsonl"))

    b, _ = make(tmp_path, spawn=spawn)
    b.canary_grace_s = 0.1
    out = [x async for x in b.ask("run it")]
    assert "Hooks aren't running on Copilot, so I've turned off its shell. Tools still work." in out
    assert out[-1] == "pineapple" and not any("canary-ok" in x for x in out)   # killed, retried tools-off
    assert saved == {"copilot_native_tools": False} and b.s.copilot_native_tools is False
    assert "--allow-all-tools" in runs[0] and "--allow-all-tools" not in runs[1]
    assert "--excluded-tools" in runs[1]


async def test_limit_raises_limit_error(tmp_path):
    async def spawn(argv, cwd, env):
        return FakeProc([json.dumps({"type": "error", "data": {"message": "You've hit your usage limit"}})])

    b, _ = make(tmp_path, spawn=spawn)
    with pytest.raises(cli.LimitError):
        [x async for x in b.ask("hi")]
    assert b._proc is None


def test_copilot_native_tools_is_a_live_bool():
    assert Settings().copilot_native_tools is True
    f = EDITABLE_SETTINGS["copilot_native_tools"]
    assert f.kind == "bool" and f.restart is False and f.label == "Copilot: allow its own shell"
    assert coerce_setting("copilot_native_tools", "off") is False
