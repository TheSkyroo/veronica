import json
import sys
from pathlib import Path

from tests.brains_fakes import FakeProc
from veronica.brain import gateclient, hook
from veronica.brain.backends import cli
from veronica.brain.backends.antigravity import AntigravityBrain
from veronica.brain.gate import ToolGate
from veronica.config import Settings

FIX = Path(__file__).parent / "fixtures" / "brains"


def make(tmp_path, spawn=None, **settings):
    s = Settings(home=tmp_path, **settings)
    cards = []
    b = AntigravityBrain(s, ToolGate(s, None, on_tool=lambda su, d: cards.append((su, d))),
                         on_tool=lambda su, d: cards.append((su, d)), spawn=spawn)
    b.hooks_file = tmp_path / "gemini" / "config" / "hooks.json"   # never the real ~/.gemini
    runs = []
    b._run = lambda argv, **kw: runs.append(argv)
    return b, runs, cards


def lines(name):
    return [l for l in (FIX / name).read_text().splitlines() if l.strip()]


def conv_id(name):
    """The fixtures are refreshed by the live tests, so ids are read, not hardcoded."""
    return json.loads(lines(name)[0])["conversation_id"]


def test_is_persistent_and_argv(tmp_path):
    b, _, _ = make(tmp_path)
    assert b.mode == "persistent" and b.binary == "agy"
    a = b.argv("", None, [], native=True)
    assert a == ["agy", "--output-format", "stream-json", "--input-format", "stream-json", "--print=",
                 "--dangerously-skip-permissions", "--effort", "low"]
    a2 = b.argv("", "conv-1", [], native=True)
    assert a2[a2.index("--conversation") + 1] == "conv-1"
    # native tools off: no auto-approve flag, so headless agy denies its own shell/file tools itself
    assert "--dangerously-skip-permissions" not in b.argv("", None, [], native=False)
    b.s.effort = "high"
    assert b.argv("", None, [], native=True)[-2:] == ["--effort", "high"]


def test_turn_message_and_image_refs(tmp_path):
    b, _, _ = make(tmp_path)
    m = json.loads(b.turn_message("hi", []))
    assert m == {"event": "user", "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}
    m = json.loads(b.turn_message("look", [tmp_path / "img-1.png"]))
    assert m["message"]["content"][0]["text"] == f"@{tmp_path / 'img-1.png'} look"


def test_workspace_files_and_hook_merge(tmp_path):
    b, runs, _ = make(tmp_path)
    b.hooks_file.parent.mkdir(parents=True)
    b.hooks_file.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "run_command", "hooks": [
        {"type": "command", "command": "/usr/bin/true"}]}], "Stop": [{"hooks": []}]}, "other": 1}))
    b.prepare_workspace("SYSTEM PROMPT", native=True)
    assert (b.workspace / "AGENTS.md").read_text() == "SYSTEM PROMPT"
    hooks = json.loads(b.hooks_file.read_text())
    assert hooks["other"] == 1 and "Stop" in hooks["hooks"]              # the user's file survives
    pre = hooks["hooks"]["PreToolUse"]
    assert pre[0]["hooks"][0]["command"] == "/usr/bin/true"
    cmd = pre[1]["hooks"][0]["command"]
    assert pre[1]["matcher"] == "*" and cmd == b.hook_command()
    # agy's own default is 30 s; ours is explicit and above the gate budget
    assert pre[1]["hooks"][0]["timeout"] == gateclient.HOOK_TIMEOUT_S
    assert gateclient.GATE_ANSWER_BUDGET_S < gateclient.HOOK_TIMEOUT_S
    assert cmd.startswith(f"{sys.executable} -m veronica.brain.hook antigravity --sock {b.s.gate_socket} ")
    assert f"--log {b.hook_log}" in cmd and cmd.endswith(f"--scope-file {b.workspace / 'active-conversation'}")
    # idempotent
    b.prepare_workspace("SYSTEM PROMPT", native=True)
    b.prepare_workspace("SYSTEM PROMPT", native=False)
    assert len(json.loads(b.hooks_file.read_text())["hooks"]["PreToolUse"]) == 2
    assert (b.workspace / "AGENTS.md").read_text().startswith("SYSTEM PROMPT\n\nDo not run shell commands")
    # MCP servers registered once per process, with the gate env
    assert len(runs) == len(hook.OUR_SERVERS)
    assert runs[0] == ["agy", "mcp", "add", "--env", f"VERONICA_GATE_SOCK={b.s.gate_socket}",
                       "--env", "VERONICA_BRAIN=antigravity", "--env", f"VERONICA_HOOK_LOG={b.hook_log}",
                       "veronica-mac", sys.executable, "-m", "veronica.tools.serve", "mac"]
    assert {r[-1] for r in runs} == set(hook.OUR_SERVERS)


def test_hook_file_created_when_missing_or_broken(tmp_path):
    b, _, _ = make(tmp_path)
    b.prepare_workspace("P", native=True)
    assert json.loads(b.hooks_file.read_text())["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == b.hook_command()
    b.hooks_file.write_text("not json")
    b.prepare_workspace("P", native=True)
    assert len(json.loads(b.hooks_file.read_text())["hooks"]["PreToolUse"]) == 1


def test_parse_plain_fixture(tmp_path):
    b, _, _ = make(tmp_path)
    events = [e for line in lines("antigravity-plain.jsonl") for e in b.parse(line)]
    assert isinstance(events[0], cli.Session) and events[0].id == conv_id("antigravity-plain.jsonl")
    text = "".join(e.delta for e in events if isinstance(e, cli.Text))
    assert "pineapple" in text.lower()
    assert isinstance(events[-1], cli.Done) and events[-1].final_text == "pineapple\n"


def test_parse_tool_fixture(tmp_path):
    b, _, _ = make(tmp_path)
    events = [e for line in lines("antigravity-tool.jsonl") for e in b.parse(line)]
    starts = [e for e in events if isinstance(e, cli.ToolStart)]
    ends = [e for e in events if isinstance(e, cli.ToolEnd)]
    assert len(starts) == 1 and starts[0].tool == "run_command" and starts[0].native
    assert starts[0].input["CommandLine"].endswith("&& echo canary-ok")
    assert starts[0].call_id == f"{conv_id('antigravity-tool.jsonl')}:2" and ends[0].call_id == starts[0].call_id
    assert b.native_key(starts[0].tool, starts[0].input) == starts[0].input["CommandLine"]


def test_parse_mcp_and_error_shapes(tmp_path):
    b, _, _ = make(tmp_path)
    conv = "c"
    su = lambda **kw: json.dumps({"event": "step_update", "step_update": {"conversation_id": conv, **kw}})
    [start] = b.parse(su(step_index=4, state="ACTIVE", step_type="tool", tool_name="call_mcp_tool",
                         tool_info={"name": "call_mcp_tool", "parameters": {"ServerName": "veronica-mac", "ToolName": "volume_get"}}))
    assert isinstance(start, cli.ToolStart) and start.native is False and start.call_id == "c:4"
    [end] = b.parse(su(step_index=4, state="ERROR", step_type="tool", tool_name="run_command", tool_info={"error": {"message": "denied"}}))
    assert isinstance(end, cli.ToolEnd) and end.call_id == "c:4"
    assert b.parse(su(step_index=0, state="DONE", step_type="user_input")) == []
    assert b.parse(su(step_index=1, state="DONE", step_type="agent_response")) == []      # no delta
    [err] = b.parse(json.dumps({"event": "result", "result": {"status": "ERROR", "error": "boom"}}))
    assert isinstance(err, cli.Error) and err.message == "boom"
    assert b.parse(json.dumps({"event": "something_new"})) == []


def test_native_key_matches_hook_for_agy_tools(tmp_path):
    b, _, _ = make(tmp_path)
    assert b.native_key("run_command", {"CommandLine": "ls -la", "Cwd": "/"}) == "ls -la"
    assert b.native_key("write_to_file", {"TargetFile": "/tmp/a", "CodeContent": "x"}) == "/tmp/a"
    assert b.native_key("view_file", {"AbsolutePath": "/tmp/a"}) == "/tmp/a"
    for t in ("view_file", "list_dir", "grep_search", "find_by_name", "search_web", "read_url_content"):
        assert t in hook.READONLY_TOOLS


async def test_end_to_end_with_fake_spawn(tmp_path):
    procs = []

    async def spawn(argv, cwd, env):
        procs.append((argv, FakeProc(lines("antigravity-plain.jsonl"), hang=True)))
        return procs[-1][1]

    b, runs, _ = make(tmp_path, spawn=spawn)
    out = [x async for x in b.ask("say pineapple")]
    assert out == ["pineapple"]
    argv, proc = procs[0]
    assert argv[0] == "agy" and "--conversation" not in argv
    # session + scope file written from init, before the prompt line went in
    sid = conv_id("antigravity-plain.jsonl")
    assert b.s.session_file_for("antigravity").read_text() == sid
    assert (b.workspace / "active-conversation").read_text() == sid
    msg = json.loads(proc.stdin.lines[0])
    assert msg["message"]["content"][0]["text"] == "say pineapple"
    assert runs and b._proc is proc
    # a second ask reuses the child; a tool turn passes the canary when the hook logged it
    b.hook_log.write_text("")
    proc.feed(lines("antigravity-tool.jsonl")[1:])
    [start] = [e for line in lines("antigravity-tool.jsonl") for e in b.parse(line) if isinstance(e, cli.ToolStart)]
    key = b.native_key(start.tool, start.input)
    orig = b.turn_message

    def turn_message(text, image_paths):   # "the hook fired": runs after the per-turn log truncation
        b.hook_log.write_text(json.dumps({"ts": 9e12, "call": "run_command", "key": key, "decision": "pending"}) + "\n")
        return orig(text, image_paths)
    b.turn_message = turn_message
    out2 = [x async for x in b.ask("run it")]
    assert out2 == ["canary-ok"] and b._proc is proc and len(procs) == 1


async def test_limit_fixture_raises_limit_error(tmp_path):
    """antigravity-limit.jsonl is synthetic (a quota error cannot be captured
    on demand): the real `result` shape with status ERROR and a 429 message."""
    import pytest

    async def spawn(argv, cwd, env):
        return FakeProc(lines("antigravity-limit.jsonl"))

    b, _, _ = make(tmp_path, spawn=spawn)
    with pytest.raises(cli.LimitError):
        [x async for x in b.ask("hi")]
    assert b._proc is None


async def test_resume_argv_after_interrupt(tmp_path):
    spawned = []

    async def spawn(argv, cwd, env):
        spawned.append(argv)
        return FakeProc(lines("antigravity-plain.jsonl"), hang=True)

    b, _, _ = make(tmp_path, spawn=spawn, interrupt_drain_s=0.05)
    [x async for x in b.ask("one")]
    await b.interrupt()
    [x async for x in b.ask("two")]
    assert "--conversation" not in spawned[0]
    assert spawned[1][spawned[1].index("--conversation") + 1] == conv_id("antigravity-plain.jsonl")


def test_hook_entry_is_replaced_not_accumulated(tmp_path):
    """A different sys.executable (dev worktree vs app bundle, or an upgrade)
    must not leave a second entry of ours behind in the user's file."""
    b, _, _ = make(tmp_path)
    b.prepare_workspace("P", native=True)
    stale = json.loads(b.hooks_file.read_text())
    stale["hooks"]["PreToolUse"][0]["hooks"][0]["command"] = (
        "/old/python -m veronica.brain.hook antigravity --sock /old/sock --log /old/log")
    stale["hooks"]["PreToolUse"].insert(0, {"matcher": "run_command", "hooks": [
        {"type": "command", "command": "/usr/bin/true"}]})
    b.hooks_file.write_text(json.dumps(stale))
    b.prepare_workspace("P", native=True)
    pre = json.loads(b.hooks_file.read_text())["hooks"]["PreToolUse"]
    ours = [e for e in pre if any("veronica.brain.hook antigravity" in h["command"] for h in e["hooks"])]
    assert len(ours) == 1 and ours[0]["hooks"][0]["command"] == b.hook_command()
    assert len(pre) == 2 and pre[0]["hooks"][0]["command"] == "/usr/bin/true"


async def test_close_removes_our_hook_entry(tmp_path):
    b, _, _ = make(tmp_path)
    b.hooks_file.parent.mkdir(parents=True)
    b.hooks_file.write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "run_command", "hooks": [
        {"type": "command", "command": "/usr/bin/true"}]}]}, "other": 1}))
    b.prepare_workspace("P", native=True)
    await b.close()
    data = json.loads(b.hooks_file.read_text())
    assert data["other"] == 1
    assert [e["hooks"][0]["command"] for e in data["hooks"]["PreToolUse"]] == ["/usr/bin/true"]
    await b.close()                                # idempotent


def test_hook_command_quotes_paths(tmp_path):
    b, _, _ = make(tmp_path)
    b.workspace = tmp_path / "with space"
    b.hook_log = b.workspace / "hook.log"
    b.scope_file = b.workspace / "active-conversation"
    assert f"'{b.hook_log}'" in b.hook_command() and f"'{b.scope_file}'" in b.hook_command()
