import asyncio
import json
import signal

import pytest

from tests.brains_fakes import FakeProc
from veronica.brain.backends import cli
from veronica.brain.gate import ToolGate
from veronica.config import Settings


class EchoBrain(cli.CliBrain):
    """A CliBrain whose parse() takes our own event JSON, to test the base alone."""
    name, label, binary = "echo", "Echo", "echo-cli"

    def argv(self, text, session_id, image_paths, native):
        return ["echo-cli", text] + (["--resume", session_id] if session_id else []) + [str(p) for p in image_paths]

    def env(self):
        return {"ECHO_EXTRA": "1"}

    def prepare_workspace(self, prompt_text, native):
        (self.workspace / "prepared").write_text(prompt_text)

    def parse(self, line):
        e = json.loads(line)
        t = e.pop("t")
        return [getattr(cli, t)(**e)]


class PersistentEchoBrain(EchoBrain):
    mode = "persistent"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.started = []

    def turn_message(self, text, image_paths):
        return json.dumps({"text": text, "images": [str(p) for p in image_paths]})

    def session_started(self, session_id):
        self.started.append(session_id)


def ev(t, **kw):
    return json.dumps({"t": t, **kw})


def build(tmp_path, lines, *, brain_cls=EchoBrain, **kw):
    s = Settings(home=tmp_path)
    spawned = []

    async def spawn(argv, cwd, env):
        spawned.append((argv, cwd, env))
        return FakeProc(lines, **kw)

    cards = []
    g = ToolGate(s, None, on_tool=lambda su, d: cards.append((su, d)))
    return brain_cls(s, g, on_tool=lambda su, d: cards.append((su, d)), spawn=spawn), spawned, cards


async def collect(brain, text, images=()):
    return [s async for s in brain.ask(text, images)]


async def test_sentences_stream_and_session_saved(tmp_path):
    b, spawned, _ = build(tmp_path, [
        ev("Session", id="abc"),
        ev("Text", delta="Hello there. How "),
        ev("Text", delta="are you?"),
        ev("Done"),
    ])
    assert await collect(b, "hi") == ["Hello there.", "How are you?"]
    assert b.s.session_file_for("echo").read_text() == "abc"
    argv, cwd, env = spawned[0]
    assert argv == ["echo-cli", "hi"] and cwd == str(b.workspace)
    assert env["VERONICA_GATE_SOCK"] == str(b.s.gate_socket) and env["VERONICA_BRAIN"] == "echo"
    assert env["VERONICA_HOOK_LOG"] == str(b.hook_log) and env["ECHO_EXTRA"] == "1"
    assert (b.workspace / "prepared").read_text().startswith("You are Veronica")
    assert b._proc is None
    # next turn resumes
    b2, spawned2, _ = build(tmp_path, [ev("Done", final_text="ok")])
    await collect(b2, "again")
    assert spawned2[0][0] == ["echo-cli", "again", "--resume", "abc"]


async def test_whole_reply_fallback_when_no_deltas(tmp_path):
    b, _, _ = build(tmp_path, [ev("Done", final_text="One. Two.")])
    assert await collect(b, "x") == ["One.", "Two."]


async def test_native_tool_card_only_for_readonly(tmp_path):
    b, _, cards = build(tmp_path, [
        ev("ToolStart", call_id="1", tool="read_file", input={"file_path": "/a"}, native=True),
        ev("ToolEnd", call_id="1"),
        ev("ToolStart", call_id="2", tool="mcp__veronica-mac__open_app", input={"name": "Safari"}, native=False),
        ev("ToolEnd", call_id="2"),
        ev("Done", final_text="done"),
    ])
    await collect(b, "x")
    assert cards == [("read_file /a", "auto")]     # the MCP call was carded by the gate in tools.serve


async def test_images_written_to_workspace(tmp_path):
    b, spawned, _ = build(tmp_path, [ev("Done", final_text="seen")])
    await collect(b, "look", images=[b"\x89PNG....", b"\xff\xd8\xff...."])
    argv = spawned[0][0]
    assert argv[-2].endswith("img-1.png") and argv[-1].endswith("img-2.jpg")
    assert (b.workspace / "img-1.png").read_bytes() == b"\x89PNG...."


async def test_error_result_spoken_and_overflow_resets_session(tmp_path):
    b, _, _ = build(tmp_path, [ev("Error", message="prompt is too long")])
    b.s.session_file_for("echo").write_text("old")
    assert await collect(b, "x") == ["My memory got full, starting a fresh conversation."]
    assert not b.s.session_file_for("echo").exists()
    b, _, _ = build(tmp_path, [ev("Error", message="boom")])
    assert await collect(b, "x") == ["Echo returned an error, check the log."]


async def test_nonzero_exit_without_result_is_an_error(tmp_path):
    b, _, _ = build(tmp_path, [ev("Text", delta="partial")], exit_code=2)
    assert await collect(b, "x") == ["Echo returned an error, check the log."]


async def test_limit_error_raises_limit_error(tmp_path):
    b, _, _ = build(tmp_path, [ev("Error", message="You have hit your usage limit")])
    with pytest.raises(cli.LimitError):
        await collect(b, "x")


async def test_timeout_kills_child(tmp_path):
    b, _, _ = build(tmp_path, [], hang=True)
    b.s.brain_timeout_s = 0.05
    assert await collect(b, "x") == ["Taking too long, cancelled."]
    assert b._proc is None


async def test_interrupt_sigint_then_kill(tmp_path):
    b, _, _ = build(tmp_path, [], hang=True)
    b.s.interrupt_drain_s = 0.05
    task = asyncio.create_task(collect(b, "x"))
    await asyncio.sleep(0.01)
    proc = b._proc
    await b.interrupt()
    assert proc.signals == [signal.SIGINT] and proc.killed and b._proc is None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_interrupt_when_idle_is_a_noop(tmp_path):
    b, _, _ = build(tmp_path, [])
    await b.interrupt()
    await b.close()


async def test_canary_kills_and_disables_native_tools(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(cli.prefs, "save_settings_override", lambda k, v: saved.update({k: v}))
    first = [
        ev("ToolStart", call_id="9", tool="run_command", input={"command": "ls"}, native=True),
        ev("ToolEnd", call_id="9"),
        ev("Done", final_text="listed"),
    ]
    second = [ev("Done", final_text="fallback answer")]
    s = Settings(home=tmp_path)
    object.__setattr__(s, "echo_native_tools", True)
    runs = []
    procs = []

    async def spawn(argv, cwd, env):
        runs.append(argv)
        procs.append(FakeProc(first if len(runs) == 1 else second))
        return procs[-1]

    b = EchoBrain(s, ToolGate(s, None), spawn=spawn)
    b.canary_grace_s = 0.1
    out = await collect(b, "list files")
    # no hook.log line for key "ls" -> canary trips
    assert out[0] == "Hooks aren't running on Echo, so I've turned off its shell. Tools still work."
    assert out[1:] == ["fallback answer"]
    assert saved == {"echo_native_tools": False} and len(runs) == 2
    assert procs[0].killed and s.echo_native_tools is False


async def test_canary_passes_when_hook_logged(tmp_path):
    lines = [
        ev("ToolStart", call_id="9", tool="run_command", input={"command": "ls"}, native=True),
        ev("ToolEnd", call_id="9"),
        ev("Done", final_text="listed"),
    ]
    runs = []

    async def spawn(argv, cwd, env):
        runs.append(argv)
        b.hook_log.write_text(json.dumps({"ts": 9e12, "call": "run_command", "key": "ls", "decision": "pending"}) + "\n")
        return FakeProc(lines)

    s = Settings(home=tmp_path)
    b = EchoBrain(s, ToolGate(s, None), spawn=spawn)
    assert await collect(b, "x") == ["listed"] and len(runs) == 1


async def test_canary_ignores_stale_log_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.prefs, "save_settings_override", lambda k, v: None)
    lines = [
        ev("ToolStart", call_id="9", tool="run_command", input={"command": "ls"}, native=True),
        ev("ToolEnd", call_id="9"),
        ev("Done", final_text="listed"),
    ]

    async def spawn(argv, cwd, env):
        # a line from long ago (the log is truncated per turn, but be strict about ts too)
        b.hook_log.write_text(json.dumps({"ts": 1.0, "call": "run_command", "key": "ls", "decision": "pending"}) + "\n")
        return FakeProc(lines)

    s = Settings(home=tmp_path)
    b = EchoBrain(s, ToolGate(s, None), spawn=spawn)
    b.canary_grace_s = 0.1
    out = await collect(b, "x")
    assert out[0].startswith("Hooks aren't running on Echo")


async def test_default_native_key_prefers_command_then_file(tmp_path):
    b, _, _ = build(tmp_path, [])
    assert b.native_key("run_command", {"command": "ls"}) == "ls"
    assert b.native_key("write_file", {"file_path": "/a", "content": "x"}) == "/a"
    assert b.native_key("weird", {"b": "second", "a": "first"}) == "first"
    assert b.native_key("weird", {}) == "weird"


# -- persistent mode ----------------------------------------------------------
async def test_persistent_two_asks_reuse_one_process(tmp_path):
    b, spawned, _ = build(tmp_path, [
        ev("Session", id="conv-1"),
        ev("Text", delta="First answer."),
        ev("Done", final_text="First answer."),
    ], brain_cls=PersistentEchoBrain)
    assert await collect(b, "one") == ["First answer."]
    proc = b._proc
    assert proc is not None and spawned[0][0] == ["echo-cli", ""]
    assert b.started == ["conv-1"] and b.s.session_file_for("echo").read_text() == "conv-1"
    assert json.loads(proc.stdin.lines[0]) == {"text": "one", "images": []}
    proc.feed([ev("Text", delta="Second answer."), ev("Done")])
    assert await collect(b, "two") == ["Second answer."]
    assert b._proc is proc and len(spawned) == 1
    assert json.loads(proc.stdin.lines[1]) == {"text": "two", "images": []}


async def test_persistent_kill_then_respawn_resumes(tmp_path):
    s = Settings(home=tmp_path, interrupt_drain_s=0.05)
    spawned, procs = [], []

    async def spawn(argv, cwd, env):
        spawned.append(argv)
        # an idle persistent child blocks on stdin (hang) and ignores SIGINT
        procs.append(FakeProc([ev("Session", id="conv-1"), ev("Done", final_text="ok" if len(spawned) == 1 else "back")],
                              hang=True))
        return procs[-1]

    b = PersistentEchoBrain(s, ToolGate(s, None), spawn=spawn)
    assert await collect(b, "one") == ["ok"]
    first = b._proc
    await b.interrupt()
    assert first.signals == [signal.SIGINT] and first.killed and b._proc is None
    assert await collect(b, "two") == ["back"]
    assert len(spawned) == 2 and spawned[1] == ["echo-cli", "", "--resume", "conv-1"]
    assert b.started == ["conv-1", "conv-1"]


async def test_persistent_failed_resume_starts_fresh(tmp_path):
    s = Settings(home=tmp_path)
    s.session_file_for("echo").write_text("stale")
    spawned = []

    async def spawn(argv, cwd, env):
        spawned.append(argv)
        if "--resume" in argv:
            return FakeProc([ev("Error", message="conversation not found")], exit_code=1)
        return FakeProc([ev("Session", id="fresh"), ev("Done", final_text="hi")])

    b = PersistentEchoBrain(s, ToolGate(s, None), spawn=spawn)
    assert await collect(b, "x") == ["hi"]
    assert spawned == [["echo-cli", "", "--resume", "stale"], ["echo-cli", ""]]
    assert s.session_file_for("echo").read_text() == "fresh"


async def test_persistent_respawns_when_native_flag_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.prefs, "save_settings_override", lambda k, v: None)
    s = Settings(home=tmp_path)
    object.__setattr__(s, "echo_native_tools", True)
    procs = []

    async def spawn(argv, cwd, env):
        procs.append(FakeProc([ev("Session", id="c"), ev("Done", final_text="ok")]))
        return procs[-1]

    b = PersistentEchoBrain(s, ToolGate(s, None), spawn=spawn)
    await collect(b, "one")
    object.__setattr__(s, "echo_native_tools", False)
    await collect(b, "two")
    assert len(procs) == 2 and procs[0].killed


async def test_persistent_error_kills_child(tmp_path):
    b, _, _ = build(tmp_path, [
        ev("Session", id="c"),
        ev("Error", message="boom"),
    ], brain_cls=PersistentEchoBrain)
    assert await collect(b, "x") == ["Echo returned an error, check the log."]
    assert b._proc is None
