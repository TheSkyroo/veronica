"""`LocalBrain` — hermetic: a fake `llama-server` process and a fake
streaming HTTP client. Nothing here loads a model; the `live` tests at the
bottom use the real binary and weights."""
import asyncio
import json
import time

import pytest

from mcp.types import CallToolResult, TextContent

from veronica.brain.backends import check_backend, local as local_mod
from veronica.brain.backends.local import START_FAILED, LocalBrain
from veronica.brain.gate import ToolGate
from veronica.config import Settings
from veronica.tools import registry


# -- fakes ---------------------------------------------------------------------
def sse(**delta) -> str:
    return "data: " + json.dumps({"choices": [{"delta": delta}]})


def tool_delta(index: int, name: str | None, args: str, call_id: str = "") -> str:
    fn = {"arguments": args}
    if name:
        fn["name"] = name
    call = {"index": index, "function": fn}
    if call_id:
        call["id"] = call_id
    return sse(tool_calls=[call])


class FakeStream:
    def __init__(self, lines, status=200):
        self.lines, self.status_code, self.closed = lines, status, False

    async def aiter_lines(self):
        for line in self.lines:
            await asyncio.sleep(0)
            if self.closed:
                return
            yield line

    async def aread(self):
        return b""

    async def aclose(self):
        self.closed = True


class _StreamCM:
    def __init__(self, stream):
        self.stream = stream

    async def __aenter__(self):
        return self.stream

    async def __aexit__(self, *exc):
        await self.stream.aclose()


class FakeClient:
    """`rounds` is one list of SSE lines per completion, in order.
    `health` is a list of booleans consumed by /health (last one repeats)."""

    def __init__(self, rounds=(), health=(True,)):
        self.rounds, self.health = list(rounds), list(health)
        self.posts, self.healths, self.streams = [], 0, []
        self.closed = False

    async def get(self, url, timeout=None):
        self.healths += 1
        ok = self.health.pop(0) if len(self.health) > 1 else self.health[0]
        return type("R", (), {"status_code": 200 if ok else 503})()

    def stream(self, method, url, json=None, timeout=None):
        self.posts.append(json)
        lines = self.rounds.pop(0) if self.rounds else ["data: [DONE]"]
        stream = FakeStream(lines)
        self.streams.append(stream)
        return _StreamCM(stream)

    async def aclose(self):
        self.closed = True


class FakeProc:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.waited = 0

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    async def wait(self):
        self.waited += 1
        return self.returncode


class FakeToolServer:
    """Stands in for an MCP server instance: records calls, replies text."""

    def __init__(self, reply="ok"):
        self.calls, self.reply = [], reply

    def get_request_handler(self, kind):
        async def handler(_ctx, params):
            self.calls.append((params.name, dict(params.arguments or {})))
            return CallToolResult(content=[TextContent(type="text", text=self.reply)])

        return type("H", (), {"handler": staticmethod(handler)})()


def make_brain(tmp_path, *, rounds=(), health=(True,), confirm=None, **kw):
    s = Settings(home=tmp_path, local_server_bin=tmp_path / "llama-server",
                 local_model=tmp_path / "m.gguf", **kw)
    async def yes(summary, detail):
        return True

    gate = ToolGate(s, confirm or yes)
    client = FakeClient(rounds, health)
    spawned = []

    async def spawn(argv):
        spawned.append(argv)
        return FakeProc()

    brain = LocalBrain(s, gate, spawn=spawn, client=client)
    return brain, client, spawned


def fake_catalog(monkeypatch, server, names=("mcp__mac__volume_get",)):
    schemas = [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
               for n in names]
    monkeypatch.setattr(local_mod, "_tools", (schemas, set(names)))
    # The tool runs through the shared registry, so that is where the
    # stand-in server has to be.
    monkeypatch.setattr(registry, "SERVERS",
                        {n.split("__")[1]: {"instance": server} for n in names})


async def drain(brain, text="hi"):
    return [s async for s in brain.ask(text)]


# -- start / health ------------------------------------------------------------
async def test_argv_carries_the_model_context_and_port(tmp_path):
    brain, *_ = make_brain(tmp_path, local_ctx=4096, local_port=9100)
    argv = brain.argv()
    assert argv[0] == str(tmp_path / "llama-server")
    assert argv[argv.index("--model") + 1] == str(tmp_path / "m.gguf")
    assert argv[argv.index("--ctx-size") + 1] == "4096"
    assert argv[argv.index("--port") + 1] == "9100"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert "--jinja" in argv and "--no-webui" in argv
    assert brain.base_url == "http://127.0.0.1:9100"


async def test_first_turn_spawns_and_waits_for_health(tmp_path, monkeypatch):
    monkeypatch.setattr(local_mod, "HEALTH_POLL_S", 0)
    brain, client, spawned = make_brain(
        tmp_path, rounds=[[sse(content="Hello.")]], health=[False, False, True])
    assert await drain(brain) == ["Hello."]
    assert len(spawned) == 1 and client.healths == 3
    await brain.close()


async def test_a_server_already_listening_is_adopted_not_spawned(tmp_path):
    brain, client, spawned = make_brain(tmp_path, rounds=[[sse(content="Hi.")]])
    assert await drain(brain) == ["Hi."]
    assert spawned == [] and brain._proc is None
    await brain.close()


async def test_the_server_stays_up_between_turns(tmp_path):
    brain, client, spawned = make_brain(
        tmp_path, rounds=[[sse(content="One.")], [sse(content="Two.")]], health=[False, True])
    await drain(brain)
    await drain(brain, "again")
    assert len(spawned) == 1
    await brain.close()


async def test_a_server_that_will_not_start_speaks_the_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(local_mod, "HEALTH_POLL_S", 0)
    brain, client, spawned = make_brain(tmp_path, health=[False])
    brain.start_timeout_s = 0
    assert await drain(brain) == [START_FAILED]
    await brain.close()


async def test_a_server_that_exits_speaks_the_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(local_mod, "HEALTH_POLL_S", 0)
    s = Settings(home=tmp_path)
    async def yes(summary, detail):
        return True

    gate = ToolGate(s, yes)

    async def spawn(argv):
        proc = FakeProc()
        proc.returncode = 1
        return proc

    brain = LocalBrain(s, gate, spawn=spawn, client=FakeClient(health=[False]))
    assert await drain(brain) == [START_FAILED]


# -- streaming -----------------------------------------------------------------
async def test_deltas_become_sentences(tmp_path):
    brain, *_ = make_brain(tmp_path, rounds=[[
        sse(content="The battery "), sse(content="is at sixty. "),
        sse(content="It is charging"), sse(content="."), "data: [DONE]",
    ]])
    assert await drain(brain) == ["The battery is at sixty.", "It is charging."]
    await brain.close()


async def test_reasoning_content_is_never_spoken(tmp_path):
    brain, *_ = make_brain(tmp_path, rounds=[[
        sse(reasoning_content="The user wants the battery. "),
        sse(reasoning_content="I should call a tool."),
        sse(content="Sixty percent."),
    ]])
    assert await drain(brain) == ["Sixty percent."]
    await brain.close()


async def test_a_tail_without_punctuation_is_flushed(tmp_path):
    brain, *_ = make_brain(tmp_path, rounds=[[sse(content="about sixty")]])
    assert await drain(brain) == ["about sixty"]
    await brain.close()


async def test_junk_lines_are_ignored(tmp_path):
    brain, *_ = make_brain(tmp_path, rounds=[[
        ": keep-alive", "", "data: not json", sse(content="Fine."),
    ]])
    assert await drain(brain) == ["Fine."]
    await brain.close()


async def test_an_http_error_says_so_once(tmp_path):
    brain, client, _ = make_brain(tmp_path)

    def stream(method, url, json=None, timeout=None):
        return _StreamCM(FakeStream([], status=500))

    client.stream = stream
    assert await drain(brain) == ["The local model returned an error, check the log."]
    await brain.close()


# -- history -------------------------------------------------------------------
async def test_history_carries_the_conversation(tmp_path):
    brain, client, _ = make_brain(
        tmp_path, rounds=[[sse(content="Hi.")], [sse(content="Still here.")]])
    await drain(brain, "hello")
    await drain(brain, "you there?")
    assert [m["role"] for m in brain._history] == ["user", "assistant", "user", "assistant"]
    # The second POST replays the first exchange, under one system prompt.
    sent = client.posts[1]["messages"]
    assert sent[0]["role"] == "system" and "Veronica" in sent[0]["content"]
    assert [m["content"] for m in sent[1:]] == ["hello", "Hi.", "you there?"]
    await brain.close()


async def test_history_is_trimmed_to_the_context_window(tmp_path):
    long = "x" * 400
    brain, *_ = make_brain(tmp_path, local_ctx=1024,
                           rounds=[[sse(content=long)] for _ in range(6)])
    for i in range(6):
        await drain(brain, f"{long} {i}")
    assert len(brain._history) < 12
    # Never cut mid-exchange: the window always starts on a user message.
    assert brain._history[0]["role"] == "user"
    await brain.close()


async def test_one_oversized_question_is_truncated_not_thrown_away(tmp_path):
    """The budget is small at the default context; a long paste must not
    empty the history and leave the POST with no user turn at all."""
    huge = "x" * 20000
    brain, client, _ = make_brain(tmp_path, local_ctx=1024, rounds=[[sse(content="Ok.")]])
    assert await drain(brain, huge) == ["Ok."]
    sent = client.posts[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    budget = int(1024 * local_mod.HISTORY_SHARE * local_mod.CHARS_PER_TOKEN)
    assert 0 < len(sent[1]["content"]) <= budget
    await brain.close()


async def test_the_system_prompt_says_it_is_offline(tmp_path):
    brain, client, _ = make_brain(tmp_path, rounds=[[sse(content="Ok.")]])
    await drain(brain)
    assert "offline on this Mac" in client.posts[0]["messages"][0]["content"]
    await brain.close()


# -- tools ---------------------------------------------------------------------
async def test_a_tool_call_goes_through_the_gate_and_runs(tmp_path, monkeypatch):
    server = FakeToolServer("Volume is 40.")
    fake_catalog(monkeypatch, server)
    brain, client, _ = make_brain(tmp_path, rounds=[
        [tool_delta(0, "mcp__mac__volume_get", "{}", "c1")],
        [sse(content="It's at forty.")],
    ])
    assert await drain(brain, "how loud is it") == ["It's at forty."]
    assert server.calls == [("volume_get", {})]
    # The tool result went back as a `tool` message keyed to the call.
    tool_msg = next(m for m in brain._history if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1" and tool_msg["content"] == "Volume is 40."
    assert client.posts[0]["tools"][0]["function"]["name"] == "mcp__mac__volume_get"
    await brain.close()


async def test_the_gate_can_refuse_and_the_model_is_told(tmp_path, monkeypatch):
    server = FakeToolServer()
    fake_catalog(monkeypatch, server, names=("mcp__mac__clipboard_write",))
    asked = []

    async def confirm(summary, detail):
        asked.append(summary)
        return False

    brain, *_ = make_brain(tmp_path, confirm=confirm, rounds=[
        [tool_delta(0, "mcp__mac__clipboard_write", '{"text": "hi"}', "c1")],
        [sse(content="Alright.")],
    ])
    assert await drain(brain, "copy hi") == ["Alright."]
    assert asked and server.calls == []       # asked, refused, never ran
    assert "Not allowed" in next(m for m in brain._history if m["role"] == "tool")["content"]
    await brain.close()


async def test_arguments_arrive_in_pieces(tmp_path, monkeypatch):
    server = FakeToolServer()
    fake_catalog(monkeypatch, server, names=("mcp__mac__open_app",))
    brain, *_ = make_brain(tmp_path, rounds=[
        [tool_delta(0, "mcp__mac__open_app", '{"name":', "c1"), tool_delta(0, None, ' "Safari"}')],
        [sse(content="Opened.")],
    ])
    await drain(brain, "open safari")
    assert server.calls == [("open_app", {"name": "Safari"})]
    await brain.close()


async def test_a_model_that_ignores_the_tools_just_answers(tmp_path, monkeypatch):
    fake_catalog(monkeypatch, FakeToolServer())
    brain, *_ = make_brain(tmp_path, rounds=[[sse(content="I can't check that offline.")]])
    assert await drain(brain) == ["I can't check that offline."]
    await brain.close()


async def test_an_unknown_tool_name_is_reported_not_raised(tmp_path, monkeypatch):
    fake_catalog(monkeypatch, FakeToolServer())
    brain, *_ = make_brain(tmp_path, rounds=[
        [tool_delta(0, "mcp__mac__teleport", "{}", "c1")],
        [sse(content="I can't do that.")],
    ])
    assert await drain(brain) == ["I can't do that."]
    assert "no tool called" in next(m for m in brain._history if m["role"] == "tool")["content"]
    await brain.close()


async def test_a_tool_loop_stops_after_a_few_rounds(tmp_path, monkeypatch):
    server = FakeToolServer()
    fake_catalog(monkeypatch, server)
    rounds = [[tool_delta(0, "mcp__mac__volume_get", "{}", f"c{i}")] for i in range(10)]
    brain, *_ = make_brain(tmp_path, rounds=rounds)
    assert await drain(brain) == []
    assert len(server.calls) == local_mod.MAX_TOOL_ROUNDS + 1
    await brain.close()


async def test_the_real_catalog_covers_our_mcp_servers():
    local_mod._tools = None
    schemas, index = await local_mod.tool_catalog()
    names = {s["function"]["name"] for s in schemas}
    assert {"mcp__mac__volume_get", "mcp__pim__mail_send", "mcp__browser__browser_read"} <= names
    assert all(n.startswith("mcp__") for n in index)
    assert schemas[0]["function"]["parameters"]["type"] == "object"


# -- interrupt / idle ----------------------------------------------------------
async def test_interrupt_closes_the_stream_and_keeps_what_was_said(tmp_path):
    brain, client, _ = make_brain(tmp_path, rounds=[
        [sse(content="One. "), sse(content="Two. "), sse(content="Three.")]])
    said = []
    gen = brain.ask("talk")
    async for sent in gen:
        said.append(sent)
        if len(said) == 1:
            await brain.interrupt()
    await gen.aclose()
    assert said == ["One."] and client.streams[0].closed
    assert brain._history[-1]["content"].startswith("One.")
    await brain.close()


async def test_interrupt_when_idle_is_a_no_op(tmp_path):
    brain, *_ = make_brain(tmp_path)
    await brain.interrupt()
    await brain.close()


async def test_the_server_is_stopped_after_an_idle_stretch(tmp_path):
    brain, client, spawned = make_brain(
        tmp_path, rounds=[[sse(content="Hi.")]], health=[False, True])
    brain.idle_shutdown_s = 0.02
    await drain(brain)
    proc = brain._proc
    assert proc is not None
    for _ in range(100):
        await asyncio.sleep(0.01)
        if proc.terminated:
            break
    assert proc.terminated and brain._proc is None
    await brain.close()


async def test_the_idle_watch_never_stops_a_server_a_turn_just_claimed(tmp_path):
    """The watcher and a starting turn can wake in the same tick. The turn
    takes the server first; the watcher has to notice and back off, or the
    model dies under the answer."""
    brain, client, _ = make_brain(
        tmp_path, rounds=[[sse(content="Hi.")]], health=[False, True])
    brain.idle_shutdown_s = 0.01
    await drain(brain)
    proc = brain._proc
    async with brain._server_lock:          # a turn is starting right now
        await asyncio.sleep(0.05)           # the idle deadline passes in here
        brain._touch()                      # ... and claims the server
    for _ in range(5):                      # let the watcher take the lock
        await asyncio.sleep(0)
    assert not proc.terminated and brain._proc is proc
    await brain.close()


async def test_a_dead_child_is_reaped_before_a_listening_server_is_adopted(tmp_path):
    brain, client, spawned = make_brain(
        tmp_path, rounds=[[sse(content="Hi.")], [sse(content="Again.")]],
        health=[False, True])
    await drain(brain)
    proc = brain._proc
    proc.returncode = 1                     # llama-server fell over on its own
    assert await drain(brain) == ["Again."]  # something is listening: adopt it
    assert brain._adopted and brain._proc is None and proc.waited == 1
    assert len(spawned) == 1
    await brain.close()
    assert brain._adopted is False


async def test_close_waits_for_the_idle_watch_to_stop(tmp_path):
    brain, *_ = make_brain(tmp_path, rounds=[[sse(content="Hi.")]], health=[False, True])
    await drain(brain)
    watch = brain._idle_task
    await brain.close()
    assert watch.done() and brain._idle_task is None


async def test_close_stops_the_server_and_the_client(tmp_path):
    brain, client, _ = make_brain(tmp_path, rounds=[[sse(content="Hi.")]], health=[False, True])
    await drain(brain)
    proc = brain._proc
    await brain.close()
    assert proc.terminated and brain._proc is None and client.closed


async def test_an_adopted_server_is_left_running(tmp_path):
    brain, client, spawned = make_brain(tmp_path, rounds=[[sse(content="Hi.")]])
    await drain(brain)
    await brain.close()
    assert spawned == []


# -- availability --------------------------------------------------------------
def test_available_when_both_files_are_there(tmp_path):
    s = Settings(home=tmp_path, local_server_bin=tmp_path / "llama-server",
                 local_model=tmp_path / "m.gguf")
    a = check_backend("local", settings=s)
    assert not a.ok and "server isn't there" in a.hint
    (tmp_path / "llama-server").write_text("")
    a = check_backend("local", settings=s)
    assert not a.ok and "model file isn't there" in a.hint
    (tmp_path / "m.gguf").write_text("")
    assert check_backend("local", settings=s).ok


def test_availability_never_looks_at_path_or_a_login(tmp_path):
    s = Settings(home=tmp_path, local_server_bin=tmp_path / "llama-server",
                 local_model=tmp_path / "m.gguf")
    (tmp_path / "llama-server").write_text("")
    (tmp_path / "m.gguf").write_text("")
    asked = []
    assert check_backend("local", which=lambda b: asked.append(b), settings=s).ok
    assert asked == []


# -- live ----------------------------------------------------------------------
@pytest.mark.live
async def test_live_plain_turn(capsys):
    """The real llama-server, the real weights, one plain sentence."""
    s = Settings()
    async def yes(summary, detail):
        return True

    gate = ToolGate(s, yes)
    brain = LocalBrain(s, gate)
    try:
        t0 = time.monotonic()
        said = [x async for x in brain.ask("Reply with exactly: pineapple.")]
        with capsys.disabled():
            print(f"\n[live plain] {time.monotonic() - t0:.1f}s -> {said!r}")
        assert said and "pineapple" in " ".join(said).lower()
    finally:
        await brain.close()


@pytest.mark.live
async def test_live_tool_turn(capsys):
    """A turn that should reach an `mcp__mac__*` tool. Every call still goes
    through the gate; the confirm here just answers yes."""
    s = Settings()
    decided = []
    async def yes(summary, detail):
        return True

    gate = ToolGate(s, yes, on_tool=lambda summary, d: decided.append((summary, d)))
    brain = LocalBrain(s, gate)
    try:
        for prompt in ("What's the battery at?", "What is the system volume set to?"):
            t0 = time.monotonic()
            said = [x async for x in brain.ask(prompt)]
            calls = [m for m in brain._history if m["role"] == "assistant" and m.get("tool_calls")]
            with capsys.disabled():
                print(f"\n[live tool] {prompt!r} {time.monotonic() - t0:.1f}s -> {said!r}")
                print(f"           tool calls: {[c['function']['name'] for m in calls for c in m['tool_calls']]}")
            assert said
    finally:
        await brain.close()


async def test_catch_all_tools_are_not_offered_to_the_local_model():
    """A 3B model treats applescript / screen control as an escape hatch when
    no tool fits, so the user gets a confirm out of nowhere. They're hidden."""
    import veronica.brain.backends.local as local_mod

    local_mod._tools = None
    try:
        schemas, index = await local_mod.tool_catalog()
    finally:
        local_mod._tools = None
    names = {s["function"]["name"] for s in schemas}
    assert "mcp__mac__applescript" not in names and "mcp__mac__applescript" not in index
    assert not any(n.startswith("mcp__computer__") for n in names)
    assert "mcp__mac__volume_get" in names        # ordinary tools still offered
