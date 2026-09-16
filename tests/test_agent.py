import asyncio
import contextlib
import datetime as dt

import pytest

from veronica.brain import agent as agent_mod
from veronica.brain.agent import Brain, summarize_detail, summarize_tool
from veronica.brain.prompts import system_prompt
from veronica.config import Settings


def test_system_prompt_has_date_and_rules():
    p = system_prompt(dt.date(2026, 9, 15))
    assert "You are Veronica" in p
    assert "2026-09-15" in p
    assert "one to three spoken sentences" in p
    assert "For information from the internet, use WebSearch or WebFetch rather than shell commands. Use shell commands only for actions on this Mac." in p


def test_summarize_tool():
    assert summarize_tool("Bash", {"command": "ls -la"}) == "Bash: ls -la"
    assert summarize_tool("Write", {"file_path": "/x/notes.txt"}) == "Write file /x/notes.txt"
    assert summarize_tool("Edit", {"file_path": "/x/a.py"}) == "Edit file /x/a.py"
    assert summarize_tool("WebSearch", {"query": "weather"}) == "WebSearch: weather"
    assert summarize_tool("Foo", {"a": 1}) == "Foo"


# ---- fake SDK client -------------------------------------------------------

class _Text:
    def __init__(self, text): self.text = text

class _Assistant:
    def __init__(self, *texts): self.content = [_Text(t) for t in texts]

class _Result:
    def __init__(self, sid):
        self.session_id = sid
        self.result = None
        self.terminal_reason = "success"
        self.is_error = False


class FakeClient:
    instances = []

    def __init__(self, options=None):
        self.options = options
        self.queries = []
        self.disconnect_calls = 0
        self.script = [_Assistant("Hello there. How "), _Assistant("are you?"), _Result("sess-1")]
        FakeClient.instances.append(self)

    async def connect(self): pass

    async def disconnect(self):
        self.disconnect_calls += 1

    async def query(self, prompt): self.queries.append(prompt)

    async def receive_response(self):
        for m in self.script:
            yield m


@pytest.fixture
def brain(tmp_home, monkeypatch):
    monkeypatch.setattr(agent_mod, "AssistantMessage", _Assistant)
    monkeypatch.setattr(agent_mod, "TextBlock", _Text)
    monkeypatch.setattr(agent_mod, "ResultMessage", _Result)
    monkeypatch.setattr(Brain, "_client_cls", FakeClient)
    FakeClient.instances.clear()

    async def confirm(summary, detail=""): return summary.startswith("Bash")

    return Brain(Settings(), confirm=confirm)


async def test_ask_yields_sentences_and_saves_session(brain, tmp_home):
    out = [s async for s in brain.ask("hi")]
    assert out == ["Hello there.", "How are you?"]
    assert (tmp_home / "session").read_text() == "sess-1"
    assert FakeClient.instances[0].queries == ["hi"]


async def test_options_wired(brain):
    [s async for s in brain.ask("x")]
    o = FakeClient.instances[0].options
    assert o.effort == "low"
    assert o.max_turns == 8
    assert o.permission_mode == "default"
    assert "You are Veronica" in o.system_prompt
    assert o.can_use_tool is not None
    assert o.setting_sources == []
    assert "mac" in o.mcp_servers
    assert not o.allowed_tools


async def test_resume_from_saved_session(brain, tmp_home):
    (tmp_home / "session").write_text("old-sess")
    [s async for s in brain.ask("x")]
    assert FakeClient.instances[0].options.resume == "old-sess"


async def test_can_use_tool_gate(brain):
    [s async for s in brain.ask("x")]
    gate = FakeClient.instances[0].options.can_use_tool
    allow = await gate("Bash", {"command": "rm x"}, None)
    deny = await gate("Write", {"file_path": "a"}, None)
    assert allow.behavior == "allow"
    assert deny.behavior == "deny" and deny.message == "user declined"


async def test_timeout_yields_message(brain, monkeypatch):
    import asyncio

    async def slow(self):
        await asyncio.sleep(10)
        yield _Result("s")

    monkeypatch.setattr(FakeClient, "receive_response", slow)
    brain.s = Settings(brain_timeout_s=0)
    out = [s async for s in brain.ask("x")]
    assert out == ["Taking too long, cancelled."]
    assert brain._client is None
    assert FakeClient.instances[0].disconnect_calls == 1


async def test_connect_failure_resets_client(brain, monkeypatch):
    class BadClient(FakeClient):
        async def connect(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(Brain, "_client_cls", BadClient)
    with pytest.raises(RuntimeError):
        [s async for s in brain.ask("x")]
    assert brain._client is None


async def test_stale_session_cleared_and_retried(brain, tmp_home, monkeypatch):
    (tmp_home / "session").write_text("old-sess")

    class FlakyClient(FakeClient):
        async def connect(self):
            if self.options.resume == "old-sess":
                raise RuntimeError("stale session rejected")

    monkeypatch.setattr(Brain, "_client_cls", FlakyClient)
    FakeClient.instances.clear()

    out = [s async for s in brain.ask("x")]

    assert out == ["Hello there.", "How are you?"]
    assert "old-sess" not in (tmp_home / "session").read_text()
    assert len(FlakyClient.instances) == 2


async def test_error_result_speaks_error(brain, tmp_home, monkeypatch):
    class ErrClient(FakeClient):
        def __init__(self, options=None):
            super().__init__(options)
            r = _Result("sess-err")
            r.is_error = True
            r.result = "rate limited"
            self.script = [r]

    monkeypatch.setattr(Brain, "_client_cls", ErrClient)
    out = [s async for s in brain.ask("x")]
    assert out == ["Claude returned an error, check the log."]
    assert brain._client is None
    assert not (tmp_home / "session").exists()


async def test_query_failure_closes_client_and_propagates(brain, monkeypatch):
    class QueryFailsClient(FakeClient):
        async def query(self, prompt):
            raise RuntimeError("dead")

    monkeypatch.setattr(Brain, "_client_cls", QueryFailsClient)
    with pytest.raises(RuntimeError):
        [s async for s in brain.ask("x")]
    assert brain._client is None


async def test_stream_exception_closes_client(brain, monkeypatch):
    async def boom(self):
        raise RuntimeError("stream broke")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(FakeClient, "receive_response", boom)
    with pytest.raises(RuntimeError):
        [s async for s in brain.ask("x")]
    assert brain._client is None


def test_summarize_tool_prefers_description():
    inp = {"command": "curl -s https://wttr.in", "description": "Fetch weather from wttr.in"}
    assert summarize_tool("Bash", inp) == "Fetch weather from wttr.in"
    assert summarize_detail("Bash", inp) == "Bash: curl -s https://wttr.in"


def test_summarize_tool_description_stripped_truncated_no_trailing_period():
    long_desc = "  " + ("a" * 90) + ".  "
    inp = {"command": "ls", "description": long_desc}
    result = summarize_tool("Bash", inp)
    assert len(result) <= 80
    assert not result.endswith(".")
    assert result == "a" * 80


def test_summarize_tool_ignores_blank_or_missing_description():
    assert summarize_tool("Bash", {"command": "ls -la", "description": ""}) == "Bash: ls -la"
    assert summarize_tool("Bash", {"command": "ls -la", "description": "   "}) == "Bash: ls -la"
    assert summarize_tool("Bash", {"command": "ls -la"}) == "Bash: ls -la"


def test_summarize_detail_always_raw():
    assert summarize_detail("Write", {"file_path": "/x/notes.txt", "description": "Save notes"}) == "Write file /x/notes.txt"
    assert summarize_detail("Foo", {"a": 1}) == "Foo"


def test_summarize_mac_tools():
    assert summarize_tool("mcp__mac__open_app", {"name": "Safari"}) == "Open Safari"
    assert summarize_tool("mcp__mac__open_url", {"url": "https://x.y"}) == "Open https://x.y"
    assert summarize_tool("mcp__mac__clipboard_write", {"text": "a" * 80}) == "Copy to clipboard: " + "a" * 60
    assert summarize_tool("mcp__mac__applescript", {"script": "tell app \"Music\" to play"}) == 'AppleScript: tell app "Music" to play'
    assert summarize_tool("mcp__mac__volume_get", {}) == "volume_get"


async def test_gate_auto_allows_safe_tools_without_confirm(brain):
    calls = []

    async def confirm(summary, detail=""):
        calls.append(summary)
        return False

    brain._confirm = confirm
    res = await brain._can_use_tool("Read", {"file_path": "/x"}, None)
    assert res.behavior == "allow" and calls == []
    res = await brain._can_use_tool("Bash", {"command": "ls"}, None)
    assert res.behavior == "allow" and calls == []
    res = await brain._can_use_tool("Bash", {"command": "rm x"}, None)
    assert res.behavior == "deny" and calls == ["Bash: rm x"]


async def test_interrupt_without_client_is_noop(brain):
    await brain.interrupt()  # must not raise


async def test_interrupt_after_completed_ask_is_noop(brain):
    [s async for s in brain.ask("x")]
    client = FakeClient.instances[0]
    client.interrupts = 0

    async def interrupt():
        client.interrupts += 1

    client.interrupt = interrupt

    await asyncio.wait_for(brain.interrupt(), 0.5)

    assert client.interrupts == 0     # no turn in flight: no control request, no drain
    assert brain._client is not None


async def _consume(agen):
    return [s async for s in agen]


async def _start_in_flight_ask(brain, monkeypatch):
    """Puts brain into _in_flight state by starting ask() against a client
    whose stream yields one message then blocks forever, and running that
    ask() as a background task. Returns (task, client)."""
    about_to_hang = asyncio.Event()

    class BlockingClient(FakeClient):
        async def receive_response(self):
            yield _Assistant("Hello there.")
            about_to_hang.set()
            await asyncio.Event().wait()   # never set: simulates a stalled turn

    monkeypatch.setattr(Brain, "_client_cls", BlockingClient)
    task = asyncio.create_task(_consume(brain.ask("x")))
    await about_to_hang.wait()   # ask() is past client.query() (sets _in_flight) and hung
    return task, FakeClient.instances[0]


async def test_interrupt_calls_client(brain, monkeypatch):
    task, client = await _start_in_flight_ask(brain, monkeypatch)
    client.interrupts = 0

    async def interrupt():
        client.interrupts += 1

    async def receive_response():
        yield _Result("s")   # drain sees the turn end immediately

    client.interrupt = interrupt
    client.receive_response = receive_response
    await brain.interrupt()
    assert client.interrupts == 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_interrupt_drains_leftover_stream(brain, monkeypatch):
    task, client = await _start_in_flight_ask(brain, monkeypatch)
    client.interrupts = 0
    client.drained = []

    async def interrupt():
        client.interrupts += 1

    async def receive_response():
        for m in [_Assistant("leftover 1"), _Assistant("leftover 2"), _Result("s")]:
            client.drained.append(m)
            yield m

    client.interrupt = interrupt
    client.receive_response = receive_response

    await brain.interrupt()

    assert client.interrupts == 1
    assert len(client.drained) == 3   # both leftover assistant messages + the ResultMessage
    assert brain._client is not None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_interrupt_after_consumer_cancelled_still_interrupts(brain, monkeypatch):
    """Mirrors the real barge path: the orchestrator cancels the turn (the
    task consuming ask()) BEFORE calling brain.interrupt(). Cancelling the
    consumer must not clear _in_flight — the SDK turn is still running from
    Claude's point of view until interrupt()+drain actually observes it end."""
    task, client = await _start_in_flight_ask(brain, monkeypatch)

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    client.interrupts = 0
    client.drained = []

    async def interrupt():
        client.interrupts += 1

    async def receive_response():
        for m in [_Assistant("leftover"), _Result("s")]:
            client.drained.append(m)
            yield m

    client.interrupt = interrupt
    client.receive_response = receive_response

    await brain.interrupt()

    assert client.interrupts == 1
    assert len(client.drained) == 2
    assert brain._in_flight is False


async def test_interrupt_drain_timeout_closes_client(brain, monkeypatch):
    task, client = await _start_in_flight_ask(brain, monkeypatch)
    client.interrupts = 0

    async def interrupt():
        client.interrupts += 1

    async def receive_response():
        await asyncio.sleep(10)
        yield _Result("s")   # pragma: no cover - unreachable, drain times out first

    client.interrupt = interrupt
    client.receive_response = receive_response
    brain.s = Settings(interrupt_drain_s=0)

    await brain.interrupt()

    assert brain._client is None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_interrupt_call_itself_timing_out_closes_client(brain, monkeypatch):
    """client.interrupt() (the control-request call, not the drain) can hang
    too — the SDK awaits an ack for up to 60s. That must also be bounded by
    interrupt_drain_s and close the client rather than hang."""
    task, client = await _start_in_flight_ask(brain, monkeypatch)

    async def slow_interrupt():
        await asyncio.sleep(10)

    client.interrupt = slow_interrupt
    brain.s = Settings(interrupt_drain_s=0)

    await asyncio.wait_for(brain.interrupt(), 1)

    assert brain._client is None

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_on_tool_auto_allow(brain):
    seen = []
    brain._on_tool = lambda s, d: seen.append((s, d))
    await brain._can_use_tool("Read", {"file_path": "/x"}, None)
    assert seen == [("Read: /x", "auto")]


async def test_on_tool_not_called_on_confirm_path(brain):
    seen = []
    brain._on_tool = lambda s, d: seen.append((s, d))
    await brain._can_use_tool("Write", {"file_path": "/a"}, None)
    assert seen == []
