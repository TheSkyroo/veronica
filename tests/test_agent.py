import datetime as dt

import pytest

from veronica.brain import agent as agent_mod
from veronica.brain.agent import Brain, summarize_tool
from veronica.brain.prompts import system_prompt
from veronica.config import Settings


def test_system_prompt_has_date_and_rules():
    p = system_prompt(dt.date(2026, 9, 15))
    assert "You are Veronica" in p
    assert "2026-09-15" in p
    assert "one to three spoken sentences" in p


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
    def __init__(self, sid): self.session_id = sid; self.result = None; self.terminal_reason = "success"


class FakeClient:
    instances = []

    def __init__(self, options=None):
        self.options = options
        self.queries = []
        self.script = [_Assistant("Hello there. How "), _Assistant("are you?"), _Result("sess-1")]
        FakeClient.instances.append(self)

    async def connect(self): pass
    async def disconnect(self): pass
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

    async def confirm(summary): return summary.startswith("Bash")

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


async def test_resume_from_saved_session(brain, tmp_home):
    (tmp_home / "session").write_text("old-sess")
    [s async for s in brain.ask("x")]
    assert FakeClient.instances[0].options.resume == "old-sess"


async def test_can_use_tool_gate(brain):
    [s async for s in brain.ask("x")]
    gate = FakeClient.instances[0].options.can_use_tool
    allow = await gate("Bash", {"command": "ls"}, None)
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
