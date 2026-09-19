import pytest
from mcp.types import CallToolRequestParams

from veronica.brain.base import Decision
from veronica.tools import serve


@pytest.fixture(autouse=True)
def _restore_mac_handler():
    # gated_server wraps the module-level server in place; put it back so
    # one test's seam doesn't become the next test's "original".
    inst = serve.server_for("mac")
    entry = inst.get_request_handler("tools/call")
    yield
    inst.add_request_handler("tools/call", entry.params_type, entry.handler)


def test_server_lookup():
    assert serve.server_for("mac").name == "mac"
    with pytest.raises(KeyError):
        serve.server_for("nope")


def _call_handler(inst):
    # mcp 2.x: handlers are (ctx, params) -> CallToolResult, keyed by method.
    return inst.get_request_handler("tools/call").handler


async def test_gated_call_tool_denies_then_allows(monkeypatch):
    answers = [Decision(False, "denied", "user declined"), Decision(True, "approved")]
    asked = []

    def fake_ask(tool, input, **kw):
        asked.append((tool, input, kw))
        return answers.pop(0)

    monkeypatch.setattr(serve, "ask_gate", fake_ask)
    monkeypatch.setenv("VERONICA_BRAIN", "codex")
    calls = []

    async def fake_handler(name, args):
        calls.append((name, args))
        return [{"type": "text", "text": "copied"}]

    inst = serve.gated_server("mac", call_tool=fake_handler)
    params = CallToolRequestParams(name="clipboard_write", arguments={"text": "hi"})
    denied = await _call_handler(inst)(None, params)
    assert denied.is_error and "Not allowed: user declined" in denied.content[0].text
    assert asked[0][0] == "mcp__mac__clipboard_write" and asked[0][2] == {"origin": "mcp", "backend": "codex"}
    assert calls == []
    ok = await _call_handler(inst)(None, params)
    assert not ok.is_error and ok.content[0].text == "copied"
    assert calls == [("clipboard_write", {"text": "hi"})]


async def test_gated_call_tool_falls_through_to_real_handler(monkeypatch):
    monkeypatch.setattr(serve, "ask_gate", lambda *a, **k: Decision(True, "auto"))
    inst = serve.gated_server("mac")
    # unknown tool -> the SDK's own error result, proving the original handler ran
    res = await _call_handler(inst)(None, CallToolRequestParams(name="nope", arguments={}))
    assert res.is_error and "not found" in res.content[0].text


async def test_list_tools_passthrough():
    inst = serve.gated_server("mac")
    res = await inst.get_request_handler("tools/list").handler(None, None)
    assert any(t.name == "clipboard_write" for t in res.tools)
