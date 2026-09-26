import pytest
from mcp.types import CallToolRequestParams

from veronica.brain.base import Decision
from veronica.tools import serve


@pytest.fixture(autouse=True)
def _restore_handlers():
    # gated_server wraps the module-level servers in place; put them back so
    # one test's seam doesn't become the next test's "original".
    saved = [(serve.server_for(n), serve.server_for(n).get_request_handler("tools/call"))
             for n in ("mac", "screen")]
    yield
    for inst, entry in saved:
        inst.add_request_handler("tools/call", entry.params_type, entry.handler)


def test_server_lookup():
    assert serve.server_for("mac").name == "mac"
    with pytest.raises(KeyError):
        serve.server_for("nope")


def _call_handler(inst):
    # mcp 2.x: handlers are (ctx, params) -> CallToolResult, keyed by method.
    return inst.get_request_handler("tools/call").handler


def _fake_gate(monkeypatch, answers, asked=None):
    def fake(tool, input, **kw):
        (asked if asked is not None else []).append((tool, input, kw))
        return answers.pop(0)

    monkeypatch.setattr(serve, "call_gate", fake)
    monkeypatch.setenv("VERONICA_BRAIN", "codex")


async def test_call_is_denied_then_run_by_the_app(monkeypatch):
    asked = []
    _fake_gate(monkeypatch, [
        (Decision(False, "denied", "user declined"), [], True),
        (Decision(True, "approved", ""), [{"type": "text", "text": "copied"}], False),
    ], asked)
    inst = serve.gated_server("mac")
    params = CallToolRequestParams(name="clipboard_write", arguments={"text": "hi"})
    denied = await _call_handler(inst)(None, params)
    assert denied.is_error and "Not allowed: user declined" in denied.content[0].text
    assert asked[0][0] == "mcp__mac__clipboard_write" and asked[0][2] == {"origin": "mcp", "backend": "codex"}
    ok = await _call_handler(inst)(None, params)
    assert not ok.is_error and ok.content[0].text == "copied"


async def test_the_proxy_never_runs_the_tool_itself(monkeypatch):
    """Even an allow must not reach the local handler: this process has no
    business doing what the app holds the permission for."""
    ran = []
    inst = serve.server_for("mac")
    original = _call_handler(inst)

    async def spy(ctx, params):
        ran.append(params.name)
        return await original(ctx, params)

    inst.add_request_handler("tools/call", CallToolRequestParams, spy)
    _fake_gate(monkeypatch, [(Decision(True, "auto", ""), [{"type": "text", "text": "42"}], False)])
    res = await _call_handler(serve.gated_server("mac"))(
        None, CallToolRequestParams(name="volume_get", arguments={}))
    assert res.content[0].text == "42" and ran == []


async def test_image_content_survives_the_proxy(monkeypatch):
    blocks = [{"type": "image", "data": "QUJD", "mimeType": "image/png"},
              {"type": "text", "text": "Screenshot of the screen."}]
    _fake_gate(monkeypatch, [(Decision(True, "auto", ""), blocks, False)])
    res = await _call_handler(serve.gated_server("screen"))(
        None, CallToolRequestParams(name="screenshot", arguments={"region": "screen"}))
    assert not res.is_error
    assert res.content[0].type == "image" and res.content[0].data == "QUJD"
    assert res.content[0].mime_type == "image/png"
    assert res.content[1].text == "Screenshot of the screen."


async def test_an_unreadable_block_becomes_text(monkeypatch):
    _fake_gate(monkeypatch, [(Decision(True, "auto", ""), [{"type": "wat"}], False)])
    res = await _call_handler(serve.gated_server("mac"))(
        None, CallToolRequestParams(name="volume_get", arguments={}))
    assert "unreadable" in res.content[0].text


async def test_the_apps_error_flag_comes_through(monkeypatch):
    _fake_gate(monkeypatch, [(Decision(True, "auto", ""), [{"type": "text", "text": "error: nope"}], True)])
    res = await _call_handler(serve.gated_server("mac"))(
        None, CallToolRequestParams(name="volume_get", arguments={}))
    assert res.is_error and res.content[0].text == "error: nope"


async def test_list_tools_passthrough():
    inst = serve.gated_server("mac")
    res = await inst.get_request_handler("tools/list").handler(None, None)
    assert any(t.name == "clipboard_write" for t in res.tools)
