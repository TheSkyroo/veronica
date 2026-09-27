"""The external-brain tool path end to end: a `tools.serve` proxy talking
to a real GateServer whose runner is the real registry. Nothing here fakes
the wire, so it is the test that the tool really runs in the app process
— which is what macOS's TCC grants, and the timer service, depend on."""
import asyncio
import base64
from pathlib import Path

import pytest
from claude_agent_sdk import create_sdk_mcp_server
from mcp.types import CallToolRequestParams

from veronica.brain.gate import GateServer, ToolGate
from veronica.config import Settings
from veronica.tools import (
    browser, computer, mac, memory_tools, music, pim, registry, screen, serve,
)
from veronica.tools.timers import TimerService


@pytest.fixture
def sock(tmp_path, monkeypatch):
    """AF_UNIX paths are capped at ~104 bytes and pytest's tmp_path on macOS
    is longer, so bind relative to it."""
    monkeypatch.chdir(tmp_path)
    return Path("gate.sock")


@pytest.fixture(autouse=True)
def app_side(monkeypatch):
    """Give the registry its own server objects. In the app the proxy lives
    in another process; in one process it and the registry would share the
    module-level instances, and the proxy would end up calling itself."""
    mods = {"mac": mac, "pim": pim, "memory": memory_tools, "screen": screen,
            "music": music, "browser": browser, "computer": computer}
    monkeypatch.setattr(registry, "SERVERS",
                        {n: create_sdk_mcp_server(n, tools=m.TOOLS) for n, m in mods.items()})


@pytest.fixture(autouse=True)
def _restore_handlers():
    # gated_server wraps the module-level servers in place; put them back.
    saved = [(serve.server_for(n), serve.server_for(n).get_request_handler("tools/call"))
             for n in ("mac", "pim", "screen")]
    yield
    for inst, entry in saved:
        inst.add_request_handler("tools/call", entry.params_type, entry.handler)


async def proxy_call(sock, monkeypatch, server, tool, args, *, answers=(True,)):
    """One `tools/call` through the real proxy -> socket -> gate -> registry."""
    monkeypatch.setenv("VERONICA_GATE_SOCK", str(sock))
    monkeypatch.setenv("VERONICA_BRAIN", "codex")
    pending = list(answers)

    async def confirm(summary, detail=""):
        return pending.pop(0)

    # nothing auto-allowed: clipboard_write is the confirm-class stand-in here
    srv = GateServer(ToolGate(Settings(auto_allow_tools=[]), confirm), sock, run_tool=registry.call_tool)
    await srv.start()
    try:
        inst = serve.gated_server(server)
        return await inst.get_request_handler("tools/call").handler(
            None, CallToolRequestParams(name=tool, arguments=args))
    finally:
        await srv.stop()


async def test_a_timer_set_through_an_external_brain_fires(sock, monkeypatch):
    """It used to be dropped: `tools.serve` bound its own TimerService in a
    child process where nobody could speak. The tool now runs in the app,
    so the app's service — and its announcement — is the one that gets it."""
    said = []

    async def announce(text):
        said.append(text)

    async def no_banner(args):
        return {"content": []}

    monkeypatch.setattr(mac.notify, "handler", no_banner)      # no real notification in a test
    monkeypatch.setattr(pim, "service", TimerService(on_fire=announce))
    res = await proxy_call(sock, monkeypatch, "pim", "timer_set", {"minutes": 0.0001, "label": "tea"})
    assert not res.is_error and res.content[0].text.startswith("Timer set for")
    for _ in range(100):
        if said:
            break
        await asyncio.sleep(0.01)
    assert said == ["Timer tea done"]


async def test_a_screenshot_comes_back_as_an_image_over_the_socket(sock, monkeypatch):
    png = b"\x89PNG not really"
    monkeypatch.setattr(screen, "capture_screenshot", lambda region: (png, None, "image/png"))
    monkeypatch.setattr(screen, "load_geometry", lambda: None)
    res = await proxy_call(sock, monkeypatch, "screen", "screenshot", {"region": "screen"}, answers=())
    assert not res.is_error
    assert res.content[0].type == "image"
    assert base64.b64decode(res.content[0].data) == png


async def test_a_denied_call_never_reaches_the_tool(sock, monkeypatch):
    ran = []
    monkeypatch.setattr(mac, "run", lambda *a, **k: ran.append(a))
    res = await proxy_call(sock, monkeypatch, "mac", "clipboard_write", {"text": "hi"}, answers=(False,))
    assert res.is_error and "Not allowed" in res.content[0].text and ran == []


async def test_the_proxy_fails_closed_when_the_app_is_gone(sock, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE_SOCK", str(sock))       # never bound
    ran = []
    monkeypatch.setattr(screen, "capture_screenshot", lambda region: ran.append(region) or "no")
    inst = serve.gated_server("screen")
    res = await inst.get_request_handler("tools/call").handler(
        None, CallToolRequestParams(name="screenshot", arguments={"region": "screen"}))
    assert res.is_error and "Not allowed" in res.content[0].text and ran == []
