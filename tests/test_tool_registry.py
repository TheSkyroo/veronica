"""The in-process tool registry: the one place a Veronica tool actually
runs outside the Claude SDK."""
import base64

import pytest
from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica.tools import registry, screen, serve


@pytest.fixture
def fake_server(monkeypatch):
    """A server named `fake` in the registry, with one echoing tool."""
    @tool("echo", "Echo the text back", {"text": str})
    async def echo(args: dict) -> dict:
        if args["text"] == "boom":
            raise RuntimeError("kaboom")
        return {"content": [{"type": "text", "text": args["text"]}], "is_error": args["text"] == "bad"}

    monkeypatch.setitem(registry.SERVERS, "fake", create_sdk_mcp_server("fake", tools=[echo]))


def test_split_name():
    assert registry.split_name("mcp__screen__screenshot") == ("screen", "screenshot")
    assert registry.split_name("Bash") is None
    assert registry.split_name("mcp__screen") is None


def test_registry_covers_every_served_server():
    # tools.serve lists the same servers in its child; the two must not drift.
    assert set(registry.SERVERS) == set(serve.SERVERS)


async def test_call_tool_returns_text_blocks(fake_server):
    assert await registry.call_tool("mcp__fake__echo", {"text": "hi"}) == (
        [{"type": "text", "text": "hi"}], False)


async def test_call_tool_keeps_the_tools_own_error_flag(fake_server):
    content, is_error = await registry.call_tool("mcp__fake__echo", {"text": "bad"})
    assert is_error and content == [{"type": "text", "text": "bad"}]


async def test_call_tool_handler_failure_is_an_error(fake_server):
    content, is_error = await registry.call_tool("mcp__fake__echo", {"text": "boom"})
    assert is_error and "kaboom" in content[0]["text"]


async def test_call_tool_unknown_name_is_an_error():
    content, is_error = await registry.call_tool("mcp__nope__nope", {})
    assert is_error and "no tool called mcp__nope__nope" in content[0]["text"]
    content, is_error = await registry.call_tool("Bash", {"command": "ls"})
    assert is_error and "no tool called Bash" in content[0]["text"]


async def test_call_tool_passes_image_blocks_through(monkeypatch):
    png = b"\x89PNG tiny"
    monkeypatch.setattr(screen, "capture_screenshot", lambda region, display="auto": (png, None, "image/png"))
    monkeypatch.setattr(screen, "load_geometry", lambda: None)
    content, is_error = await registry.call_tool("mcp__screen__screenshot", {"region": "screen"})
    assert not is_error
    assert content[0] == {"type": "image", "data": base64.b64encode(png).decode(), "mimeType": "image/png"}
    assert content[1]["type"] == "text"


async def test_oversized_image_is_replaced_not_sent(monkeypatch):
    # screen.py already caps a capture at MAX_PNG_BYTES; anything past that
    # ceiling would make a response line nobody can read, so it never goes
    # on the wire.
    huge = b"x" * (registry.MAX_IMAGE_BYTES + 1)
    monkeypatch.setattr(screen, "capture_screenshot", lambda region, display="auto": (huge, None, "image/png"))
    monkeypatch.setattr(screen, "load_geometry", lambda: None)
    content, is_error = await registry.call_tool("mcp__screen__screenshot", {"region": "screen"})
    assert is_error and content[0]["type"] == "text" and "too large" in content[0]["text"]
