"""`python -m veronica.tools.serve <name>`: one of Veronica's MCP servers
over stdio for an external brain (Codex/Antigravity/Copilot/Qwen).

This process is a proxy, not an implementation. It advertises the server's
tools (tools/list is pure, so it is answered from the same server objects
the app uses) and forwards every tools/call over the gate socket
(VERONICA_GATE_SOCK): the app asks the same question the in-process gate
would — policy, trust window, voice confirm — and, when it allows, runs the
tool ITSELF and sends the content back. Nothing here touches the screen,
the keyboard or Apple Events, because macOS attributes what this process
does to the CLI that spawned it, and that binary holds none of Veronica's
TCC grants. Running the tool in the app is also why a timer set from an
external brain announces like any other.

stdout is the protocol; log to stderr only."""
import asyncio
import importlib
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ContentBlock, TextContent
from pydantic import TypeAdapter

from veronica.brain.gateclient import call_gate

log = logging.getLogger("veronica.tools.serve")

SERVERS = {
    "mac": "veronica.tools.mac",
    "pim": "veronica.tools.pim",
    "memory": "veronica.tools.memory_tools",
    "screen": "veronica.tools.screen",
    "music": "veronica.tools.music",
    "browser": "veronica.tools.browser",
    "computer": "veronica.tools.computer",
}

_block = TypeAdapter(ContentBlock)


def server_for(name: str) -> Server:
    mod = importlib.import_module(SERVERS[name])       # KeyError for unknown names
    return getattr(mod, f"{name}_server")["instance"]


def _content(blocks: list[dict]) -> list[ContentBlock]:
    """The app's content dicts back as MCP blocks (text, and the image the
    screenshot tool returns). A block we can't parse is described rather
    than dropped, so the brain never gets a silently empty result."""
    out: list[ContentBlock] = []
    for b in blocks:
        try:
            out.append(_block.validate_python(b))
        except Exception:
            log.warning("unparseable content block %r", b.get("type"))
            out.append(TextContent(type="text", text=f"error: unreadable {b.get('type')} result"))
    return out


def gated_server(name: str) -> Server:
    """Wrap `name`'s tools/call handler so every call is gated and run by
    the app instead of here."""
    inst = server_for(name)
    backend = os.environ.get("VERONICA_BRAIN", "external")

    async def gated(ctx, params: CallToolRequestParams) -> CallToolResult:
        tool, args = params.name, dict(params.arguments or {})
        d, blocks, is_error = await asyncio.to_thread(
            call_gate, f"mcp__{name}__{tool}", args, origin="mcp", backend=backend)
        if not d.allow:
            return CallToolResult(content=[TextContent(type="text", text=f"Not allowed: {d.message}")], is_error=True)
        return CallToolResult(content=_content(blocks), is_error=is_error)

    inst.add_request_handler("tools/call", CallToolRequestParams, gated)
    return inst


async def _main(name: str) -> None:
    inst = gated_server(name)
    async with stdio_server() as (read, write):
        await inst.run(read, write, inst.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    asyncio.run(_main(sys.argv[1]))
