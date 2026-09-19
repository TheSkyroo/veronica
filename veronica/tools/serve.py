"""`python -m veronica.tools.serve <name>`: one of Veronica's MCP servers
over stdio for an external brain (Codex/Antigravity/Copilot/Qwen). Every
tools/call first asks the gate socket (VERONICA_GATE_SOCK) the same
question the in-process gate would ask: policy, trust window, voice
confirm. stdout is the protocol; log to stderr only."""
import asyncio
import importlib
import logging
import os
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, TextContent

from veronica.brain.gateclient import ask_gate

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


def server_for(name: str) -> Server:
    mod = importlib.import_module(SERVERS[name])       # KeyError for unknown names
    return getattr(mod, f"{name}_server")["instance"]


def gated_server(name: str, call_tool=None) -> Server:
    """Wrap `name`'s tools/call handler so every call goes through the gate
    first. `call_tool(tool, args) -> [content dict]` is a test seam that
    stands in for the real handler."""
    inst = server_for(name)
    original = inst.get_request_handler("tools/call").handler
    backend = os.environ.get("VERONICA_BRAIN", "external")

    async def gated(ctx, params: CallToolRequestParams) -> CallToolResult:
        tool, args = params.name, dict(params.arguments or {})
        d = await asyncio.to_thread(ask_gate, f"mcp__{name}__{tool}", args, origin="mcp", backend=backend)
        if not d.allow:
            return CallToolResult(content=[TextContent(type="text", text=f"Not allowed: {d.message}")], is_error=True)
        if call_tool is not None:
            content = await call_tool(tool, args)
            return CallToolResult(content=[TextContent(**c) for c in content])
        return await original(ctx, params)

    inst.add_request_handler("tools/call", CallToolRequestParams, gated)
    return inst


def _bind(name: str) -> None:
    """The module-level services `__main__` binds in the app process. In a
    stdio child there is nobody to speak, so timers set from an external
    brain are recorded but never announce."""
    if name == "memory":
        from veronica.config import Settings
        from veronica.memory.store import MemoryStore
        from veronica.tools import memory_tools

        memory_tools.bind(MemoryStore(Settings().memory_path))
    elif name == "pim":
        from veronica.tools import pim
        from veronica.tools.timers import TimerService

        async def _silent(_msg: str) -> None:
            log.warning("timer fired in a tools.serve child; nothing to announce it")

        pim.bind(TimerService(on_fire=_silent))
        log.warning("timers set through an external brain don't announce")


async def _main(name: str) -> None:
    _bind(name)
    inst = gated_server(name)
    async with stdio_server() as (read, write):
        await inst.run(read, write, inst.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    asyncio.run(_main(sys.argv[1]))
