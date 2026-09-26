"""Veronica's own MCP servers, and the one way to run a tool from them.

Everything that isn't the Claude SDK calls in here: `LocalBrain` directly,
and an external brain's `tools.serve` child indirectly, over the gate
socket. Running the handlers in the app process and nowhere else is what
keeps macOS's TCC grants pointed at Veronica.app — a capture (or a key
press, or an Apple Event) issued from a helper the CLI spawned is
attributed to that helper, which was never granted anything."""
import base64
import logging

from mcp.types import CallToolRequestParams

from veronica.tools.browser import browser_server
from veronica.tools.computer import computer_server
from veronica.tools.mac import mac_server
from veronica.tools.memory_tools import memory_server
from veronica.tools.music import music_server
from veronica.tools.pim import pim_server
from veronica.tools.screen import MAX_PNG_BYTES, screen_server

log = logging.getLogger(__name__)

SERVERS = {
    "mac": mac_server, "pim": pim_server, "memory": memory_server,
    "screen": screen_server, "music": music_server,
    "browser": browser_server, "computer": computer_server,
}

# A result travels to `tools.serve` as one JSON line, and a screenshot's
# base64 rides inside it. screen.py already re-encodes anything over
# MAX_PNG_BYTES, so a block past that ceiling means something went wrong
# upstream: say so rather than write a line the other end can't read.
MAX_IMAGE_BYTES = MAX_PNG_BYTES
TOO_LARGE = "error: the image was too large to return"


def split_name(full: str) -> tuple[str, str] | None:
    """`mcp__<server>__<tool>` -> (server, tool), or None if it isn't one."""
    if not full.startswith("mcp__"):
        return None
    server, sep, tool = full[len("mcp__"):].partition("__")
    return (server, tool) if sep and server and tool else None


def _block(content) -> dict:
    """One MCP content object as a JSON-safe dict."""
    d = content.model_dump(mode="json", by_alias=True, exclude_none=True)   # wire names: mimeType, not mime_type
    if d.get("type") == "image" and len(base64.b64decode(d.get("data") or "", validate=False)) > MAX_IMAGE_BYTES:
        log.warning("dropping an oversized %s image block", d.get("mimeType"))
        return {"type": "text", "text": TOO_LARGE}
    return d


async def call_tool(full: str, args: dict) -> tuple[list[dict], bool]:
    """Run one of our tools here, in the app process. Returns its content
    blocks as plain dicts plus the is_error flag. Never raises: an unknown
    name or a handler that blew up comes back as an error result, the same
    way the SDK reports one, so the brain can re-plan."""
    parts = split_name(full)
    if parts is None or parts[0] not in SERVERS:
        return [{"type": "text", "text": f"error: there is no tool called {full}"}], True
    server, bare = parts
    try:
        inst = SERVERS[server]["instance"]
        result = await inst.get_request_handler("tools/call").handler(
            None, CallToolRequestParams(name=bare, arguments=args))
    except Exception as exc:
        log.exception("tool %s failed", full)
        return [{"type": "text", "text": f"error: {exc}"}], True
    blocks = [_block(c) for c in (result.content or [])]
    return blocks, bool(result.is_error) or any(b.get("text") == TOO_LARGE for b in blocks)
