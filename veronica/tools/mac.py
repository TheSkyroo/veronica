"""Typed macOS actions exposed to Claude as in-process MCP tools."""
import subprocess

from claude_agent_sdk import create_sdk_mcp_server, tool

TIMEOUT_S = 10


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def run(argv: list[str], stdin: str | None = None, ok_text: str | None = None) -> dict:
    """Run argv (never a shell string) and map the result to MCP content.

    On success, `ok_text` (if given) is returned verbatim instead of stdout —
    used by tools where stdout is not meaningful output (e.g. `open`).
    """
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _err(f"timed out after {TIMEOUT_S}s")
    except Exception as exc:  # e.g. FileNotFoundError
        return _err(str(exc))
    if done.returncode != 0:
        return _err(done.stderr.strip() or f"exit {done.returncode}")
    if ok_text is not None:
        return _ok(ok_text)
    out = done.stdout.strip()
    return _ok(out or "ok")


def _q(s: str) -> str:
    """Quote for an AppleScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _guard(fn):
    """Wrap a handler so malformed args (missing keys, bad types) return
    `_err(...)` instead of raising — argument extraction happens before
    `run()`'s own error handling, so it needs its own net."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


@tool("open_app", "Open a macOS application by name, e.g. Safari", {"name": str})
@_guard
async def open_app(args: dict) -> dict:
    name = str(args["name"])
    if "/" in name or name.startswith(".") or name.startswith("-"):
        return _err("app name must be a bare application name")
    return run(["open", "-a", name], ok_text="ok")


@tool("open_url", "Open an http(s) URL in the default browser", {"url": str})
@_guard
async def open_url(args: dict) -> dict:
    url = str(args.get("url", ""))
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    return run(["open", url])


@tool("clipboard_read", "Read the current clipboard text", {})
@_guard
async def clipboard_read(args: dict) -> dict:
    return run(["pbpaste"])


@tool("clipboard_write", "Replace the clipboard with the given text", {"text": str})
@_guard
async def clipboard_write(args: dict) -> dict:
    return run(["pbcopy"], stdin=str(args.get("text", "")))


@tool("notify", "Show a macOS notification banner", {"title": str, "message": str})
@_guard
async def notify(args: dict) -> dict:
    script = f'display notification "{_q(str(args.get("message", "")))}" with title "{_q(str(args.get("title", "")))}"'
    return run(["osascript", "-e", script])


@tool("volume_get", "Get system output volume (0-100)", {})
@_guard
async def volume_get(args: dict) -> dict:
    return run(["osascript", "-e", "output volume of (get volume settings)"])


@tool("volume_set", "Set system output volume (0-100)", {"level": int})
@_guard
async def volume_set(args: dict) -> dict:
    try:
        level = int(float(args.get("level", 0)))
    except (TypeError, ValueError):
        return _err("level must be a number 0-100")
    level = max(0, min(100, level))
    return run(["osascript", "-e", f"set volume output volume {level}"])


@tool("applescript", "Run an AppleScript snippet (powerful; user must confirm)", {"script": str})
@_guard
async def applescript(args: dict) -> dict:
    return run(["osascript", "-e", str(args.get("script", ""))])


TOOLS = [open_app, open_url, clipboard_read, clipboard_write, notify, volume_get, volume_set, applescript]
MAC_TOOL_NAMES = [t.name for t in TOOLS]
mac_server = create_sdk_mcp_server(name="mac", version="1.0.0", tools=TOOLS)
