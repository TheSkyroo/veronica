"""Screen awareness: capture what's on the user's screen and hand it to
Claude as an image, exposed as an in-process MCP tool. Read-only and local
(no network), so it's allow-class — see `veronica.brain.policy`.
"""
import asyncio
import base64
import datetime as dt
import subprocess
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica.config import settings

TIMEOUT_S = 15
DOWNSCALE_MAX_PX = 1568
REGIONS = ("screen", "window", "selection")


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _image_result(png_bytes: bytes, path: Path) -> dict:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "content": [
            {"type": "image", "data": b64, "mimeType": "image/png"},
            {"type": "text", "text": f"Screenshot saved to {path}"},
        ]
    }


def _guard(fn):
    """Wrap a handler so malformed args or unexpected failures return
    `_err(...)` instead of raising (same pattern as tools/mac.py)."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


def _screens_dir() -> Path:
    d = settings.home / "screens"
    d.mkdir(parents=True, exist_ok=True)
    return d


def front_window_id() -> int | None:
    """The window id (kCGWindowNumber) of the frontmost normal (layer 0)
    on-screen window, via Quartz's CGWindowListCopyWindowInfo. Returns None
    if Quartz isn't importable (e.g. non-macOS test environment) or no such
    window is found."""
    try:
        import Quartz
    except Exception:
        return None
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    info = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
    for w in info:
        if w.get("kCGWindowLayer", 0) == 0:
            wid = w.get("kCGWindowNumber")
            if wid is not None:
                return int(wid)
    return None


def _capture_argv(region: str, out_path: Path) -> list[str] | None:
    """Build the `screencapture` argv for `region`, or None if a window
    capture was requested but no front window id could be found."""
    argv = ["screencapture", "-x", "-t", "png"]
    if region == "window":
        wid = front_window_id()
        if wid is None:
            return None
        argv += ["-l", str(wid)]
    elif region == "selection":
        argv += ["-i"]
    argv.append(str(out_path))
    return argv


def capture_screenshot(region: str = "screen") -> tuple[bytes, Path] | str:
    """Take a screenshot via `screencapture`, downscale it via `sips`, and
    return (png_bytes, path) — or an error string on failure. Synchronous;
    run via asyncio.to_thread from the tool handler."""
    region = region if region in REGIONS else "screen"
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_path = _screens_dir() / f"{ts}.png"
    argv = _capture_argv(region, out_path)
    if argv is None:
        return "could not determine the front window"
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"timed out after {TIMEOUT_S}s"
    except Exception as exc:
        return str(exc)
    if done.returncode != 0:
        return done.stderr.strip() or f"exit {done.returncode}"
    if not out_path.exists():
        return "screencapture produced no file (selection cancelled?)"
    try:
        subprocess.run(
            ["sips", "--resampleHeightWidthMax", str(DOWNSCALE_MAX_PX), str(out_path)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except Exception:
        pass  # downscaling is best-effort; fall back to the original file
    try:
        data = out_path.read_bytes()
    except Exception as exc:
        return str(exc)
    return data, out_path


@tool(
    "screenshot",
    "Take a screenshot to see what's on the user's screen. region: "
    "'screen' (default, whole main display), 'window' (frontmost window "
    "only), or 'selection' (user drags to pick an area).",
    {"region": str},
)
@_guard
async def screenshot(args: dict) -> dict:
    region = str(args.get("region", "screen") or "screen")
    result = await asyncio.to_thread(capture_screenshot, region)
    if isinstance(result, str):
        return _err(result)
    data, path = result
    return _image_result(data, path)


TOOLS = [screenshot]
SCREEN_TOOL_NAMES = [t.name for t in TOOLS]
screen_server = create_sdk_mcp_server(name="screen", version="1.0.0", tools=TOOLS)
