"""Screen awareness: capture what's on the user's screen and hand it to
Claude as an image, exposed as an in-process MCP tool. Read-only and local
(no network), so it's allow-class — see `veronica.brain.policy`.

Privacy: nothing accumulates on disk. Each capture overwrites the single
file ~/.veronica/screens/latest.png (directory 0700, file 0600), which is
kept only so the brain's text-only fallback (if sending the image block
fails) can point Claude's Read tool at it; a transient JPEG re-encode of
an oversized capture is deleted as soon as its bytes are read.
"""
import asyncio
import base64
import contextlib
import os
import subprocess
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica.config import settings

TIMEOUT_S = 15
SELECTION_TIMEOUT_S = 60   # the user has to drag out a region first
DOWNSCALE_MAX_PX = 1568
# PNGs bigger than this (busy 4K/5K screens) get re-encoded as JPEG q80
# so the image block stays well inside the API's per-image limit.
MAX_PNG_BYTES = 3 * 1024 * 1024
JPEG_QUALITY = 80
REGIONS = ("screen", "window", "selection")
LATEST_NAME = "latest.png"


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _image_result(image_bytes: bytes, mime: str, region: str) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "content": [
            {"type": "image", "data": b64, "mimeType": mime},
            {"type": "text", "text": f"Screenshot of the {region}."},
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
    with contextlib.suppress(OSError):
        os.chmod(d, 0o700)
    return d


def latest_screenshot_path() -> Path:
    """Where the most recent capture lives (overwritten every time)."""
    return settings.home / "screens" / LATEST_NAME


def _window_list() -> list[dict]:
    """On-screen windows, front to back, via Quartz — [] if Quartz isn't
    importable (e.g. non-macOS test environment)."""
    try:
        import Quartz
    except Exception:
        return []
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    return list(Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or [])


def _frontmost_pid() -> int | None:
    """PID of the frontmost application, or None if AppKit isn't
    available / nothing is frontmost."""
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else None
    except Exception:
        return None


MIN_WINDOW_PX = 50


def front_window_id() -> int | None:
    """The window id (kCGWindowNumber) of the frontmost app's frontmost
    *real* window: layer 0, visible (alpha > 0), bigger than a helper
    sliver (> MIN_WINDOW_PX on both sides), and — when the frontmost app's
    PID can be determined — owned by that app. Without those filters the
    first layer-0 entry is often an invisible alpha-0 helper window (menu
    bar extras, input-method panels, screen-recording overlays), whose
    capture is a blank image. Returns None if nothing qualifies."""
    pid = _frontmost_pid()
    for w in _window_list():
        if w.get("kCGWindowLayer", 0) != 0:
            continue
        if float(w.get("kCGWindowAlpha", 1) or 0) <= 0:
            continue
        if pid is not None and w.get("kCGWindowOwnerPID") != pid:
            continue
        bounds = w.get("kCGWindowBounds") or {}
        if float(bounds.get("Width", 0) or 0) <= MIN_WINDOW_PX or float(bounds.get("Height", 0) or 0) <= MIN_WINDOW_PX:
            continue
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


def _reencode_jpeg(png_path: Path) -> bytes | None:
    """Re-encode `png_path` as JPEG q80 via sips into a sibling temp file,
    return its bytes and delete it. None if anything fails (caller keeps
    the PNG)."""
    jpg_path = png_path.with_suffix(".jpg")
    try:
        done = subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(JPEG_QUALITY),
             str(png_path), "--out", str(jpg_path)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        if done.returncode != 0 or not jpg_path.exists():
            return None
        return jpg_path.read_bytes()
    except Exception:
        return None
    finally:
        with contextlib.suppress(OSError):
            jpg_path.unlink()


def capture_screenshot(region: str = "screen") -> tuple[bytes, Path, str] | str:
    """Take a screenshot via `screencapture` into the single latest.png
    (0600, overwritten each time), downscale it via `sips`, and return
    (image_bytes, path, mime) — mime is image/png, or image/jpeg if the
    PNG was over MAX_PNG_BYTES and got re-encoded — or an error string on
    failure. Synchronous; run via asyncio.to_thread from the tool handler."""
    region = region if region in REGIONS else "screen"
    out_path = _screens_dir() / LATEST_NAME
    with contextlib.suppress(OSError):
        out_path.unlink()   # never serve a stale capture if this one fails
    argv = _capture_argv(region, out_path)
    if argv is None:
        return "could not determine the front window"
    timeout = SELECTION_TIMEOUT_S if region == "selection" else TIMEOUT_S
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"timed out after {timeout}s"
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
    # After sips: it rewrites the file (fresh inode, default umask mode), so
    # a chmod before it would be undone.
    with contextlib.suppress(OSError):
        os.chmod(out_path, 0o600)
    try:
        data = out_path.read_bytes()
    except Exception as exc:
        return str(exc)
    mime = "image/png"
    if len(data) > MAX_PNG_BYTES:
        jpeg = _reencode_jpeg(out_path)
        if jpeg is not None:
            data, mime = jpeg, "image/jpeg"
    return data, out_path, mime


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
    data, _path, mime = result
    return _image_result(data, mime, region if region in REGIONS else "screen")


TOOLS = [screenshot]
SCREEN_TOOL_NAMES = [t.name for t in TOOLS]
screen_server = create_sdk_mcp_server(name="screen", version="1.0.0", tools=TOOLS)
