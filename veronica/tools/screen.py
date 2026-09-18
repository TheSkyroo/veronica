"""Screen awareness: capture what's on the user's screen and hand it to
Claude as an image, exposed as an in-process MCP tool. Read-only and local
(no network), so it's allow-class — see `veronica.brain.policy`.

Privacy: nothing accumulates on disk. Each capture overwrites the single
file ~/.veronica/screens/latest.png (directory 0700, file 0600), which is
kept only so the brain's text-only fallback (if sending the image block
fails) can point Claude's Read tool at it; a transient JPEG re-encode of
an oversized capture is deleted as soon as its bytes are read.

Geometry: next to latest.png lives latest.json (0600), describing the
captured area in screen points and the final PNG size, so the computer_*
tools can turn a pixel Claude points at in the image back into a screen
coordinate — see `Geometry`, `load_geometry`.
"""
import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import struct
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica.config import settings

TIMEOUT_S = 15
SELECTION_TIMEOUT_S = 60   # the user has to drag out a region first
DOWNSCALE_MAX_PX = 1568
# PNGs bigger than this get re-encoded as JPEG q80 (then q60 if still too
# big). The Agent SDK's stream-json reader caps one JSON line at 1 MiB and
# the base64 image travels inside it (~37% inflation plus the rest of the
# message), so the raw image must stay well under that: a 407 KB PNG
# already tripped "JSON message exceeded maximum buffer size of 1048576".
MAX_PNG_BYTES = 300 * 1024
JPEG_QUALITY = 80
JPEG_QUALITY_LOW = 60
REGIONS = ("screen", "window", "selection")
LATEST_NAME = "latest.png"
GEOMETRY_NAME = "latest.json"
GEOMETRY_PATH: Path = settings.home / "screens" / GEOMETRY_NAME
# A screenshot older than this is not a safe basis for clicking.
GEOMETRY_MAX_AGE_S = 120

log = logging.getLogger(__name__)
_now = time.time   # swapped in tests


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _image_result(image_bytes: bytes, mime: str, region: str, geometry: "Geometry | None" = None) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    if geometry is None:
        text = f"Screenshot of the {region}."
    else:
        text = (
            f"Screenshot of the {region}: {geometry.image_w}\u00d7{geometry.image_h} px "
            f"(screen {geometry.width_pt:.0f}\u00d7{geometry.height_pt:.0f} pt). "
            "Coordinates you pass to computer_* tools are in these image pixels."
        )
    return {
        "content": [
            {"type": "image", "data": b64, "mimeType": mime},
            {"type": "text", "text": text},
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


# ---- geometry sidecar -----------------------------------------------------

@dataclass
class Geometry:
    """What latest.png shows, in screen points, plus its pixel size.

    `origin_*`/`width_pt`/`height_pt` are the captured area on screen (the
    whole main display, or the front window's bounds for a window capture);
    `image_w/h` the final — downscaled — PNG; `scale = image_w / width_pt`
    (pixels per point). `window` is the captured window's
    {id, app, title, x, y, w, h} or None."""
    region: str
    image_w: int
    image_h: int
    origin_x: float
    origin_y: float
    width_pt: float
    height_pt: float
    scale: float
    captured_at: float
    window: dict | None

    def to_screen(self, x_img: float, y_img: float) -> tuple[float, float]:
        """Image pixel (top-left origin) → global screen point."""
        return (self.origin_x + x_img / self.scale, self.origin_y + y_img / self.scale)

    @property
    def age_s(self) -> float:
        return _now() - self.captured_at


def _quartz():
    import Quartz
    return Quartz


def _display_bounds(quartz=None) -> tuple[float, float, float, float]:
    """Main display bounds in points: (x, y, w, h)."""
    q = quartz if quartz is not None else _quartz()
    r = q.CGDisplayBounds(q.CGMainDisplayID())
    return (float(r.origin.x), float(r.origin.y), float(r.size.width), float(r.size.height))


def _window_bounds(window_id: int, quartz=None) -> dict | None:
    """{"id","app","title","x","y","w","h"} for `window_id` (points), or
    None if the window is gone."""
    q = quartz if quartz is not None else _quartz()
    info = q.CGWindowListCopyWindowInfo(q.kCGWindowListOptionIncludingWindow, window_id) or []
    for w in info:
        if w.get("kCGWindowNumber") != window_id:
            continue
        b = w.get("kCGWindowBounds") or {}
        return {
            "id": int(window_id),
            "app": str(w.get("kCGWindowOwnerName") or ""),
            "title": str(w.get("kCGWindowName") or ""),
            "x": float(b.get("X", 0) or 0), "y": float(b.get("Y", 0) or 0),
            "w": float(b.get("Width", 0) or 0), "h": float(b.get("Height", 0) or 0),
        }
    return None


_SIPS_DIM = re.compile(r"pixel(Width|Height):\s*(\d+)")


def _png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) of `path`: `sips -g` first, then the PNG IHDR
    header as a fallback. None if neither works."""
    try:
        done = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        dims = dict(_SIPS_DIM.findall(done.stdout or ""))
        if "Width" in dims and "Height" in dims:
            return int(dims["Width"]), int(dims["Height"])
    except Exception:
        pass
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
    except Exception:
        pass
    return None


def _geometry_path() -> Path:
    return _screens_dir() / GEOMETRY_NAME


def write_geometry(geometry: Geometry, path: Path | None = None) -> None:
    path = path or _geometry_path()
    path.write_text(json.dumps(asdict(geometry)))
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def load_geometry(path: Path | None = None) -> Geometry | None:
    """The sidecar for the latest capture, or None if missing/corrupt."""
    path = path or _geometry_path()
    try:
        raw = json.loads(path.read_text())
        return Geometry(
            region=str(raw["region"]), image_w=int(raw["image_w"]), image_h=int(raw["image_h"]),
            origin_x=float(raw["origin_x"]), origin_y=float(raw["origin_y"]),
            width_pt=float(raw["width_pt"]), height_pt=float(raw["height_pt"]),
            scale=float(raw["scale"]), captured_at=float(raw["captured_at"]),
            window=dict(raw["window"]) if raw.get("window") else None,
        )
    except Exception:
        return None


def _build_geometry(region: str, png_path: Path, window_id: int | None) -> Geometry | None:
    """Compute the sidecar for the capture that just landed at `png_path`.
    None (no sidecar) if the image size or the display bounds can't be
    determined — the screenshot itself is still fine to show."""
    size = _png_size(png_path)
    if size is None:
        return None
    try:
        window = _window_bounds(window_id) if region == "window" and window_id is not None else None
        if window is not None and window["w"] > 0 and window["h"] > 0:
            ox, oy, w_pt, h_pt = window["x"], window["y"], window["w"], window["h"]
        else:
            ox, oy, w_pt, h_pt = _display_bounds()
            region, window = "screen", None
    except Exception as exc:
        log.warning("screenshot geometry unavailable: %s", exc)
        return None
    if w_pt <= 0:
        return None
    return Geometry(
        region=region, image_w=size[0], image_h=size[1],
        origin_x=ox, origin_y=oy, width_pt=w_pt, height_pt=h_pt,
        scale=size[0] / w_pt, captured_at=_now(), window=window,
    )


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


def _capture_argv(region: str, out_path: Path, window_id: int | None = None) -> list[str] | None:
    """Build the `screencapture` argv for `region`, or None if a window
    capture was requested but no front window id could be found."""
    argv = ["screencapture", "-x", "-t", "png"]
    if region == "window":
        wid = window_id if window_id is not None else front_window_id()
        if wid is None:
            return None
        # -o: no drop shadow, so the image edges are the window bounds
        # and the geometry sidecar's scale is exact.
        argv += ["-l", str(wid), "-o"]
    elif region == "selection":
        argv += ["-i"]
    argv.append(str(out_path))
    return argv


# screencapture's stderr when its notion of the displays is stale (a
# monitor was just plugged/unplugged) — or when Screen Recording is denied.
DISPLAY_CHANGE_MARKER = "could not create image"
DISPLAY_CHANGE_ERROR = (
    "Couldn't capture the screen — the display setup just changed (or Screen Recording "
    "isn't granted to Veronica). Try again in a moment."
)


def _main_display_argv(out_path: Path) -> list[str]:
    """Retry argv: the whole main display, explicitly (-D 1)."""
    return ["screencapture", "-x", "-t", "png", "-D", "1", str(out_path)]


def _reencode_jpeg(png_path: Path, quality: int = JPEG_QUALITY) -> bytes | None:
    """Re-encode `png_path` as JPEG q80 via sips into a sibling temp file,
    return its bytes and delete it. None if anything fails (caller keeps
    the PNG)."""
    jpg_path = png_path.with_suffix(".jpg")
    try:
        done = subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(quality),
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
    with contextlib.suppress(OSError):
        _geometry_path().unlink()
    window_id = front_window_id() if region == "window" else None
    argv = _capture_argv(region, out_path, window_id)
    if argv is None:
        return "could not determine the front window"
    captured_region = region
    timeout = SELECTION_TIMEOUT_S if region == "selection" else TIMEOUT_S
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if done.returncode != 0 and DISPLAY_CHANGE_MARKER in (done.stderr or "").lower():
            # Right after a monitor is (un)plugged, the display list
            # screencapture consults can be stale and it fails with "could
            # not create image from display"; one retry pinned to the
            # main display (-D 1) usually succeeds.
            done = subprocess.run(_main_display_argv(out_path), capture_output=True, text=True, timeout=timeout)
            if done.returncode != 0:
                return DISPLAY_CHANGE_ERROR
            captured_region = "screen"   # the retry grabbed the whole display
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
    try:
        geometry = _build_geometry(captured_region, out_path, window_id)
        if geometry is not None:
            write_geometry(geometry)
    except Exception as exc:
        log.warning("screenshot geometry sidecar not written: %s", exc)
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
        if jpeg is not None and len(jpeg) > MAX_PNG_BYTES:
            jpeg = _reencode_jpeg(out_path, quality=JPEG_QUALITY_LOW) or jpeg
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
    return _image_result(data, mime, region if region in REGIONS else "screen", load_geometry())


TOOLS = [screenshot]
SCREEN_TOOL_NAMES = [t.name for t in TOOLS]
screen_server = create_sdk_mcp_server(name="screen", version="1.0.0", tools=TOOLS)
