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

Displays: `screencapture` grabs the main display unless told otherwise,
so a `screen` capture is pinned to a display with `-D <index>` — by
default the one the frontmost window sits on, which is the monitor the
user is working on. See `displays`, `pick_display`; the geometry's
`origin_*` is that display's origin in the global point space, so a
click mapped back out of the image lands on the right monitor.
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
# display="all" puts one image block per display in the same JSON line, so
# they share a budget well under that 1 MiB ceiling and each is downscaled
# harder; a display whose image doesn't fit is named in the text rather
# than dropped silently.
MAX_ALL_BYTES = 600 * 1024
DOWNSCALE_ALL_MAX_PX = 1024
MAX_DISPLAYS = 16
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


def _display_phrase(geometry: "Geometry | None", count: int, with_size: bool = False) -> str | None:
    """"display 2 of 2 (external)" for the display a capture came from, or
    None when there's only one display (or no display info) \u2014 then the
    wording stays as it always was, with no screen to disambiguate."""
    d = geometry.display if geometry is not None else None
    if not d or count <= 1:
        return None
    kind = "main" if d.get("main") else "external"
    if with_size and geometry is not None:
        kind += f", {geometry.width_pt:.0f}\u00d7{geometry.height_pt:.0f} pt"
    return f"display {d.get('index')} of {count} ({kind})"


def _image_result(
    image_bytes: bytes, mime: str, region: str,
    geometry: "Geometry | None" = None, display_count: int = 1,
) -> dict:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    coords = "Coordinates you pass to computer_* tools are in these image pixels."
    if geometry is None:
        text = f"Screenshot of the {region}."
    elif region == "screen" and (phrase := _display_phrase(geometry, display_count, with_size=True)):
        text = f"Screenshot of {phrase}: {geometry.image_w}\u00d7{geometry.image_h} px. {coords}"
    else:
        phrase = _display_phrase(geometry, display_count)
        where = f"the {region} on {phrase}" if phrase else f"the {region}"
        text = (
            f"Screenshot of {where}: {geometry.image_w}\u00d7{geometry.image_h} px "
            f"(screen {geometry.width_pt:.0f}\u00d7{geometry.height_pt:.0f} pt). {coords}"
        )
    return {
        "content": [
            {"type": "image", "data": b64, "mimeType": mime},
            {"type": "text", "text": text},
        ]
    }


def _all_result(
    shots: "list[tuple[Display | None, bytes, str]]",
    chosen: "Display | None",
    geometry: "Geometry | None",
) -> dict:
    """One result carrying every display's image. Images go in display
    order and share `MAX_ALL_BYTES`; the coordinates belong to `chosen` \u2014
    the display the sidecar (and latest.png) describes."""
    content: list[dict] = []
    total, dropped = 0, []
    for d, data, mime in shots:
        if content and total + len(data) > MAX_ALL_BYTES:
            dropped.append(d)
            continue
        total += len(data)
        content.append({
            "type": "image", "data": base64.b64encode(data).decode("ascii"), "mimeType": mime,
        })
    listing = ", ".join(
        f"display {d.index} ({d.label}, {d.w:.0f}\u00d7{d.h:.0f} pt)" if d else "the screen"
        for d, _data, _mime in shots
    )
    parts = [f"Screenshot of all {len(shots)} displays, in order: {listing}."]
    if dropped:
        names = ", ".join(f"display {d.index}" for d in dropped if d)
        parts.append(
            f"{names} didn't fit in one message \u2014 ask for it on its own with display=<number>."
        )
    if chosen is not None and geometry is not None:
        parts.append(
            f"Coordinates you pass to computer_* tools are pixels of the display "
            f"{chosen.index} ({chosen.label}) image."
        )
    content.append({"type": "text", "text": " ".join(parts)})
    return {"content": content}


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
    {id, app, title, x, y, w, h} or None; `display` the screen it came
    from as {id, index, main}, or None when that isn't known.

    `origin_*` is in the *global* point space, so on a second monitor it
    is that display's origin (e.g. (1470, 0)) and `to_screen` lands on
    the right screen."""
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
    display: dict | None = None

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


@dataclass
class Display:
    """One active display: its CoreGraphics id, its 1-based place in the
    active display list — which is what `screencapture -D <n>` counts —
    and its bounds in the global point space."""
    id: int
    index: int
    x: float
    y: float
    w: float
    h: float
    main: bool

    @property
    def label(self) -> str:
        return "main" if self.main else "external"

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def as_sidecar(self) -> dict:
        return {"id": self.id, "index": self.index, "main": self.main}


def _active_displays(quartz=None) -> list[Display]:
    """Every active display in `CGGetActiveDisplayList` order."""
    q = quartz if quartz is not None else _quartz()
    err, ids, _count = q.CGGetActiveDisplayList(MAX_DISPLAYS, None, None)
    if err:
        return []
    main = int(q.CGMainDisplayID())
    out = []
    for i, did in enumerate(ids or (), start=1):
        r = q.CGDisplayBounds(did)
        out.append(Display(
            id=int(did), index=i,
            x=float(r.origin.x), y=float(r.origin.y),
            w=float(r.size.width), h=float(r.size.height),
            main=int(did) == main,
        ))
    return out


def displays() -> list[Display]:
    """The active displays, or a single entry built from the main
    display's bounds when the list can't be read. [] when Quartz isn't
    reachable at all (non-macOS test environment) — callers then fall
    back to screencapture's own default, the main display."""
    try:
        found = _active_displays()
    except Exception as exc:
        log.warning("active display list unavailable: %s", exc)
        found = []
    if found:
        return found
    try:
        x, y, w, h = _display_bounds()
    except Exception:
        return []
    return [Display(id=0, index=1, x=x, y=y, w=w, h=h, main=True)]


def main_display(found: list[Display] | None = None) -> Display | None:
    found = displays() if found is None else found
    return next((d for d in found if d.main), found[0] if found else None)


def _front_window_display(found: list[Display]) -> Display | None:
    """The display holding the centre of the frontmost window — the screen
    the user is actually working on — or None if there's no front window
    (or its centre isn't on any display)."""
    try:
        wid = front_window_id()
        if wid is None:
            return None
        w = _window_bounds(wid)
        if not w or w["w"] <= 0 or w["h"] <= 0:
            return None
        cx, cy = w["x"] + w["w"] / 2, w["y"] + w["h"] / 2
        return next((d for d in found if d.contains(cx, cy)), None)
    except Exception as exc:
        log.warning("front window display unknown: %s", exc)
        return None


def pick_display(spec: str = "auto", found: list[Display] | None = None) -> Display | None:
    """Which display a region='screen' capture should grab: 'auto' (the
    one the frontmost window is on, else the main one), 'main', 'all'
    (same as auto — it picks the display the coordinates will belong to),
    or a 1-based display number. None when no display is known."""
    found = displays() if found is None else found
    if not found:
        return None
    spec = str(spec or "auto").strip().lower()
    if spec.isdigit():
        index = int(spec)
        chosen = next((d for d in found if d.index == index), None)
        if chosen is None:
            raise ValueError(f"there is no display {index} — this Mac has {len(found)}")
        return chosen
    if spec == "main":
        return main_display(found)
    if spec not in ("", "auto", "all"):
        raise ValueError(f"display must be 'auto', 'main', 'all' or a number, not {spec!r}")
    return _front_window_display(found) or main_display(found)


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
            display=dict(raw["display"]) if raw.get("display") else None,
        )
    except Exception:
        return None


def _build_geometry(
    region: str, png_path: Path, window_id: int | None, display: Display | None = None,
) -> Geometry | None:
    """Compute the sidecar for the capture that just landed at `png_path`.
    None (no sidecar) if the image size or the display bounds can't be
    determined — the screenshot itself is still fine to show — and always
    None for a `selection` capture: the dragged rectangle's screen origin
    isn't known, so no geometry is the honest answer (the computer tools
    then refuse with "take a screenshot first" instead of clicking the
    wrong place)."""
    if region == "selection":
        return None
    size = _png_size(png_path)
    if size is None:
        return None
    try:
        window = _window_bounds(window_id) if region == "window" and window_id is not None else None
        if window is not None and window["w"] > 0 and window["h"] > 0:
            ox, oy, w_pt, h_pt = window["x"], window["y"], window["w"], window["h"]
        elif display is not None:
            # The captured display's own origin in the global space, so a
            # click mapped back through `to_screen` lands on that monitor.
            ox, oy, w_pt, h_pt = display.x, display.y, display.w, display.h
            region, window = "screen", None
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
        display=display.as_sidecar() if display is not None else None,
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
# A window whose shorter side is under this is a strip, not somewhere the
# user works: Chrome publishes a 2560x115 untitled layer-0 window in front
# of the real one.
STRIP_PX = 200
# Among the app's real windows, the frontmost one at least this share of the
# largest one's area wins, so a size-alike window behind never beats the one
# in front, but a popup or panel never beats the main window.
MAIN_AREA_SHARE = 0.25


def front_window_id() -> int | None:
    """The window id (kCGWindowNumber) of the frontmost app's frontmost
    *real* window: layer 0, visible (alpha > 0), bigger than a helper
    sliver (> MIN_WINDOW_PX on both sides), and — when the frontmost app's
    PID can be determined — owned by that app. Without those filters the
    first layer-0 entry is often an invisible alpha-0 helper window (menu
    bar extras, input-method panels, screen-recording overlays), whose
    capture is a blank image.

    Of those, strips (shorter side under STRIP_PX) are passed over and the
    frontmost window with at least MAIN_AREA_SHARE of the largest one's
    area wins; if only strips are left, the first as before. Returns None
    if nothing qualifies."""
    pid = _frontmost_pid()
    found: list[tuple[int, float, float]] = []      # (id, w, h), z-order
    for w in _window_list():
        if w.get("kCGWindowLayer", 0) != 0:
            continue
        if float(w.get("kCGWindowAlpha", 1) or 0) <= 0:
            continue
        if pid is not None and w.get("kCGWindowOwnerPID") != pid:
            continue
        bounds = w.get("kCGWindowBounds") or {}
        ww, hh = float(bounds.get("Width", 0) or 0), float(bounds.get("Height", 0) or 0)
        if ww <= MIN_WINDOW_PX or hh <= MIN_WINDOW_PX:
            continue
        wid = w.get("kCGWindowNumber")
        if wid is not None:
            found.append((int(wid), ww, hh))
    if not found:
        return None
    real = [(wid, ww * hh) for wid, ww, hh in found if min(ww, hh) >= STRIP_PX]
    if not real:
        return found[0][0]
    largest = max(area for _, area in real)
    return next(wid for wid, area in real if area >= largest * MAIN_AREA_SHARE)


def _capture_argv(
    region: str, out_path: Path, window_id: int | None = None, display: Display | None = None,
) -> list[str] | None:
    """Build the `screencapture` argv for `region`, or None if a window
    capture was requested but no front window id could be found. Without
    `-D` screencapture grabs the main display only, which is how the
    external monitor used to be invisible."""
    argv = ["screencapture", "-x", "-t", "png"]
    if region == "screen" and display is not None:
        argv += ["-D", str(display.index)]
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


def _main_display_argv(out_path: Path, index: int = 1) -> list[str]:
    """Retry argv: the whole display at `index`, explicitly (-D n)."""
    return ["screencapture", "-x", "-t", "png", "-D", str(index), str(out_path)]


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


def _clear_latest(out_path: Path) -> None:
    """Never serve a stale capture (or its sidecar) if this one fails."""
    with contextlib.suppress(OSError):
        out_path.unlink()
    with contextlib.suppress(OSError):
        _geometry_path().unlink()


def _capture_one(
    region: str, display: Display | None, out_path: Path,
    max_px: int = DOWNSCALE_MAX_PX, max_bytes: int = MAX_PNG_BYTES,
) -> tuple[bytes, str, Geometry | None] | str:
    """One `screencapture` run into `out_path`, downscaled to `max_px` and
    re-encoded as JPEG if still over `max_bytes`. Returns
    (image_bytes, mime, geometry) or an error string."""
    window_id = front_window_id() if region == "window" else None
    argv = _capture_argv(region, out_path, window_id, display)
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
            # display (-D n) usually succeeds.
            retry = _main_display_argv(out_path, display.index if display is not None else 1)
            done = subprocess.run(retry, capture_output=True, text=True, timeout=timeout)
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
            ["sips", "--resampleHeightWidthMax", str(max_px), str(out_path)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except Exception:
        pass  # downscaling is best-effort; fall back to the original file
    try:
        geometry = _build_geometry(captured_region, out_path, window_id, display)
    except Exception as exc:
        log.warning("screenshot geometry unavailable: %s", exc)
        geometry = None
    # After sips: it rewrites the file (fresh inode, default umask mode), so
    # a chmod before it would be undone.
    with contextlib.suppress(OSError):
        os.chmod(out_path, 0o600)
    try:
        data = out_path.read_bytes()
    except Exception as exc:
        return str(exc)
    mime = "image/png"
    if len(data) > max_bytes:
        jpeg = _reencode_jpeg(out_path)
        if jpeg is not None and len(jpeg) > max_bytes:
            jpeg = _reencode_jpeg(out_path, quality=JPEG_QUALITY_LOW) or jpeg
        if jpeg is not None:
            data, mime = jpeg, "image/jpeg"
    return data, mime, geometry


def _store_geometry(geometry: Geometry | None) -> None:
    if geometry is None:
        return
    try:
        write_geometry(geometry)
    except Exception as exc:
        log.warning("screenshot geometry sidecar not written: %s", exc)


def capture_screenshot(region: str = "screen", display: str = "auto") -> tuple[bytes, Path, str] | str:
    """Take a screenshot via `screencapture` into the single latest.png
    (0600, overwritten each time), downscale it via `sips`, and return
    (image_bytes, path, mime) — mime is image/png, or image/jpeg if the
    PNG was over MAX_PNG_BYTES and got re-encoded — or an error string on
    failure. `display` picks the screen a region='screen' capture grabs
    ('auto', 'main', or a 1-based number — see `pick_display`); a window
    or selection capture spans whatever display it's on anyway.
    Synchronous; run via asyncio.to_thread from the tool handler."""
    region = region if region in REGIONS else "screen"
    out_path = _screens_dir() / LATEST_NAME
    _clear_latest(out_path)
    try:
        # A dragged selection has no display to pin: not even asking keeps
        # it geometry-free (see `_build_geometry`).
        target = pick_display(display) if region != "selection" else None
    except ValueError as exc:
        return str(exc)
    result = _capture_one(region, target, out_path)
    if isinstance(result, str):
        return result
    data, mime, geometry = result
    _store_geometry(geometry)
    return data, out_path, mime


def capture_all_displays() -> tuple[list[tuple[Display | None, bytes, str]], Display | None] | str:
    """Capture every active display. Returns the shots in display order
    plus the display latest.png and the sidecar describe — the one holding
    the frontmost window, captured last so it's the one left on disk and
    the one the computer_* tools act on."""
    found = displays()
    out_path = _screens_dir() / LATEST_NAME
    _clear_latest(out_path)
    chosen = pick_display("all", found)
    # Capture the chosen display last: each run overwrites latest.png, so
    # whatever went last is what the sidecar and OCR see.
    order: list[Display | None] = [d for d in found if d is not chosen]
    order.append(chosen)   # None when no display is known: one plain capture
    n = len(order)
    max_px = DOWNSCALE_ALL_MAX_PX if n > 1 else DOWNSCALE_MAX_PX
    max_bytes = min(MAX_PNG_BYTES, max(MAX_ALL_BYTES // n, 1))
    shots: list[tuple[Display | None, bytes, str]] = []
    geometry = None
    for d in order:
        result = _capture_one("screen", d, out_path, max_px=max_px, max_bytes=max_bytes)
        if isinstance(result, str):
            return result
        data, mime, geometry = result
        shots.append((d, data, mime))
    _store_geometry(geometry)
    shots.sort(key=lambda s: s[0].index if s[0] is not None else 1)
    return shots, chosen


@tool(
    "screenshot",
    "Take a screenshot to see what's on the user's screen. region: "
    "'screen' (default, a whole display), 'window' (frontmost window "
    "only), or 'selection' (user drags to pick an area). With more than "
    "one monitor, display picks which screen region='screen' grabs: "
    "'auto' (default, the one the frontmost window is on), 'main', 'all' "
    "(every display, one image each) or a 1-based display number.",
    # Spelled out as JSON Schema (not {"region": str, ...}) so `display`
    # stays optional — the dict form makes every key required.
    {
        "type": "object",
        "properties": {"region": {"type": "string"}, "display": {"type": "string"}},
        "required": ["region"],
    },
)
@_guard
async def screenshot(args: dict) -> dict:
    region = str(args.get("region", "screen") or "screen")
    display = str(args.get("display") or "auto").strip().lower()
    if display == "all" and region not in ("window", "selection"):
        result = await asyncio.to_thread(capture_all_displays)
        if isinstance(result, str):
            return _err(result)
        shots, chosen = result
        if len(shots) == 1:   # one monitor: "all" is just a screen capture
            _d, data, mime = shots[0]
            return _image_result(data, mime, "screen", load_geometry(), 1)
        return _all_result(shots, chosen, load_geometry())
    result = await asyncio.to_thread(capture_screenshot, region, display)
    if isinstance(result, str):
        return _err(result)
    data, _path, mime = result
    return _image_result(
        data, mime, region if region in REGIONS else "screen",
        load_geometry(), len(displays()),
    )


TOOLS = [screenshot]
SCREEN_TOOL_NAMES = [t.name for t in TOOLS]
screen_server = create_sdk_mcp_server(name="screen", version="1.0.0", tools=TOOLS)
