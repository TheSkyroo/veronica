"""Screen awareness: capture what's on the user's screen and hand it to
Claude as an image, exposed as an in-process MCP tool. Read-only and local
(no network), so it's allow-class — see `veronica.brain.policy`.

Privacy: nothing accumulates on disk. Each capture overwrites the single
file ~/.veronica/screens/latest.png (directory 0700, file 0600 — on
Windows the user profile's ACLs are what keep it private, the mode bits
are best-effort), which is kept only so the brain's text-only fallback
(if sending the image block fails) and OCR can read it; a JPEG re-encode
of an oversized capture is made in memory and never written.

Capture: `mss` grabs the pixels of a monitor, of the foreground window's
rectangle, or of every monitor in turn; Pillow downscales and encodes.
A `selection` capture opens Windows' own snipping overlay
(ms-screenclip:) and picks the snip up from the clipboard. Windows needs
no permission for any of this.

Coordinates: the process is per-monitor DPI aware
(`computer_events.ensure_dpi_awareness`), so every rectangle here is in
physical pixels of the virtual screen — (0, 0) is the primary monitor's
top-left and a monitor left of or above it has negative coordinates.

Geometry: next to latest.png lives latest.json (0600), describing the
captured area in screen pixels and the final (possibly downscaled) image
size, so the computer_* tools can turn a pixel Claude points at in the
image back into a screen coordinate — see `Geometry`, `load_geometry`.

Displays: a `screen` capture grabs one monitor — by default the one the
foreground window sits on, which is the one the user is working on. See
`displays`, `pick_display`; the geometry's `origin_*` is that monitor's
origin in the virtual screen, so a click mapped back out of the image
lands on the right monitor.
"""
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica.config import settings
from veronica.tools import computer_events

SELECTION_TIMEOUT_S = 60   # the user has to drag out a region first
SELECTION_POLL_S = 0.25
DOWNSCALE_MAX_PX = 1568
# Images bigger than this get re-encoded as JPEG q80 (then q60 if still too
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
_now = time.time          # swapped in tests
_poll_sleep = time.sleep  # swapped in tests


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _display_phrase(geometry: "Geometry | None", count: int, with_size: bool = False) -> str | None:
    """"display 2 of 2 (secondary)" for the display a capture came from,
    or None when there's only one display (or no display info) — then
    the wording stays as it always was, with no screen to disambiguate."""
    d = geometry.display if geometry is not None else None
    if not d or count <= 1:
        return None
    kind = "primary" if d.get("main") else "secondary"
    if with_size and geometry is not None:
        kind += f", {geometry.screen_w:.0f}×{geometry.screen_h:.0f} px"
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
        text = f"Screenshot of {phrase}: {geometry.image_w}×{geometry.image_h} px. {coords}"
    else:
        phrase = _display_phrase(geometry, display_count)
        where = f"the {region} on {phrase}" if phrase else f"the {region}"
        text = (
            f"Screenshot of {where}: {geometry.image_w}×{geometry.image_h} px "
            f"(screen {geometry.screen_w:.0f}×{geometry.screen_h:.0f} px). {coords}"
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
    order and share `MAX_ALL_BYTES`; the coordinates belong to `chosen` —
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
        f"display {d.index} ({d.label}, {d.w:.0f}×{d.h:.0f} px)" if d else "the screen"
        for d, _data, _mime in shots
    )
    parts = [f"Screenshot of all {len(shots)} displays, in order: {listing}."]
    if dropped:
        names = ", ".join(f"display {d.index}" for d in dropped if d)
        parts.append(
            f"{names} didn't fit in one message — ask for it on its own with display=<number>."
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
    `_err(...)` instead of raising."""
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


# ---- framework seams --------------------------------------------------------

def _win32():
    """The Win32 backend shared with `computer_events` (tests swap it)."""
    return computer_events._win32()


def _mss():
    import mss
    return mss


def _pil():
    from PIL import Image
    return Image


# ---- geometry sidecar -----------------------------------------------------

@dataclass
class Geometry:
    """What latest.png shows, in screen pixels, plus the image's size.

    `origin_*`/`screen_w`/`screen_h` are the captured area on screen (a
    whole monitor, or the foreground window's bounds for a window
    capture) in physical virtual-screen pixels; `image_w/h` the final —
    possibly downscaled — image; `scale = image_w / screen_w` (image
    pixels per screen pixel, 1.0 unless downscaled). `window` is the
    captured window's {id, app, title, x, y, w, h} or None; `display` the
    monitor it came from as {id, index, main}, or None when not known.

    `origin_*` is in the *virtual screen*, so on a second monitor it is
    that monitor's origin (e.g. (1920, 0), or (-1920, 0) for one left of
    the primary) and `to_screen` lands on the right monitor."""
    region: str
    image_w: int
    image_h: int
    origin_x: float
    origin_y: float
    screen_w: float
    screen_h: float
    scale: float
    captured_at: float
    window: dict | None
    display: dict | None = None

    def to_screen(self, x_img: float, y_img: float) -> tuple[float, float]:
        """Image pixel (top-left origin) → virtual-screen pixel."""
        return (self.origin_x + x_img / self.scale, self.origin_y + y_img / self.scale)

    @property
    def age_s(self) -> float:
        return _now() - self.captured_at


@dataclass
class Display:
    """One monitor: its HMONITOR, its 1-based number (primary first, then
    the others left to right, top to bottom), and its bounds in the
    virtual screen (physical pixels; negative left of / above the
    primary)."""
    id: int
    index: int
    x: float
    y: float
    w: float
    h: float
    main: bool
    name: str = ""

    @property
    def label(self) -> str:
        return "primary" if self.main else "secondary"

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h

    def as_sidecar(self) -> dict:
        return {"id": self.id, "index": self.index, "main": self.main}


def _active_displays(win32=None) -> list[Display]:
    """Every monitor via EnumDisplayMonitors/GetMonitorInfoW, numbered
    primary first, then by position (x, then y)."""
    w = win32 if win32 is not None else _win32()
    mons = list(w.monitors() or ())[:MAX_DISPLAYS]
    mons.sort(key=lambda m: (not m["primary"], m["rect"][0], m["rect"][1]))
    out = []
    for i, m in enumerate(mons, start=1):
        left, top, right, bottom = m["rect"]
        out.append(Display(
            id=int(m.get("handle") or 0), index=i,
            x=float(left), y=float(top), w=float(right - left), h=float(bottom - top),
            main=bool(m["primary"]), name=str(m.get("name") or ""),
        ))
    if out and not any(d.main for d in out):
        out[0].main = True
    return out


def displays() -> list[Display]:
    """The monitors, or [] when Win32 isn't reachable (non-Windows test
    environment) — callers then fall back to mss' own primary monitor."""
    try:
        return _active_displays()
    except Exception as exc:
        log.warning("display list unavailable: %s", exc)
        return []


def main_display(found: list[Display] | None = None) -> Display | None:
    found = displays() if found is None else found
    return next((d for d in found if d.main), found[0] if found else None)


def _front_window_display(found: list[Display]) -> Display | None:
    """The display holding the centre of the foreground window — the
    screen the user is actually working on — or None if there's no front
    window (or its centre isn't on any display)."""
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
    one the foreground window is on, else the primary one), 'main' /
    'primary', 'all' (same as auto — it picks the display the coordinates
    will belong to), or a 1-based display number. None when no display
    is known."""
    found = displays() if found is None else found
    if not found:
        return None
    spec = str(spec or "auto").strip().lower()
    if spec.isdigit():
        index = int(spec)
        chosen = next((d for d in found if d.index == index), None)
        if chosen is None:
            raise ValueError(f"there is no display {index} — this PC has {len(found)}")
        return chosen
    if spec in ("main", "primary"):
        return main_display(found)
    if spec not in ("", "auto", "all"):
        raise ValueError(f"display must be 'auto', 'main', 'all' or a number, not {spec!r}")
    return _front_window_display(found) or main_display(found)


def _window_bounds(window_id: int, win32=None) -> dict | None:
    """{"id","app","title","x","y","w","h"} for the window `window_id`
    (an HWND), in screen pixels — DWM's visible frame, without the
    invisible resize border — or None if the window is gone."""
    w = win32 if win32 is not None else _win32()
    rect = w.window_rect(window_id)
    if not rect:
        return None
    left, top, right, bottom = rect
    try:
        app, _exe = computer_events.app_for_pid(w.window_pid(window_id), win32=w)
    except Exception:
        app = ""
    return {
        "id": int(window_id), "app": str(app or ""), "title": str(w.window_title(window_id) or ""),
        "x": float(left), "y": float(top), "w": float(right - left), "h": float(bottom - top),
    }


def _png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) of the PNG at `path`, read from its IHDR header.
    None if it isn't a readable PNG."""
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
    path.write_text(json.dumps(asdict(geometry)), encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def load_geometry(path: Path | None = None) -> Geometry | None:
    """The sidecar for the latest capture, or None if missing/corrupt."""
    path = path or _geometry_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Geometry(
            region=str(raw["region"]), image_w=int(raw["image_w"]), image_h=int(raw["image_h"]),
            origin_x=float(raw["origin_x"]), origin_y=float(raw["origin_y"]),
            screen_w=float(raw["screen_w"]), screen_h=float(raw["screen_h"]),
            scale=float(raw["scale"]), captured_at=float(raw["captured_at"]),
            window=dict(raw["window"]) if raw.get("window") else None,
            display=dict(raw["display"]) if raw.get("display") else None,
        )
    except Exception:
        return None


def _build_geometry(
    region: str, image_size: tuple[int, int] | None,
    rect: tuple[float, float, float, float] | None,
    window: dict | None = None, display: Display | None = None,
) -> Geometry | None:
    """The sidecar for a capture of `rect` = (x, y, w, h) on screen that
    ended up `image_size` pixels. None (no sidecar) when either is unknown
    — the screenshot itself is still fine to show — and always None for a
    `selection` capture: the snipped rectangle's screen origin isn't
    known, so no geometry is the honest answer (the computer tools then
    refuse with "take a screenshot first" instead of clicking the wrong
    place)."""
    if region == "selection" or image_size is None or rect is None:
        return None
    ox, oy, w, h = rect
    if w <= 0 or h <= 0:
        return None
    return Geometry(
        region=region, image_w=int(image_size[0]), image_h=int(image_size[1]),
        origin_x=float(ox), origin_y=float(oy), screen_w=float(w), screen_h=float(h),
        scale=image_size[0] / w, captured_at=_now(), window=window,
        display=display.as_sidecar() if display is not None else None,
    )


MIN_WINDOW_PX = 50
# The desktop and the taskbar can be the foreground window; neither is a
# window the user works in.
SHELL_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})


def _real_window(w, hwnd: int) -> bool:
    """Visible (not minimised/cloaked), not the shell, and bigger than a
    helper sliver on both sides."""
    if not hwnd or not w.window_visible(hwnd) or w.window_class(hwnd) in SHELL_CLASSES:
        return False
    rect = w.window_rect(hwnd)
    if not rect:
        return False
    left, top, right, bottom = rect
    return right - left > MIN_WINDOW_PX and bottom - top > MIN_WINDOW_PX


def front_window_id() -> int | None:
    """The HWND of the foreground window when it's a real window (see
    `_real_window`); when the foreground is a tiny helper or a hidden
    owner window instead, the frontmost real top-level window of the same
    process (EnumWindows z-order). None for the desktop/taskbar, when
    nothing qualifies, or when Win32 isn't reachable."""
    try:
        w = _win32()
        hwnd = w.foreground_window()
        if not hwnd or w.window_class(hwnd) in SHELL_CLASSES:
            return None
        if _real_window(w, hwnd):
            return int(hwnd)
        pid = w.window_pid(hwnd)
        for other in w.top_level_windows():
            if other != hwnd and w.window_pid(other) == pid and _real_window(w, other):
                return int(other)
    except Exception as exc:
        log.debug("front window unknown: %s", exc)
    return None


DISPLAY_CHANGE_ERROR = (
    "Couldn't capture the screen — the display setup may have just changed. Try again in a moment."
)


def _grab(left: int, top: int, width: int, height: int):
    """The screen pixels of that virtual-screen rectangle as a Pillow RGB
    image, via mss. A fresh mss instance per call: its device contexts
    belong to the thread that made them, and captures run in a worker
    thread."""
    mss, Image = _mss(), _pil()
    with mss.mss() as sct:
        shot = sct.grab({"left": int(left), "top": int(top), "width": int(width), "height": int(height)})
        return Image.frombytes("RGB", shot.size, shot.rgb)


def _primary_rect() -> tuple[int, int, int, int]:
    """mss' idea of the primary monitor, when Win32 gave no display list."""
    with _mss().mss() as sct:
        m = sct.monitors[1]
        return (m["left"], m["top"], m["width"], m["height"])


def _virtual_rect() -> tuple[int, int, int, int] | None:
    try:
        return tuple(int(v) for v in _win32().virtual_screen())
    except Exception:
        return None


def _clip(rect, bounds) -> tuple[int, int, int, int] | None:
    """`rect` ∩ `bounds` (both (x, y, w, h)), or None if they don't meet."""
    if bounds is None:
        return rect
    x1, y1 = max(rect[0], bounds[0]), max(rect[1], bounds[1])
    x2 = min(rect[0] + rect[2], bounds[0] + bounds[2])
    y2 = min(rect[1] + rect[3], bounds[1] + bounds[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


# ---- selection (Windows' snipping overlay) ---------------------------------

def _start_snip() -> None:
    os.startfile("ms-screenclip:")   # type: ignore[attr-defined]  (Windows only)


def _clipboard_image():
    from PIL import Image, ImageGrab
    img = ImageGrab.grabclipboard()
    return img if isinstance(img, Image.Image) else None


def _grab_selection(timeout: float = SELECTION_TIMEOUT_S):
    """Open the snipping overlay and wait for the user's snip to land on
    the clipboard (a new clipboard sequence number holding an image).
    Returns the image, or an error string on timeout / cancel. The snip
    stays on the clipboard afterwards — that's where Windows puts it."""
    try:
        before = _win32().clipboard_sequence()
        _start_snip()
    except Exception as exc:
        return f"couldn't open the snipping overlay: {exc}"
    deadline = _now() + timeout
    while _now() < deadline:
        _poll_sleep(SELECTION_POLL_S)
        try:
            if _win32().clipboard_sequence() == before:
                continue
            img = _clipboard_image()
        except Exception as exc:
            log.debug("clipboard read failed: %s", exc)
            continue
        if img is not None:
            return img.convert("RGB")
    return "no area was selected (the snip was cancelled or timed out)"


# ---- encode -----------------------------------------------------------------

def _downscale(img, max_px: int):
    """`img` shrunk (never enlarged) so its longer side is at most `max_px`."""
    if max(img.size) <= max_px:
        return img
    img = img.copy()
    img.thumbnail((max_px, max_px), _pil().LANCZOS)
    return img


def _encode(img, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _reencode_jpeg(img, quality: int = JPEG_QUALITY) -> bytes | None:
    """`img` as JPEG bytes, in memory. None if encoding fails (the caller
    keeps the PNG)."""
    try:
        return _encode(img.convert("RGB"), "JPEG", quality=quality)
    except Exception:
        return None


def _write_private(path: Path, data: bytes) -> None:
    """Write `data` to `path`, created 0600 (not umask-wide, then chmod)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)


def _clear_latest(out_path: Path) -> None:
    """Never serve a stale capture (or its sidecar) if this one fails."""
    with contextlib.suppress(OSError):
        out_path.unlink()
    with contextlib.suppress(OSError):
        _geometry_path().unlink()


def _grab_rect(rect: tuple[int, int, int, int]):
    """`_grab` with one retry: right after a monitor is (un)plugged the
    first grab can fail on stale device contexts."""
    try:
        return _grab(*rect)
    except Exception as exc:
        log.info("screen grab failed, retrying once: %s", exc)
    try:
        return _grab(*rect)
    except Exception as exc:
        log.warning("screen grab failed: %s", exc)
        return None


def _capture_one(
    region: str, display: Display | None, out_path: Path,
    max_px: int | None = None, max_bytes: int | None = None,
) -> tuple[bytes, str, Geometry | None] | str:
    """One capture into `out_path`, downscaled to `max_px` (default
    DOWNSCALE_MAX_PX) and re-encoded as JPEG if still over `max_bytes`
    (default MAX_PNG_BYTES). Returns (image_bytes, mime, geometry) or an
    error string."""
    max_px = DOWNSCALE_MAX_PX if max_px is None else max_px
    max_bytes = MAX_PNG_BYTES if max_bytes is None else max_bytes
    window = None
    rect = None
    if region == "selection":
        img = _grab_selection()
        if isinstance(img, str):
            return img
    else:
        if region == "window":
            wid = front_window_id()
            window = _window_bounds(wid) if wid is not None else None
            if window is None or window["w"] <= 0 or window["h"] <= 0:
                return "could not determine the front window"
            rect = _clip((int(window["x"]), int(window["y"]), int(window["w"]), int(window["h"])),
                         _virtual_rect())
            if rect is None:
                return "the front window is off screen"
        elif display is not None:
            rect = (int(display.x), int(display.y), int(display.w), int(display.h))
        else:
            try:
                rect = _primary_rect()
            except Exception as exc:
                return f"couldn't find a screen to capture: {exc}"
        img = _grab_rect(rect)
        if img is None:
            return DISPLAY_CHANGE_ERROR
    img = _downscale(img, max_px)
    try:
        data = _encode(img, "PNG")
        _write_private(out_path, data)
    except Exception as exc:
        return str(exc)
    geometry = _build_geometry(region, img.size, rect, window, display)
    mime = "image/png"
    if len(data) > max_bytes:
        jpeg = _reencode_jpeg(img)
        if jpeg is not None and len(jpeg) > max_bytes:
            jpeg = _reencode_jpeg(img, quality=JPEG_QUALITY_LOW) or jpeg
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
    """Take a screenshot into the single latest.png (0600, overwritten
    each time), downscaled to `DOWNSCALE_MAX_PX`, and return (image_bytes,
    path, mime) — mime is image/png, or image/jpeg if the PNG was over
    MAX_PNG_BYTES and got re-encoded (latest.png stays the PNG) — or an
    error string on failure. `display` picks the monitor a
    region='screen' capture grabs ('auto', 'main', or a 1-based number —
    see `pick_display`); a window capture is wherever its window is, and
    a selection wherever the user snips. Synchronous; run via
    asyncio.to_thread from the tool handler."""
    region = region if region in REGIONS else "screen"
    computer_events.ensure_dpi_awareness()
    out_path = _screens_dir() / LATEST_NAME
    _clear_latest(out_path)
    target = None
    if region == "screen":
        try:
            target = pick_display(display)
        except ValueError as exc:
            return str(exc)
    elif region == "window":
        # recorded in the sidecar: the monitor the window is on
        target = _front_window_display(displays())
    # A snipped selection has no display to pin: not even asking keeps it
    # geometry-free (see `_build_geometry`).
    result = _capture_one(region, target, out_path)
    if isinstance(result, str):
        return result
    data, mime, geometry = result
    _store_geometry(geometry)
    return data, out_path, mime


def capture_all_displays() -> tuple[list[tuple[Display | None, bytes, str]], Display | None] | str:
    """Capture every monitor. Returns the shots in display order plus the
    display latest.png and the sidecar describe — the one holding the
    foreground window, captured last so it's the one left on disk and the
    one the computer_* tools act on."""
    computer_events.ensure_dpi_awareness()
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
    "'screen' (default, a whole display), 'window' (foreground window "
    "only), or 'selection' (the user snips an area). With more than "
    "one monitor, display picks which screen region='screen' grabs: "
    "'auto' (default, the one the foreground window is on), 'main' (the "
    "primary monitor), 'all' (every display, one image each) or a "
    "1-based display number (1 = primary).",
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
