"""Computer use: act on the frontmost app using the last screenshot —
move/click/drag/scroll the mouse, type, press keys, and find text on
screen via OCR — exposed as the in-process `computer` MCP server.

Coordinates default to **pixels of latest.png** (what Claude reads off
the screenshot); `space="screen"` accepts screen points instead. The
geometry sidecar (`screen.load_geometry`) maps one to the other, and
must be fresh (`GEOMETRY_MAX_AGE_S`) for anything positional — a stale
view is not a safe basis for clicking. `computer_type`/`computer_key`
need no geometry.

Safety, in the tool (in addition to the confirm gate in the brain):
Accessibility must be granted (the first call asks macOS to prompt),
password fields are never typed into, "Allow"-style buttons are never
clicked while a system permission dialog is up, and app-quit/lock
combos are refused by `computer_events.key`.

Every action ends with a short settle, then reports the frontmost app so
the brain can tell whether focus moved (it should still screenshot to
verify). Primitives are called through the `computer_events` module
attribute so tests can swap in recording fakes.
"""
import asyncio
import logging

from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica.tools import computer_events as events
from veronica.tools import ocr, screen
from veronica.tools.computer_events import (
    DISALLOWED_DIALOG_TARGETS,
    PERMISSION_HINT,
    DangerousCombo,
)
from veronica.tools.screen import GEOMETRY_MAX_AGE_S, Geometry

log = logging.getLogger("veronica.tools.computer")

STALE_HINT = "Take a screenshot first — I need a fresh view of the screen to know where things are."
SECURE_HINT = "That's a password field — I won't type into it."
DIALOG_HINT = "I won't click through a system permission dialog — please do that one yourself."
DANGEROUS_HINT = "I won't press that — it would quit or lock the Mac."
FIND_MAX = 10
SETTLE_S = 0.15          # let the app react before reporting the frontmost window
SPACES = ("image", "screen")
_sleep = asyncio.sleep   # module attr so tests can stub the settle


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def _guard(fn):
    """Wrap a handler so malformed args/unexpected failures return
    `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            log.exception("computer tool failed")
            return _err(f"{type(exc).__name__}: {exc}")
    wrapper.__name__ = fn.__name__
    return wrapper


# --- gates -------------------------------------------------------------------

def _trusted() -> bool:
    """Accessibility granted? `prompt=True` makes macOS show the grant
    prompt the first time it isn't."""
    return events.accessibility_trusted(prompt=True)


def _fresh_geometry() -> Geometry | str:
    """The latest screenshot's geometry, or the spoken error when there is
    none or it is older than `GEOMETRY_MAX_AGE_S`."""
    geometry = screen.load_geometry()
    if geometry is None or geometry.age_s > GEOMETRY_MAX_AGE_S:
        return STALE_HINT
    return geometry


def _num(args: dict, key: str) -> float:
    if key not in args or args[key] is None:
        raise ValueError(f"{key} is required")
    try:
        return float(args[key])
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, not {args[key]!r}") from None


def _coords(args: dict, geometry: Geometry, xk: str = "x", yk: str = "y") -> tuple[float, float]:
    """(x_pt, y_pt) for the `xk`/`yk` args: image pixels mapped through
    `geometry` by default, or passed through when `space="screen"`."""
    x, y = _num(args, xk), _num(args, yk)
    space = str(args.get("space") or "image").lower()
    if space not in SPACES:
        raise ValueError(f"space must be one of {', '.join(SPACES)}")
    if space == "screen":
        return (x, y)
    return geometry.to_screen(x, y)


async def _done() -> dict:
    """Settle, then report where focus is so the brain can notice it moved."""
    await _sleep(SETTLE_S)
    front = events.frontmost()
    if not front.app:
        return _ok("done")
    if front.window_title:
        return _ok(f"done — frontmost: {front.app} — {front.window_title}")
    return _ok(f"done — frontmost: {front.app}")


async def _run(fn, *a) -> None:
    """Primitives block (they sleep between events); keep the loop free."""
    await asyncio.to_thread(fn, *a)


def _find_words(geometry: Geometry, query: str) -> list[ocr.Word]:
    words = ocr.recognize_text(
        screen.latest_screenshot_path(), image_size=(geometry.image_w, geometry.image_h)
    )
    return ocr.find_text(words, query)


# --- tools -------------------------------------------------------------------

@tool(
    "computer_move",
    "Move the mouse pointer to (x, y). Coordinates are pixels of the last "
    "screenshot by default; space='screen' means screen points. Needs a "
    "screenshot from the last two minutes.",
    {"x": float, "y": float, "space": str},
)
@_guard
async def computer_move(args: dict) -> dict:
    if not _trusted():
        return _err(PERMISSION_HINT)
    geometry = _fresh_geometry()
    if isinstance(geometry, str):
        return _err(geometry)
    x, y = _coords(args, geometry)
    await _run(events.move, x, y)
    return await _done()


@tool(
    "computer_scroll",
    "Scroll at (x, y) by (dx, dy) pixels in content direction: positive dy "
    "scrolls DOWN (shows content further down the page), negative dy scrolls "
    "up; positive dx scrolls right. Coordinates are pixels of the last "
    "screenshot by default; space='screen' means screen points.",
    {"x": float, "y": float, "dx": float, "dy": float, "space": str},
)
@_guard
async def computer_scroll(args: dict) -> dict:
    if not _trusted():
        return _err(PERMISSION_HINT)
    geometry = _fresh_geometry()
    if isinstance(geometry, str):
        return _err(geometry)
    x, y = _coords(args, geometry)
    dx = float(args.get("dx") or 0)
    dy = float(args.get("dy") or 0)
    if dx == 0 and dy == 0:
        return _err("dx or dy is required (positive dy scrolls down)")
    # CGEvent's sign is the opposite of content direction: positive dy
    # scrolls up, positive dx scrolls left.
    await _run(events.scroll, x, y, -dx, -dy)
    return await _done()


@tool(
    "computer_find",
    "Find text on the last screenshot with OCR. Returns up to 10 matches "
    "with their centre (x, y) in screenshot pixels — pass those to "
    "computer_click — or 'not found'. Needs a screenshot from the last two minutes.",
    {"text": str},
)
@_guard
async def computer_find(args: dict) -> dict:
    query = str(args.get("text", "")).strip()
    if not query:
        return _err("text is required")
    if not _trusted():
        return _err(PERMISSION_HINT)
    geometry = _fresh_geometry()
    if isinstance(geometry, str):
        return _err(geometry)
    matches = await asyncio.to_thread(_find_words, geometry, query)
    if not matches:
        return _ok("not found")
    lines = []
    for i, w in enumerate(matches[:FIND_MAX], start=1):
        cx, cy = w.center
        lines.append(
            f"{i}. '{w.text}' at ({round(cx)}, {round(cy)}) size "
            f"{round(w.w)}×{round(w.h)} (conf {w.confidence:.2f})"
        )
    return _ok("\n".join(lines))


@tool(
    "computer_click",
    "Click at (x, y): button 'left' (default), 'right' or 'middle'; "
    "double=true for a double-click. Coordinates are pixels of the last "
    "screenshot by default; space='screen' means screen points. Needs a "
    "screenshot from the last two minutes; take another one afterwards to verify.",
    {"x": float, "y": float, "button": str, "double": bool, "space": str},
)
@_guard
async def computer_click(args: dict) -> dict:
    if not _trusted():
        return _err(PERMISSION_HINT)
    geometry = _fresh_geometry()
    if isinstance(geometry, str):
        return _err(geometry)
    button = str(args.get("button") or "left").lower()
    if button not in events._BUTTONS:
        return _err(f"button must be one of {', '.join(events._BUTTONS)}")
    x, y = _coords(args, geometry)
    await _run(events.click, x, y, button, bool(args.get("double", False)))
    return await _done()


@tool(
    "computer_click_text",
    "Find text on the last screenshot with OCR and click the centre of the "
    "match (index picks which match when there are several, 0 = first, "
    "top-to-bottom); double=true for a double-click. Needs a screenshot "
    "from the last two minutes; take another one afterwards to verify.",
    {"text": str, "index": int, "double": bool},
)
@_guard
async def computer_click_text(args: dict) -> dict:
    query = str(args.get("text", "")).strip()
    if not query:
        return _err("text is required")
    if not _trusted():
        return _err(PERMISSION_HINT)
    if query.lower() in DISALLOWED_DIALOG_TARGETS and events.is_system_dialog(events.frontmost()):
        return _err(DIALOG_HINT)
    geometry = _fresh_geometry()
    if isinstance(geometry, str):
        return _err(geometry)
    matches = await asyncio.to_thread(_find_words, geometry, query)
    if not matches:
        return _err(f"no match for '{query}'")
    index = int(args.get("index") or 0)
    if not 0 <= index < len(matches):
        n = len(matches)
        return _err(f"index {index} is out of range: {n} match{'es' if n != 1 else ''} for '{query}'")
    cx, cy = matches[index].center
    x, y = geometry.to_screen(cx, cy)
    await _run(events.click, x, y, "left", bool(args.get("double", False)))
    return await _done()


@tool(
    "computer_drag",
    "Left-drag from (x1, y1) to (x2, y2). Coordinates are pixels of the last "
    "screenshot by default; space='screen' means screen points. Needs a "
    "screenshot from the last two minutes.",
    {"x1": float, "y1": float, "x2": float, "y2": float, "space": str},
)
@_guard
async def computer_drag(args: dict) -> dict:
    if not _trusted():
        return _err(PERMISSION_HINT)
    geometry = _fresh_geometry()
    if isinstance(geometry, str):
        return _err(geometry)
    x1, y1 = _coords(args, geometry, "x1", "y1")
    x2, y2 = _coords(args, geometry, "x2", "y2")
    await _run(events.drag, x1, y1, x2, y2)
    return await _done()


@tool(
    "computer_type",
    "Type text into whatever has keyboard focus (click a field first); "
    "submit=true presses Enter afterwards. Refuses password fields.",
    {"text": str, "submit": bool},
)
@_guard
async def computer_type(args: dict) -> dict:
    text = str(args.get("text") or "")
    submit = bool(args.get("submit", False))
    if not text and not submit:
        return _err("text is required")
    if not _trusted():
        return _err(PERMISSION_HINT)
    if events.focused_is_secure():
        return _err(SECURE_HINT)
    if text:
        await _run(events.type_text, text)
    if submit:
        await _run(events.key, "enter")
    return await _done()


@tool(
    "computer_key",
    "Press a key or combo, e.g. 'enter', 'esc', 'tab', 'cmd+s', "
    "'cmd+shift+t', 'ctrl+alt+left'. Quit/lock combos are refused.",
    {"combo": str},
)
@_guard
async def computer_key(args: dict) -> dict:
    combo = str(args.get("combo") or "").strip()
    if not combo:
        return _err("combo is required")
    if not _trusted():
        return _err(PERMISSION_HINT)
    try:
        await _run(events.key, combo)
    except DangerousCombo:
        return _err(DANGEROUS_HINT)
    except ValueError as exc:
        return _err(f"Unknown key: {exc}")
    return await _done()


TOOLS = [
    computer_move, computer_scroll, computer_find, computer_click,
    computer_click_text, computer_drag, computer_type, computer_key,
]
COMPUTER_TOOL_NAMES = [t.name for t in TOOLS]
computer_server = create_sdk_mcp_server(name="computer", version="1.0.0", tools=TOOLS)
