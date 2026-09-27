"""Low-level input primitives for computer use: synthetic mouse/keyboard
events via Quartz CGEvent, Accessibility (AX) checks, and a frontmost-app
lookup.

Coordinates here are **screen points** in the global space: top-left
origin on the main display, running on across every attached one, so a
point on a second monitor is simply past the main display's width and
nothing here clamps it. The `computer` tools convert screenshot pixels
before calling in. Every
framework call goes through `_quartz()` / `_ax()` / `_appkit()` so tests
can swap in recording fakes — nothing in this module is exercised for
real under pytest.
"""
import logging
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

PERMISSION_HINT = (
    "Veronica isn't allowed to control this Mac yet — enable it in "
    "System Settings > Privacy & Security > Accessibility, then try again."
)

DRAG_STEPS = 8
DRAG_DURATION_S = 0.2
TYPE_CHUNK_UTF16 = 20
TYPE_CHUNK_GAP_S = 0.01
CLICK_GAP_S = 0.02

# Bundle ids whose windows are OS permission/security dialogs. System
# Settings counts unconditionally: its window title is unreliable (often
# empty, or the sidebar section rather than the pane), and any pane is
# one click from Privacy & Security.
SYSTEM_DIALOG_BUNDLES = frozenset({
    "com.apple.SecurityAgent",
    "com.apple.UserNotificationCenter",
    "com.apple.coreservices.uiagent",
    "com.apple.accessibility.universalAccessAuthWarn",
    "com.apple.systempreferences",
})

# Button labels Veronica must never click while a system dialog is up.
DISALLOWED_DIALOG_TARGETS = frozenset({
    "allow", "always allow", "ok", "open system settings", "continue", "install", "trust",
})

# US-layout virtual keycodes (Carbon kVK_*).
KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "equal": 24, "9": 25, "7": 26,
    "minus": 27, "8": 28, "0": 29, "rightbracket": 30, "o": 31, "u": 32, "leftbracket": 33,
    "i": 34, "p": 35, "enter": 36, "return": 36, "l": 37, "j": 38, "quote": 39, "k": 40,
    "semicolon": 41, "backslash": 42, "comma": 43, "slash": 44, "n": 45, "m": 46,
    "period": 47, "tab": 48, "space": 49, "grave": 50, "backspace": 51, "delete": 51,
    "esc": 53, "escape": 53,
    "f5": 96, "f6": 97, "f7": 98, "f3": 99, "f8": 100, "f9": 101, "f11": 103, "f10": 109,
    "f12": 111, "f4": 118, "f2": 120, "f1": 122,
    "home": 115, "pageup": 116, "forwarddelete": 117, "end": 119, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
}
# Punctuation by literal character, plus a few spoken aliases.
KEYCODES.update({
    "-": 27, "=": 24, "[": 33, "]": 30, "\\": 42, ";": 41, "'": 39, ",": 43, ".": 47,
    "/": 44, "`": 50,
    "pgup": 116, "pgdn": 121, "pgdown": 121, "del": 51, "bksp": 51, "spacebar": 49,
    "arrowup": 126, "arrowdown": 125, "arrowleft": 123, "arrowright": 124,
})

_BUTTONS = ("left", "right", "middle")


@dataclass(frozen=True)
class Front:
    """The frontmost app and its main window at the time of the lookup."""
    app: str
    bundle_id: str
    window_title: str
    pid: int


# --- framework seams ---------------------------------------------------------

def _quartz():
    import Quartz
    return Quartz


def _ax():
    import ApplicationServices
    return ApplicationServices


def _appkit():
    import AppKit
    return AppKit


_sleep = time.sleep


def post(event) -> None:
    """Post one CGEvent to the HID event tap (i.e. as if from hardware)."""
    q = _quartz()
    q.CGEventPost(q.kCGHIDEventTap, event)


def _release(event, what: str) -> None:
    """Post a key-up / mouse-up from a `finally`: a failure here is logged
    rather than raised so it never masks the original error."""
    try:
        post(event)
    except Exception as e:  # noqa: BLE001
        log.warning("failed to release %s: %s", what, e)


# --- modifiers ---------------------------------------------------------------

# CGEventFlags masks (fixed CoreGraphics header values; same as
# Quartz.kCGEventFlagMask{Shift,Control,Alternate,Command}). Literals so
# importing this module never touches Quartz.
_MASK_SHIFT = 1 << 17
_MASK_CONTROL = 1 << 18
_MASK_ALTERNATE = 1 << 19
_MASK_COMMAND = 1 << 20

MODIFIERS: dict[str, int] = {
    "cmd": _MASK_COMMAND, "command": _MASK_COMMAND,
    "shift": _MASK_SHIFT,
    "alt": _MASK_ALTERNATE, "option": _MASK_ALTERNATE,
    "ctrl": _MASK_CONTROL, "control": _MASK_CONTROL,
}


# --- mouse -------------------------------------------------------------------

def _button_consts(button: str):
    q = _quartz()
    if button == "left":
        return q.kCGMouseButtonLeft, q.kCGEventLeftMouseDown, q.kCGEventLeftMouseUp
    if button == "right":
        return q.kCGMouseButtonRight, q.kCGEventRightMouseDown, q.kCGEventRightMouseUp
    if button == "middle":
        return q.kCGMouseButtonCenter, q.kCGEventOtherMouseDown, q.kCGEventOtherMouseUp
    raise ValueError(f"unknown mouse button {button!r} (use one of {', '.join(_BUTTONS)})")


def _mouse_event(type_, x: float, y: float, button_const):
    return _quartz().CGEventCreateMouseEvent(None, type_, (float(x), float(y)), button_const)


def move(x: float, y: float) -> None:
    """Move the pointer to (x, y) screen points."""
    q = _quartz()
    post(_mouse_event(q.kCGEventMouseMoved, x, y, q.kCGMouseButtonLeft))


def click(x: float, y: float, button: str = "left", double: bool = False) -> None:
    """Move to (x, y) then press and release `button`; `double` sends a
    second down/up pair tagged with click state 2 so apps see a real
    double-click."""
    q = _quartz()
    btn, down, up = _button_consts(button)
    move(x, y)

    def pair(click_state: int | None) -> None:
        def make(type_):
            ev = _mouse_event(type_, x, y, btn)
            if click_state is not None:
                q.CGEventSetIntegerValueField(ev, q.kCGMouseEventClickState, click_state)
            return ev
        try:
            post(make(down))
        finally:
            _release(make(up), f"{button} button")

    pair(None)
    if double:
        _sleep(CLICK_GAP_S)
        pair(2)


def drag(x1: float, y1: float, x2: float, y2: float) -> None:
    """Left-drag from (x1, y1) to (x2, y2): press, `DRAG_STEPS`
    interpolated dragged-moves spread over `DRAG_DURATION_S`, release."""
    q = _quartz()
    btn = q.kCGMouseButtonLeft
    move(x1, y1)
    step_s = DRAG_DURATION_S / DRAG_STEPS
    try:
        post(_mouse_event(q.kCGEventLeftMouseDown, x1, y1, btn))
        for i in range(1, DRAG_STEPS + 1):
            t = i / DRAG_STEPS
            _sleep(step_s)
            post(_mouse_event(q.kCGEventLeftMouseDragged, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, btn))
    finally:
        _release(_mouse_event(q.kCGEventLeftMouseUp, x2, y2, btn), "left button (drag)")


def scroll(x: float, y: float, dx: float = 0, dy: float = 0) -> None:
    """Move to (x, y) then scroll by (dx, dy) pixels. CGEvent's sign
    convention: positive dy scrolls up (toward the top of the document),
    negative scrolls down; positive dx scrolls left."""
    q = _quartz()
    move(x, y)
    post(q.CGEventCreateScrollWheelEvent(None, q.kCGScrollEventUnitPixel, 2, int(dy), int(dx)))


# --- keyboard ----------------------------------------------------------------

def _utf16_units(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _chunks(text: str, limit: int = TYPE_CHUNK_UTF16) -> list[str]:
    """Split `text` into pieces of at most `limit` UTF-16 code units,
    never splitting a surrogate pair (astral characters count as 2)."""
    out: list[str] = []
    cur, cur_units = [], 0
    for ch in text:
        units = 2 if ord(ch) > 0xFFFF else 1
        if cur and cur_units + units > limit:
            out.append("".join(cur))
            cur, cur_units = [], 0
        cur.append(ch)
        cur_units += units
    if cur:
        out.append("".join(cur))
    return out


def type_text(text: str) -> None:
    """Type `text` as Unicode key events (layout-independent), in chunks
    of `TYPE_CHUNK_UTF16` code units with a short gap between them."""
    q = _quartz()
    chunks = _chunks(text)
    for i, chunk in enumerate(chunks):
        n = _utf16_units(chunk)
        for keydown in (True, False):
            ev = q.CGEventCreateKeyboardEvent(None, 0, keydown)
            q.CGEventKeyboardSetUnicodeString(ev, n, chunk)
            post(ev)
        if i < len(chunks) - 1:
            _sleep(TYPE_CHUNK_GAP_S)


_MODIFIER_CANON = {
    "cmd": "cmd", "command": "cmd",
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "alt", "option": "alt",
    "shift": "shift",
}
_MODIFIER_ORDER = ("cmd", "ctrl", "alt", "shift")
_KEY_CANON = {"escape": "esc"}


class DangerousCombo(ValueError):
    """Raised by `key()` for combos that quit/force-quit apps or lock or
    power off the Mac — never sent, whatever the caller asked."""


# Canonical form (see `normalize_combo`).
DANGEROUS_COMBOS = frozenset({
    "cmd+q",                 # quit app
    "cmd+shift+q",           # log out
    "cmd+ctrl+q",            # lock screen
    "cmd+alt+esc",           # force quit
    "cmd+alt+shift+esc",     # force quit frontmost immediately
    "cmd+ctrl+power",        # restart
})


def normalize_combo(combo: str) -> str:
    """Canonical form of a combo: lowercase, modifier aliases folded
    (command→cmd, control→ctrl, option→alt), modifiers deduplicated and
    ordered cmd, ctrl, alt, shift, then the key. Raises ValueError when
    the shape isn't (modifiers)+one key; the key name itself is *not*
    validated here (see `key()`)."""
    parts = [p.strip().lower() for p in combo.split("+")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"can't parse key combo {combo!r}")
    *mods, name = parts
    canon = set()
    for m in mods:
        if m not in _MODIFIER_CANON:
            raise ValueError(f"unknown modifier {m!r} in {combo!r}")
        canon.add(_MODIFIER_CANON[m])
    if name in _MODIFIER_CANON:
        raise ValueError(f"key combo {combo!r} has modifiers but no key to press")
    ordered = [m for m in _MODIFIER_ORDER if m in canon]
    return "+".join([*ordered, _KEY_CANON.get(name, name)])


def _parse_combo(combo: str) -> tuple[int, int]:
    """`"cmd+shift+s"` → (keycode, flags). Raises DangerousCombo for the
    denylist and ValueError for anything that isn't (modifiers)+one key
    from `KEYCODES`."""
    norm = normalize_combo(combo)
    if norm in DANGEROUS_COMBOS:
        raise DangerousCombo(f"refusing to press {norm} (quits, locks or powers off)")
    *mods, name = norm.split("+")
    flags = 0
    for m in mods:
        flags |= MODIFIERS[m]
    if name not in KEYCODES:
        raise ValueError(f"unknown key {name!r} in {combo!r}")
    return KEYCODES[name], flags


def key(combo: str) -> None:
    """Press and release a key combo such as "enter", "cmd+s" or
    "ctrl+shift+tab". Modifier flags are set explicitly on both events
    (so a stuck hardware modifier — e.g. the push-to-talk key — never
    leaks in), and the key-up is always attempted even if the key-down
    fails, so no modifier is left stuck."""
    q = _quartz()
    keycode, flags = _parse_combo(combo)

    def make(keydown: bool):
        ev = q.CGEventCreateKeyboardEvent(None, keycode, keydown)
        q.CGEventSetFlags(ev, flags)
        return ev

    try:
        post(make(True))
    finally:
        _release(make(False), f"key {combo!r}")


# --- accessibility -----------------------------------------------------------

def accessibility_trusted(prompt: bool = False) -> bool:
    """True when this process may post events / read AX. `prompt=True`
    makes macOS show the "allow Veronica to control this computer"
    prompt (once) when it isn't."""
    try:
        ax = _ax()
        return bool(ax.AXIsProcessTrustedWithOptions({ax.kAXTrustedCheckOptionPrompt: bool(prompt)}))
    except Exception as e:  # noqa: BLE001 — permission checks never raise
        log.warning("accessibility check failed: %s", e)
        return False


def focused_is_secure() -> bool:
    """True when the system-wide focused UI element is a password field
    (role or subrole AXSecureTextField). False on any error."""
    try:
        ax = _ax()
        system = ax.AXUIElementCreateSystemWide()
        err, elem = ax.AXUIElementCopyAttributeValue(system, ax.kAXFocusedUIElementAttribute, None)
        if err or elem is None:
            return False
        for attr in (ax.kAXRoleAttribute, ax.kAXSubroleAttribute):
            err, value = ax.AXUIElementCopyAttributeValue(elem, attr, None)
            if not err and value == "AXSecureTextField":
                return True
        return False
    except Exception as e:  # noqa: BLE001
        log.debug("focused element check failed: %s", e)
        return False


# --- frontmost ---------------------------------------------------------------

def _on_screen(info: dict) -> bool:
    """Cheap sanity check: windows parked at wildly negative coords are
    hidden helpers (e.g. Electron's offscreen renderer)."""
    bounds = info.get("kCGWindowBounds") or {}
    try:
        x, y = float(bounds.get("X", 0)), float(bounds.get("Y", 0))
        w, h = float(bounds.get("Width", 0)), float(bounds.get("Height", 0))
    except (TypeError, ValueError):
        return True
    return x + w > 0 and y + h > 0


def _window_title(pid: int) -> str:
    q = _quartz()
    try:
        infos = q.CGWindowListCopyWindowInfo(q.kCGWindowListOptionOnScreenOnly, 0) or []
    except Exception as e:  # noqa: BLE001
        log.debug("window list failed: %s", e)
        return ""
    fallback = None
    for info in infos:
        if info.get("kCGWindowOwnerPID") != pid or info.get("kCGWindowLayer", 0) != 0:
            continue
        if _on_screen(info):
            return str(info.get("kCGWindowName") or "")
        if fallback is None:
            fallback = info
    return str(fallback.get("kCGWindowName") or "") if fallback else ""


def frontmost() -> Front:
    """The frontmost application and the title of its first layer-0
    on-screen window (empty strings when there's nothing to report)."""
    try:
        app = _appkit().NSWorkspace.sharedWorkspace().frontmostApplication()
    except Exception as e:  # noqa: BLE001
        log.debug("frontmost app lookup failed: %s", e)
        app = None
    if app is None:
        return Front(app="", bundle_id="", window_title="", pid=0)
    pid = int(app.processIdentifier() or 0)
    return Front(
        app=str(app.localizedName() or ""),
        bundle_id=str(app.bundleIdentifier() or ""),
        window_title=_window_title(pid),
        pid=pid,
    )


def is_system_dialog(front: Front) -> bool:
    """True when `front` is an OS permission/security dialog — where
    "Allow"-style clicks are refused and every action needs a confirm."""
    return front.bundle_id in SYSTEM_DIALOG_BUNDLES
