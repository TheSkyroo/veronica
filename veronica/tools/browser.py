"""Browser control for Chrome and Microsoft Edge on Windows through the
Veronica browser extension (veronica/browser_extension, a Manifest V3
extension the user loads unpacked once).

The extension's service worker keeps a WebSocket open to a small server
this module runs inside the app (`start_bridge()`: a daemon thread with its
own asyncio loop, bound to 127.0.0.1:VERONICA_BROWSER_PORT, default 8765).
Veronica sends {id, op, ...args}; the extension answers {id, ok, result} or
{id, ok: false, error}. The ops are a fixed list implemented in the
extension (tabs, open, ready_state, read, find, click, type, scroll, back),
so Veronica never ships code into the page.

Who may connect: the WebSocket Origin must be the extension's
(chrome-extension://<EXTENSION_ID>, fixed by the "key" in manifest.json;
VERONICA_BROWSER_EXTENSION_ID adds others), and the first message must
carry the pairing token Veronica keeps in ~/.veronica/browser_token, which
the user pastes once into the extension's options page. Anything else is
closed with code 4401.

With both Chrome and Edge connected, tools drive the one whose window was
focused most recently (the extension reports focus changes); before any
focus report, the one that connected last. Within that browser the target
is the active tab of its last-focused normal window.

Page text is untrusted: it comes back as data, whitespace-collapsed and
capped (READ_MAX / FIND_MAX_LINES), never as instructions."""
import asyncio
import concurrent.futures
import hmac
import itertools
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool

log = logging.getLogger("veronica.tools.browser")

TIMEOUT_S = 20             # one request, end to end
CONNECT_WAIT_S = 2.0       # grace for an extension that is mid-(re)connect
AUTH_TIMEOUT_S = 5.0       # the hello must arrive this soon after connecting
DEFAULT_PORT = 8765
MAX_MESSAGE = 4 * 1024 * 1024
READ_MIN = 50
READ_DEFAULT = 6000
READ_MAX = 20000
FIND_MAX_LINES = 10
FIND_LINE_MAX = 160
# After a navigation (open/click/back/submit) wait for the tab to finish
# loading so a following browser_read sees the new page.
NAV_WAIT_S = 3.0
NAV_POLL_S = 0.25
_sleep = asyncio.sleep   # module attr so tests can stub the poll's delay

# The extension's ID, derived from the public key in manifest.json, so it is
# the same wherever the folder is loaded from, in Chrome and in Edge.
EXTENSION_ID = "kbjjfiapkdanjhojndmgepdlnokoilhe"
EXTENSION_DIR = Path(__file__).resolve().parent.parent / "browser_extension"
AUTH_CLOSE_CODE = 4401


class BrowserUnavailable(RuntimeError):
    """No usable browser connection (none connected, timed out, dropped)."""


class BrowserError(RuntimeError):
    """The extension ran the request and reported an error."""


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


# -- configuration ----------------------------------------------------------------
def _home() -> Path:
    return Path(os.environ.get("VERONICA_HOME") or Path.home() / ".veronica")


def token_path() -> Path:
    return _home() / "browser_token"


def pairing_token() -> str:
    """The pairing secret the extension must present; created on first use
    (and recreated if the file is empty or unreadable)."""
    p = token_path()
    try:
        tok = p.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(24)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tok + "\n", encoding="utf-8")
    try:
        os.chmod(p, 0o600)       # best effort; on Windows the profile ACL protects it
    except OSError:
        pass
    return tok


def bridge_port() -> int:
    try:
        port = int(os.environ.get("VERONICA_BROWSER_PORT", DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT
    return port if 0 < port < 65536 else DEFAULT_PORT


def allowed_origins() -> set[str]:
    ids = {EXTENSION_ID}
    ids |= {s.strip() for s in os.environ.get("VERONICA_BROWSER_EXTENSION_ID", "").split(",") if s.strip()}
    return {f"chrome-extension://{i}" for i in ids}


def origin_allowed(origin: str | None) -> bool:
    return bool(origin) and origin in allowed_origins()


def no_browser_message() -> str:
    return (
        "No browser is connected. Install the Veronica extension in Chrome or Edge: open "
        "chrome://extensions (or edge://extensions), turn on Developer mode, choose Load unpacked "
        f"and pick {EXTENSION_DIR}; then paste the pairing code from {token_path()} into the "
        "extension's options page."
    )


# -- the bridge -------------------------------------------------------------------
@dataclass(eq=False)
class _Client:
    conn: Any                      # websockets ServerConnection, or a fake in tests
    browser: str
    connected_at: float
    focused_at: float = 0.0
    pending: dict[int, asyncio.Future] = field(default_factory=dict)

    @property
    def recency(self) -> float:
        return max(self.focused_at, self.connected_at)


class Bridge:
    """Connected extensions plus request/response bookkeeping. Every
    coroutine here runs on `self.loop`; `request()` may be awaited from any
    loop (the tools run on the agent's)."""

    def __init__(self, token: str, loop: asyncio.AbstractEventLoop | None = None):
        self.token = token
        self.loop = loop
        self.clients: list[_Client] = []
        self.error: str | None = None          # why the server isn't listening, if it isn't
        self._ids = itertools.count(1)
        self._changed: asyncio.Event | None = None
        self._server = None

    def _event(self) -> asyncio.Event:
        if self._changed is None:
            self._changed = asyncio.Event()
        return self._changed

    # connection side
    async def handle(self, conn) -> None:
        """Serve one extension connection: authenticate, then route replies
        and focus events until it closes."""
        client = await self._authenticate(conn)
        if client is None:
            return
        self.clients.append(client)
        self._event().set()
        log.info("browser extension connected (%s)", client.browser)
        try:
            while True:
                raw = await conn.recv()
                self._on_message(client, raw)
        except Exception as exc:  # noqa: BLE001 — ConnectionClosed (or a test fake's EOF) ends the session
            log.debug("browser connection ended: %r", exc)
        finally:
            self.clients.remove(client)
            for fut in client.pending.values():
                if not fut.done():
                    fut.set_exception(BrowserUnavailable(f"{client.browser} disconnected before answering."))
            client.pending.clear()
            log.info("browser extension disconnected (%s)", client.browser)

    async def _authenticate(self, conn) -> _Client | None:
        try:
            raw = await asyncio.wait_for(conn.recv(), AUTH_TIMEOUT_S)
            hello = json.loads(raw)
        except Exception:  # noqa: BLE001 — silence, garbage or a drop all mean "not paired"
            hello = None
        tok = hello.get("token") if isinstance(hello, dict) and hello.get("type") == "hello" else None
        if not isinstance(tok, str) or not hmac.compare_digest(tok.encode(), self.token.encode()):
            log.warning("browser extension rejected: bad or missing pairing token")
            try:
                await conn.send(json.dumps({"type": "hello", "ok": False, "error": "bad pairing token"}))
                await conn.close(AUTH_CLOSE_CODE, "bad pairing token")
            except Exception:  # already gone; nothing to refuse
                log.debug("closing a rejected browser connection failed", exc_info=True)
            return None
        await conn.send(json.dumps({"type": "hello", "ok": True}))
        browser = str(hello.get("browser") or "browser")[:20]
        return _Client(conn=conn, browser=browser, connected_at=time.monotonic())

    def _on_message(self, client: _Client, raw) -> None:
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        if kind == "focus":
            client.focused_at = time.monotonic()
            return
        if kind == "ping":
            asyncio.ensure_future(self._send_quiet(client, {"type": "pong"}))
            return
        fut = client.pending.pop(msg.get("id"), None) if isinstance(msg.get("id"), int) else None
        if fut is None or fut.done():
            return
        if msg.get("ok"):
            fut.set_result(msg.get("result"))
        else:
            fut.set_exception(BrowserError(str(msg.get("error") or "the browser reported an error")))

    @staticmethod
    async def _send_quiet(client: _Client, obj: dict) -> None:
        try:
            await client.conn.send(json.dumps(obj))
        except Exception:  # a dropped socket is noticed by handle()
            log.debug("browser send failed", exc_info=True)

    # request side
    def target(self) -> _Client | None:
        return max(self.clients, key=lambda c: c.recency, default=None)

    async def _request(self, op: str, params: dict, timeout: float) -> Any:
        if self.error:
            raise BrowserUnavailable(self.error)
        client = self.target()
        if client is None:
            ev = self._event()
            ev.clear()
            try:
                await asyncio.wait_for(ev.wait(), CONNECT_WAIT_S)
            except TimeoutError:
                pass
            client = self.target()
            if client is None:
                raise BrowserUnavailable(no_browser_message())
        rid = next(self._ids)
        fut = asyncio.get_running_loop().create_future()
        client.pending[rid] = fut
        try:
            await client.conn.send(json.dumps({"id": rid, "op": op, **params}))
            return await asyncio.wait_for(fut, timeout)
        except TimeoutError:
            raise BrowserUnavailable(f"{client.browser} didn't answer within {timeout:g}s.") from None
        except (BrowserError, BrowserUnavailable):
            raise
        except Exception as exc:  # noqa: BLE001 — any send failure means the socket is gone
            raise BrowserUnavailable(f"Lost the connection to {client.browser}: {exc}") from None
        finally:
            client.pending.pop(rid, None)

    async def request(self, op: str, timeout: float | None = None, **params) -> Any:
        """Send one op to the target browser and return its result. Raises
        BrowserUnavailable (no browser / timeout / dropped) or BrowserError."""
        timeout = timeout or TIMEOUT_S
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if self.loop is None or self.loop is running:
            return await self._request(op, params, timeout)
        cf = asyncio.run_coroutine_threadsafe(self._request(op, params, timeout), self.loop)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(cf), timeout + CONNECT_WAIT_S + 1)
        except (TimeoutError, concurrent.futures.TimeoutError):
            cf.cancel()
            raise BrowserUnavailable(f"The browser didn't answer within {timeout:g}s.") from None


_bridge: Bridge | None = None
_bridge_lock = threading.Lock()


def _process_request(connection, request):
    """websockets hook: refuse any handshake not from our extension."""
    if not origin_allowed(request.headers.get("Origin")):
        log.warning("browser bridge refused origin %r", request.headers.get("Origin"))
        return connection.respond(403, "Forbidden\n")
    return None


def _run_server(bridge: Bridge, port: int, ready: threading.Event) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bridge.loop = loop

    async def main():
        from websockets.asyncio.server import (
            serve,  # lazy: only the app process needs it
        )
        try:
            bridge._server = await serve(
                bridge.handle, "127.0.0.1", port, process_request=_process_request,
                max_size=MAX_MESSAGE, ping_interval=None,
            )
        except OSError as exc:
            bridge.error = (f"Veronica couldn't listen for the browser extension on 127.0.0.1:{port} "
                            f"({exc.strerror or exc}). Is another copy running? Set VERONICA_BROWSER_PORT "
                            "to a free port here and in the extension's options.")
            log.error(bridge.error)
        finally:
            ready.set()

    try:
        loop.run_until_complete(main())
        if bridge.error is None:
            log.info("browser bridge listening on 127.0.0.1:%d", port)
            loop.run_forever()
    except Exception:
        log.exception("browser bridge crashed")
        bridge.error = bridge.error or "The browser bridge stopped unexpectedly; restart Veronica."
        ready.set()


def start_bridge(port: int | None = None) -> Bridge:
    """Start the extension's WebSocket server (idempotent, thread-safe).
    Call once at app startup; the tools also start it on first use."""
    global _bridge
    with _bridge_lock:
        if _bridge is not None:
            return _bridge
        bridge = Bridge(pairing_token())
        ready = threading.Event()
        threading.Thread(target=_run_server, args=(bridge, port or bridge_port(), ready),
                         name="veronica-browser-bridge", daemon=True).start()
        ready.wait(5)
        _bridge = bridge
        return bridge


def stop_bridge() -> None:
    """Close the server and its loop (app shutdown, tests). A later
    start_bridge() starts a fresh one."""
    global _bridge
    with _bridge_lock:
        bridge, _bridge = _bridge, None
    loop = bridge.loop if bridge else None
    if loop is None or not loop.is_running():
        return

    async def close():
        if bridge._server is not None:
            bridge._server.close()
            await bridge._server.wait_closed()

    try:
        asyncio.run_coroutine_threadsafe(close(), loop).result(5)
    except Exception:
        log.debug("browser bridge close failed", exc_info=True)
    loop.call_soon_threadsafe(loop.stop)


async def _call(op: str, **params) -> Any:
    bridge = _bridge or await asyncio.to_thread(start_bridge)
    return await bridge.request(op, **params)


async def _wait_for_load() -> None:
    """Poll the tab's load status until 'complete' or NAV_WAIT_S elapses.
    Best effort: any error (tab closed, browser gone) just ends the wait;
    it never changes the calling tool's result."""
    polls = int(NAV_WAIT_S / NAV_POLL_S)
    for i in range(polls):
        try:
            state = await _call("ready_state", timeout=NAV_WAIT_S)
        except Exception:
            log.debug("readyState poll failed", exc_info=True)
            return
        if state == "complete":
            return
        if i < polls - 1:
            await _sleep(NAV_POLL_S)


def _guard(fn):
    """Wrap a handler so bridge failures, malformed args and unexpected
    errors return `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except (BrowserUnavailable, BrowserError) as exc:
            return _err(str(exc))
        except Exception as exc:
            log.exception("browser tool failed")
            return _err(f"{type(exc).__name__}: {exc}")
    wrapper.__name__ = fn.__name__
    return wrapper


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def _clean(s, limit: int) -> str:
    return " ".join(str(s or "").split())[:limit]


# -- tools -----------------------------------------------------------------------
@tool("browser_tabs", "List the open tabs of the front window of Chrome or Edge (current tab marked *)", {})
@_guard
async def browser_tabs(args: dict) -> dict:
    tabs = _dict(await _call("tabs")).get("tabs") or []
    out = []
    for i, t in enumerate(tabs, start=1):
        t = _dict(t)
        mark = "* " if t.get("active") else ""
        out.append(f"{i}. {mark}{_clean(t.get('title'), 200)} — {_clean(t.get('url'), 500)}")
    return _ok("\n".join(out) if out else "No tabs.")


@tool("browser_open", "Open an http(s) URL in the current browser (new tab by default)", {"url": str, "new_tab": bool})
@_guard
async def browser_open(args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    new_tab = bool(args.get("new_tab", True))
    await _call("open", url=url, new_tab=new_tab)
    await _wait_for_load()
    return _ok(f"Opened {url}")


@tool("browser_read", "Read the current tab: title, URL and visible text (capped)", {"max_chars": int})
@_guard
async def browser_read(args: dict) -> dict:
    try:
        max_chars = int(args.get("max_chars") or READ_DEFAULT)
    except (TypeError, ValueError):
        max_chars = READ_DEFAULT
    max_chars = max(READ_MIN, min(READ_MAX, max_chars))
    data = _dict(await _call("read"))
    text = " ".join(str(data.get("text", "")).split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…[truncated]"
    return _ok(f"{_clean(data.get('title'), 300)}\n{_clean(data.get('url'), 1000)}\n{text}")


@tool("browser_find", "Find lines on the current page containing text (case-insensitive)", {"text": str})
@_guard
async def browser_find(args: dict) -> dict:
    needle = str(args.get("text", "")).strip()
    if not needle:
        return _err("text is required")
    lines = _dict(await _call("find", text=needle, max_lines=FIND_MAX_LINES)).get("lines") or []
    out = []
    for item in lines[:FIND_MAX_LINES]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append(f"{item[0]}: {_clean(item[1], FIND_LINE_MAX)}")
    return _ok("\n".join(out) if out else "not found")


ELEMENTS_DEFAULT = 150
# Browsers whose window a real mouse click may be aimed at.
BROWSER_EXES = frozenset({"chrome.exe", "msedge.exe", "brave.exe", "vivaldi.exe", "opera.exe"})


def _ref_or_target(args: dict) -> tuple[int | None, str]:
    """(ref, target) from a click/type request; ValueError if neither."""
    ref = args.get("ref")
    if ref in ("", None):
        ref = None
    elif isinstance(ref, bool) or not str(ref).strip().lstrip("#").isdigit():
        raise ValueError("ref must be an element number from browser_elements")
    else:
        ref = int(str(ref).strip().lstrip("#"))
    target = str(args.get("target") or "").strip()
    if ref is None and not target:
        raise ValueError("give ref (a number from browser_elements) or target (the element's visible text)")
    return ref, target


@tool("browser_elements",
      "List what can be clicked or typed into on the current page — links, buttons, video players, "
      "search boxes — each with a number (ref) to pass to browser_click / browser_type. Use it whenever "
      "you need to click something whose exact text you don't know (a video, an icon button, a result).",
      {"type": "object", "properties": {"max": {"type": "integer"}}, "required": []})
@_guard
async def browser_elements(args: dict) -> dict:
    try:
        max_items = max(10, min(300, int(args.get("max") or ELEMENTS_DEFAULT)))
    except (TypeError, ValueError):
        max_items = ELEMENTS_DEFAULT
    res = _dict(await _call("elements", max=max_items))
    items = [e for e in (res.get("elements") or []) if isinstance(e, dict)]
    lines = [f"Title: {_clean(res.get('title'), 200)}", f"URL: {_clean(res.get('url'), 500)}",
             "Elements (ref role \"name\"; * = on screen now):"]
    for e in items:
        bits = [f"[{e.get('ref')}]", "*" if e.get("in_view") else " ", str(e.get("role") or ""),
                f"\"{_clean(e.get('name'), 100)}\""]
        if e.get("href"):
            bits.append(f"-> {_clean(e['href'], 80)}")
        for flag in ("disabled", "checked", "playing"):
            if flag in e:
                bits.append(f"{flag}={str(e[flag]).lower()}")
        if e.get("value"):
            bits.append(f"value=\"{_clean(e['value'], 60)}\"")
        lines.append(" ".join(bits))
    total = res.get("total")
    if isinstance(total, int) and total > len(items):
        lines.append(f"(+{total - len(items)} more further down; scroll and list again)")
    if not items:
        lines.append("(nothing clickable found — the page may still be loading)")
    return _ok("\n".join(lines))


# -- real clicks ---------------------------------------------------------------------
# A script's element.click() is not a user gesture: sites that only react to
# real input, and anything the browser gates on a gesture (starting a video
# with sound, fullscreen, some popups), ignore it. So a click goes where the
# element is on screen, with the actual mouse — when that is certain to land
# on the browser — and falls back to the script click otherwise.

def _screen_point(loc: dict) -> tuple[int, int] | None:
    """The physical screen pixel for the page point `loc` describes (CSS
    pixels inside the viewport, plus the window geometry the page reported).
    The browser window's own borders are inferred from outer - inner size."""
    try:
        zoom = float(loc.get("zoom") or 1.0)
        dpr = float(loc["dpr"])
        scale = dpr / zoom                         # Windows display scaling (DIP -> physical)
        border = max(0.0, (float(loc["outerWidth"]) - float(loc["innerWidth"]) * zoom) / 2)
        top = float(loc["outerHeight"]) - float(loc["innerHeight"]) * zoom - border
        x = (float(loc["screenX"]) + border) * scale + float(loc["x"]) * dpr
        y = (float(loc["screenY"]) + top) * scale + float(loc["y"]) * dpr
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return round(x), round(y)


def _window_pid_at(x: int, y: int) -> int | None:
    """The process owning the top-level window under screen point (x, y)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32")
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.GetAncestor.restype = wintypes.HWND
    user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
    hwnd = user32.WindowFromPoint(wintypes.POINT(x, y))
    if not hwnd:
        return None
    root = user32.GetAncestor(hwnd, 2) or hwnd          # GA_ROOT
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(root, ctypes.byref(pid))
    return pid.value or None


def _mouse_click(x: int, y: int) -> bool:
    """A real left click at (x, y) if it will land on the browser in front;
    False (nothing done) otherwise."""
    from veronica.tools import computer_events as ce

    ce.ensure_dpi_awareness()
    front = ce.frontmost()
    if front.bundle_id.lower() not in BROWSER_EXES:
        log.info("real click skipped: %s is in front, not the browser", front.bundle_id or "?")
        return False
    if ce.input_blocked():
        log.info("real click skipped: the browser runs as administrator")
        return False
    if _window_pid_at(x, y) != front.pid:
        log.info("real click skipped: something else is on top at (%d, %d)", x, y)
        return False
    ce.click(x, y)
    return True


async def _click(ref: int | None, target: str) -> str | None:
    """Click the element; what was clicked (or None if there's no such
    element). A real mouse click where possible, a script click if not."""
    loc = _dict(await _call("locate", ref=ref, target=target))
    if not loc.get("found"):
        return None
    point = None if loc.get("covered") else _screen_point(loc)
    if point is not None:
        try:
            if await asyncio.to_thread(_mouse_click, *point):
                log.info("browser click: real mouse at %s on %s", point, loc.get("described"))
                return str(loc.get("described") or "")
        except Exception:
            log.warning("real browser click failed; using a script click", exc_info=True)
    clicked = _dict(await _call("click", ref=ref, target=target)).get("clicked")
    return str(clicked) if clicked else None


_CLICK_SCHEMA = {
    "type": "object",
    "properties": {
        "ref": {"type": "integer", "description": "Element number from browser_elements (preferred)"},
        "target": {"type": "string", "description": "The element's visible text or label (also shown when asking)"},
    },
    "required": [],
}


@tool("browser_click",
      "Click something on the current page: pass ref (a number from browser_elements) — and target, its "
      "name — or just target, its visible text or label. Uses the real mouse when it can, so video players "
      "and other controls that ignore scripted clicks still work.",
      _CLICK_SCHEMA)
@_guard
async def browser_click(args: dict) -> dict:
    try:
        ref, target = _ref_or_target(args)
    except ValueError as exc:
        return _err(str(exc))
    clicked = await _click(ref, target)
    if not clicked:
        return _err(f"no element {ref}" if ref is not None else f"no element matching '{target}'")
    await _wait_for_load()
    return _ok(f"Clicked {_clean(clicked, 80)}")


@tool("browser_type",
      "Type text into a field on the current page — by ref (a number from browser_elements) or target "
      "(its placeholder, label or name) — optionally pressing Enter",
      {"type": "object",
       "properties": {"ref": {"type": "integer"}, "target": {"type": "string"},
                      "text": {"type": "string"}, "submit": {"type": "boolean"}},
       "required": ["text"]})
@_guard
async def browser_type(args: dict) -> dict:
    try:
        ref, target = _ref_or_target(args)
    except ValueError as exc:
        return _err(str(exc))
    text = str(args.get("text", ""))
    submit = bool(args.get("submit", False))
    typed = _dict(await _call("type", ref=ref, target=target, text=text, submit=submit)).get("typed")
    if not typed:
        return _err(f"no field {ref}" if ref is not None else f"no field matching '{target}'")
    if submit:
        await _wait_for_load()
    return _ok(f"Typed into {_clean(typed, 80)}")


_SCROLL_DIRECTIONS = ("up", "down", "top", "bottom")


@tool("browser_scroll", "Scroll the current page: up, down, top or bottom", {"direction": str})
@_guard
async def browser_scroll(args: dict) -> dict:
    direction = str(args.get("direction", "down")).strip().lower()
    if direction not in _SCROLL_DIRECTIONS:
        return _err("direction must be up, down, top or bottom")
    await _call("scroll", direction=direction)
    return _ok(f"Scrolled {direction}")


@tool("browser_back", "Go back one page in the current tab", {})
@_guard
async def browser_back(args: dict) -> dict:
    await _call("back")
    await _wait_for_load()
    return _ok("Went back")


TOOLS = [browser_tabs, browser_open, browser_read, browser_find, browser_elements, browser_click, browser_type,
         browser_scroll, browser_back]
BROWSER_TOOL_NAMES = [t.name for t in TOOLS]
browser_server = create_sdk_mcp_server(name="browser", version="1.0.0", tools=TOOLS)
