"""Browser tools over the extension bridge, with a fake extension
connection (no real browser). One test drives the real websockets server
over loopback to check the Origin gate and the pairing handshake."""
import asyncio
import json

import pytest

from veronica.tools import browser as b

TOKEN = "s3cret-token"
EOF = object()


class FakeConn:
    """Stands in for a websockets ServerConnection. `responder(msg)` answers
    each request: ("ok", result), ("err", text) or None (never answers)."""

    def __init__(self, responder=None):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = None
        self.responder = responder or (lambda msg: ("ok", "complete") if msg["op"] == "ready_state" else ("ok", None))

    async def recv(self):
        m = await self.inbox.get()
        if m is EOF:
            raise EOFError("closed")
        return m

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        if "id" in msg:
            reply = self.responder(msg)
            if reply is not None:
                kind, val = reply
                out = {"id": msg["id"], "ok": kind == "ok"}
                out["result" if kind == "ok" else "error"] = val
                self.inbox.put_nowait(json.dumps(out))

    async def close(self, code=1000, reason=""):
        self.closed = code
        self.inbox.put_nowait(EOF)

    def drop(self):
        self.inbox.put_nowait(EOF)

    def requests(self, op=None):
        return [m for m in self.sent if "id" in m and (op is None or m["op"] == op)]


@pytest.fixture
def bridge(monkeypatch):
    br = b.Bridge(TOKEN)            # loop=None: runs on the test's loop
    monkeypatch.setattr(b, "_bridge", br)

    async def no_sleep(_s):
        pass

    monkeypatch.setattr(b, "_sleep", no_sleep)
    monkeypatch.setattr(b, "CONNECT_WAIT_S", 0.05)
    yield br


async def connect(br, responder=None, browser="Chrome", token=TOKEN):
    conn = FakeConn(responder)
    conn.inbox.put_nowait(json.dumps({"type": "hello", "token": token, "browser": browser, "version": "1.0.0"}))
    task = asyncio.ensure_future(br.handle(conn))
    for _ in range(50):
        if any(c.conn is conn for c in br.clients) or task.done():
            break
        await asyncio.sleep(0)
    conn.task = task
    return conn


def replies(table):
    """Responder from {op: reply}; ready_state defaults to 'complete'."""
    def responder(msg):
        if msg["op"] in table:
            r = table[msg["op"]]
            return r(msg) if callable(r) else r
        return ("ok", "complete") if msg["op"] == "ready_state" else ("ok", None)
    return responder


def text(res):
    return res["content"][0]["text"]


# -- security ---------------------------------------------------------------------
async def test_hello_with_right_token_is_accepted(bridge):
    conn = await connect(bridge)
    assert conn.sent[0] == {"type": "hello", "ok": True}
    assert bridge.target().conn is conn and bridge.target().browser == "Chrome"


async def test_bad_token_is_rejected_and_closed(bridge):
    conn = await connect(bridge, token="wrong")
    await conn.task
    assert conn.closed == b.AUTH_CLOSE_CODE
    assert conn.sent[0]["ok"] is False
    assert bridge.clients == []


async def test_first_message_must_be_hello(bridge):
    conn = FakeConn()
    conn.inbox.put_nowait(json.dumps({"id": 1, "ok": True, "result": None}))
    await bridge.handle(conn)
    assert conn.closed == b.AUTH_CLOSE_CODE and bridge.clients == []

    conn = FakeConn()
    conn.inbox.put_nowait("not json")
    await bridge.handle(conn)
    assert conn.closed == b.AUTH_CLOSE_CODE


async def test_silent_client_times_out_of_auth(bridge, monkeypatch):
    monkeypatch.setattr(b, "AUTH_TIMEOUT_S", 0.01)
    conn = FakeConn()
    await bridge.handle(conn)
    assert conn.closed == b.AUTH_CLOSE_CODE


def test_origin_gate(monkeypatch):
    monkeypatch.delenv("VERONICA_BROWSER_EXTENSION_ID", raising=False)
    assert b.origin_allowed(f"chrome-extension://{b.EXTENSION_ID}")
    assert not b.origin_allowed("chrome-extension://abcdefghijklmnopabcdefghijklmnop")
    assert not b.origin_allowed("https://evil.example")
    assert not b.origin_allowed(None) and not b.origin_allowed("")
    monkeypatch.setenv("VERONICA_BROWSER_EXTENSION_ID", "abcdefghijklmnopabcdefghijklmnop")
    assert b.origin_allowed("chrome-extension://abcdefghijklmnopabcdefghijklmnop")


def test_extension_id_matches_manifest_key():
    import base64
    import hashlib
    manifest = json.loads((b.EXTENSION_DIR / "manifest.json").read_text())
    digest = hashlib.sha256(base64.b64decode(manifest["key"])).hexdigest()[:32]
    assert "".join(chr(ord("a") + int(c, 16)) for c in digest) == b.EXTENSION_ID
    assert {"tabs", "scripting", "storage", "alarms"} <= set(manifest["permissions"])


def test_pairing_token_created_once(tmp_path, monkeypatch):
    monkeypatch.setenv("VERONICA_HOME", str(tmp_path))
    tok = b.pairing_token()
    assert len(tok) >= 24
    assert (tmp_path / "browser_token").read_text().strip() == tok
    assert b.pairing_token() == tok


def test_bridge_port_env(monkeypatch):
    monkeypatch.delenv("VERONICA_BROWSER_PORT", raising=False)
    assert b.bridge_port() == 8765
    monkeypatch.setenv("VERONICA_BROWSER_PORT", "9123")
    assert b.bridge_port() == 9123
    monkeypatch.setenv("VERONICA_BROWSER_PORT", "nope")
    assert b.bridge_port() == 8765


# -- no browser / timeouts ---------------------------------------------------------
async def test_no_browser_connected_is_a_clear_error(bridge):
    res = await b.browser_read.handler({})
    assert res.get("is_error")
    assert "Install the Veronica extension in Chrome or Edge" in text(res)
    assert "Load unpacked" in text(res)


async def test_request_times_out(bridge, monkeypatch):
    monkeypatch.setattr(b, "TIMEOUT_S", 0.05)
    await connect(bridge, responder=lambda msg: None)
    res = await b.browser_tabs.handler({})
    assert res.get("is_error") and "didn't answer" in text(res)


async def test_disconnect_fails_pending_request(bridge):
    holder = {}

    def responder(msg):
        holder["conn"].drop()

    conn = await connect(bridge, responder=responder)
    holder["conn"] = conn
    res = await b.browser_tabs.handler({})
    assert res.get("is_error") and "disconnected" in text(res)
    await conn.task
    assert bridge.clients == []


async def test_bridge_listen_error_is_reported(bridge):
    bridge.error = "Veronica couldn't listen on 127.0.0.1:8765"
    res = await b.browser_read.handler({})
    assert res.get("is_error") and "couldn't listen" in text(res)


async def test_extension_error_is_passed_through(bridge):
    await connect(bridge, replies({"read": ("err", "Veronica can't access this page")}))
    res = await b.browser_read.handler({})
    assert res.get("is_error") and text(res) == "Veronica can't access this page"


# -- target selection -----------------------------------------------------------
async def test_most_recently_focused_browser_wins(bridge):
    chrome = await connect(bridge, browser="Chrome")
    edge = await connect(bridge, browser="Edge")
    assert bridge.target().conn is edge                  # last connected
    chrome.inbox.put_nowait(json.dumps({"type": "focus"}))
    await asyncio.sleep(0.01)
    assert bridge.target().conn is chrome
    await b.browser_tabs.handler({})
    assert chrome.requests("tabs") and not edge.requests("tabs")


async def test_ping_gets_pong(bridge):
    conn = await connect(bridge)
    conn.inbox.put_nowait(json.dumps({"type": "ping"}))
    await asyncio.sleep(0.01)
    assert {"type": "pong"} in conn.sent


# -- request/response mapping per tool -------------------------------------------
async def test_tabs_lists_and_marks_current(bridge):
    conn = await connect(bridge, replies({"tabs": ("ok", {"browser": "Edge", "tabs": [
        {"title": "GitHub", "url": "https://github.com", "active": False},
        {"title": "Docs", "url": "https://docs.example", "active": True}]})}))
    res = await b.browser_tabs.handler({})
    assert text(res) == "1. GitHub — https://github.com\n2. * Docs — https://docs.example"
    assert conn.requests("tabs")[0]["op"] == "tabs"


async def test_tabs_empty(bridge):
    await connect(bridge, replies({"tabs": ("ok", {"tabs": []})}))
    assert text(await b.browser_tabs.handler({})) == "No tabs."


async def test_open_rejects_non_http(bridge):
    conn = await connect(bridge)
    res = await b.browser_open.handler({"url": "file:///C:/Windows/win.ini"})
    assert res.get("is_error") and "http" in text(res)
    assert conn.requests() == []


async def test_open_sends_url_and_new_tab(bridge):
    conn = await connect(bridge)
    res = await b.browser_open.handler({"url": "https://example.com", "new_tab": False})
    assert text(res) == "Opened https://example.com"
    req = conn.requests("open")[0]
    assert req["url"] == "https://example.com" and req["new_tab"] is False
    await b.browser_open.handler({"url": "https://example.com"})
    assert conn.requests("open")[1]["new_tab"] is True


async def test_read_caps_and_truncates(bridge):
    await connect(bridge, replies({"read": ("ok", {"title": "T", "url": "https://x", "text": "a " * 5000})}))
    res = await b.browser_read.handler({"max_chars": 100})
    t = text(res)
    assert t.startswith("T\nhttps://x\n")
    assert t.endswith("…[truncated]")
    assert len(t) < 200


async def test_read_collapses_whitespace_in_title(bridge):
    await connect(bridge, replies({"read": ("ok", {"title": "A\nB", "url": "https://x", "text": "hi\n\n there"})}))
    assert text(await b.browser_read.handler({})) == "A B\nhttps://x\nhi there"


async def test_read_tolerates_garbage_result(bridge):
    await connect(bridge, replies({"read": ("ok", "not a dict")}))
    res = await b.browser_read.handler({})
    assert not res.get("is_error")


async def test_find_returns_numbered_lines(bridge):
    conn = await connect(bridge, replies({"find": ("ok", {"lines": [[3, "Pricing plans"], [9, "See pricing"]]})}))
    res = await b.browser_find.handler({"text": "pricing"})
    assert text(res) == "3: Pricing plans\n9: See pricing"
    req = conn.requests("find")[0]
    assert req["text"] == "pricing" and req["max_lines"] == b.FIND_MAX_LINES


async def test_find_caps_lines_and_handles_none(bridge):
    many = [[i, "x" * 500] for i in range(50)]
    await connect(bridge, replies({"find": ("ok", {"lines": many})}))
    lines = text(await b.browser_find.handler({"text": "x"})).split("\n")
    assert len(lines) == b.FIND_MAX_LINES and all(len(ln) < 200 for ln in lines)
    assert (await b.browser_find.handler({"text": " "})).get("is_error")


async def test_find_not_found(bridge):
    await connect(bridge, replies({"find": ("ok", {"lines": []})}))
    assert text(await b.browser_find.handler({"text": "zzz"})) == "not found"


async def test_click_reports_element(bridge):
    conn = await connect(bridge, replies({"click": ("ok", {"clicked": "BUTTON Log in"})}))
    res = await b.browser_click.handler({"target": "Log in"})
    assert text(res) == "Clicked BUTTON Log in"
    assert conn.requests("click")[0]["target"] == "Log in"


async def test_click_no_match(bridge):
    await connect(bridge, replies({"click": ("ok", {"clicked": None})}))
    res = await b.browser_click.handler({"target": "Nope"})
    assert res.get("is_error") and "no element matching" in text(res)


async def test_type_sends_fields(bridge):
    conn = await connect(bridge, replies({"type": ("ok", {"typed": "INPUT search"})}))
    res = await b.browser_type.handler({"target": "search", "text": "hello", "submit": True})
    assert text(res) == "Typed into INPUT search"
    req = conn.requests("type")[0]
    assert (req["target"], req["text"], req["submit"]) == ("search", "hello", True)


async def test_type_no_match(bridge):
    await connect(bridge, replies({"type": ("ok", {"typed": None})}))
    res = await b.browser_type.handler({"target": "nope", "text": "x"})
    assert res.get("is_error") and "no field matching" in text(res)


async def test_scroll_and_back(bridge):
    conn = await connect(bridge, replies({"scroll": ("ok", "ok"), "back": ("ok", "ok")}))
    assert text(await b.browser_scroll.handler({"direction": "bottom"})) == "Scrolled bottom"
    assert conn.requests("scroll")[0]["direction"] == "bottom"
    assert text(await b.browser_back.handler({})) == "Went back"
    assert conn.requests("back")
    res = await b.browser_scroll.handler({"direction": "sideways"})
    assert res.get("is_error")
    assert len(conn.requests("scroll")) == 1


# -- waiting for navigation ---------------------------------------------------------
def ready_sequence(states):
    states = list(states)

    def reply(_msg):
        return states.pop(0) if states else ("ok", "complete")
    return reply


async def test_open_waits_for_page_to_load(bridge):
    conn = await connect(bridge, replies({"ready_state": ready_sequence([("ok", "loading"), ("ok", "loading"), ("ok", "complete")])}))
    res = await b.browser_open.handler({"url": "https://example.com"})
    assert text(res) == "Opened https://example.com"
    assert len(conn.requests("ready_state")) == 3


async def test_click_waits_for_page_to_load(bridge):
    conn = await connect(bridge, replies({"click": ("ok", {"clicked": "A Docs"}),
                                          "ready_state": ready_sequence([("ok", "loading"), ("ok", "complete")])}))
    assert text(await b.browser_click.handler({"target": "Docs"})) == "Clicked A Docs"
    assert len(conn.requests("ready_state")) == 2


async def test_back_and_submit_wait_for_page_to_load(bridge):
    conn = await connect(bridge, replies({"back": ("ok", "ok"), "type": ("ok", {"typed": "INPUT q"})}))
    assert text(await b.browser_back.handler({})) == "Went back"
    assert len(conn.requests("ready_state")) == 1
    await b.browser_type.handler({"target": "q", "text": "x", "submit": True})
    assert len(conn.requests("ready_state")) == 2
    await b.browser_type.handler({"target": "q", "text": "x", "submit": False})
    assert len(conn.requests("ready_state")) == 2      # no submit: nothing to wait for


async def test_nav_wait_error_does_not_fail_tool(bridge):
    conn = await connect(bridge, replies({"ready_state": ready_sequence([("ok", "loading"), ("err", "No open browser tab."), ("ok", "complete")])}))
    res = await b.browser_open.handler({"url": "https://example.com"})
    assert not res.get("is_error") and text(res) == "Opened https://example.com"
    assert len(conn.requests("ready_state")) == 2


async def test_nav_wait_gives_up_after_deadline(bridge, monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(b, "_sleep", fake_sleep)
    conn = await connect(bridge, replies({"ready_state": ("ok", "loading")}))
    assert text(await b.browser_open.handler({"url": "https://example.com"})) == "Opened https://example.com"
    assert len(conn.requests("ready_state")) == int(b.NAV_WAIT_S / b.NAV_POLL_S)
    assert slept and all(s == b.NAV_POLL_S for s in slept)


# -- the real server over loopback ------------------------------------------------
def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_real_websocket_server(tmp_path, monkeypatch):
    ws_client = pytest.importorskip("websockets.asyncio.client")
    from websockets.exceptions import InvalidStatus
    monkeypatch.setenv("VERONICA_HOME", str(tmp_path))
    monkeypatch.setattr(b, "_bridge", None)
    port = _free_port()
    br = b.start_bridge(port)
    try:
        assert br.error is None and br.loop is not None
        tok = (tmp_path / "browser_token").read_text().strip()
        url = f"ws://127.0.0.1:{port}"

        with pytest.raises(InvalidStatus):
            await ws_client.connect(url, origin="https://evil.example")

        async with ws_client.connect(url, origin=f"chrome-extension://{b.EXTENSION_ID}") as ws:
            await ws.send(json.dumps({"type": "hello", "token": "wrong"}))
            assert json.loads(await ws.recv())["ok"] is False
            await ws.wait_closed()
            assert ws.close_code == b.AUTH_CLOSE_CODE

        async with ws_client.connect(url, origin=f"chrome-extension://{b.EXTENSION_ID}") as ws:
            await ws.send(json.dumps({"type": "hello", "token": tok, "browser": "Edge"}))
            assert json.loads(await ws.recv()) == {"type": "hello", "ok": True}

            async def extension():
                req = json.loads(await ws.recv())
                assert req["op"] == "read"
                await ws.send(json.dumps({"id": req["id"], "ok": True,
                                          "result": {"title": "T", "url": "https://x", "text": "hello"}}))

            ext = asyncio.ensure_future(extension())
            res = await b.browser_read.handler({})     # crosses from this loop to the bridge's
            await ext
            assert text(res) == "T\nhttps://x\nhello"
    finally:
        b.stop_bridge()
