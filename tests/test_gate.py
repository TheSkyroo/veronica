import time
from pathlib import Path

import pytest

from veronica.brain.gate import ToolGate
from veronica.config import Settings
from veronica.orchestrator import ConfirmResult
from veronica.tools.computer_events import Front

FINDER = Front(app="Finder", bundle_id="com.apple.finder", window_title="Desktop", pid=1)


def make(answers, *, front=FINDER, now=None, **settings):
    calls, cards = [], []

    async def confirm(summary, detail=""):
        calls.append((summary, detail))
        a = answers.pop(0)
        return a if isinstance(a, ConfirmResult) else ConfirmResult("approved" if a else "denied")

    clock = (lambda: now[0]) if now is not None else time.monotonic
    g = ToolGate(Settings(**settings), confirm, on_tool=lambda s, d: cards.append((s, d)),
                 frontmost=lambda: front, clock=clock)
    return g, calls, cards


async def test_allow_class_is_auto_without_asking():
    g, calls, cards = make([])
    d = await g.decide("mcp__mac__volume_get", {})
    assert d.allow and d.kind == "auto" and calls == [] and cards == [("volume_get", "auto")]


async def test_confirm_class_asks_and_yes_allows():
    g, calls, _ = make([True])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert d.allow and d.kind == "approved" and calls[0][0] == "Copy to clipboard: hi"


async def test_no_denies_with_user_declined():
    g, _, _ = make([False])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert not d.allow and d.kind == "denied" and d.message == "user declined"


async def test_other_answer_becomes_redirect():
    g, _, _ = make([ConfirmResult("other", "open it in the other profile")])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert not d.allow and d.kind == "other"
    assert d.message == "user declined and said: 'open it in the other profile'"
    assert g.pending_redirect == "open it in the other profile"


async def test_screencapture_bash_is_redirected():
    g, calls, _ = make([])
    d = await g.decide("Bash", {"command": "screencapture x.png"})
    assert not d.allow and d.kind == "redirect" and "screenshot tool" in d.message and calls == []


async def test_trust_window_allows_second_click_same_app():
    now = [100.0]
    g, calls, cards = make([True], now=now, computer_trust_s=90)
    assert (await g.decide("mcp__computer__computer_click", {"x": 1, "y": 2})).allow
    now[0] = 130.0
    d = await g.decide("mcp__computer__computer_click", {"x": 3, "y": 4})
    assert d.allow and d.kind == "trusted" and len(calls) == 1 and cards[-1] == ("Click (3, 4)", "auto")
    g.clear_trust()
    now[0] = 131.0
    # a second confirm is needed after clear_trust; answers list is empty -> IndexError proves it asked
    with pytest.raises(IndexError):
        await g.decide("mcp__computer__computer_click", {"x": 5, "y": 6})
    assert calls[-1][0] == "Click (5, 6)"


async def test_preapproval_covers_first_confirm_call_only():
    now = [10.0]
    g, calls, cards = make([True], now=now)
    g.begin_turn(7)
    g.preapprove(7, until=30.0)
    d1 = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert d1.allow and d1.kind == "preapproved" and calls == [] and cards[-1] == ("Copy to clipboard: hi", "preapproved")
    d2 = await g.decide("mcp__mac__clipboard_write", {"text": "yo"})
    assert d2.allow and d2.kind == "approved" and len(calls) == 1


async def test_preapproval_never_for_always_confirm():
    now = [10.0]
    g, calls, _ = make([True], now=now)
    g.begin_turn(1); g.preapprove(1, until=30.0)
    d = await g.decide("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"})
    assert d.kind == "approved" and len(calls) == 1


async def test_allowlisted_shortcut_runs_without_asking():
    g, calls, cards = make([], shortcut_allowlist=["Morning", "Pay Rent"])
    d = await g.decide("mcp__mac__run_shortcut", {"name": "morning"})
    assert d.allow and d.kind == "auto" and calls == []
    assert cards[-1] == ("Run the shortcut 'morning'", "auto")


async def test_unlisted_shortcut_asks():
    g, calls, _ = make([True], shortcut_allowlist=["Morning"])
    d = await g.decide("mcp__mac__run_shortcut", {"name": "Wipe Disk"})
    assert d.allow and d.kind == "approved"
    assert calls[0][0] == "Run the shortcut 'Wipe Disk'"


async def test_shortcuts_ask_by_default():
    g, calls, _ = make([True])
    assert (await g.decide("mcp__mac__run_shortcut", {"name": "Morning"})).kind == "approved"
    assert len(calls) == 1


async def test_message_send_is_asked_even_when_preapproved():
    now = [10.0]
    g, calls, _ = make([True], now=now)
    g.begin_turn(1); g.preapprove(1, until=30.0)
    d = await g.decide("mcp__pim__message_send", {"to": "Priya", "body": "on my way"})
    assert d.kind == "approved" and calls[0][0] == "Message Priya: on my way"


# -- GateServer: the socket front for out-of-process callers ------------------
import asyncio
import json

from veronica.brain.gate import GateServer


@pytest.fixture
def sock(tmp_path, monkeypatch):
    """AF_UNIX paths are capped at ~104 bytes and pytest's tmp_path on macOS
    is longer, so bind relative to it."""
    monkeypatch.chdir(tmp_path)
    return Path("gate.sock")


async def _roundtrip(path, req):
    r, w = await asyncio.open_unix_connection(str(path))
    w.write((json.dumps(req) + "\n").encode())
    await w.drain()
    line = await r.readline()
    w.close()
    await w.wait_closed()
    return json.loads(line)


async def test_gate_server_allow_and_deny(sock):
    g, _, _ = make([True, False])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        assert (sock).stat().st_mode & 0o777 == 0o600
        ok = await _roundtrip(sock,
                              {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "a"}, "origin": "mcp", "backend": "codex"})
        assert ok == {"allow": True, "kind": "approved", "reason": ""}
        no = await _roundtrip(sock,
                              {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "b"}, "origin": "hook", "backend": "codex"})
        assert no == {"allow": False, "kind": "denied", "reason": "user declined"}
    finally:
        await srv.stop()
    assert not (sock).exists()


async def test_gate_server_allow_class_needs_no_confirm(sock):
    g, calls, _ = make([])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        ok = await _roundtrip(sock,
                              {"v": 1, "tool": "mcp__mac__volume_get", "input": {}, "origin": "mcp", "backend": "codex"})
        assert ok == {"allow": True, "kind": "auto", "reason": ""} and calls == []
    finally:
        await srv.stop()


async def test_gate_server_malformed_request_is_denied(sock):
    g, _, _ = make([])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        r, w = await asyncio.open_unix_connection(str(sock))
        w.write(b"not json\n")
        await w.drain()
        assert json.loads(await r.readline()) == {"allow": False, "kind": "denied", "reason": "bad request"}
        w.close()
        await w.wait_closed()
    finally:
        await srv.stop()


async def test_gate_server_serializes_confirms(sock):
    order = []

    async def confirm(summary, detail=""):
        order.append(("start", summary))
        await asyncio.sleep(0.05)
        order.append(("end", summary))
        return True

    g = ToolGate(Settings(), confirm)
    srv = GateServer(g, sock)
    await srv.start()
    try:
        await asyncio.gather(
            _roundtrip(sock, {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "A"}, "origin": "mcp", "backend": "x"}),
            _roundtrip(sock, {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "B"}, "origin": "mcp", "backend": "x"}),
        )
    finally:
        await srv.stop()
    assert [o[0] for o in order] == ["start", "end", "start", "end"]
