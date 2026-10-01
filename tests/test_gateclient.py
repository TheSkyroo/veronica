import asyncio
import json
import threading

import pytest

from veronica.brain import gateclient
from veronica.brain.gateclient import ask_gate


TOKEN = "s3cret-token"


@pytest.fixture
def sock(tmp_path):
    """The gate's endpoint file (name kept from the Unix-socket days)."""
    return tmp_path / "gate.json"


async def _listen(handler, path):
    """A loopback server like GateServer's, with its endpoint file written."""
    srv = await asyncio.start_server(handler, "127.0.0.1", 0)
    path.write_text(json.dumps({"v": 1, "port": srv.sockets[0].getsockname()[1], "token": TOKEN, "pid": 1}))
    return srv


def _serve_once(path, reply):
    """A one-shot fake gate on a thread so the sync client has something to
    talk to. Returns once it is listening; gives up after two seconds. It
    records the token line the client sent first."""
    ready = threading.Event()

    async def main():
        got = {}

        async def h(r, w):
            got["token"] = (await r.readline()).decode().strip()
            got["req"] = json.loads(await r.readline())
            w.write((json.dumps(reply) + "\n").encode())
            await w.drain()
            w.close()

        srv = await _listen(h, path)
        async with srv:
            ready.set()
            for _ in range(200):
                if "req" in got:
                    break
                await asyncio.sleep(0.01)
        return got

    box = {}
    t = threading.Thread(target=lambda: box.update(asyncio.run(main())))
    t.start()
    assert ready.wait(2)
    return t, box


def test_ask_gate_allow(sock):
    t, box = _serve_once(sock, {"allow": True, "kind": "approved", "reason": ""})
    d = ask_gate("mcp__system__clipboard_write", {"text": "hi"}, origin="mcp", backend="codex", gate=str(sock))
    t.join(2)
    assert d.allow and d.kind == "approved"
    assert box["token"] == TOKEN
    assert box["req"] == {"v": 1, "tool": "mcp__system__clipboard_write", "input": {"text": "hi"}, "origin": "mcp",
                          "backend": "codex", "budget": gateclient.GATE_ANSWER_BUDGET_S}


def test_ask_gate_fails_closed_without_an_endpoint_file(sock):
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex", gate=str(sock))
    assert not d.allow and d.kind == "denied" and "gate isn't reachable" in d.message


@pytest.mark.parametrize("content", [
    "not json", "{}", '{"port": "80", "token": "t"}', '{"port": 0, "token": "t"}',
    '{"port": 70000, "token": "t"}', '{"port": 8080, "token": ""}', '{"port": 8080}',
])
def test_ask_gate_fails_closed_on_a_bad_endpoint_file(sock, content):
    sock.write_text(content)
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex", gate=str(sock))
    assert not d.allow and d.message == gateclient.UNREACHABLE


def test_ask_gate_fails_closed_when_nothing_listens(sock):
    import socket
    with socket.socket() as s:            # a port that was free a moment ago
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    sock.write_text(json.dumps({"port": port, "token": TOKEN}))
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex", gate=str(sock))
    assert not d.allow and d.message == gateclient.UNREACHABLE


def test_ask_gate_fails_closed_against_a_real_gate_with_the_wrong_token(sock):
    """End to end with GateServer: a stale or forged token is a deny."""
    from veronica.brain.gate import GateServer, ToolGate
    from veronica.config import Settings

    async def confirm(*a, **kw):
        raise AssertionError("must not be asked")

    async def main():
        srv = GateServer(ToolGate(Settings(), confirm), sock)
        await srv.start()
        try:
            data = json.loads(sock.read_text())
            sock.write_text(json.dumps({**data, "token": "forged"}))
            return await asyncio.to_thread(ask_gate, "mcp__system__volume_get", {}, origin="mcp",
                                           backend="codex", gate=str(sock))
        finally:
            await srv.stop()

    d = asyncio.run(main())
    assert not d.allow and d.message == "bad token"


def test_ask_gate_fails_closed_without_env(monkeypatch):
    monkeypatch.delenv("VERONICA_GATE", raising=False)
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex")
    assert not d.allow and d.kind == "denied"


def test_ask_gate_reads_env(sock, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE", str(sock))
    t, _ = _serve_once(sock, {"allow": False, "kind": "denied", "reason": "user declined"})
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex")
    t.join(2)
    assert not d.allow and d.message == "user declined"


def _serve_silently(path):
    """A gate that accepts the connection and never answers — the user is
    taking longer than the budget to say yes or no."""
    ready = threading.Event()
    stop = threading.Event()

    async def main():
        async def h(r, w):
            await r.readline()
            await r.readline()
            while not stop.is_set():
                await asyncio.sleep(0.01)
            w.close()

        srv = await _listen(h, path)
        async with srv:
            ready.set()
            while not stop.is_set():
                await asyncio.sleep(0.01)

    t = threading.Thread(target=lambda: asyncio.run(main()))
    t.start()
    assert ready.wait(2)
    return t, stop


def test_ask_gate_denies_when_the_answer_takes_too_long(sock):
    t, stop = _serve_silently(sock)
    try:
        d = ask_gate("Bash", {"command": "rm -rf x"}, origin="hook", backend="copilot",
                     gate=str(sock), timeout=0.2)
    finally:
        stop.set()
        t.join(2)
    assert not d.allow and d.kind == "denied" and d.message == gateclient.NO_ANSWER


def test_ask_gate_defaults_to_the_answer_budget(sock, monkeypatch):
    """Unset $VERONICA_GATE_TIMEOUT_S: the budget applies anyway, so the hook
    denies before the CLI's own hook timeout lets the tool run ungated."""
    monkeypatch.delenv("VERONICA_GATE_TIMEOUT_S", raising=False)
    monkeypatch.setattr(gateclient, "GATE_ANSWER_BUDGET_S", 0.2)
    assert gateclient.GATE_ANSWER_BUDGET_S < gateclient.HOOK_TIMEOUT_S
    t, stop = _serve_silently(sock)
    try:
        d = ask_gate("Bash", {"command": "rm -rf x"}, origin="hook", backend="copilot", gate=str(sock))
    finally:
        stop.set()
        t.join(2)
    assert not d.allow and d.message == gateclient.NO_ANSWER


def test_ask_gate_timeout_env_overrides_the_budget(sock, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE_TIMEOUT_S", "0.2")
    monkeypatch.setattr(gateclient, "GATE_ANSWER_BUDGET_S", 30.0)
    t, stop = _serve_silently(sock)
    try:
        d = ask_gate("Bash", {"command": "rm -rf x"}, origin="hook", backend="copilot", gate=str(sock))
    finally:
        stop.set()
        t.join(2)
    assert not d.allow and d.message == gateclient.NO_ANSWER


# -- call_gate: decide, then run the tool in the app process ------------------


def test_call_gate_returns_the_tools_content(sock):
    blocks = [{"type": "image", "data": "QUJD", "mimeType": "image/png"},
              {"type": "text", "text": "Screenshot of the screen."}]
    t, box = _serve_once(sock, {"allow": True, "kind": "auto", "reason": "",
                                "content": blocks, "is_error": False})
    d, content, is_error = gateclient.call_gate(
        "mcp__screen__screenshot", {"region": "screen"}, origin="mcp", backend="codex", gate=str(sock))
    t.join(2)
    assert d.allow and content == blocks and not is_error
    assert box["req"] == {"v": 1, "op": "call", "tool": "mcp__screen__screenshot",
                          "input": {"region": "screen"}, "origin": "mcp", "backend": "codex",
                          "budget": gateclient.GATE_CALL_BUDGET_S}


def test_call_gate_deny_has_no_content(sock):
    t, _ = _serve_once(sock, {"allow": False, "kind": "denied", "reason": "user declined"})
    d, content, is_error = gateclient.call_gate(
        "mcp__system__clipboard_write", {"text": "hi"}, origin="mcp", backend="codex", gate=str(sock))
    t.join(2)
    assert not d.allow and d.message == "user declined" and content == [] and is_error


def test_call_gate_fails_closed_without_socket():
    d, content, is_error = gateclient.call_gate(
        "mcp__screen__screenshot", {}, origin="mcp", backend="codex", gate="missing.json")
    assert not d.allow and d.message == gateclient.UNREACHABLE and content == [] and is_error


def test_call_gate_fails_closed_on_an_allow_without_content(sock):
    t, _ = _serve_once(sock, {"allow": True, "kind": "auto", "reason": ""})
    d, content, _ = gateclient.call_gate(
        "mcp__screen__screenshot", {}, origin="mcp", backend="codex", gate=str(sock))
    t.join(2)
    assert not d.allow and content == []


def test_call_gate_budget_covers_the_confirm_and_the_tool():
    assert gateclient.GATE_CALL_BUDGET_S > gateclient.GATE_ANSWER_BUDGET_S


def test_call_gate_denies_when_the_tool_takes_too_long(sock):
    t, stop = _serve_silently(sock)
    try:
        d, content, _ = gateclient.call_gate(
            "mcp__screen__screenshot", {}, origin="mcp", backend="codex", gate=str(sock), timeout=0.2)
    finally:
        stop.set()
        t.join(2)
    assert not d.allow and d.message == gateclient.NO_ANSWER and content == []


def test_the_request_carries_the_callers_own_timeout(sock, monkeypatch):
    """The gate answers before this runs out (GateServer), so a slow answer
    comes back as a spoken, explicit skip instead of a silent timeout."""
    monkeypatch.setenv("VERONICA_GATE_TIMEOUT_S", "7")
    t, box = _serve_once(sock, {"allow": False, "kind": "denied", "reason": "user declined"})
    ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex", gate=str(sock))
    t.join(2)
    assert box["req"]["budget"] == 7.0
