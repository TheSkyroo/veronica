import asyncio
import json
import threading
from pathlib import Path

import pytest

from veronica.brain import gateclient
from veronica.brain.gateclient import ask_gate


@pytest.fixture
def sock(tmp_path, monkeypatch):
    """AF_UNIX paths are capped at ~104 bytes and pytest's tmp_path on macOS
    is longer, so bind relative to it."""
    monkeypatch.chdir(tmp_path)
    return Path("g.sock")


def _serve_once(path, reply):
    """A one-shot fake gate on a thread so the sync client has something to
    talk to. Returns once it is listening; gives up after two seconds."""
    ready = threading.Event()

    async def main():
        got = {}

        async def h(r, w):
            got["req"] = json.loads(await r.readline())
            w.write((json.dumps(reply) + "\n").encode())
            await w.drain()
            w.close()

        srv = await asyncio.start_unix_server(h, path=str(path))
        async with srv:
            ready.set()
            for _ in range(200):
                if "req" in got:
                    break
                await asyncio.sleep(0.01)
        return got.get("req")

    box = {}
    t = threading.Thread(target=lambda: box.update(req=asyncio.run(main())))
    t.start()
    assert ready.wait(2)
    return t, box


def test_ask_gate_allow(sock):
    t, box = _serve_once(sock, {"allow": True, "kind": "approved", "reason": ""})
    d = ask_gate("mcp__mac__clipboard_write", {"text": "hi"}, origin="mcp", backend="codex", sock=str(sock))
    t.join(2)
    assert d.allow and d.kind == "approved"
    assert box["req"] == {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "hi"}, "origin": "mcp", "backend": "codex"}


def test_ask_gate_fails_closed_without_socket(sock):
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex", sock="missing.sock")
    assert not d.allow and d.kind == "denied" and "gate isn't reachable" in d.message


def test_ask_gate_fails_closed_without_env(monkeypatch):
    monkeypatch.delenv("VERONICA_GATE_SOCK", raising=False)
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex")
    assert not d.allow and d.kind == "denied"


def test_ask_gate_reads_env(sock, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE_SOCK", str(sock))
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
            while not stop.is_set():
                await asyncio.sleep(0.01)
            w.close()

        srv = await asyncio.start_unix_server(h, path=str(path))
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
                     sock=str(sock), timeout=0.2)
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
        d = ask_gate("Bash", {"command": "rm -rf x"}, origin="hook", backend="copilot", sock=str(sock))
    finally:
        stop.set()
        t.join(2)
    assert not d.allow and d.message == gateclient.NO_ANSWER


def test_ask_gate_timeout_env_overrides_the_budget(sock, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE_TIMEOUT_S", "0.2")
    monkeypatch.setattr(gateclient, "GATE_ANSWER_BUDGET_S", 30.0)
    t, stop = _serve_silently(sock)
    try:
        d = ask_gate("Bash", {"command": "rm -rf x"}, origin="hook", backend="copilot", sock=str(sock))
    finally:
        stop.set()
        t.join(2)
    assert not d.allow and d.message == gateclient.NO_ANSWER
