"""Sync client for the gate socket, used from processes Veronica spawns
(tools.serve, brain.hook). Fails closed: no socket, no answer, bad JSON
-> deny."""
import json
import os
import socket

from veronica.brain.base import Decision

UNREACHABLE = "Veronica's gate isn't reachable"


def ask_gate(
    tool: str,
    input: dict,
    *,
    origin: str,
    backend: str,
    sock: str | None = None,
    timeout: float | None = None,
) -> Decision:
    """Ask the gate (GateServer) about one tool call. `sock` defaults to
    $VERONICA_GATE_SOCK, `timeout` to $VERONICA_GATE_TIMEOUT_S (unset =
    wait as long as the user takes to answer)."""
    path = sock or os.environ.get("VERONICA_GATE_SOCK", "")
    if timeout is None:
        env = os.environ.get("VERONICA_GATE_TIMEOUT_S")
        timeout = float(env) if env else None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            req = {"v": 1, "tool": tool, "input": input, "origin": origin, "backend": backend}
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        resp = json.loads(buf)
        return Decision(bool(resp["allow"]), resp.get("kind", "denied"), str(resp.get("reason", "")))
    except Exception:
        return Decision(False, "denied", UNREACHABLE)
