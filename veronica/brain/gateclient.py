"""Sync client for the gate socket, used from processes Veronica spawns
(tools.serve, brain.hook). Fails closed: no socket, no answer, bad JSON
-> deny."""
import json
import os
import socket

from veronica.brain.base import Decision

UNREACHABLE = "Veronica's gate isn't reachable"
NO_ANSWER = "Veronica didn't get an answer in time"

# The per-hook timeout we configure in every CLI that takes one, and the
# budget the hook gives the user to answer. A hook that outlives the CLI's
# timeout is not just slow: Copilot FAILS OPEN on one, so the command would
# run ungated. The budget is well under the timeout, so the hook always
# answers first — with a deny when nobody said yes.
HOOK_TIMEOUT_S = 60
GATE_ANSWER_BUDGET_S = 45.0


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
    $VERONICA_GATE_SOCK, `timeout` to $VERONICA_GATE_TIMEOUT_S, else
    GATE_ANSWER_BUDGET_S."""
    path = sock or os.environ.get("VERONICA_GATE_SOCK", "")
    if timeout is None:
        env = os.environ.get("VERONICA_GATE_TIMEOUT_S")
        timeout = float(env) if env else GATE_ANSWER_BUDGET_S
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
    except TimeoutError:
        return Decision(False, "denied", NO_ANSWER)
    except Exception:
        return Decision(False, "denied", UNREACHABLE)
