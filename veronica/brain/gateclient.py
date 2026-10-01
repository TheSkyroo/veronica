"""Sync client for the gate (GateServer), used from processes Veronica
spawns (tools.serve, brain.hook). The gate listens on a loopback port;
where, and the token it wants first, are in its endpoint file, named by
`gate=` or $VERONICA_GATE. Fails closed: no endpoint file, a bad one, no
connection, no answer, bad JSON -> deny.

Two questions: `ask_gate` is "may I" (what brain.hook needs — the CLI runs
its own tool itself), `call_gate` is "may I, and if so run it for me" (what
tools.serve needs — our tools must run in the app process, which owns the
screen, the clipboard, the Outlook session and the timers)."""
import json
import os
import socket
from pathlib import Path

from veronica.brain.base import Decision

UNREACHABLE = "Veronica's gate isn't reachable"
# The gate only ever listens here; an endpoint file naming anything else
# is not ours to trust.
LOOPBACK = "127.0.0.1"
NO_ANSWER = "Veronica didn't get an answer in time"

# The per-hook timeout we configure in every CLI that takes one, and the
# budget the hook gives the user to answer. A hook that outlives the CLI's
# timeout is not just slow: Copilot FAILS OPEN on one, so the command would
# run ungated. The budget is well under the timeout, so the hook always
# answers first — with a deny when nobody said yes.
HOOK_TIMEOUT_S = 60
GATE_ANSWER_BUDGET_S = 45.0
# A `call` waits for the same confirm AND then for the tool to run, and some
# of ours are slow on purpose (a `selection` screenshot gives the user a
# minute to drag a rectangle). No CLI times these out the way hooks are
# timed out, so the budget is just long enough to cover the slowest tool.
GATE_CALL_BUDGET_S = GATE_ANSWER_BUDGET_S + 90.0


def read_endpoint(path: str | os.PathLike) -> tuple[int, str]:
    """(port, token) from the gate's endpoint file. Raises on a missing,
    unreadable or malformed one."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    port, token = data["port"], data["token"]
    if not isinstance(port, int) or not 0 < port < 65536 or not isinstance(token, str) or not token:
        raise ValueError("bad gate endpoint")
    return port, token


def _exchange(req: dict, gate: str | None, timeout: float | None, budget: float) -> dict:
    """Token line, one request line out, one response line back. Raises on
    anything the caller should treat as a deny."""
    path = gate or os.environ.get("VERONICA_GATE", "")
    if not path:
        raise FileNotFoundError("no gate endpoint")
    port, token = read_endpoint(path)
    if timeout is None:
        env = os.environ.get("VERONICA_GATE_TIMEOUT_S")
        timeout = float(env) if env else budget
    # The gate answers a little before this runs out (GateServer._decide),
    # so a slow answer is an explicit deny rather than our timeout.
    req = {**req, "budget": timeout}
    with socket.create_connection((LOOPBACK, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall((token + "\n" + json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf)


def _decision(resp: dict) -> Decision:
    return Decision(bool(resp["allow"]), resp.get("kind", "denied"), str(resp.get("reason", "")))


def ask_gate(
    tool: str,
    input: dict,
    *,
    origin: str,
    backend: str,
    gate: str | None = None,
    timeout: float | None = None,
) -> Decision:
    """Ask the gate (GateServer) about one tool call. `gate` (the endpoint
    file) defaults to $VERONICA_GATE, `timeout` to $VERONICA_GATE_TIMEOUT_S,
    else GATE_ANSWER_BUDGET_S."""
    req = {"v": 1, "tool": tool, "input": input, "origin": origin, "backend": backend}
    try:
        return _decision(_exchange(req, gate, timeout, GATE_ANSWER_BUDGET_S))
    except TimeoutError:
        return Decision(False, "denied", NO_ANSWER)
    except Exception:
        return Decision(False, "denied", UNREACHABLE)


def call_gate(
    tool: str,
    input: dict,
    *,
    origin: str,
    backend: str,
    gate: str | None = None,
    timeout: float | None = None,
) -> tuple[Decision, list[dict], bool]:
    """Ask the gate to decide AND, if it allows, run the tool in the app
    process. Returns (decision, MCP content blocks, is_error); the blocks
    are empty on a deny, and the caller turns the decision into the refusal
    the brain sees. Timeout defaults to $VERONICA_GATE_TIMEOUT_S, else
    GATE_CALL_BUDGET_S."""
    req = {"v": 1, "op": "call", "tool": tool, "input": input, "origin": origin, "backend": backend}
    try:
        resp = _exchange(req, gate, timeout, GATE_CALL_BUDGET_S)
        d = _decision(resp)
    except TimeoutError:
        return Decision(False, "denied", NO_ANSWER), [], True
    except Exception:
        return Decision(False, "denied", UNREACHABLE), [], True
    if not d.allow:
        return d, [], True
    content = resp.get("content")
    if not isinstance(content, list):
        return Decision(False, "denied", UNREACHABLE), [], True
    return d, content, bool(resp.get("is_error"))
