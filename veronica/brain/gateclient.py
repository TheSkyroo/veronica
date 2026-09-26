"""Sync client for the gate socket, used from processes Veronica spawns
(tools.serve, brain.hook). Fails closed: no socket, no answer, bad JSON
-> deny.

Two questions: `ask_gate` is "may I" (what brain.hook needs — the CLI runs
its own tool itself), `call_gate` is "may I, and if so run it for me" (what
tools.serve needs — our tools must run in the app process or macOS hands
their TCC grants to the wrong binary)."""
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
# A `call` waits for the same confirm AND then for the tool to run, and some
# of ours are slow on purpose (a `selection` screenshot gives the user a
# minute to drag a rectangle). No CLI times these out the way hooks are
# timed out, so the budget is just long enough to cover the slowest tool.
GATE_CALL_BUDGET_S = GATE_ANSWER_BUDGET_S + 90.0


def _exchange(req: dict, sock: str | None, timeout: float | None, budget: float) -> dict:
    """One request line out, one response line back. Raises on anything the
    caller should treat as a deny."""
    path = sock or os.environ.get("VERONICA_GATE_SOCK", "")
    if timeout is None:
        env = os.environ.get("VERONICA_GATE_TIMEOUT_S")
        timeout = float(env) if env else budget
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
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
    sock: str | None = None,
    timeout: float | None = None,
) -> Decision:
    """Ask the gate (GateServer) about one tool call. `sock` defaults to
    $VERONICA_GATE_SOCK, `timeout` to $VERONICA_GATE_TIMEOUT_S, else
    GATE_ANSWER_BUDGET_S."""
    req = {"v": 1, "tool": tool, "input": input, "origin": origin, "backend": backend}
    try:
        return _decision(_exchange(req, sock, timeout, GATE_ANSWER_BUDGET_S))
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
    sock: str | None = None,
    timeout: float | None = None,
) -> tuple[Decision, list[dict], bool]:
    """Ask the gate to decide AND, if it allows, run the tool in the app
    process. Returns (decision, MCP content blocks, is_error); the blocks
    are empty on a deny, and the caller turns the decision into the refusal
    the brain sees. Timeout defaults to $VERONICA_GATE_TIMEOUT_S, else
    GATE_CALL_BUDGET_S."""
    req = {"v": 1, "op": "call", "tool": tool, "input": input, "origin": origin, "backend": backend}
    try:
        resp = _exchange(req, sock, timeout, GATE_CALL_BUDGET_S)
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
