# Veronica Brains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Veronica's brain a swappable backend — Claude (default), Codex, Antigravity, Copilot, Qwen — switchable by voice/menu/settings, with every backend going through the same confirm gate, and automatic failover to the next available brain when one hits its usage limit.

**Architecture:** The confirm/trust/pre-approval logic leaves the Claude-specific class and becomes `ToolGate`; the Claude backend calls it in-process, external CLIs reach it out-of-process over a Unix socket (`GateServer` ↔ `gateclient`) from two entry points: `veronica.tools.serve` (our MCP servers over stdio, gated per call) and `veronica.brain.hook` (the CLI's own shell/edit tools, via each CLI's pre-tool hook). External backends are subprocess-per-turn adapters (`CliBrain`) that parse the CLI's JSON stream into sentences and HUD tool cards. `BrainSwitcher` owns the active backend, availability checks, and failover.

**Tech Stack:** Python 3.12, `uv`, `claude-agent-sdk` (Claude), the `mcp` package (`mcp.server.stdio`), asyncio subprocesses, the vendor CLIs `codex` 0.155.1 / `agy` 1.2.7 / `copilot` 1.0.86 / `qwen` 0.24.0 (installed globally on the dev Mac), pytest with the `live` marker.

**Spec:** `docs/superpowers/specs/2026-09-19-veronica-brains-design.md`

## Global Constraints

- Every backend uses the vendor CLI's own login. **No API key**, ever. Claude stays on the Claude Code subscription via `claude-agent-sdk`.
- Confirm-gate stays strict: only `policy.classify()` and the designed exemptions (computer trust window, one-shot pre-approval) may auto-allow. Never `allowed_tools`. Never run an external CLI in auto-approve mode without our hook as the gate **and** the canary watching that the hook fires.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits. Verify with `git log -1 --format=%B` after every commit.
- `uv run pytest -q` stays hermetic: no real CLI spawned, no socket outside `tmp_path`. Real-CLI tests carry `@pytest.mark.live` (deselected by default via `addopts = "-m 'not live'"`).
- Python 3.12; match existing patterns (`_ok`/`_err` in tools, `Settings` + `EDITABLE_SETTINGS` + `SETTING_SECTIONS` + hand-listed `settingRow(...)` in `settings.js`, local intents in `brain/intents.py`, spoken copy short).
- Branch: git worktree `brains` off `master` (`git worktree add ../veronica-brains -b brains master`). Run tests inside the worktree with `uv run pytest -q` (uv resolves the worktree's own `.venv`; run `uv sync` once there).
- The `on_tool(summary, decision)` HUD card decisions in use today: `auto`, `ask`, `allowed`, `declined`, `redirected`, `preapproved`. Don't invent new ones except `limit` (Task 7).

---

## File map

| File | Responsibility |
|---|---|
| `veronica/brain/base.py` (new) | `Brain` protocol, `Decision` dataclass |
| `veronica/brain/gate.py` (new) | `ToolGate` (all confirm/trust/pre-approval/redirect state, backend-agnostic), `GateServer` (Unix socket front for out-of-process callers) |
| `veronica/brain/gateclient.py` (new) | `ask_gate(...)` sync client used by `serve` and `hook`; fail-closed |
| `veronica/brain/hook.py` (new) | `python -m veronica.brain.hook <backend>`: pre-tool hook executable |
| `veronica/tools/serve.py` (new) | `python -m veronica.tools.serve <name>`: our MCP server over stdio, gated |
| `veronica/brain/backends/__init__.py` (new) | `BACKENDS`, `check_backend`, `make_brain` |
| `veronica/brain/backends/claude.py` (new) | `ClaudeBrain` (today's `Brain`, gate delegated) |
| `veronica/brain/backends/cli.py` (new) | `CliBrain` base: spawn, stream, sentences, canary, limit detection, workspace |
| `veronica/brain/backends/antigravity.py`, `codex.py`, `copilot.py`, `qwen.py` (new) | argv/config/parse per CLI |
| `veronica/brain/switch.py` (new) | `BrainSwitcher`: current brain, switch, failover, cooldowns |
| `veronica/brain/agent.py` | keeps summaries/constants; `Brain = ClaudeBrain` re-export |
| `veronica/config.py`, `veronica/ui/settings/bridge.py`, `veronica/ui/settings/settings.js` | new settings + rows |
| `veronica/brain/intents.py`, `veronica/orchestrator.py`, `veronica/__main__.py` | switch/which intents, gate server lifecycle, failover hook, HUD `backend` event |
| `veronica/ui/hud/hud.js`, `veronica/ui/menubar.py` | `Brain: …` label, Brain submenu |
| `tests/fixtures/brains/*.jsonl` | captured CLI output |
| `README.md` | "Brains" section |

---

### Task 1: `ToolGate` extraction + `ClaudeBrain` + `Brain` protocol

**Files:**
- Create: `veronica/brain/base.py`, `veronica/brain/gate.py` (ToolGate only; GateServer comes in Task 2), `veronica/brain/backends/__init__.py` (empty for now), `veronica/brain/backends/claude.py`
- Modify: `veronica/brain/agent.py` (remove the `Brain` class body; keep helpers; re-export), `tests/test_agent.py` (private-state pokes)
- Test: `tests/test_gate.py` (new), `tests/test_agent.py`

**Interfaces:**
- Produces:
  ```python
  # veronica/brain/base.py
  @dataclass(frozen=True)
  class Decision:
      allow: bool
      kind: Literal["auto", "preapproved", "trusted", "approved", "denied", "other", "redirect"]
      message: str = ""        # deny reason for the model ("user declined", "user declined and said: '…'", redirect hint)
      heard: str = ""          # what the user said when kind == "other"

  class Brain(Protocol):
      name: str
      gate: "ToolGate"
      def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]: ...
      async def interrupt(self) -> None: ...
      async def close(self) -> None: ...
  ```
  ```python
  # veronica/brain/gate.py
  class ToolGate:
      def __init__(self, settings, confirm: Confirm, on_tool=None, frontmost=frontmost, clock=time.monotonic): ...
      async def decide(self, tool_name: str, input: dict) -> Decision
      # state carried over from Brain, same names/semantics:
      def clear_trust(self) -> None
      def begin_turn(self, turn_id: int) -> None
      def preapprove(self, turn_id: int, until: float) -> None
      pending_redirect: str | None
  ```
  `ClaudeBrain(settings, confirm=None, on_tool=None, memory=None, frontmost=frontmost, clock=time.monotonic, *, gate: ToolGate | None = None)` — when `gate` is None it builds one from the other args (today's constructor keeps working). `name = "claude"`. Exposes proxies `clear_trust`, `begin_turn`, `preapprove`, `pending_redirect` (property get/set) that forward to `self.gate` so `orchestrator.py` is untouched in this task.

- [ ] **Step 1: Write the failing gate tests**

`tests/test_gate.py` — port the gate cases from `tests/test_agent.py` (search there for `_can_use_tool`, `test_trust_*`, `test_preapprov*`, `test_deny_other*`) onto `ToolGate.decide`. Minimum set:

```python
import time
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
    d = await g.decide("mcp__mac__read_battery", {})
    assert d.allow and d.kind == "auto" and calls == [] and cards == [("Read battery", "auto")]


async def test_confirm_class_asks_and_yes_allows():
    g, calls, _ = make([True])
    d = await g.decide("mcp__mac__open_app", {"name": "Safari"})
    assert d.allow and d.kind == "approved" and calls[0][0] == "Open Safari"


async def test_no_denies_with_user_declined():
    g, _, _ = make([False])
    d = await g.decide("mcp__mac__open_app", {"name": "Safari"})
    assert not d.allow and d.kind == "denied" and d.message == "user declined"


async def test_other_answer_becomes_redirect():
    g, _, _ = make([ConfirmResult("other", "open it in the other profile")])
    d = await g.decide("mcp__mac__open_app", {"name": "Safari"})
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
    g2_calls_before = len(calls)
    # a second confirm is needed after clear_trust; answers list is empty -> IndexError proves it asked
    with pytest.raises(IndexError):
        await g.decide("mcp__computer__computer_click", {"x": 5, "y": 6})
    assert len(calls) == g2_calls_before  # the fake raised before recording


async def test_preapproval_covers_first_confirm_call_only():
    now = [10.0]
    g, calls, cards = make([True], now=now)
    g.begin_turn(7)
    g.preapprove(7, until=30.0)
    d1 = await g.decide("mcp__mac__open_app", {"name": "Safari"})
    assert d1.allow and d1.kind == "preapproved" and calls == [] and cards[-1] == ("Open Safari", "preapproved")
    d2 = await g.decide("mcp__mac__open_app", {"name": "Notes"})
    assert d2.allow and d2.kind == "approved" and len(calls) == 1


async def test_preapproval_never_for_always_confirm():
    now = [10.0]
    g, calls, _ = make([True], now=now)
    g.begin_turn(1); g.preapprove(1, until=30.0)
    d = await g.decide("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"})
    assert d.kind == "approved" and len(calls) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest -q tests/test_gate.py`
Expected: FAIL — `ModuleNotFoundError: veronica.brain.gate`

- [ ] **Step 3: Create `base.py` and `gate.py`**

`veronica/brain/base.py`:
```python
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

DecisionKind = Literal["auto", "preapproved", "trusted", "approved", "denied", "other", "redirect"]


@dataclass(frozen=True)
class Decision:
    """What the gate decided about one tool call. `message` is what the
    model is told on a deny; `heard` carries the user's words when they
    answered with something other than yes/no."""
    allow: bool
    kind: DecisionKind
    message: str = ""
    heard: str = ""


class Brain(Protocol):
    """What the orchestrator needs from any backend: sentences streamed
    from ask(), an interrupt, a close, and the shared gate."""
    name: str
    gate: "ToolGate"  # noqa: F821  (veronica.brain.gate; string to avoid the import cycle)

    def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]: ...
    async def interrupt(self) -> None: ...
    async def close(self) -> None: ...
```

`veronica/brain/gate.py` — move, verbatim where possible, from `agent.py`'s `Brain`: `_REDIRECT_BASH`, `_bash_redirect`, `_confirm_outcome` (module function in agent.py today — move it here and import it back into agent.py), `_deny` → returns `Decision`, `begin_turn`, `preapprove`, `_preapproved`, `clear_trust`, `_trustable`, `_trusted`, `_gate_computer` → returns `Decision`. The public method:

```python
class ToolGate:
    """The one place a tool call is allowed or refused, for every backend.
    In-process for Claude (ClaudeBrain.can_use_tool wraps decide()), over
    the gate socket for external CLIs (GateServer, Task 2)."""

    def __init__(self, settings, confirm, on_tool=None, frontmost=frontmost, clock=time.monotonic):
        self.s = settings
        self._confirm = confirm
        self._on_tool = on_tool
        self._frontmost = frontmost
        self._clock = clock
        self._trust_until = 0.0
        self._trust_app: str | None = None
        self._current_turn = 0
        self._asked_this_turn = 0
        self._preapproved_turn: int | None = None
        self._preapproved_until = 0.0
        self.pending_redirect: str | None = None

    def _card(self, summary: str, decision: str) -> None:
        if self._on_tool:
            self._on_tool(summary, decision)

    async def decide(self, tool_name: str, input: dict) -> Decision:
        summary = summarize_tool(tool_name, input)
        redirect = self._bash_redirect(tool_name, input)
        if redirect is not None:
            log.info("tool redirected: %s -> %s", summary, redirect)
            return Decision(False, "redirect", redirect)
        if classify(tool_name, input) == "allow":
            log.info("auto-allow: %s", summary)
            self._card(summary, "auto")
            return Decision(True, "auto")
        front = self._frontmost() if tool_name.startswith(COMPUTER_PREFIX) else None
        if self._preapproved(tool_name, input, front):
            log.info("pre-approved by request wording: %s", summary)
            self._card(summary, "preapproved")
            return Decision(True, "preapproved")
        if front is not None:
            return await self._gate_computer(tool_name, input, summary, front)
        log.info("tool request: %s", summary)
        outcome, heard = _confirm_outcome(await self._confirm(summary, summarize_detail(tool_name, input)))
        if outcome == "approved":
            return Decision(True, "approved")
        return self._deny(outcome, heard)

    def _deny(self, outcome: str, heard: str) -> Decision:
        if outcome == "other":
            self.pending_redirect = heard
            return Decision(False, "other", f"user declined and said: {heard!r}", heard)
        return Decision(False, "denied", "user declined")
```
`_gate_computer` returns `Decision(True, "trusted")` on the trust path (card `"auto"` — unchanged wire value), `Decision(True, "approved")` after a yes, `self._deny(...)` otherwise (after `clear_trust()`). Import `summarize_tool`, `summarize_detail`, `COMPUTER_PREFIX` from `veronica.brain.agent` **lazily inside functions or at module bottom** — `agent.py` will import `ClaudeBrain` from `backends.claude`, which imports `gate.py`; avoid the cycle by having `agent.py` import `backends.claude` at the *end* of the module (after the helpers are defined), and `gate.py` import from `agent` at top (helpers exist by then).

- [ ] **Step 4: Run the gate tests**

Run: `uv run pytest -q tests/test_gate.py`
Expected: PASS

- [ ] **Step 5: Create `backends/claude.py` and slim `agent.py`**

`veronica/brain/backends/claude.py`: the `Brain` class from `agent.py` renamed `ClaudeBrain`, with:
```python
class ClaudeBrain:
    name = "claude"
    _client_cls = ClaudeSDKClient

    def __init__(self, settings, confirm=None, on_tool=None, memory=None,
                 frontmost=frontmost, clock=time.monotonic, *, gate: ToolGate | None = None):
        self.s = settings
        self.gate = gate or ToolGate(settings, confirm, on_tool=on_tool, frontmost=frontmost, clock=clock)
        self._memory = memory
        self._client = None
        self._in_flight = False

    # proxies so the orchestrator keeps calling brain.<x>
    def clear_trust(self): self.gate.clear_trust()
    def begin_turn(self, turn_id): self.gate.begin_turn(turn_id)
    def preapprove(self, turn_id, until): self.gate.preapprove(turn_id, until)
    @property
    def pending_redirect(self): return self.gate.pending_redirect
    @pending_redirect.setter
    def pending_redirect(self, v): self.gate.pending_redirect = v

    async def _can_use_tool(self, tool_name, input, context):
        d = await self.gate.decide(tool_name, input)
        if d.allow:
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message=d.message)
```
`_options`, `_ensure_client`, `_build_prompt`, `_image_fallback_text`, `ask`, `close`, `interrupt`, session file helpers: unchanged (they read `self.s.session_file`). `ask()` still resets `self.pending_redirect = None` at its start (it did already).

`veronica/brain/agent.py`: delete the `Brain` class; keep everything above it; add at the bottom:
```python
from veronica.brain.backends.claude import ClaudeBrain  # noqa: E402  (after the helpers it imports)

Brain = ClaudeBrain
```
Move `_confirm_outcome` into `gate.py` and import it in `agent.py` only if something else there uses it (nothing should).

- [ ] **Step 6: Fix `tests/test_agent.py` private-state pokes**

`grep -n "_trust_until\|_trust_app\|_preapproved_turn\|_asked_this_turn\|_current_turn" tests/test_agent.py` — rewrite each `b._x` as `b.gate._x`. Constructor calls (`Brain(Settings(), confirm=confirm, ...)`) stay. Tests that call `b._can_use_tool(...)` stay (it still exists on `ClaudeBrain`).

- [ ] **Step 7: Full suite**

Run: `uv run pytest -q`
Expected: all pass (2214 + the new gate tests). Also `uv run python -c "from veronica.brain.agent import Brain; from veronica.brain.backends.claude import ClaudeBrain; assert Brain is ClaudeBrain"`.

- [ ] **Step 8: Commit**

```bash
git add veronica/brain/base.py veronica/brain/gate.py veronica/brain/backends/ veronica/brain/agent.py tests/test_gate.py tests/test_agent.py
git commit -m "refactor(brain): extract ToolGate and ClaudeBrain; Brain protocol"
```

---

### Task 2: Gate socket — `GateServer`, `gateclient.ask_gate`, `tools/serve`, `brain/hook`

**Files:**
- Modify: `veronica/brain/gate.py` (add `GateServer`), `veronica/config.py` (`gate_socket`, `backend_dir`)
- Create: `veronica/brain/gateclient.py`, `veronica/tools/serve.py`, `veronica/brain/hook.py`
- Test: `tests/test_gate.py` (server), `tests/test_gateclient.py`, `tests/test_tools_serve.py`, `tests/test_hook.py`

**Interfaces:**
- Consumes: `ToolGate.decide`, `veronica.tools.<name>.<name>_server.instance` (an `mcp.server.Server`), `policy.classify`.
- Produces:
  ```python
  # config.py
  Settings.gate_socket: Path            # default home / "gate.sock"
  Settings.backend_dir(name: str) -> Path   # home / "backends" / name, mkdir 0700
  Settings.session_file_for(name) -> Path   # "claude" -> session_file; else backend_dir(name)/"session"

  # gate.py
  class GateServer:
      def __init__(self, gate: ToolGate, path: Path): ...
      async def start(self) -> None      # unlink stale, bind, chmod 0600
      async def stop(self) -> None
  # wire: one JSON line in, one out. Request {"v":1,"tool":str,"input":dict,"origin":"mcp"|"hook","backend":str}
  # Response {"allow":bool,"kind":str,"reason":str}

  # gateclient.py
  def ask_gate(tool: str, input: dict, *, origin: str, backend: str,
               sock: str | None = None, timeout: float | None = None) -> Decision
  # sock defaults to env VERONICA_GATE_SOCK; timeout defaults to env VERONICA_GATE_TIMEOUT_S or None
  # fail-closed: any error -> Decision(False, "denied", "Veronica's gate isn't reachable")

  # hook.py
  CANONICAL: dict[str, str]   # vendor tool name -> "Bash" | "Write" | "Edit" | "read"
  def canonical_tool(backend: str, tool_name: str, tool_input: dict) -> tuple[str, dict] | None
      # None = read-only native tool, allow without the gate;
      # ("mcp__<server>__<tool>", input) for our servers; ("Bash", {"command": ...}) etc.
  def run(backend: str, stdin_text: str, *, ask=ask_gate, log_path: Path | None = None) -> tuple[str, int]  # (stdout, exit code)
  def emit(backend: str, allow: bool, reason: str) -> str   # per-backend JSON shape
  ```

- [ ] **Step 1: Failing tests — GateServer round trip and serialization**

Append to `tests/test_gate.py`:
```python
import asyncio, json
from veronica.brain.gate import GateServer


async def _roundtrip(path, req):
    r, w = await asyncio.open_unix_connection(str(path))
    w.write((json.dumps(req) + "\n").encode()); await w.drain()
    line = await r.readline()
    w.close(); await w.wait_closed()
    return json.loads(line)


async def test_gate_server_allow_and_deny(tmp_path):
    g, _, _ = make([True, False])
    srv = GateServer(g, tmp_path / "gate.sock")
    await srv.start()
    try:
        assert (tmp_path / "gate.sock").stat().st_mode & 0o777 == 0o600
        ok = await _roundtrip(tmp_path / "gate.sock",
                              {"v": 1, "tool": "mcp__mac__open_app", "input": {"name": "Safari"}, "origin": "mcp", "backend": "codex"})
        assert ok == {"allow": True, "kind": "approved", "reason": ""}
        no = await _roundtrip(tmp_path / "gate.sock",
                              {"v": 1, "tool": "mcp__mac__open_app", "input": {"name": "Notes"}, "origin": "hook", "backend": "codex"})
        assert no == {"allow": False, "kind": "denied", "reason": "user declined"}
    finally:
        await srv.stop()
    assert not (tmp_path / "gate.sock").exists()


async def test_gate_server_malformed_request_is_denied(tmp_path):
    g, _, _ = make([])
    srv = GateServer(g, tmp_path / "gate.sock"); await srv.start()
    try:
        r, w = await asyncio.open_unix_connection(str(tmp_path / "gate.sock"))
        w.write(b"not json\n"); await w.drain()
        assert json.loads(await r.readline()) == {"allow": False, "kind": "denied", "reason": "bad request"}
        w.close(); await w.wait_closed()
    finally:
        await srv.stop()


async def test_gate_server_serializes_confirms(tmp_path):
    order = []
    async def confirm(summary, detail=""):
        order.append(("start", summary)); await asyncio.sleep(0.05); order.append(("end", summary))
        return True
    g = ToolGate(Settings(), confirm)
    srv = GateServer(g, tmp_path / "gate.sock"); await srv.start()
    try:
        await asyncio.gather(
            _roundtrip(tmp_path / "gate.sock", {"v": 1, "tool": "mcp__mac__open_app", "input": {"name": "A"}, "origin": "mcp", "backend": "x"}),
            _roundtrip(tmp_path / "gate.sock", {"v": 1, "tool": "mcp__mac__open_app", "input": {"name": "B"}, "origin": "mcp", "backend": "x"}),
        )
    finally:
        await srv.stop()
    assert [o[0] for o in order] == ["start", "end", "start", "end"]
```

- [ ] **Step 2: Implement `GateServer` + settings fields**

```python
class GateServer:
    """Unix-socket front for ToolGate.decide, for the out-of-process
    callers (tools.serve, brain.hook). One JSON line per request; confirms
    are serialized because the orchestrator can only ask one question at a
    time."""

    def __init__(self, gate: ToolGate, path: Path) -> None:
        self.gate, self.path = gate, path
        self._server: asyncio.AbstractServer | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        if self.path.exists():
            self.path.unlink()
        self._server = await asyncio.start_unix_server(self._handle, path=str(self.path))
        self.path.chmod(0o600)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self.path.exists():
            self.path.unlink()

    async def _handle(self, reader, writer):
        try:
            line = await reader.readline()
            try:
                req = json.loads(line)
                tool, inp = str(req["tool"]), dict(req.get("input") or {})
            except Exception:
                resp = {"allow": False, "kind": "denied", "reason": "bad request"}
            else:
                log.info("gate request from %s/%s: %s", req.get("backend"), req.get("origin"), tool)
                async with self._lock:
                    d = await self.gate.decide(tool, inp)
                resp = {"allow": d.allow, "kind": d.kind, "reason": d.message}
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()
        except Exception:
            log.exception("gate request failed")
        finally:
            writer.close()
```
`config.py`: add `gate_socket: Path = Field(default_factory=lambda: Path.home() / ".veronica" / "gate.sock")` next to `home` (not editable), and
```python
    def backend_dir(self, name: str) -> Path:
        d = self.home / "backends" / name
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        return d

    def session_file_for(self, name: str) -> Path:
        return self.session_file if name == "claude" else self.backend_dir(name) / "session"
```
Tests for these two in `tests/test_config.py` (`tmp_path` home; `backend_dir("codex")` exists with mode 0o700; `session_file_for("claude") == session_file`).

- [ ] **Step 3: Run**  `uv run pytest -q tests/test_gate.py tests/test_config.py` → PASS

- [ ] **Step 4: Failing tests — `ask_gate`**

`tests/test_gateclient.py`:
```python
import asyncio, json, os, threading
from veronica.brain.gateclient import ask_gate


def _serve_once(path, reply):
    """A one-shot fake gate on a thread so the sync client has something to talk to."""
    async def main():
        got = {}
        async def h(r, w):
            got["req"] = json.loads(await r.readline())
            w.write((json.dumps(reply) + "\n").encode()); await w.drain(); w.close()
        srv = await asyncio.start_unix_server(h, path=str(path))
        async with srv:
            while "req" not in got:
                await asyncio.sleep(0.01)
        return got["req"]
    box = {}
    t = threading.Thread(target=lambda: box.update(req=asyncio.run(main())))
    t.start()
    return t, box


def test_ask_gate_allow(tmp_path):
    t, box = _serve_once(tmp_path / "g.sock", {"allow": True, "kind": "approved", "reason": ""})
    d = ask_gate("mcp__mac__open_app", {"name": "Safari"}, origin="mcp", backend="codex", sock=str(tmp_path / "g.sock"))
    t.join(2)
    assert d.allow and d.kind == "approved"
    assert box["req"] == {"v": 1, "tool": "mcp__mac__open_app", "input": {"name": "Safari"}, "origin": "mcp", "backend": "codex"}


def test_ask_gate_fails_closed_without_socket(tmp_path):
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex", sock=str(tmp_path / "missing.sock"))
    assert not d.allow and d.kind == "denied" and "gate isn't reachable" in d.message


def test_ask_gate_reads_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VERONICA_GATE_SOCK", str(tmp_path / "g.sock"))
    t, _ = _serve_once(tmp_path / "g.sock", {"allow": False, "kind": "denied", "reason": "user declined"})
    d = ask_gate("Bash", {"command": "ls"}, origin="hook", backend="codex")
    t.join(2)
    assert not d.allow and d.message == "user declined"
```

- [ ] **Step 5: Implement `gateclient.py`**

```python
"""Sync client for the gate socket, used from processes Veronica spawns
(tools.serve, brain.hook). Fails closed: no socket, no answer, bad JSON
-> deny."""
import json, os, socket
from veronica.brain.base import Decision

UNREACHABLE = "Veronica's gate isn't reachable"


def ask_gate(tool, input, *, origin, backend, sock=None, timeout=None) -> Decision:
    path = sock or os.environ.get("VERONICA_GATE_SOCK", "")
    if timeout is None:
        env = os.environ.get("VERONICA_GATE_TIMEOUT_S")
        timeout = float(env) if env else None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(path)
            s.sendall((json.dumps({"v": 1, "tool": tool, "input": input, "origin": origin, "backend": backend}) + "\n").encode())
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
```

- [ ] **Step 6: Run** `uv run pytest -q tests/test_gateclient.py` → PASS

- [ ] **Step 7: Failing tests — `tools/serve`**

`tests/test_tools_serve.py`:
```python
import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest
from veronica.brain.base import Decision
from veronica.tools import serve


def test_server_lookup():
    assert serve.server_for("mac").name == "mac"
    with pytest.raises(KeyError):
        serve.server_for("nope")


async def test_gated_call_tool_denies_then_allows(monkeypatch):
    answers = [Decision(False, "denied", "user declined"), Decision(True, "approved")]
    asked = []
    def fake_ask(tool, input, **kw):
        asked.append((tool, input, kw)); return answers.pop(0)
    monkeypatch.setattr(serve, "ask_gate", fake_ask)
    monkeypatch.setenv("VERONICA_BRAIN", "codex")
    calls = []
    async def fake_handler(name, args):
        calls.append((name, args)); return [ {"type": "text", "text": "opened"} ]
    inst = serve.gated_server("mac", call_tool=fake_handler)
    denied = await inst.request_handlers[CallToolRequest](
        CallToolRequest(method="tools/call", params=CallToolRequestParams(name="open_app", arguments={"name": "Safari"})))
    assert denied.root.isError and "Not allowed: user declined" in denied.root.content[0].text
    assert asked[0][0] == "mcp__mac__open_app" and asked[0][2] == {"origin": "mcp", "backend": "codex"}
    assert calls == []
    ok = await inst.request_handlers[CallToolRequest](
        CallToolRequest(method="tools/call", params=CallToolRequestParams(name="open_app", arguments={"name": "Safari"})))
    assert not ok.root.isError and calls == [("open_app", {"name": "Safari"})]


async def test_list_tools_passthrough():
    inst = serve.gated_server("mac")
    res = await inst.request_handlers[ListToolsRequest](ListToolsRequest(method="tools/list"))
    assert any(t.name == "open_app" for t in res.root.tools)
```
(Look at how `claude_agent_sdk.create_sdk_mcp_server` registers handlers: it calls `server.list_tools()` / `server.call_tool()` decorators on an `mcp.server.Server`; `request_handlers[CallToolRequest]` is the low-level entry the test drives. Adjust the exact request/response object names to the installed `mcp` version — `uv run python -c "import mcp.types as t; print(t.CallToolRequest, t.ServerResult)"` — the shape above is the 1.x API.)

- [ ] **Step 8: Implement `veronica/tools/serve.py`**

```python
"""`python -m veronica.tools.serve <name>`: one of Veronica's MCP servers
over stdio for an external brain (Codex/Antigravity/Copilot/Qwen). Every
call_tool first asks the gate socket (VERONICA_GATE_SOCK) the same
question the in-process gate would ask: policy, trust window, voice
confirm. stdout is the protocol; log to stderr only."""
import asyncio, importlib, logging, os, sys
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequest, CallToolResult, ServerResult, TextContent
from veronica.brain.gateclient import ask_gate

SERVERS = {"mac": "veronica.tools.mac", "pim": "veronica.tools.pim", "memory": "veronica.tools.memory_tools",
           "screen": "veronica.tools.screen", "music": "veronica.tools.music",
           "browser": "veronica.tools.browser", "computer": "veronica.tools.computer"}


def server_for(name: str) -> Server:
    mod = importlib.import_module(SERVERS[name])       # KeyError for unknown names
    return getattr(mod, f"{name}_server").instance


def gated_server(name: str, call_tool=None) -> Server:
    inst = server_for(name)
    original = inst.request_handlers[CallToolRequest]
    backend = os.environ.get("VERONICA_BRAIN", "external")

    async def gated(req):
        tool, args = req.params.name, dict(req.params.arguments or {})
        d = await asyncio.to_thread(ask_gate, f"mcp__{name}__{tool}", args, origin="mcp", backend=backend)
        if not d.allow:
            return ServerResult(CallToolResult(content=[TextContent(type="text", text=f"Not allowed: {d.message}")], isError=True))
        if call_tool is not None:   # test seam
            content = await call_tool(tool, args)
            return ServerResult(CallToolResult(content=[TextContent(**c) for c in content]))
        return await original(req)

    inst.request_handlers[CallToolRequest] = gated
    return inst


async def _main(name: str) -> None:
    inst = gated_server(name)
    async with stdio_server() as (read, write):
        await inst.run(read, write, inst.create_initialization_options())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    asyncio.run(_main(sys.argv[1]))
```
Memory server note: `memory_tools.bind(store)` is normally called by `__main__`; in the stdio process bind a `MemoryStore(Settings().memory_path)` at startup when `name == "memory"` (same for `pim.bind(TimerService(...))` — timers can't announce from a child process, so bind a `TimerService(on_fire=lambda *_: None)` and log a warning that timers set from an external brain don't fire; document in README).

- [ ] **Step 9: Run** `uv run pytest -q tests/test_tools_serve.py` → PASS. Smoke by hand: `printf '' | VERONICA_GATE_SOCK=/nonexistent uv run python -m veronica.tools.serve mac` exits cleanly on EOF.

- [ ] **Step 10: Failing tests — hook**

`tests/test_hook.py`:
```python
import json
import pytest
from veronica.brain import hook
from veronica.brain.base import Decision


@pytest.mark.parametrize("backend,tool,inp,expected", [
    ("antigravity", "run_command", {"command": "ls -la"}, ("Bash", {"command": "ls -la"})),
    ("qwen", "run_shell_command", {"command": "echo hi"}, ("Bash", {"command": "echo hi"})),
    ("codex", "shell", {"command": ["bash", "-lc", "ls"]}, ("Bash", {"command": "bash -lc ls"})),
    ("copilot", "bash", {"command": "pwd"}, ("Bash", {"command": "pwd"})),
    ("antigravity", "write_file", {"file_path": "/tmp/x", "content": "y"}, ("Write", {"file_path": "/tmp/x", "content": "y"})),
    ("codex", "apply_patch", {"patch": "*** Begin Patch"}, ("Edit", {"patch": "*** Begin Patch"})),
    ("antigravity", "read_file", {"file_path": "/tmp/x"}, None),
    ("copilot", "grep", {"pattern": "x"}, None),
    ("qwen", "google_web_search", {"query": "x"}, None),
    ("codex", "mcp__veronica-mac__open_app", {"name": "Safari"}, ("mcp__mac__open_app", {"name": "Safari"})),
    ("copilot", "veronica-mac__open_app", {"name": "Safari"}, ("mcp__mac__open_app", {"name": "Safari"})),
    ("qwen", "mac__open_app", {"name": "Safari"}, ("mcp__mac__open_app", {"name": "Safari"})),
    ("antigravity", "some_new_tool", {"a": 1}, ("some_new_tool", {"a": 1})),
])
def test_canonical_tool(backend, tool, inp, expected):
    assert hook.canonical_tool(backend, tool, inp) == expected


def test_run_logs_before_asking_and_allows(tmp_path):
    seen = []
    def ask(tool, input, **kw):
        seen.append((tool, input, kw, (tmp_path / "hook.log").read_text()))
        return Decision(True, "approved")
    out, code = hook.run("codex", json.dumps({"tool_name": "shell", "tool_input": {"command": "ls"}}), ask=ask, log_path=tmp_path / "hook.log")
    assert code == 0
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert seen[0][0] == "Bash" and seen[0][2] == {"origin": "hook", "backend": "codex"}
    assert '"key": "ls"' in seen[0][3]           # the log line existed before the gate answered


def test_run_deny_shapes(tmp_path):
    deny = lambda *a, **k: Decision(False, "denied", "user declined")
    out, _ = hook.run("antigravity", json.dumps({"tool_name": "run_command", "tool_input": {"command": "rm x"}}), ask=deny, log_path=tmp_path / "l")
    assert json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"] == "user declined"
    out, _ = hook.run("qwen", json.dumps({"tool_name": "run_shell_command", "tool_input": {"command": "rm x"}}), ask=deny, log_path=tmp_path / "l")
    assert json.loads(out) == {"decision": "deny", "reason": "user declined"}
    out, _ = hook.run("copilot", json.dumps({"toolName": "bash", "toolArgs": {"command": "rm x"}}), ask=deny, log_path=tmp_path / "l")
    assert json.loads(out)["permissionDecision"] == "deny"


def test_run_mcp_and_readonly_skip_gate(tmp_path):
    def boom(*a, **k): raise AssertionError("gate must not be asked")
    out, _ = hook.run("codex", json.dumps({"tool_name": "mcp__veronica-mac__open_app", "tool_input": {"name": "x"}}), ask=boom, log_path=tmp_path / "l")
    assert json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "allow"
    out, _ = hook.run("qwen", json.dumps({"tool_name": "read_file", "tool_input": {}}), ask=boom, log_path=tmp_path / "l")
    assert out == ""


def test_run_exception_fails_closed(tmp_path):
    def boom(*a, **k): raise RuntimeError("x")
    out, code = hook.run("codex", json.dumps({"tool_name": "shell", "tool_input": {"command": "ls"}}), ask=boom, log_path=tmp_path / "l")
    assert code == 0 and json.loads(out)["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"] == "gate error"
```

- [ ] **Step 11: Implement `veronica/brain/hook.py`**

```python
"""`python -m veronica.brain.hook <backend>`: the pre-tool hook Veronica
installs in each external CLI's workspace. Reads the hook payload on
stdin, maps the CLI's tool to our canonical name, logs it (the canary in
backends/cli.py checks this log), asks the gate socket, and prints the
CLI's decision shape. Anything unexpected -> deny."""
import json, os, shlex, sys, time
from pathlib import Path
from veronica.brain.gateclient import ask_gate

SHELL_TOOLS = {"run_command", "run_shell_command", "Bash", "shell", "local_shell", "bash"}
WRITE_TOOLS = {"write_file", "write_to_file", "Write"}
EDIT_TOOLS = {"replace", "edit", "edit_file", "apply_patch", "str_replace_editor", "Edit"}
READONLY_TOOLS = {"read_file", "view", "glob", "grep", "list_directory", "find", "web_fetch",
                  "web_search", "google_web_search", "fetch", "Read", "Glob", "Grep", "ls"}
OUR_SERVERS = ("mac", "pim", "memory", "screen", "music", "browser", "computer")


def _ours(tool_name: str) -> str | None:
    """'mcp__veronica-mac__open_app' / 'veronica-mac__open_app' / 'mac__open_app' -> 'mcp__mac__open_app'."""
    t = tool_name[5:] if tool_name.startswith("mcp__") else tool_name
    if t.startswith("veronica-"):
        t = t[len("veronica-"):]
    server, sep, short = t.partition("__")
    return f"mcp__{server}__{short}" if sep and server in OUR_SERVERS else None


def canonical_tool(backend, tool_name, tool_input):
    ours = _ours(tool_name)
    if ours:
        return ours, dict(tool_input)
    if tool_name in READONLY_TOOLS:
        return None
    if tool_name in SHELL_TOOLS:
        cmd = tool_input.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(shlex.quote(str(c)) if " " in str(c) else str(c) for c in cmd)
        return "Bash", {"command": str(cmd)}
    if tool_name in WRITE_TOOLS:
        return "Write", dict(tool_input)
    if tool_name in EDIT_TOOLS:
        return "Edit", dict(tool_input)
    return tool_name, dict(tool_input)


def emit(backend, allow, reason=""):
    if backend == "qwen":
        return "" if allow else json.dumps({"decision": "deny", "reason": reason})
    if backend == "copilot":
        return json.dumps({"permissionDecision": "allow" if allow else "deny", "permissionDecisionReason": reason})
    return json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                              "permissionDecision": "allow" if allow else "deny",
                                              "permissionDecisionReason": reason}})


def _key(canon, inp):
    return inp.get("command") or inp.get("file_path") or inp.get("path") or canon


def run(backend, stdin_text, *, ask=ask_gate, log_path=None):
    try:
        payload = json.loads(stdin_text or "{}")
        tool = payload.get("tool_name") or payload.get("toolName") or ""
        inp = payload.get("tool_input") or payload.get("toolArgs") or payload.get("toolInput") or {}
        canon = canonical_tool(backend, str(tool), dict(inp))
        if canon is None:
            return "", 0
        name, cinp = canon
        if name.startswith("mcp__"):
            return emit(backend, True), 0     # gated inside tools.serve already
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), "call": tool, "key": _key(name, cinp), "decision": "pending"}) + "\n")
        d = ask(name, cinp, origin="hook", backend=backend)
        return emit(backend, d.allow, d.message), 0
    except Exception:
        return emit(backend, False, "gate error"), 0


if __name__ == "__main__":
    backend = sys.argv[1]
    log_path = Path(os.environ.get("VERONICA_HOOK_LOG", "")) if os.environ.get("VERONICA_HOOK_LOG") else None
    out, code = run(backend, sys.stdin.read(), log_path=log_path)
    sys.stdout.write(out)
    sys.exit(code)
```
(Copilot's exact decision key is confirmed in Task 5 against the hooks reference; `emit` is the one place to change.)

- [ ] **Step 12: Run** `uv run pytest -q tests/test_hook.py` then the full suite → PASS

- [ ] **Step 13: Commit**

```bash
git add veronica/brain/gate.py veronica/brain/gateclient.py veronica/brain/hook.py veronica/tools/serve.py veronica/config.py tests/test_gate.py tests/test_gateclient.py tests/test_tools_serve.py tests/test_hook.py tests/test_config.py
git commit -m "feat(brain): gate socket, stdio MCP serving and pre-tool hook for external brains"
```

---

### Task 3: `CliBrain` base + `AntigravityBrain` (+ fixture capture)

**Files:**
- Create: `veronica/brain/backends/cli.py`, `veronica/brain/backends/antigravity.py`, `tests/fixtures/brains/antigravity-plain.jsonl`, `tests/fixtures/brains/antigravity-tool.jsonl`, `tests/fixtures/brains/antigravity-limit.jsonl`, `tests/test_backend_cli.py`, `tests/test_backend_antigravity.py`, `tests/test_brains_live.py`
- Modify: `veronica/config.py` (`antigravity_native_tools`)

**Interfaces:**
- Consumes: `ToolGate`, `Settings.backend_dir/session_file_for/gate_socket`, `SentenceSplitter`, `system_prompt`, `summarize_detail`.
- Produces:
  ```python
  # cli.py
  @dataclass
  class Text:  delta: str
  @dataclass
  class ToolStart: call_id: str; tool: str; input: dict; native: bool
  @dataclass
  class ToolEnd:  call_id: str
  @dataclass
  class Session: id: str
  @dataclass
  class Done:   final_text: str = ""
  @dataclass
  class Error:  message: str
  Event = Text | ToolStart | ToolEnd | Session | Done | Error

  LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "429", "quota", "resource exhausted",
                   "too many requests", "limit reached", "out of credits", "insufficient_quota", "overloaded")
  OVERFLOW_MARKERS = ("context", "compact", "too long", "prompt is too long", "max_tokens")

  class LimitError(Exception): ...          # raised out of ask() so the switcher can fail over (Task 7)

  class CliBrain:
      name: str                              # subclass sets
      label: str
      binary: str
      def __init__(self, settings, gate: ToolGate, on_tool=None, memory=None, *, spawn=None, clock=time.monotonic): ...
      # subclass hooks
      def argv(self, text: str, session_id: str | None, image_paths: list[Path], native: bool) -> list[str]
      def env(self) -> dict[str, str]                       # extra env for the child
      def prepare_workspace(self, prompt_text: str, native: bool) -> None   # write config/hook files
      def parse(self, line: str) -> list[Event]
      # provided
      def ask(self, text, images=()) -> AsyncIterator[str]
      async def interrupt(self) -> None
      async def close(self) -> None
      def clear_trust / begin_turn / preapprove / pending_redirect  (proxies to gate)
      workspace: Path                          # settings.backend_dir(name)
      hook_log: Path                           # workspace / "hook.log"
      native_tools_enabled() -> bool           # getattr(settings, f"{name}_native_tools", True)
  ```

- [ ] **Step 1: Failing tests for the base (`tests/test_backend_cli.py`)**

Use a `FakeProc` + `fake_spawn` that streams canned lines:
```python
import asyncio, json, signal
import pytest
from veronica.brain.backends import cli
from veronica.brain.gate import ToolGate
from veronica.config import Settings


class FakeProc:
    def __init__(self, lines, *, exit_code=0, hang=False):
        self._lines = [l if l.endswith("\n") else l + "\n" for l in lines]
        self.returncode = None; self.signals = []; self.killed = False; self._hang = hang; self._exit = exit_code
        self.stdout = self; self.stderr = asyncio.StreamReader(); self.stderr.feed_eof()
    async def readline(self):
        if self._lines:
            await asyncio.sleep(0); return self._lines.pop(0).encode()
        if self._hang and not self.killed:
            await asyncio.sleep(3600)
        self.returncode = self._exit; return b""
    def send_signal(self, sig): self.signals.append(sig)
    def kill(self): self.killed = True; self.returncode = -9
    async def wait(self): return self.returncode if self.returncode is not None else 0


class EchoBrain(cli.CliBrain):
    """A CliBrain whose parse() takes our own event JSON, to test the base alone."""
    name, label, binary = "echo", "Echo", "echo-cli"
    def argv(self, text, session_id, image_paths, native):
        return ["echo-cli", text] + (["--resume", session_id] if session_id else []) + [str(p) for p in image_paths]
    def env(self): return {}
    def prepare_workspace(self, prompt_text, native): (self.workspace / "prepared").write_text(prompt_text)
    def parse(self, line):
        e = json.loads(line); t = e.pop("t")
        return [getattr(cli, t)(**e)]


def build(tmp_path, lines, **kw):
    s = Settings(home=tmp_path)
    spawned = []
    async def spawn(argv, cwd, env):
        spawned.append((argv, cwd, env)); return FakeProc(lines, **kw)
    cards = []
    g = ToolGate(s, None, on_tool=lambda su, d: cards.append((su, d)))
    return EchoBrain(s, g, on_tool=lambda su, d: cards.append((su, d)), spawn=spawn), spawned, cards


async def collect(brain, text, images=()):
    return [s async for s in brain.ask(text, images)]


async def test_sentences_stream_and_session_saved(tmp_path):
    b, spawned, _ = build(tmp_path, [
        json.dumps({"t": "Session", "id": "abc"}),
        json.dumps({"t": "Text", "delta": "Hello there. How "}),
        json.dumps({"t": "Text", "delta": "are you?"}),
        json.dumps({"t": "Done"}),
    ])
    assert await collect(b, "hi") == ["Hello there.", "How are you?"]
    assert b.s.session_file_for("echo").read_text() == "abc"
    assert spawned[0][0] == ["echo-cli", "hi"] and spawned[0][1] == str(b.workspace)
    assert (b.workspace / "prepared").read_text().startswith("You are Veronica")
    # next turn resumes
    b2, spawned2, _ = build(tmp_path, [json.dumps({"t": "Done", "final_text": "ok"})])
    await collect(b2, "again")
    assert spawned2[0][0] == ["echo-cli", "again", "--resume", "abc"]


async def test_whole_reply_fallback_when_no_deltas(tmp_path):
    b, _, _ = build(tmp_path, [json.dumps({"t": "Done", "final_text": "One. Two."})])
    assert await collect(b, "x") == ["One.", "Two."]


async def test_native_tool_card_only_for_readonly(tmp_path):
    b, _, cards = build(tmp_path, [
        json.dumps({"t": "ToolStart", "call_id": "1", "tool": "read_file", "input": {"file_path": "/a"}, "native": True}),
        json.dumps({"t": "ToolEnd", "call_id": "1"}),
        json.dumps({"t": "Done", "final_text": "done"}),
    ])
    await collect(b, "x")
    assert cards == [("read_file /a", "auto")]


async def test_images_written_to_workspace(tmp_path):
    b, spawned, _ = build(tmp_path, [json.dumps({"t": "Done", "final_text": "seen"})])
    await collect(b, "look", images=[b"\x89PNG....", b"\xff\xd8\xff...."])
    argv = spawned[0][0]
    assert argv[-2].endswith("img-1.png") and argv[-1].endswith("img-2.jpg")
    assert (b.workspace / "img-1.png").read_bytes() == b"\x89PNG...."


async def test_error_result_spoken_and_overflow_resets_session(tmp_path):
    b, _, _ = build(tmp_path, [json.dumps({"t": "Error", "message": "prompt is too long"})])
    b.s.session_file_for("echo").write_text("old")
    assert await collect(b, "x") == ["My memory got full, starting a fresh conversation."]
    assert not b.s.session_file_for("echo").exists()
    b, _, _ = build(tmp_path, [json.dumps({"t": "Error", "message": "boom"})])
    assert await collect(b, "x") == ["Echo returned an error, check the log."]


async def test_limit_error_raises_limit_error(tmp_path):
    b, _, _ = build(tmp_path, [json.dumps({"t": "Error", "message": "You have hit your usage limit"})])
    with pytest.raises(cli.LimitError):
        await collect(b, "x")


async def test_timeout_kills_child(tmp_path):
    b, spawned, _ = build(tmp_path, [], hang=True)
    b.s.brain_timeout_s = 0.05
    assert await collect(b, "x") == ["Taking too long, cancelled."]
    # spawn returned the proc; it was killed
    assert b._proc is None


async def test_interrupt_sigint_then_kill(tmp_path):
    b, _, _ = build(tmp_path, [], hang=True)
    b.s.interrupt_drain_s = 0.05
    task = asyncio.create_task(collect(b, "x"))
    await asyncio.sleep(0.01)
    proc = b._proc
    await b.interrupt()
    assert proc.signals == [signal.SIGINT] and proc.killed
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_canary_kills_and_disables_native_tools(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(cli.prefs, "save_settings_override", lambda k, v: saved.update({k: v}))
    first = [
        json.dumps({"t": "ToolStart", "call_id": "9", "tool": "run_command", "input": {"command": "ls"}, "native": True}),
        json.dumps({"t": "ToolEnd", "call_id": "9"}),
        json.dumps({"t": "Done", "final_text": "listed"}),
    ]
    second = [json.dumps({"t": "Done", "final_text": "fallback answer"})]
    s = Settings(home=tmp_path); s.echo_native_tools = True  # type: ignore[attr-defined]
    runs = []
    async def spawn(argv, cwd, env):
        runs.append(argv); return FakeProc(first if len(runs) == 1 else second)
    b = EchoBrain(s, ToolGate(s, None), spawn=spawn)
    out = await collect(b, "list files")
    # no hook.log line for key "ls" -> canary trips
    assert out[0] == "Hooks aren't running on Echo, so I've turned off its shell. Tools still work."
    assert out[1:] == ["fallback answer"]
    assert saved == {"echo_native_tools": False} and len(runs) == 2


async def test_canary_passes_when_hook_logged(tmp_path):
    lines = [
        json.dumps({"t": "ToolStart", "call_id": "9", "tool": "run_command", "input": {"command": "ls"}, "native": True}),
        json.dumps({"t": "ToolEnd", "call_id": "9"}),
        json.dumps({"t": "Done", "final_text": "listed"}),
    ]
    runs = []
    async def spawn(argv, cwd, env):
        runs.append(argv)
        b.hook_log.write_text(json.dumps({"ts": 9e12, "call": "run_command", "key": "ls", "decision": "pending"}) + "\n")
        return FakeProc(lines)
    s = Settings(home=tmp_path)
    b = EchoBrain(s, ToolGate(s, None), spawn=spawn)
    assert await collect(b, "x") == ["listed"] and len(runs) == 1
```
`Settings` has `extra="ignore"`; for the fake `echo_native_tools` attribute the test sets it after construction — `validate_assignment=True` rejects unknown fields, so in `CliBrain.native_tools_enabled()` use `getattr(self.s, f"{self.name}_native_tools", True)` and in the test use `object.__setattr__(s, "echo_native_tools", True)`; likewise `monkeypatch.setattr(cli.prefs, ...)` requires `cli.py` to `from veronica import prefs`.

- [ ] **Step 2: Run** → FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `cli.py`**

Core of `ask()` (subclass hooks around it):
```python
async def ask(self, text, images=()):
    self.pending_redirect = None
    splitter = SentenceSplitter()
    native = self.native_tools_enabled()
    for attempt in range(2):                      # second pass = canary fallback
        outcome = await self._run_once(text, tuple(images), native, splitter)
        if outcome.kind == "ok":
            for sent in outcome.sentences: yield sent
            for sent in splitter.flush(): yield sent
            return
        if outcome.kind == "canary" and attempt == 0:
            yield f"Hooks aren't running on {self.label}, so I've turned off its shell. Tools still work."
            prefs.save_settings_override(f"{self.name}_native_tools", False)
            try: setattr(self.s, f"{self.name}_native_tools", False)
            except ValueError: pass
            native = False; splitter = SentenceSplitter()
            continue
        if outcome.kind == "limit":
            raise LimitError(outcome.message)
        for sent in outcome.sentences: yield sent   # error / timeout messages
        return
```
`_run_once`: writes images (`img-N.png|jpg` by magic bytes), `prepare_workspace(system_prompt(dt.date.today(), facts, recent), native)` (facts/recent from `self._memory` exactly like `ClaudeBrain._options`), truncates `hook_log`, spawns via `self._spawn(argv, cwd=str(self.workspace), env={**os.environ, **self.env(), "VERONICA_GATE_SOCK": str(self.s.gate_socket), "VERONICA_BRAIN": self.name, "VERONICA_HOOK_LOG": str(self.hook_log)})`, then reads lines with `asyncio.timeout(self.s.brain_timeout_s)` per line; dispatches events:
- `Text` → `splitter.feed` → sentences appended to the outcome as they arrive (yielded live: make `_run_once` an async generator that yields sentences and finally returns the outcome via a holder — or simpler: make `ask()` itself contain the loop and factor the spawn into `_spawn_turn()`; either is fine as long as sentences stream before `Done`).
- `ToolStart(native=True)` and tool in `hook.READONLY_TOOLS` → `on_tool(f"{tool} {input.get('file_path') or input.get('pattern') or input.get('query') or ''}".strip(), "auto")`; other native tools → remember `(call_id, key)` for the canary; MCP tools → nothing (the gate carded them).
- `ToolEnd` → if the call was native non-readonly: check `hook_log` for a line with that key and `ts >= turn_start` (allow 3 s grace via a small await loop); missing → kill, outcome `canary`.
- `Session` → `_save_session(id)`.
- `Error` → limit markers → outcome `limit`; overflow markers → clear session, sentences `["My memory got full, starting a fresh conversation."]`; else `[f"{self.label} returned an error, check the log."]`.
- `Done(final_text)` → if no `Text` seen and `final_text` → feed it through the splitter.
- EOF without `Done` and `returncode != 0` → treat as `Error(stderr tail)`.
- `TimeoutError` → kill → `["Taking too long, cancelled."]`.
`interrupt()`: if `self._proc` alive: `send_signal(SIGINT)`, `wait` with `asyncio.timeout(self.s.interrupt_drain_s)`, on timeout `kill()`; clear `_proc`. `close()` = `interrupt()`. Default `spawn`:
```python
async def _default_spawn(argv, cwd, env):
    return await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)
```
stderr is drained to `workspace / "last-stderr.log"` by a side task.

- [ ] **Step 4: Run base tests** → PASS

- [ ] **Step 5: Live probe + fixture capture for `agy`** (requires the user's `agy` login; skip-marked)

`tests/test_brains_live.py` (create; add backends in later tasks):
```python
import json, os, shutil, subprocess
import pytest
from pathlib import Path

pytestmark = pytest.mark.live
FIX = Path(__file__).parent / "fixtures" / "brains"


def _capture(name, argv, cwd):
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=180)
    (FIX / f"{name}.jsonl").write_text(p.stdout)
    return p


@pytest.mark.skipif(shutil.which("agy") is None, reason="agy not installed")
def test_agy_plain_turn(tmp_path):
    p = _capture("antigravity-plain", ["agy", "-p", "Reply with exactly: pineapple. Nothing else.",
                                      "--output-format", "stream-json"], tmp_path)
    assert p.returncode == 0 and "pineapple" in p.stdout.lower()
```
Run: `uv run pytest -q -m live tests/test_brains_live.py -k agy`. Read the captured `tests/fixtures/brains/antigravity-plain.jsonl` and note the real event names. Then capture a tool turn: create `<tmp>/.agy/hooks.json` + `agy mcp add veronica-mac <venv python> -m veronica.tools.serve mac` and prompt `"Use the veronica-mac read_battery tool and tell me the percentage"` with a fake gate socket that always allows (a 10-line asyncio server in the test) → `antigravity-tool.jsonl`; and one with the native shell: `"Run the shell command: echo canary-ok"` → confirms the hook fires (`hook.log` non-empty). Decide during this step (and write the answers into `antigravity.py` docstring): (a) does `.agy/` in cwd hold `hooks.json`/`settings.json` (else use `~/.gemini/antigravity-cli/`), (b) does `@path` attach an image, (c) is there a system-prompt flag (`agy --help` showed none: use `AGENTS.md` in the workspace).

- [ ] **Step 6: Failing tests for `AntigravityBrain`** (`tests/test_backend_antigravity.py`)

Written against the captured fixtures:
```python
from pathlib import Path
from veronica.brain.backends.antigravity import AntigravityBrain
FIX = Path(__file__).parent / "fixtures" / "brains"


def test_argv_first_and_resumed_turns(tmp_path):
    s = Settings(home=tmp_path); b = AntigravityBrain(s, ToolGate(s, None))
    a = b.argv("hi", None, [], native=True)
    assert a[:2] == ["agy", "-p"] and "hi" in a and "--output-format" in a and a[a.index("--output-format") + 1] == "stream-json"
    assert "--conversation" not in a
    a2 = b.argv("hi", "conv-1", [tmp_path / "img-1.png"], native=True)
    assert a2[a2.index("--conversation") + 1] == "conv-1" and any(str(tmp_path / "img-1.png") in x for x in a2)


def test_workspace_files(tmp_path):
    s = Settings(home=tmp_path); b = AntigravityBrain(s, ToolGate(s, None))
    b.prepare_workspace("SYSTEM PROMPT", native=True)
    assert (b.workspace / "AGENTS.md").read_text() == "SYSTEM PROMPT"
    hooks = json.loads((b.workspace / ".agy" / "hooks.json").read_text())
    cmd = hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert cmd.endswith("-m veronica.brain.hook antigravity")
    perms = json.loads((b.workspace / ".agy" / "settings.json").read_text())["permissions"]
    assert "run_command" in perms["allow"]
    b.prepare_workspace("SYSTEM PROMPT", native=False)
    perms = json.loads((b.workspace / ".agy" / "settings.json").read_text())["permissions"]
    assert "run_command" in perms["deny"]


def test_parse_plain_fixture(tmp_path):
    s = Settings(home=tmp_path); b = AntigravityBrain(s, ToolGate(s, None))
    events = [e for line in (FIX / "antigravity-plain.jsonl").read_text().splitlines() if line.strip() for e in b.parse(line)]
    text = "".join(e.delta for e in events if isinstance(e, cli.Text))
    assert "pineapple" in text.lower()
    assert any(isinstance(e, cli.Session) for e in events) and isinstance(events[-1], cli.Done)


def test_parse_tool_fixture(tmp_path):
    s = Settings(home=tmp_path); b = AntigravityBrain(s, ToolGate(s, None))
    events = [e for line in (FIX / "antigravity-tool.jsonl").read_text().splitlines() if line.strip() for e in b.parse(line)]
    starts = [e for e in events if isinstance(e, cli.ToolStart)]
    assert starts and starts[0].tool.endswith("read_battery") and starts[0].native is False


async def test_end_to_end_with_fake_spawn(tmp_path):
    s = Settings(home=tmp_path)
    async def spawn(argv, cwd, env): return FakeProc((FIX / "antigravity-plain.jsonl").read_text().splitlines())
    b = AntigravityBrain(s, ToolGate(s, None), spawn=spawn)
    out = [x async for x in b.ask("say pineapple")]
    assert any("pineapple" in x.lower() for x in out)
```
(Import `FakeProc` from `tests/test_backend_cli.py` — move it to `tests/brains_fakes.py` so all backend tests share it.)

- [ ] **Step 7: Implement `antigravity.py`**

```python
class AntigravityBrain(CliBrain):
    name, label, binary = "antigravity", "Antigravity", "agy"

    def argv(self, text, session_id, image_paths, native):
        prompt = " ".join(f"@{p}" for p in image_paths) + (" " if image_paths else "") + text
        a = ["agy", "-p", prompt, "--output-format", "stream-json", "--print-timeout", "0"]
        if session_id:
            a += ["--conversation", session_id]
        if self.s.effort in ("low", "medium", "high"):
            a += ["--effort", self.s.effort]
        return a

    def env(self):
        return {}

    def prepare_workspace(self, prompt_text, native):
        (self.workspace / "AGENTS.md").write_text(prompt_text)
        d = self.workspace / ".agy"; d.mkdir(exist_ok=True)
        py = sys.executable
        (d / "hooks.json").write_text(json.dumps({"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [
            {"type": "command", "command": f"{py} -m veronica.brain.hook antigravity"}]}]}}, indent=2))
        native_tools = ["run_command", "write_file", "replace", "edit", "edit_file"]
        mcp = [f"mcp__veronica-{s}__*" for s in hook.OUR_SERVERS]
        perms = {"allow": mcp + (native_tools if native else []), "deny": [] if native else native_tools}
        (d / "settings.json").write_text(json.dumps({"permissions": perms}, indent=2))
        self._register_mcp()

    def _register_mcp(self):
        """`agy mcp add` is add-or-update; run once per process."""
        if getattr(self, "_mcp_registered", False): return
        for s in hook.OUR_SERVERS:
            subprocess.run(["agy", "mcp", "add", "--env", f"VERONICA_GATE_SOCK={self.s.gate_socket}",
                            "--env", "VERONICA_BRAIN=antigravity", f"veronica-{s}", sys.executable, "-m", "veronica.tools.serve", s],
                           capture_output=True, timeout=30)
        self._mcp_registered = True

    def parse(self, line):  # shapes from tests/fixtures/brains/antigravity-*.jsonl
        ...
```
`parse` maps: `init` → `Session(conversation_id)`; `step_update`/message deltas → `Text`; tool call events → `ToolStart(call_id, tool, input, native=not tool.startswith(("mcp__veronica-", "veronica-")))` / `ToolEnd`; `result` → `Done(final_text)` or `Error(message)` when status is error. `_register_mcp` is skipped in tests via `spawn` injection? No — it uses `subprocess.run`; give `CliBrain` an injectable `self._run = subprocess.run` and assert in `test_workspace_files` that it was called with `["agy", "mcp", "add", ...]` (patch to a recorder). Only `sys.executable` from inside the app bundle is the venv python (the launcher sets argv0 to the venv python) — document.

- [ ] **Step 8: Run** `uv run pytest -q tests/test_backend_cli.py tests/test_backend_antigravity.py` and the full suite → PASS. Add `antigravity_native_tools: bool = True` to `Settings` + `EditableField("bool", "Antigravity: allow its own shell", "Off = only Veronica's tools; on = its shell and file edits too, each asked through Veronica.", restart=False)`; test in `tests/test_config.py`.

- [ ] **Step 9: Commit**

```bash
git add veronica/brain/backends/cli.py veronica/brain/backends/antigravity.py veronica/config.py tests/brains_fakes.py tests/test_backend_cli.py tests/test_backend_antigravity.py tests/test_brains_live.py tests/fixtures/brains/ tests/test_config.py
git commit -m "feat(brain): CliBrain base with canary and limit detection; Antigravity backend"
```

---

### Task 4: `CodexBrain`

**Files:**
- Create: `veronica/brain/backends/codex.py`, `tests/test_backend_codex.py`, `tests/fixtures/brains/codex-plain.jsonl`, `codex-tool.jsonl`
- Modify: `tests/test_brains_live.py`, `veronica/config.py` (`codex_native_tools`)

**Interfaces:** Consumes `CliBrain`. Produces `CodexBrain(name="codex", label="Codex", binary="codex")`.

- [ ] **Step 1: Live capture** — add to `tests/test_brains_live.py`:
```python
@pytest.mark.skipif(shutil.which("codex") is None, reason="codex not installed")
def test_codex_plain_turn(tmp_path):
    p = _capture("codex-plain", ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(tmp_path),
                                 "Reply with exactly: pineapple. Nothing else."], tmp_path)
    assert p.returncode == 0 and "pineapple" in p.stdout.lower()
```
Run it (`-m live -k codex`), read the JSONL: expect `thread.started{thread_id}`, `turn.started`, `item.completed{item:{type:"agent_message", text}}`, `turn.completed`. Then capture a tool turn with `-c 'mcp_servers.veronica-mac.command="…"'` overrides and the always-allow fake gate → `codex-tool.jsonl` (`item.*{item:{type:"mcp_tool_call", server, tool, arguments}}`), and a shell turn with `.codex/hooks.json` + `-c features.hooks=true --dangerously-bypass-hook-trust` to confirm the hook fires (hook.log non-empty). If hooks don't fire in exec mode on 0.155.1, record that in the docstring and force `native=False` for codex (`-c sandbox_mode="read-only"`), keeping the canary.

- [ ] **Step 2: Failing tests** (`tests/test_backend_codex.py`), same shape as Task 3 Step 6:
```python
def test_argv(tmp_path):
    s = Settings(home=tmp_path); b = CodexBrain(s, ToolGate(s, None))
    a = b.argv("hi", None, [], native=True)
    assert a[:3] == ["codex", "exec", "--json"] and "-c" in a and 'approval_policy="never"' in a
    assert 'sandbox_mode="workspace-write"' in a and "--dangerously-bypass-hook-trust" in a and a[-1] == "hi"
    assert any(x.startswith("mcp_servers.veronica-mac.command=") for x in a)
    a2 = b.argv("hi", "thr-1", [tmp_path / "img-1.png"], native=False)
    assert a2[1:4] == ["exec", "resume", "thr-1"] and "-i" in a2 and 'sandbox_mode="read-only"' in a2
    assert f'model_reasoning_effort="{s.effort}"' in a2


def test_workspace_hooks(tmp_path):
    s = Settings(home=tmp_path); b = CodexBrain(s, ToolGate(s, None))
    b.prepare_workspace("SYS", native=True)
    hooks = json.loads((b.workspace / ".codex" / "hooks.json").read_text())
    assert hooks["hooks"]["PreToolUse"][0]["hooks"][0]["command"].endswith("-m veronica.brain.hook codex")


def test_parse_plain_fixture(...)   # Session(thread_id), Text from agent_message, Done
def test_parse_tool_fixture(...)    # ToolStart(tool="mcp__veronica-mac__read_battery" or f"{server}__{tool}", native=False); command_execution -> native=True with input {"command": ...}
def test_turn_failed_is_error(tmp_path):
    line = json.dumps({"type": "turn.failed", "error": {"message": "usage limit reached"}})
    assert isinstance(b.parse(line)[0], cli.Error)
```

- [ ] **Step 3: Implement `codex.py`**

```python
class CodexBrain(CliBrain):
    name, label, binary = "codex", "Codex", "codex"

    def _mcp_overrides(self):
        out = []
        for s in hook.OUR_SERVERS:
            out += ["-c", f'mcp_servers.veronica-{s}.command="{sys.executable}"',
                    "-c", f'mcp_servers.veronica-{s}.args=["-m","veronica.tools.serve","{s}"]',
                    "-c", f'mcp_servers.veronica-{s}.env.VERONICA_GATE_SOCK="{self.s.gate_socket}"',
                    "-c", f'mcp_servers.veronica-{s}.env.VERONICA_BRAIN="codex"',
                    "-c", f'mcp_servers.veronica-{s}.env.VERONICA_HOOK_LOG="{self.hook_log}"']
        return out

    def argv(self, text, session_id, image_paths, native):
        a = ["codex", "exec"] + (["resume", session_id] if session_id else []) + [
            "--json", "--skip-git-repo-check", "-C", str(self.workspace),
            "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="workspace-write"' if native else 'sandbox_mode="read-only"',
            "-c", f'sandbox_workspace_write.writable_roots=["{self.s.brain_cwd}"]',
            "-c", "features.hooks=true", "--dangerously-bypass-hook-trust",
            "-c", f"developer_instructions={_toml_str(self._prompt)}",
            "-c", "include_permissions_instructions=false",
            "-c", f'model_reasoning_effort="{self.s.effort}"',
        ] + self._mcp_overrides()
        for p in image_paths:
            a += ["-i", str(p)]
        return a + [text]
```
`prepare_workspace` stores `self._prompt = prompt_text` and writes `.codex/hooks.json`. `_toml_str` = `json.dumps(s)` (a JSON string is valid TOML basic string for our content — escape control chars; verify with `uv run python -c "import tomllib; tomllib.loads('x=' + json.dumps(prompt))"` in a test). `parse`: `thread.started` → `Session`; `item.completed` `agent_message` → `Text(text)` (Codex emits the whole message; `Text` still streams per item); `item.started` `command_execution` → `ToolStart(id, "shell", {"command": cmd}, native=True)`; `mcp_tool_call` → `ToolStart(id, f"mcp__{server}__{tool}", arguments, native=False)`; `item.completed` of those → `ToolEnd`; `turn.completed` → `Done`; `turn.failed`/`error` → `Error(message)`.

- [ ] **Step 4: Run tests + full suite** → PASS. Add `codex_native_tools` setting + `EditableField` (same copy pattern as Task 3).

- [ ] **Step 5: Commit**  `git commit -m "feat(brain): Codex backend"`

---

### Task 5: `CopilotBrain`

**Files:**
- Create: `veronica/brain/backends/copilot.py`, `tests/test_backend_copilot.py`, `tests/fixtures/brains/copilot-plain.jsonl`, `copilot-tool.jsonl`
- Modify: `tests/test_brains_live.py`, `veronica/config.py` (`copilot_native_tools`), `veronica/brain/hook.py` (`emit` for copilot if the key differs)

- [ ] **Step 1: Live capture** — `copilot -p "Reply with exactly: pineapple." --output-format json --silent --no-ask-user` in `tmp_path` (needs `/login` done). Then with `--additional-mcp-config '<json>'` (Task-3-style fake gate) and `.github/hooks/veronica.json` + `--allow-all-tools` for a shell turn. Confirm from the captured hook stdin (log it to a file in the test) whether the payload keys are `toolName`/`toolArgs` and from https://docs.github.com/en/copilot/reference/hooks-reference what the output key is; fix `hook.emit("copilot", ...)` and the `test_hook.py` copilot case if it differs from `{"permissionDecision": ...}`. Confirm `.github/copilot-instructions.md` is read in `-p` mode (ask "what is the secret word in your instructions?"); if not, prefix the prompt with the system text.

- [ ] **Step 2: Failing tests** (`tests/test_backend_copilot.py`) mirroring Task 4 Step 2: argv has `-p`, `--output-format json`, `--silent`, `--no-ask-user`, `--session-id <uuid>` on first turn (uuid4 stored in `self._new_session_id` so `Session` can be emitted even if the stream doesn't echo it), `--resume <id>` after, `--attachment <path>` per image, `--allow-all-tools` when native else `--allow-tool veronica-*`; `--additional-mcp-config` JSON contains all 7 servers with `env.VERONICA_GATE_SOCK`; workspace has `.github/hooks/veronica.json` and `.github/copilot-instructions.md`; parse of fixtures.

- [ ] **Step 3: Implement `copilot.py`** per the spec table (§3.2). `parse` per the captured JSONL (assistant deltas → `Text`; tool call events → `ToolStart` with `native = not tool.startswith("veronica-")`; final result → `Done`, error → `Error`).

- [ ] **Step 4: Run + full suite** → PASS. Add `copilot_native_tools` setting.

- [ ] **Step 5: Commit**  `git commit -m "feat(brain): Copilot backend"`

---

### Task 6: `QwenBrain`

**Files:**
- Create: `veronica/brain/backends/qwen.py`, `tests/test_backend_qwen.py`, `tests/fixtures/brains/qwen-plain.jsonl`, `qwen-tool.jsonl`
- Modify: `tests/test_brains_live.py`, `veronica/config.py` (`qwen_native_tools`)

- [ ] **Step 1: Live capture** — `qwen "Reply with exactly: pineapple." -o stream-json --approval-mode plan` (login needed). Tool turn with `.qwen/settings.json` (`mcpServers`, `trust: true`, `hooks.BeforeTool`) and `--approval-mode yolo`.

- [ ] **Step 2: Failing tests** — argv: positional prompt (no `-p`), `-o stream-json`, `--approval-mode yolo|plan`, `--system-prompt <text>`, `--session-id` first / `--resume` later, `@path` images; `.qwen/settings.json` content (mcpServers ×7 with env, `trust: true`, `timeout: 120000`, `hooks.BeforeTool` command ends with `-m veronica.brain.hook qwen`, `tools.exclude` when not native); parse fixtures (`init` → `Session(session_id)`, `message{role:assistant, delta}` → `Text`, `tool_use` → `ToolStart(native = tool not in our set)`, `tool_result` → `ToolEnd`, `result{status}` → `Done`/`Error`).

- [ ] **Step 3: Implement `qwen.py`** per the spec table.

- [ ] **Step 4: Run + full suite** → PASS. Add `qwen_native_tools` setting.

- [ ] **Step 5: Commit**  `git commit -m "feat(brain): Qwen Code backend"`

---

### Task 7: Registry, availability, `BrainSwitcher` with failover

**Files:**
- Create: `veronica/brain/switch.py`, `tests/test_backends_registry.py`, `tests/test_switch.py`
- Modify: `veronica/brain/backends/__init__.py`, `veronica/config.py`

**Interfaces:**
```python
# backends/__init__.py
@dataclass(frozen=True)
class BackendInfo:
    name: str; label: str; binary: str; install_cmd: str; login_cmd: str
    login_markers: tuple[str, ...]      # relative to home: files whose existence means "logged in"; "" = none checkable
    cls: type | None                    # ClaudeBrain / CodexBrain / ...

BACKENDS: dict[str, BackendInfo]        # ordered: claude, codex, antigravity, copilot, qwen

@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: Literal["ok", "not installed", "not logged in"]
    hint: str                            # spoken sentence

def check_backend(name, *, which=shutil.which, exists=None, home=None) -> Availability
def make_brain(name, settings, *, gate, on_tool=None, memory=None) -> Brain

# switch.py
class BrainSwitcher:
    def __init__(self, settings, *, gate, on_tool=None, memory=None, factory=make_brain, check=check_backend,
                 clock=time.monotonic, say=None, on_backend=None): ...
    brain: Brain                         # active
    preferred: str                       # settings.brain_backend
    standing_in: bool                    # active != preferred
    limited_until: dict[str, float]
    async def switch(self, name: str, *, manual: bool = True) -> Availability
    async def maybe_return(self) -> None            # before each turn: back to preferred if its cooldown passed
    async def failover(self, reason: str) -> str | None   # returns the new backend name or None
    def status_label(self) -> str                   # "Claude" / "Codex (for Claude)"
```
Hints (spoken): `"{label} isn't installed — run {install_cmd}, then {login_cmd}."`, `"{label} isn't logged in — run {login_cmd} in a terminal."`.

- [ ] **Step 1: Failing tests** (`tests/test_backends_registry.py`):
```python
def test_check_backend_matrix(tmp_path):
    assert check_backend("codex", which=lambda b: None).reason == "not installed"
    assert check_backend("codex", which=lambda b: "/x/codex", home=tmp_path).reason == "not logged in"
    (tmp_path / ".codex").mkdir(); (tmp_path / ".codex" / "auth.json").write_text("{}")
    assert check_backend("codex", which=lambda b: "/x/codex", home=tmp_path).ok
    hint = check_backend("copilot", which=lambda b: None).hint
    assert "npm i -g @github/copilot" in hint
    # claude: the SDK finds the CLI; only the binary is checked, login is assumed (the SDK errors loudly otherwise)
    assert check_backend("claude", which=lambda b: "/x/claude", home=tmp_path).ok


def test_make_brain_builds_each(tmp_path):
    s = Settings(home=tmp_path); g = ToolGate(s, None)
    for name in BACKENDS:
        b = make_brain(name, s, gate=g)
        assert b.name == name and b.gate is g
```
Login markers: codex `~/.codex/auth.json`; antigravity `~/.gemini/antigravity-cli/conversations` **or** a Keychain item — implement `exists` for antigravity as: marker dir exists, else `security find-generic-password -s <service>` (service name found in Task 3 by `security dump-keychain | grep -i antigravity`; if none found, use the directory marker only); copilot `~/.copilot/` file found in Task 5 (fallback: any file in `~/.copilot` newer than install); qwen `~/.qwen/oauth_creds.json`; claude: none.

`tests/test_switch.py`:
```python
class FakeBrain:
    def __init__(self, name, *, limit=False, reply=("ok.",)):
        self.name, self.limit, self.reply, self.closed, self.gate = name, limit, list(reply), False, None
    async def ask(self, text, images=()):
        if self.limit: raise LimitError("usage limit")
        for s in self.reply: yield s
    async def interrupt(self): pass
    async def close(self): self.closed = True


def make_switcher(tmp_path, avail, now, **settings):
    s = Settings(home=tmp_path, **settings); said = []; events = []
    factory = lambda name, settings, **kw: FakeBrain(name, limit=(name in settings_limits))
    ...
```
Cases: `switch("codex")` when available → new brain, old closed, prefs override `brain_backend="codex"` saved, `on_backend("codex", standing_in=False)` called; unavailable → returns the Availability, brain unchanged, nothing saved; `failover("usage limit")` picks the next in `brain_failover_order` skipping unavailable and cooled-down ones, sets `limited_until[preferred] = now + cooldown*60`, says "Claude hit its usage limit — switching to Codex.", `standing_in` True, `status_label() == "Codex (for Claude)"`; second failover from Codex goes to Antigravity and never back to Claude while cooled; none available → says "Claude hit its usage limit and no other brain is ready." and returns None; `maybe_return()` before cooldown expiry does nothing, after expiry switches back silently (`on_backend` called, nothing said, `standing_in` False); manual `switch("claude")` clears `limited_until["claude"]`; `brain_failover=False` → `failover` returns None without switching.

- [ ] **Step 2: Implement** `backends/__init__.py` and `switch.py` per the interfaces. `failover` does **not** re-run the user text itself — it only switches; the orchestrator (Task 8) re-runs. Settings added (all editable, `restart=False`, section "brain"):
  - `brain_backend: str = "claude"` → `EditableField("choice", "Brain", "Which assistant runs the thinking. Each uses its own login.", choices=("claude","codex","antigravity","copilot","qwen"))`
  - `brain_failover: bool = True` → `("bool", "Switch brains on usage limits", "When the current brain hits its usage limit, hand the request to the next available one and come back later.")`
  - `brain_failover_order: str = "claude,codex,antigravity,copilot,qwen"` → `("str", "Failover order", "Comma-separated backend names, tried in order.")`
  - `brain_limit_cooldown_min: int = 60` → `("int", "Limit cooldown (minutes)", "How long to wait before trying a brain that hit its limit again.", min=5, max=1440)`
  Tests in `tests/test_config.py` for defaults + choice validation (`Settings(brain_backend="nope")` → `ValueError` via a `field_validator`).

- [ ] **Step 3: Run + full suite** → PASS

- [ ] **Step 4: Commit**  `git commit -m "feat(brain): backend registry, availability checks, BrainSwitcher with usage-limit failover"`

---

### Task 8: Wire it in — orchestrator, intents, `__main__`, settings page, HUD label, menu bar

**Files:**
- Modify: `veronica/brain/intents.py`, `veronica/orchestrator.py`, `veronica/__main__.py`, `veronica/ui/settings/bridge.py`, `veronica/ui/settings/settings.js`, `veronica/ui/hud/hud.js`, `veronica/ui/hud/index.html` (if the label needs a node), `veronica/ui/menubar.py`
- Test: `tests/test_intents.py`, `tests/test_orchestrator.py`, `tests/test_settings_bridge.py`, `tests/test_settings_web.py` (live), `tests/test_hud_web.py` (live), `tests/test_menubar.py`

**Interfaces:**
```python
# intents.py
BrainAction = tuple[Literal["switch", "which"], str | None]
def match_brain_intent(text: str) -> BrainAction | None
# "switch to codex" / "use gemini"→None (dropped) / "use copilot" / "use antigravity" / "switch brain to qwen" /
# "back to claude" / "go back to claude" / Hinglish "codex pe switch karo", "copilot use karo" -> ("switch", name)
# "which brain are you on" / "which brain is this" / "which model are you using" / "who am i talking to" -> ("which", None)
```
Orchestrator: constructor gains `switcher: BrainSwitcher | None = None` (when given, `self.brain` becomes a property returning `self.switcher.brain`; `__main__` passes both). New `_brain_switch_turn(action)`; `run_forever` starts `GateServer(self.gate, self.s.gate_socket)` and stops it on exit; `_brain_turn` calls `await self.switcher.maybe_return()` first and catches `LimitError` around `handle_text`: `new = await self.switcher.failover(str(e))`; if `new`: `self._emit("tool", {"summary": f"{old_label}: usage limit — on {new_label}", "decision": "limit"})` and re-run `handle_text(text, images, lang=lang)` once (a second `LimitError` → failover again, at most `len(BACKENDS)` hops, then say the "no other brain is ready" line). `on_backend` callback → `self._emit("hud", {"backend": label})`, also emitted once at `run_forever` start.

- [ ] **Step 1: Failing intent tests** (`tests/test_intents.py`):
```python
@pytest.mark.parametrize("text,expected", [
    ("switch to codex", ("switch", "codex")), ("use copilot", ("switch", "copilot")),
    ("switch brain to antigravity", ("switch", "antigravity")), ("use qwen please", ("switch", "qwen")),
    ("back to claude", ("switch", "claude")), ("go back to claude", ("switch", "claude")),
    ("codex pe switch karo", ("switch", "codex")), ("copilot use karo", ("switch", "copilot")),
    ("which brain are you on", ("which", None)), ("which model are you using", ("which", None)),
    ("who am i talking to", ("which", None)),
    ("use gemini", None), ("switch to spanish", None), ("use a british voice", None), ("open codex", None),
])
def test_match_brain_intent(text, expected):
    assert match_brain_intent(text) == expected
```

- [ ] **Step 2: Implement `match_brain_intent`** with the same `_candidates_for`/normalize pipeline the other matchers use (see `match_voice_intent`): regexes `^(?:switch(?: brain)? to|use|change(?: brain)? to)\s+(claude|codex|antigravity|copilot|qwen)$`, `^(?:go )?back to claude$`, Hinglish `^(claude|codex|antigravity|copilot|qwen)\s+(?:pe|par)\s+switch\s+karo$`, `^(…)\s+use\s+karo$`; `which` phrases as a frozenset.

- [ ] **Step 3: Failing orchestrator tests** — in `tests/test_orchestrator.py` using the existing `build`/`build3` helpers plus a `FakeSwitcher`:
```python
class FakeSwitcher:
    def __init__(self, brain, avail=True, fail_to=None):
        self.brain, self.avail, self.fail_to = brain, avail, fail_to
        self.switched, self.returned, self.failovers = [], 0, []
        self.preferred = "claude"; self.standing_in = False
    async def switch(self, name, *, manual=True):
        self.switched.append(name)
        if not self.avail: return Availability(False, "not installed", "Codex isn't installed — run npm i -g @openai/codex, then codex login.")
        self.brain.name = name; return Availability(True, "ok", "")
    async def maybe_return(self): self.returned += 1
    async def failover(self, reason):
        self.failovers.append(reason)
        if self.fail_to: self.brain = self.fail_to; return self.fail_to.name
        return None
    def status_label(self): return self.brain.name.title()


async def test_switch_intent_speaks_and_switches():
    o, _, ev = build3(stt_texts=["switch to codex"]); o.switcher = FakeSwitcher(o.brain)
    await o.one_turn()
    assert o.switcher.switched == ["codex"] and o.tts.said == ["Switched to Codex."]
    assert o.brain.asked == []            # no brain round trip


async def test_switch_intent_unavailable_speaks_hint_and_keeps_brain():
    o, _, _ = build3(stt_texts=["use codex"]); o.switcher = FakeSwitcher(o.brain, avail=False)
    await o.one_turn()
    assert o.tts.said == ["Codex isn't installed — run npm i -g @openai/codex, then codex login."]


async def test_which_brain():
    o, _, _ = build3(stt_texts=["which brain are you on"]); o.switcher = FakeSwitcher(o.brain)
    await o.one_turn()
    assert o.tts.said == ["I'm on Claude."]


async def test_limit_error_fails_over_and_reruns_once():
    o, _, ev = build3(stt_texts=["what time is it in tokyo"])
    limited = LimitBrain(o)                       # ask() raises LimitError
    standin = FakeBrain2(o, replies=["It's 9 pm in Tokyo."]); standin.name = "codex"
    o.brain = limited; o.switcher = FakeSwitcher(limited, fail_to=standin)
    await o.one_turn()
    assert o.switcher.failovers == ["usage limit"]
    assert standin.asked == ["what time is it in tokyo"]
    assert "It's 9 pm in Tokyo." in o.tts.said
    assert ("tool", {"summary": "Claude: usage limit — on Codex", "decision": "limit"}) in ev


async def test_limit_error_without_standin_speaks_error():
    o, _, _ = build3(stt_texts=["hello"]); o.brain = LimitBrain(o); o.switcher = FakeSwitcher(o.brain)
    await o.one_turn()
    assert o.tts.said == ["Claude hit its usage limit and no other brain is ready."]


async def test_run_forever_starts_gate_server(tmp_path): ...   # GateServer started at s.gate_socket, socket file exists during run, removed after stop
async def test_hud_backend_event_on_switch(): ...             # ("hud", {"backend": "Codex"}) emitted
```
(`build3` returns `(orch, brain, events)`; the FakeBrain in that file records `asked`. Add `LimitBrain` there.)

- [ ] **Step 4: Implement the orchestrator changes**

In `one_turn`'s intent ladder add `brain_action = None if (<all earlier hits>) else match_brain_intent(text)` right before `quick_hit` (and add `brain_action is not None` to the later guards), with the branch:
```python
elif brain_action is not None:
    self.player.reset()
    await self._brain_switch_turn(brain_action)
```
```python
async def _brain_switch_turn(self, action):
    kind, name = action
    if self.switcher is None:
        await self.say("I can only use Claude right now."); return
    if kind == "which":
        label = BACKENDS[self.switcher.brain.name].label
        if self.switcher.standing_in:
            until = self.switcher.limited_until.get(self.switcher.preferred, 0)
            mins = max(1, round((until - self.switcher._clock()) / 60))
            await self.say(f"I'm on {label} — {BACKENDS[self.switcher.preferred].label} hit its limit, I'll try it again in {mins} minutes.")
        else:
            await self.say(f"I'm on {label}.")
        return
    if self.switcher.brain.name == name and not self.switcher.standing_in:
        await self.say(f"Already on {BACKENDS[name].label}."); return
    await self.brain.interrupt()
    self.brain.clear_trust()
    avail = await self.switcher.switch(name)
    await self.say(f"Switched to {BACKENDS[name].label}." if avail.ok else avail.hint)
```
`_brain_turn`:
```python
async def _brain_turn(self, text, images=(), *, lang=None):
    if self.switcher is not None:
        await self.switcher.maybe_return()
    for _hop in range(len(BACKENDS)):
        try:
            spoken = await self.handle_text(text, images, lang=lang)
            break
        except LimitError as e:
            if self.switcher is None:
                await self.say("Claude hit its usage limit."); return []
            old = BACKENDS[self.switcher.brain.name].label
            new = await self.switcher.failover(str(e))
            if new is None:
                return []            # the switcher already said "no other brain is ready"
            self._emit("tool", {"summary": f"{old}: usage limit — on {BACKENDS[new].label}", "decision": "limit"})
    else:
        return []
    ... (existing redirect handling unchanged)
```
`self.brain` property: `return self.switcher.brain if self.switcher is not None else self._brain` with a setter used by tests. `run_forever`: `self._gate_server = GateServer(self.gate, self.s.gate_socket); await self._gate_server.start()` in a try/finally with `stop()`; `self.gate` = `self.switcher.gate` if a switcher, else `self.brain.gate`. `on_backend` → `_emit("hud", {"backend": label})`; emit at start of `run_forever`.

`__main__.py` `build_orchestrator`: build `gate = ToolGate(s, confirm, on_tool=on_tool)` first, then `switcher = BrainSwitcher(s, gate=gate, on_tool=on_tool, memory=store, say=lambda t: holder["orch"].say(t), on_backend=lambda label, standing_in: holder["orch"].backend_changed(label, standing_in))`, pass `brain=switcher.brain, switcher=switcher`. Text mode (`--text`) gets the same switcher.

- [ ] **Step 5: Settings page** — `SETTING_SECTIONS["brain"]` += `("brain_backend", "brain_failover", "brain_failover_order", "brain_limit_cooldown_min", "codex_native_tools", "antigravity_native_tools", "copilot_native_tools", "qwen_native_tools")` and the matching `settingRow('brain', '…')` lines in `settings.js`; `_set_setting("brain_backend", …)` additionally calls `await orch.switcher.switch(value)` via `asyncio.run_coroutine_threadsafe(..., orch.loop)` (look at how `test_voice` / the update flow hand work to the loop) and returns `_fail(hint)` when unavailable. Tests: `tests/test_settings_bridge.py` (row present in state; set → switch called; unavailable → fail with hint), `tests/test_settings_web.py` live (rows render).

- [ ] **Step 6: HUD + menu bar** — `hud.js`: handle `payload.backend` in the `hud` case: set `brainEl.textContent = 'Brain: ' + payload.backend` (add `<div id="brain" class="brain"></div>` under `#status` in `index.html`, small muted text; hide when empty). `menubar.py` `_drain`: `hud` payload with `backend` → `self._brain_item.title = f"Brain: {payload['backend']}"`. Submenu `Brain` (built like `voice_menu`): one item per `BACKENDS`, callback `self._pick_brain` → `orch.request_brain_switch(name)` (schedules `_brain_switch_turn(("switch", name))` on the loop via the same mechanism the voice menu uses — `self._voice_action`); `_refresh_brain_menu` (on the 0.25 s timer, cheap): check-mark the active one, title `"Codex — standing in for Claude"` when standing in, disabled items titled `"Copilot (not installed)"`/`"(not logged in)"` from a cached `check_backend` result refreshed every 60 s. Tests in `tests/test_menubar.py` mirroring the voice-menu tests (`_drain` maps the backend payload; picking calls the orchestrator hook).

- [ ] **Step 7: Run** full suite + `uv run pytest -q -m live tests/test_hud_web.py tests/test_settings_web.py` → PASS

- [ ] **Step 8: Commit**  `git commit -m "feat(brain): switch brains by voice, menu and settings; HUD brain label; usage-limit failover wired into turns"`

---

### Task 9: README + live end-to-end tests + memory server binding notes

**Files:**
- Modify: `README.md`, `tests/test_brains_live.py`

- [ ] **Step 1: README "Brains" section** — table of the five backends with install + login commands (verbatim from `BACKENDS`), voice phrases ("switch to codex", "back to claude", "which brain are you on"), how the gate applies to external brains (our tools over stdio + the hook for their shell; the canary; the `… allow its own shell` toggles), failover (what she says, cooldown setting, silent return), known limits: timers set while on an external brain don't fire (the `pim` server runs in a child process), Antigravity's MCP entries are added to the user-level `agy` config under `veronica-*` names, Codex falls back to a read-only sandbox when its shell is off.

- [ ] **Step 2: Live end-to-end** — for each installed+logged-in CLI, one test that builds the real backend with the real `GateServer` on a `tmp_path` socket whose confirm auto-approves, asks "Use the read_battery tool and tell me the percentage", and asserts a `%` in the spoken sentences and a gate request logged; and one that asks "Run the shell command echo canary-ok" and asserts `hook.log` has a `canary-ok` key. Skip cleanly when `check_backend(name).ok` is False.

- [ ] **Step 3: Run** `uv run pytest -q` (hermetic) and `uv run pytest -q -m live tests/test_brains_live.py` (with logins) → report exact numbers.

- [ ] **Step 4: Commit**  `git commit -m "docs: brains — install, login, gate and failover"`

---

## Self-review

- **Spec coverage:** §1 protocol/package → T1, T7; §2 gate + server + client → T1, T2; §3.1 plumbing/canary/limits → T3; §3.2 per-backend → T3–T6; §3.3 hook → T2 (copilot key confirmed T5); §3.4 canary → T3 (base) + settings T3–T6; §3.5 serve → T2; §4 settings/HUD/menu/intents → T7, T8; §4a failover → T7 (switcher), T8 (re-run + card + which-brain wording); §5 docs → T9; §6 tests → each task, live → T3–T6, T9; §7 order → task order.
- **Placeholders:** the "…" in T3 Step 7 `parse` and T4/T5/T6 parse descriptions are deliberately bound to captured fixtures (spec §3.2: parsers are written against captured output); the mapping rules for each are stated in prose per event type. No TBDs.
- **Type consistency:** `Decision(allow, kind, message, heard)` used identically in gate/gateclient/serve/hook; `ToolGate` ctor `(settings, confirm, on_tool, frontmost, clock)` in T1/T2/T3 tests; `CliBrain(settings, gate, on_tool, memory, *, spawn, clock)` in T3–T6; `check_backend(name, *, which, exists, home)` in T7/T8; `BrainSwitcher.switch/maybe_return/failover/status_label/standing_in/preferred/limited_until` in T7/T8; HUD decisions `auto|ask|allowed|declined|redirected|preapproved|limit`.
