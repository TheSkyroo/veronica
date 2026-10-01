# Veronica Phase 2 — Mac Tools, Barge-in, Speech Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Veronica controls the Mac by voice through a risk-classified confirmation gate, can be interrupted by the wake word mid-answer, hears better (`small.en`, chime), and answers faster (pre-warm, pipelined TTS).

**Architecture:** A pure-function risk classifier decides allow/confirm per tool call; the existing voice gate only runs for "confirm". Mac actions are typed in-process MCP tools (`create_sdk_mcp_server`) that shell out with argv lists (never a shell string). Barge-in runs the wake listener as a concurrent task during thinking/speaking and, on detection, stops playback and interrupts the SDK turn. TTS is pipelined with a bounded queue.

**Tech Stack:** Python 3.12, `claude-agent-sdk` 0.2.152 (`tool`, `create_sdk_mcp_server`, `ClaudeSDKClient.interrupt`), `openwakeword`, `sounddevice`, `faster-whisper`, `kokoro-onnx`, `rumps`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-veronica-phase2-mac-tools-design.md` (builds on `2026-09-15-veronica-voice-agent-design.md`; Phase 1 merged at `e57267e`)

## Global Constraints

- macOS, Python ≥ 3.12, `uv run pytest -q` must stay pristine (no warnings). Live tests marked `@pytest.mark.live`.
- No API key; Claude Code login. `ANTHROPIC_API_KEY` is popped in `main()`.
- Tool gate: `classify(tool_name, input) -> "allow" | "confirm"`. Allow set: `Read, Glob, Grep, WebSearch, WebFetch`. Confirm: `Write, Edit, NotebookEdit`, unknown tools, `Bash` unless single simple safe command. `SAFE_BASH = {ls, cat, head, tail, pwd, date, cal, whoami, pbpaste, df, du, ps, which, echo, uptime, wc, file, stat}`; `open` only as `open -a <App>` or `open http(s)://…`. Forbidden chars anywhere in a Bash command: `| & ; > < $( ` (backtick) and newline.
- Mac tools (server name `mac`): `open_app`(allow), `open_url`(allow, http/https only), `clipboard_read`(allow), `clipboard_write`(confirm), `notify`(allow), `volume_get`(allow), `volume_set`(allow, clamp 0–100), `applescript`(confirm). `subprocess.run(argv, capture_output=True, text=True, timeout=10)`; errors → `is_error: True` content.
- Deny message stays exactly `user declined`. Auto-allow logs `auto-allow: <summary>` at INFO.
- Barge-in: only while state ∈ {thinking, speaking}; detection threshold `Settings.barge_threshold = 0.8`; on barge: `player.stop()`, cancel turn, `brain.interrupt()`, chime, state `listening`, no follow-up window. (Plan ruling: spec's "followup" barge dropped — that window already listens.)
- Speech: `whisper_model` default `small.en`; `vad_silence_ms` 800; chime 120 ms 880 Hz after wake, 100 ms 660 Hz when follow-up opens; both float32 24 kHz via `Player`.
- Existing spoken strings unchanged: "Sorry, didn't catch that.", "I have nothing to say to that.", "Taking too long, cancelled.", "Claude returned an error, check the log.", "Something went wrong, check the log.", "Claude Code isn't logged in.", `Run <summary>?`.
- Commit after every task; conventional `feat:`/`fix:` messages; NO `Co-Authored-By` trailer.
- Tests inject fakes with `monkeypatch.setattr`, never by direct class-attribute assignment.

---

## File structure

```
veronica/brain/policy.py          # classify(), SAFE_BASH, FORBIDDEN chars, MAC_TOOL_RISK   (new)
veronica/brain/agent.py           # gate uses classify; summarize_tool for mac tools; interrupt(); mcp_servers/allowed_tools
veronica/tools/__init__.py        # (new)
veronica/tools/mac.py             # MCP tools + mac_server + MAC_TOOL_NAMES                   (new)
veronica/audio/chime.py           # tone(freq_hz, ms) -> np.float32                            (new)
veronica/audio/wake.py            # wait(threshold=None) -> bool; stop()
veronica/orchestrator.py          # chime, pipelined TTS, warmup(), barge-in, confirming state, muted confirm
veronica/config.py                # whisper_model small.en, vad 800, barge_threshold, chime settings
veronica/__main__.py              # --text y/N prompt for confirm-class tools
veronica/ui/menubar.py            # "confirming" icon, warming icon
README.md                         # mic permission, whisper download, barge-in
scripts/check_dual_input.py       # live: two RawInputStreams at once                        (new)
tests/test_policy.py, tests/test_mac_tools.py, tests/test_chime.py (new); test_agent/test_orchestrator/test_wake/test_main/test_menubar/test_config (extended)
```

---

### Task 1: Risk classifier

**Files:**
- Create: `veronica/brain/policy.py`, `tests/test_policy.py`

**Interfaces:**
- Produces: `classify(tool_name: str, input: dict) -> Literal["allow","confirm"]`; constants `SAFE_BASH: frozenset[str]`, `FORBIDDEN = "|&;><`$\n"` (note `$(` is covered by `$`), `MAC_TOOL_RISK: dict[str, str]` keyed by short name (`open_app`…), `ALLOW_TOOLS = frozenset({"Read","Glob","Grep","WebSearch","WebFetch"})`.

- [ ] **Step 1: Write failing tests**

`tests/test_policy.py`:
```python
import pytest

from veronica.brain.policy import classify

ALLOW = "allow"
CONFIRM = "confirm"

CASES = [
    # built-ins
    ("Read", {"file_path": "/a"}, ALLOW),
    ("Glob", {"pattern": "*.py"}, ALLOW),
    ("Grep", {"pattern": "x"}, ALLOW),
    ("WebSearch", {"query": "weather"}, ALLOW),
    ("WebFetch", {"url": "https://x"}, ALLOW),
    ("Write", {"file_path": "/a"}, CONFIRM),
    ("Edit", {"file_path": "/a"}, CONFIRM),
    ("NotebookEdit", {}, CONFIRM),
    ("SomethingNew", {}, CONFIRM),
    # bash safe
    ("Bash", {"command": "ls -la ~/Desktop"}, ALLOW),
    ("Bash", {"command": "pbpaste"}, ALLOW),
    ("Bash", {"command": "date"}, ALLOW),
    ("Bash", {"command": "open -a Safari"}, ALLOW),
    ("Bash", {"command": "open -a 'Google Chrome'"}, ALLOW),
    ("Bash", {"command": "open https://example.com"}, ALLOW),
    ("Bash", {"command": "cat /etc/hosts"}, ALLOW),
    # bash confirm
    ("Bash", {"command": "rm -rf ~/x"}, CONFIRM),
    ("Bash", {"command": "sudo ls"}, CONFIRM),
    ("Bash", {"command": "ls; rm -rf ~"}, CONFIRM),
    ("Bash", {"command": "ls && rm x"}, CONFIRM),
    ("Bash", {"command": "cat a | grep b"}, CONFIRM),
    ("Bash", {"command": "echo hi > f"}, CONFIRM),
    ("Bash", {"command": "ls $(rm x)"}, CONFIRM),
    ("Bash", {"command": "ls `rm x`"}, CONFIRM),
    ("Bash", {"command": "ls\nrm x"}, CONFIRM),
    ("Bash", {"command": "open file:///etc/passwd"}, CONFIRM),
    ("Bash", {"command": "open -a Safari --args x"}, CONFIRM),
    ("Bash", {"command": "open /Applications"}, CONFIRM),
    ("Bash", {"command": "curl https://x"}, CONFIRM),
    ("Bash", {"command": "git push"}, CONFIRM),
    ("Bash", {"command": ""}, CONFIRM),
    ("Bash", {"command": "ls 'unterminated"}, CONFIRM),
    ("Bash", {}, CONFIRM),
    # mac tools
    ("mcp__mac__open_app", {"name": "Safari"}, ALLOW),
    ("mcp__mac__open_url", {"url": "https://x"}, ALLOW),
    ("mcp__mac__clipboard_read", {}, ALLOW),
    ("mcp__mac__clipboard_write", {"text": "x"}, CONFIRM),
    ("mcp__mac__notify", {"title": "a", "message": "b"}, ALLOW),
    ("mcp__mac__volume_get", {}, ALLOW),
    ("mcp__mac__volume_set", {"level": 30}, ALLOW),
    ("mcp__mac__applescript", {"script": "beep"}, CONFIRM),
    ("mcp__mac__unknown", {}, CONFIRM),
]


@pytest.mark.parametrize("tool,inp,expected", CASES)
def test_classify(tool, inp, expected):
    assert classify(tool, inp) == expected
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_policy.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'veronica.brain.policy'`

- [ ] **Step 3: Implement**

`veronica/brain/policy.py`:
```python
"""Risk classifier for tool calls: decide whether a call runs without asking."""
import shlex
from typing import Literal

Decision = Literal["allow", "confirm"]

ALLOW_TOOLS = frozenset({"Read", "Glob", "Grep", "WebSearch", "WebFetch"})

SAFE_BASH = frozenset({
    "ls", "cat", "head", "tail", "pwd", "date", "cal", "whoami", "pbpaste", "df",
    "du", "ps", "which", "echo", "uptime", "wc", "file", "stat",
})

# any of these anywhere in a Bash command → confirm (covers $(…), pipes, chains, redirects)
FORBIDDEN = "|&;><`$\n"

MAC_TOOL_RISK: dict[str, Decision] = {
    "open_app": "allow",
    "open_url": "allow",
    "clipboard_read": "allow",
    "clipboard_write": "confirm",
    "notify": "allow",
    "volume_get": "allow",
    "volume_set": "allow",
    "applescript": "confirm",
}

MAC_PREFIX = "mcp__mac__"


def _bash_is_safe(command: str) -> bool:
    if not command or any(ch in command for ch in FORBIDDEN):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    head = argv[0]
    if head == "open":
        if len(argv) == 3 and argv[1] == "-a":
            return True
        if len(argv) == 2 and argv[1].startswith(("http://", "https://")):
            return True
        return False
    return head in SAFE_BASH


def classify(tool_name: str, input: dict) -> Decision:
    if tool_name in ALLOW_TOOLS:
        return "allow"
    if tool_name == "Bash":
        return "allow" if _bash_is_safe(str(input.get("command", ""))) else "confirm"
    if tool_name.startswith(MAC_PREFIX):
        return MAC_TOOL_RISK.get(tool_name[len(MAC_PREFIX):], "confirm")
    return "confirm"
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_policy.py -q` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add veronica/brain/policy.py tests/test_policy.py && git commit -m "feat: risk classifier for tool calls"
```

---

### Task 2: Wire classifier into the Brain gate; mac summaries; interrupt()

**Files:**
- Modify: `veronica/brain/agent.py` (`summarize_tool`, `_can_use_tool`, add `interrupt`)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `classify` (Task 1).
- Produces: `Brain.interrupt() -> None` (async; safe when no client); `summarize_tool` handles `mcp__mac__*`.

- [ ] **Step 1: Write failing tests** (append to `tests/test_agent.py`; reuse existing `brain` fixture and fakes)

```python
def test_summarize_mac_tools():
    assert summarize_tool("mcp__mac__open_app", {"name": "Safari"}) == "Open Safari"
    assert summarize_tool("mcp__mac__open_url", {"url": "https://x.y"}) == "Open https://x.y"
    assert summarize_tool("mcp__mac__clipboard_write", {"text": "a" * 80}) == "Copy to clipboard: " + "a" * 60
    assert summarize_tool("mcp__mac__applescript", {"script": "tell app \"Music\" to play"}) == 'Run AppleScript: tell app "Music" to play'
    assert summarize_tool("mcp__mac__volume_get", {}) == "volume_get"


async def test_gate_auto_allows_safe_tools_without_confirm(brain):
    calls = []

    async def confirm(summary):
        calls.append(summary)
        return False

    brain._confirm = confirm
    res = await brain._can_use_tool("Read", {"file_path": "/x"}, None)
    assert res.behavior == "allow" and calls == []
    res = await brain._can_use_tool("Bash", {"command": "ls"}, None)
    assert res.behavior == "allow" and calls == []
    res = await brain._can_use_tool("Bash", {"command": "rm x"}, None)
    assert res.behavior == "deny" and calls == ["Bash: rm x"]


async def test_interrupt_without_client_is_noop(brain):
    await brain.interrupt()  # must not raise


async def test_interrupt_calls_client(brain):
    [s async for s in brain.ask("x")]
    client = FakeClient.instances[0]
    client.interrupts = 0

    async def interrupt():
        client.interrupts += 1

    client.interrupt = interrupt
    await brain.interrupt()
    assert client.interrupts == 1
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_agent.py -q` → 4 new FAIL (summaries wrong; auto-allow calls confirm; `Brain` has no `interrupt`).

- [ ] **Step 3: Implement**

In `veronica/brain/agent.py`:

```python
from veronica.brain.policy import classify

MAC_PREFIX = "mcp__mac__"


def summarize_tool(tool_name: str, input: dict) -> str:
    if tool_name.startswith(MAC_PREFIX):
        short = tool_name[len(MAC_PREFIX):]
        if short == "open_app":
            return f"Open {input.get('name', '')}"
        if short == "open_url":
            return f"Open {input.get('url', '')}"
        if short == "clipboard_write":
            return "Copy to clipboard: " + str(input.get("text", ""))[:60]
        if short == "applescript":
            return "Run AppleScript: " + str(input.get("script", ""))[:60]
        return short
    if tool_name in ("Write", "Edit") and "file_path" in input:
        return f"{tool_name} file {input['file_path']}"
    for key in ("command", "query", "url", "pattern", "file_path"):
        if key in input:
            return f"{tool_name}: {input[key]}"
    return tool_name
```

Replace `_can_use_tool`:
```python
    async def _can_use_tool(self, tool_name: str, input: dict, context):
        summary = summarize_tool(tool_name, input)
        if classify(tool_name, input) == "allow":
            log.info("auto-allow: %s", summary)
            return PermissionResultAllow(updated_input=input)
        log.info("tool request: %s", summary)
        if await self._confirm(summary):
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message="user declined")
```

Add after `close()`:
```python
    async def interrupt(self) -> None:
        """Stop the in-flight turn, if any. Safe to call when idle."""
        if self._client is None:
            return
        try:
            await self._client.interrupt()
        except Exception:
            log.exception("interrupt failed; closing client")
            await self.close()
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_agent.py -q` → all PASS. Check `test_can_use_tool_gate` still passes (it uses `Bash` `ls` → now auto-allowed → still `allow`; `Write` → confirm path → deny). If `test_can_use_tool_gate` asserted the confirm fake was called for Bash, update that assertion to use `rm x` instead of `ls`.

- [ ] **Step 5: Commit**

```bash
git add veronica/brain/agent.py tests/test_agent.py && git commit -m "feat: classify tool risk in brain gate, mac summaries, interrupt"
```

---

### Task 3: Mac MCP tools

**Files:**
- Create: `veronica/tools/__init__.py`, `veronica/tools/mac.py`, `tests/test_mac_tools.py`
- Modify: `veronica/brain/agent.py` `_options` (register server); `tests/test_agent.py` (`test_options_wired` add asserts)

**Interfaces:**
- Produces: `veronica.tools.mac.mac_server` (McpSdkServerConfig), `MAC_TOOL_NAMES: list[str]`, `run(argv: list[str], stdin: str | None = None) -> dict` helper, tool handlers `open_app, open_url, clipboard_read, clipboard_write, notify, volume_get, volume_set, applescript` (each `async (args: dict) -> dict`). Module attribute `_run_cls`-style injection: tests monkeypatch `veronica.tools.mac.subprocess.run`.

- [ ] **Step 1: Write failing tests**

`tests/test_mac_tools.py`:
```python
import subprocess

import pytest

from veronica.tools import mac


class Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done(out="OUT")

    monkeypatch.setattr(mac.subprocess, "run", run)
    return calls


def text(res):
    return res["content"][0]["text"]


async def test_open_app(fake_run):
    res = await mac.open_app.handler({"name": "Safari"})
    assert fake_run[0][0] == ["open", "-a", "Safari"]
    assert fake_run[0][1]["timeout"] == 10 and "shell" not in fake_run[0][1]
    assert text(res) == "ok" and not res.get("is_error")


async def test_open_url_rejects_non_http(fake_run):
    res = await mac.open_url.handler({"url": "file:///etc/passwd"})
    assert res["is_error"] and fake_run == []


async def test_open_url(fake_run):
    await mac.open_url.handler({"url": "https://x.y"})
    assert fake_run[0][0] == ["open", "https://x.y"]


async def test_clipboard_read(fake_run):
    assert text(await mac.clipboard_read.handler({})) == "OUT"
    assert fake_run[0][0] == ["pbpaste"]


async def test_clipboard_write(fake_run):
    await mac.clipboard_write.handler({"text": "hello"})
    assert fake_run[0][0] == ["pbcopy"] and fake_run[0][1]["input"] == "hello"


async def test_notify(fake_run):
    await mac.notify.handler({"title": "T", "message": "M"})
    assert fake_run[0][0][:2] == ["osascript", "-e"]
    assert 'display notification "M" with title "T"' in fake_run[0][0][2]


async def test_notify_escapes_quotes(fake_run):
    await mac.notify.handler({"title": 'a"b', "message": "m"})
    assert '\\"' in fake_run[0][0][2]


async def test_volume_set_clamps(fake_run):
    await mac.volume_set.handler({"level": 250})
    assert fake_run[0][0] == ["osascript", "-e", "set volume output volume 100"]
    await mac.volume_set.handler({"level": -5})
    assert fake_run[1][0] == ["osascript", "-e", "set volume output volume 0"]


async def test_volume_get(fake_run):
    await mac.volume_get.handler({})
    assert fake_run[0][0] == ["osascript", "-e", "output volume of (get volume settings)"]


async def test_applescript(fake_run):
    await mac.applescript.handler({"script": 'tell application "Music" to play'})
    assert fake_run[0][0] == ["osascript", "-e", 'tell application "Music" to play']


async def test_nonzero_exit_is_error(monkeypatch):
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: Done(rc=1, err="nope"))
    res = await mac.open_app.handler({"name": "Nope"})
    assert res["is_error"] and "nope" in text(res)


async def test_timeout_is_error(monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=10)

    monkeypatch.setattr(mac.subprocess, "run", run)
    res = await mac.applescript.handler({"script": "delay 100"})
    assert res["is_error"]


def test_server_and_names():
    assert mac.mac_server["name"] == "mac"
    assert set(mac.MAC_TOOL_NAMES) == {
        "open_app", "open_url", "clipboard_read", "clipboard_write",
        "notify", "volume_get", "volume_set", "applescript",
    }


@pytest.mark.live
async def test_live_open_finder():
    res = await mac.open_app.handler({"name": "Finder"})
    assert not res.get("is_error")
```

Note: `@tool`-decorated objects are `SdkMcpTool` with a `.handler` attribute (verify with `uv run python -c "from claude_agent_sdk import tool; t = tool('a','b',{})(lambda a: a); print([x for x in dir(t) if not x.startswith('_')])"` — if the attribute is named differently, use that name in tests and report it).

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_mac_tools.py -q` → FAIL `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`veronica/tools/__init__.py`: empty.

`veronica/tools/mac.py`:
```python
"""Typed macOS actions exposed to Claude as in-process MCP tools."""
import subprocess

from claude_agent_sdk import create_sdk_mcp_server, tool

TIMEOUT_S = 10


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def run(argv: list[str], stdin: str | None = None) -> dict:
    """Run argv (never a shell string) and map the result to MCP content."""
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _err(f"timed out after {TIMEOUT_S}s")
    except Exception as exc:  # e.g. FileNotFoundError
        return _err(str(exc))
    if done.returncode != 0:
        return _err(done.stderr.strip() or f"exit {done.returncode}")
    out = done.stdout.strip()
    return _ok(out or "ok")


def _q(s: str) -> str:
    """Quote for an AppleScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


@tool("open_app", "Open a macOS application by name, e.g. Safari", {"name": str})
async def open_app(args: dict) -> dict:
    return run(["open", "-a", str(args["name"])])


@tool("open_url", "Open an http(s) URL in the default browser", {"url": str})
async def open_url(args: dict) -> dict:
    url = str(args.get("url", ""))
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    return run(["open", url])


@tool("clipboard_read", "Read the current clipboard text", {})
async def clipboard_read(args: dict) -> dict:
    return run(["pbpaste"])


@tool("clipboard_write", "Replace the clipboard with the given text", {"text": str})
async def clipboard_write(args: dict) -> dict:
    return run(["pbcopy"], stdin=str(args.get("text", "")))


@tool("notify", "Show a macOS notification banner", {"title": str, "message": str})
async def notify(args: dict) -> dict:
    script = f'display notification "{_q(str(args.get("message", "")))}" with title "{_q(str(args.get("title", "")))}"'
    return run(["osascript", "-e", script])


@tool("volume_get", "Get system output volume (0-100)", {})
async def volume_get(args: dict) -> dict:
    return run(["osascript", "-e", "output volume of (get volume settings)"])


@tool("volume_set", "Set system output volume (0-100)", {"level": int})
async def volume_set(args: dict) -> dict:
    level = max(0, min(100, int(args.get("level", 0))))
    return run(["osascript", "-e", f"set volume output volume {level}"])


@tool("applescript", "Run an AppleScript snippet (powerful; user must confirm)", {"script": str})
async def applescript(args: dict) -> dict:
    return run(["osascript", "-e", str(args.get("script", ""))])


TOOLS = [open_app, open_url, clipboard_read, clipboard_write, notify, volume_get, volume_set, applescript]
MAC_TOOL_NAMES = [t.name for t in TOOLS]
mac_server = create_sdk_mcp_server(name="mac", version="1.0.0", tools=TOOLS)
```

Register in `veronica/brain/agent.py` `_options`:
```python
from veronica.tools.mac import MAC_TOOL_NAMES, mac_server
...
            mcp_servers={"mac": mac_server},
            allowed_tools=[f"mcp__mac__{n}" for n in MAC_TOOL_NAMES],
```
(`allowed_tools` only makes the tools available; `can_use_tool` still gates them.)

Append to `test_options_wired` in `tests/test_agent.py`:
```python
    assert "mac" in o.mcp_servers
    assert "mcp__mac__open_app" in o.allowed_tools
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_mac_tools.py tests/test_agent.py -q` → PASS. Run live once: `uv run pytest tests/test_mac_tools.py -m live -q` (Finder comes to front).

- [ ] **Step 5: Live text-mode smoke**

Run: `uv run python -m veronica --text "open Safari"` → expect `[tool] Open Safari -> allowed` (auto-allow logs at INFO; text mode prints), Safari opens, spoken reply.

- [ ] **Step 6: Commit**

```bash
git add veronica/tools veronica/brain/agent.py tests/test_mac_tools.py tests/test_agent.py && git commit -m "feat: mac MCP tools (open, clipboard, notify, volume, applescript)"
```

---

### Task 4: Speech settings + chime

**Files:**
- Create: `veronica/audio/chime.py`, `tests/test_chime.py`
- Modify: `veronica/config.py`, `tests/test_config.py`, `veronica/orchestrator.py` (chime after wake + on follow-up), `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `chime.tone(freq_hz: float, ms: int, sample_rate: int = 24000, volume: float = 0.3) -> np.ndarray` (float32, with 5 ms fade in/out); `Settings.barge_threshold=0.8`, `Settings.chime_wake_hz=880`, `Settings.chime_followup_hz=660`; `Orchestrator.chime(freq_hz, ms)` async (plays via `player`, skipped when muted).

- [ ] **Step 1: Write failing tests**

`tests/test_chime.py`:
```python
import numpy as np

from veronica.audio.chime import tone


def test_tone_shape_and_range():
    t = tone(880, 120)
    assert t.dtype == np.float32
    assert len(t) == 24000 * 120 // 1000
    assert np.abs(t).max() <= 0.3 + 1e-6
    assert abs(t[0]) < 1e-3 and abs(t[-1]) < 1e-3  # faded ends
```

Append to `tests/test_config.py::test_defaults`:
```python
    assert s.whisper_model == "small.en"
    assert s.vad_silence_ms == 800
    assert s.barge_threshold == 0.8
    assert s.chime_wake_hz == 880 and s.chime_followup_hz == 660
```
(and change the existing `vad_silence_ms == 700` assertion to 800).

Append to `tests/test_orchestrator.py` (fakes from that file):
```python
async def test_chime_after_wake_and_on_followup():
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    await o.one_turn()
    # chime samples go through player.play like speech; count plays: wake chime + 2 sentences + followup chime
    assert o.player.played == 4


async def test_chime_skipped_when_muted():
    o, _ = build()
    o.muted = True
    await o.chime(880, 120)
    assert o.player.played == 0
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_chime.py tests/test_config.py tests/test_orchestrator.py -q` → FAIL (no module; defaults differ; play count 2).

- [ ] **Step 3: Implement**

`veronica/audio/chime.py`:
```python
import numpy as np


def tone(freq_hz: float, ms: int, sample_rate: int = 24000, volume: float = 0.3) -> np.ndarray:
    """Short sine burst with 5 ms fades, float32 mono."""
    n = sample_rate * ms // 1000
    t = np.arange(n, dtype=np.float32) / sample_rate
    y = np.sin(2 * np.pi * freq_hz * t).astype(np.float32) * volume
    fade = max(1, sample_rate * 5 // 1000)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y
```

`veronica/config.py` changes: `vad_silence_ms: int = 800`; `whisper_model: str = "small.en"`; add under wake word: `barge_threshold: float = 0.8`; add section:
```python
    # chimes (Hz)
    chime_wake_hz: int = 880
    chime_followup_hz: int = 660
```

`veronica/orchestrator.py`: import `from veronica.audio.chime import tone`; add method:
```python
    async def chime(self, freq_hz: float, ms: int) -> None:
        if self.muted:
            return
        async with self._speech_lock:
            self.player.reset()
            await self.player.play(tone(freq_hz, ms))
```
In `one_turn`: first line after `_set("listening")` → `await self.chime(self.s.chime_wake_hz, 120)`; right after `self._set("followup")` → `await self.chime(self.s.chime_followup_hz, 100)`.

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest -q` → all PASS (check `test_full_turn`'s `player.played` expectations if any; update counts only where chimes now play).

- [ ] **Step 5: First-run download**

Run: `uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8'); print('small.en ready')"` (downloads ~470 MB once).
Run: `uv run pytest tests/test_stt.py -m live -q` → PASS with small.en.

- [ ] **Step 6: Commit**

```bash
git add veronica/audio/chime.py veronica/config.py veronica/orchestrator.py tests && git commit -m "feat: small.en STT, longer VAD tail, wake and follow-up chimes"
```

---

### Task 5: Warm-up + pipelined TTS

**Files:**
- Modify: `veronica/orchestrator.py` (`warmup()`, `handle_text` pipeline), `veronica/speech/stt.py` (nothing), `veronica/ui/menubar.py` (warming icon), `veronica/__main__.py` (`_text_mode` calls warmup? no — text mode is one-shot, skip)
- Test: `tests/test_orchestrator.py`, `tests/test_menubar.py`

**Interfaces:**
- Produces: `Orchestrator.warmup() -> None` (async; one dummy synth + one dummy transcribe of 1 s zeros; logs timings; tolerant of `stt is None`); `Orchestrator.ready: bool`; menubar state `"warming"` → icon `…` shown as `V …` before ready (reuse thinking icon).

- [ ] **Step 1: Write failing tests** (append to `tests/test_orchestrator.py`)

```python
async def test_warmup_touches_tts_and_stt():
    o, _ = build()
    await o.warmup()
    assert o.tts.said == ["ok"] and o.ready is True


async def test_warmup_without_stt():
    o, _ = build()
    o.stt = None
    await o.warmup()
    assert o.ready is True


async def test_pipelined_tts_preserves_order_and_overlaps_synth_with_play():
    import asyncio

    order = []

    class SlowTTS:
        async def asynth(self, text):
            order.append(("synth", text))
            await asyncio.sleep(0.01)
            return np.zeros(10, dtype=np.float32), 24000

    class SlowPlayer:
        def __init__(self): self.played = []; self.resets = 0
        async def play(self, s):
            order.append(("play", len(self.played))); self.played.append(s); await asyncio.sleep(0.02)
        def stop(self): pass
        def reset(self): self.resets += 1

    class Brain3:
        async def ask(self, text):
            for s in ["A.", "B.", "C."]:
                yield s

    o, _ = build()
    o.tts, o.player, o.brain = SlowTTS(), SlowPlayer(), Brain3()
    out = await o.handle_text("x")
    assert out == ["A.", "B.", "C."]
    assert [t for k, t in order if k == "synth"] == ["A.", "B.", "C."]
    assert len(o.player.played) == 3
    # synth of B must start before play of A finishes: synth B appears before play 1
    assert order.index(("synth", "B.")) < order.index(("play", 1))
```

`tests/test_menubar.py`: extend the state test: `app._on_state("warming"); app._refresh(None); assert app.title == "V …"`.

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_orchestrator.py tests/test_menubar.py -q` → FAIL (no `warmup`/`ready`; ordering assertion fails because synth B happens after play A completes).

- [ ] **Step 3: Implement**

In `Orchestrator.__init__`: `self.ready = False`.

```python
    async def warmup(self) -> None:
        """Load models before the first turn so the first answer isn't slow."""
        self._set("warming")
        t0 = time.monotonic()
        await self.tts.asynth("ok")
        if self.stt is not None:
            await self.stt.atranscribe(np.zeros(16000, dtype=np.int16))
        log.info("warmup done in %.1fs", time.monotonic() - t0)
        self.ready = True
        self._set("idle")
```
(add `import numpy as np` at top.)

Replace the body of `handle_text` with a producer/consumer pipeline:
```python
    async def handle_text(self, text: str) -> list[str]:
        """Ask the brain and speak each sentence; synth N+1 overlaps playback of N."""
        self._set("thinking")
        t0 = time.monotonic()
        spoken: list[str] = []
        self.player.reset()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        async def producer():
            try:
                async for sent in self.brain.ask(text):
                    spoken.append(sent)
                    samples, _ = await self.tts.asynth(sent)
                    await queue.put((sent, samples))
            finally:
                await queue.put(None)

        prod = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                sent, samples = item
                if len(spoken) and sent == spoken[0]:
                    self._set("speaking")
                    log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
                if not self.muted:
                    async with self._speech_lock:
                        await self.player.play(samples)
            await prod
        except BaseException:
            prod.cancel()
            with contextlib.suppress(BaseException):
                await prod
            raise
        if not spoken:
            await self.say("I have nothing to say to that.")
        return spoken
```
(add `import contextlib`.) Note: `say()` is still used by chime/confirm/errors; `handle_text` no longer calls `say()` for sentences.

`veronica/ui/menubar.py`: `ICONS["warming"] = "…"`; in `_run_loop` before `run_forever`: `self._loop.run_until_complete(self._orch.warmup())` then `run_until_complete(self._orch.run_forever())`.

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest -q` → all PASS. Existing `test_handle_text_speaks_each_sentence` asserts `tts.said == ["Sure.", "Done."]` and `player.played == 2` — still true. `test_confirm_does_not_race_handle_text_speech` (ordering `["First.", "Run Bash: ls?", "Second."]`) — the producer synthesizes "Second." before confirm's say; if that test now fails on `tts.said` order, change its assertion to check the **play** order (add a `played_texts` list to the test's player fake fed via a dict from samples id → text) rather than synth order, and note it in the report.

- [ ] **Step 5: Commit**

```bash
git add veronica/orchestrator.py veronica/ui/menubar.py tests && git commit -m "feat: model warm-up and pipelined TTS"
```

---

### Task 6: WakeWord `stop()` + threshold override; dual-input live check

**Files:**
- Modify: `veronica/audio/wake.py`, `tests/test_wake.py`
- Create: `scripts/check_dual_input.py`

**Interfaces:**
- Produces: `WakeWord.wait(threshold: float | None = None) -> bool` (True = detected, False = stopped); `WakeWord.stop() -> None` (thread-safe flag; `_wait` checks it every frame and clears it on entry).

- [ ] **Step 1: Write failing tests** (append to `tests/test_wake.py`)

```python
async def test_wait_returns_true_on_detection(monkeypatch):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    w = WakeWord(Settings(), frames=lambda: frames("..w"))
    assert await w.wait() is True


async def test_stop_returns_false(monkeypatch):
    import asyncio
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)
    w = WakeWord(Settings(), frames=lambda: frames("." * 100000))
    task = asyncio.create_task(w.wait())
    await asyncio.sleep(0.01)
    w.stop()
    assert await task is False


async def test_threshold_override(monkeypatch):
    monkeypatch.setattr(WakeWord, "_model_cls", FakeModel)  # scores 0.9 on 'w'
    w = WakeWord(Settings(), frames=lambda: frames("w....."))
    task = __import__("asyncio").create_task(w.wait(threshold=0.95))
    await __import__("asyncio").sleep(0.01)
    w.stop()
    assert await task is False   # 0.9 < 0.95 → never detected
```
(`frames()` in that file yields zeros forever after the pattern, so the stop path terminates within one frame.)

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_wake.py -q` → FAIL (`wait()` returns None; no `stop`; no `threshold` kwarg).

- [ ] **Step 3: Implement** (in `veronica/audio/wake.py`)

```python
import threading
...
    def __init__(...):
        ...
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    async def wait(self, threshold: float | None = None) -> bool:
        return await asyncio.to_thread(self._wait, threshold if threshold is not None else self.s.wake_threshold)

    def _wait(self, threshold: float) -> bool:
        self._stop.clear()
        self._model.reset()
        for frame in self._frames():
            if self._stop.is_set():
                return False
            chunk = np.frombuffer(frame, dtype=np.int16)
            scores = self._model.predict(chunk)
            if scores[self._key] >= threshold:
                return True
        return False
```
Update the two original tests to `assert await w.wait() is True` where they just awaited.

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_wake.py -q` → PASS.

- [ ] **Step 5: Dual-input live check script**

`scripts/check_dual_input.py`:
```python
"""Live check: can two RawInputStreams read the mic concurrently on this Mac?"""
import threading
import time

import sounddevice as sd


def reader(tag, n=1280, secs=2.0):
    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=n) as s:
        t0 = time.time()
        frames = 0
        while time.time() - t0 < secs:
            s.read(n)
            frames += 1
        print(f"{tag}: {frames} frames ok")


a = threading.Thread(target=reader, args=("A",))
b = threading.Thread(target=reader, args=("B",))
a.start(); time.sleep(0.2); b.start(); a.join(); b.join()
print("DUAL INPUT OK")
```
Run: `uv run python scripts/check_dual_input.py`. Record the result in the report. Expected on macOS/CoreAudio: both print ~25 frames and `DUAL INPUT OK`. If it raises `PortAudioError`, Task 7 must implement the fallback (confirm pauses the barge listener) — flag it as a concern.

- [ ] **Step 6: Commit**

```bash
git add veronica/audio/wake.py tests/test_wake.py scripts/check_dual_input.py && git commit -m "feat: stoppable wake listener with threshold override"
```

---

### Task 7: Barge-in

**Files:**
- Modify: `veronica/orchestrator.py` (`_run_with_barge`, `one_turn`), `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `WakeWord.wait(threshold)->bool`, `WakeWord.stop()`, `Brain.interrupt()`, `Player.stop()/reset()`.
- Produces: `Orchestrator._run_with_barge(coro) -> bool` (True if barged); `one_turn` loop restarts recording after a barge (no follow-up chime).

- [ ] **Step 1: Write failing tests** (append to `tests/test_orchestrator.py`)

```python
class BargeWake:
    """wait() returns True after `after` calls when barge=True, else blocks until stop()."""
    def __init__(self, barge_on_call=None):
        self.calls = 0; self.stops = 0; self.barge_on_call = barge_on_call
        self._ev = __import__("asyncio").Event()
    async def wait(self, threshold=None):
        self.calls += 1
        if self.barge_on_call == self.calls:
            await __import__("asyncio").sleep(0.005)
            return True
        await self._ev.wait(); self._ev.clear(); return False
    def stop(self):
        self.stops += 1; self._ev.set()


class SlowBrain:
    def __init__(self): self.interrupts = 0
    async def ask(self, text):
        yield "One."
        await __import__("asyncio").sleep(0.05)
        yield "Two."
    async def interrupt(self): self.interrupts += 1


async def test_barge_in_stops_speech_and_relistens():
    # call 1 = main wake (we call one_turn directly, so calls start at the barge listener)
    o, states = build(rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16), None], stt_texts=["first", "second"])
    o.wake = BargeWake(barge_on_call=1)
    o.brain = SlowBrain()
    await o.one_turn()
    assert o.brain.interrupts == 1
    assert o.player.stops >= 1
    assert "listening" in states[states.index("speaking") + 1:]     # re-listened after barge
    assert o.brain.interrupts == 1


async def test_no_barge_listener_stopped_when_turn_ends():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    o.wake = BargeWake()  # never barges
    await o.one_turn()
    assert o.wake.stops == 1  # listener stopped once when handle_text finished


async def test_barge_uses_barge_threshold():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hi"])
    seen = []
    class W(BargeWake):
        async def wait(self, threshold=None):
            seen.append(threshold); return await super().wait(threshold)
    o.wake = W()
    await o.one_turn()
    assert seen == [0.8]
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_orchestrator.py -q` → new tests FAIL (`BargeWake.wait` never awaited; `stops == 0`).

- [ ] **Step 3: Implement** (in `veronica/orchestrator.py`)

```python
    async def _run_with_barge(self, coro) -> bool:
        """Run a turn coroutine; return True if the wake word interrupted it."""
        turn = asyncio.ensure_future(coro)
        listener = asyncio.create_task(self.wake.wait(threshold=self.s.barge_threshold))
        try:
            done, _ = await asyncio.wait({turn, listener}, return_when=asyncio.FIRST_COMPLETED)
            if listener in done and listener.result():
                log.info("barge-in")
                self.player.stop()
                turn.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await turn
                await self.brain.interrupt()
                return True
            self.wake.stop()
            await listener
            turn.result()  # re-raise turn errors
            return False
        finally:
            if not listener.done():
                self.wake.stop()
                with contextlib.suppress(BaseException):
                    await listener
```
`one_turn` becomes:
```python
    async def one_turn(self) -> None:
        self._set("listening")
        await self.chime(self.s.chime_wake_hz, 120)
        pcm = await self.recorder.capture(max_s=self.s.listen_wait_s)
        if pcm is None:
            self._set("idle")
            return
        while True:
            text = await self.stt.atranscribe(pcm)
            log.info("heard=%r", text)
            if not text:
                self.player.reset()
                await self.say("Sorry, didn't catch that.")
            else:
                barged = await self._run_with_barge(self.handle_text(text))
                if barged:
                    self._set("listening")
                    await self.chime(self.s.chime_wake_hz, 120)
                    pcm = await self.recorder.capture(max_s=self.s.listen_wait_s)
                    if pcm is None:
                        break
                    continue
            self._set("followup")
            await self.chime(self.s.chime_followup_hz, 100)
            pcm = await self.recorder.capture(max_s=max(1, self.s.followup_window_s))
            if pcm is None:
                break
        self._set("idle")
```

Self-trigger note: the barge listener runs at `barge_threshold` (0.8) for the whole turn, including while Veronica speaks. (Plan ruling: constant threshold instead of the spec's "only while playing" — simpler, same effect.)

If Task 6's dual-input check FAILED: additionally, in `confirm()`, before `recorder.capture`, call `self.wake.stop()` and after transcription restart is NOT possible from inside (the listener task belongs to `_run_with_barge`) — instead set `self._barge_paused = True` so `_run_with_barge`, on seeing the listener return False while the turn is alive, waits for `self._barge_resume` (an `asyncio.Event` set by `confirm()` after capture) and then re-creates the listener task. Implement that only if needed and add a test.

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest -q` → all PASS, no "Task was destroyed but it is pending" warnings.

- [ ] **Step 5: Live manual check** (needs a human): `uv run python -m veronica`; ask something long ("tell me about the moon in three sentences"); say "hey jarvis" while she speaks → speech stops, chime, listening. Record what happened in the report (or "not run — needs human").

- [ ] **Step 6: Commit**

```bash
git add veronica/orchestrator.py tests/test_orchestrator.py && git commit -m "feat: barge-in via wake word during thinking and speaking"
```

---

### Task 8: Backlog fixes, `--text` y/N, confirming state, README

**Files:**
- Modify: `veronica/orchestrator.py`, `veronica/ui/menubar.py`, `veronica/__main__.py`, `README.md`, `tests/test_orchestrator.py`, `tests/test_menubar.py`, `tests/test_main.py`

**Interfaces:**
- Produces: state `"confirming"` (icon `?`); `confirm()` returns False immediately when muted; `run_forever` sets `idle` after a wake retry sleep; `_text_mode` prompts `Run <summary>? [y/N] ` for confirm-class tools using `classify`.

- [ ] **Step 1: Write failing tests**

`tests/test_orchestrator.py`:
```python
async def test_confirm_when_muted_denies_silently():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    o.muted = True
    assert await o.confirm("Bash: rm x") is False
    assert o.tts.said == []


async def test_confirming_state_emitted():
    o, states = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=["yes"])
    await o.confirm("Bash: rm x")
    assert "confirming" in states


async def test_wake_retry_returns_to_idle(monkeypatch):
    import asyncio
    o, states = build()
    class W:
        n = 0
        async def wait(self, threshold=None):
            self.n += 1
            if self.n == 1: raise RuntimeError("no mic")
            raise asyncio.CancelledError
        def stop(self): pass
    o.wake = W()
    async def nosleep(_): pass
    monkeypatch.setattr(asyncio, "sleep", nosleep)
    import pytest
    with pytest.raises(asyncio.CancelledError):
        await o.run_forever()
    assert states[-2:] == ["error", "idle"]
```

`tests/test_main.py`:
```python
async def test_text_mode_prompts_for_confirm_class(monkeypatch, tmp_home, capsys):
    import veronica.__main__ as m
    prompts = []
    monkeypatch.setattr(m, "_ask_stdin", lambda prompt: (prompts.append(prompt), "y")[1])
    stub = _stub_orchestrator(monkeypatch, handle=lambda: ["Hi."])   # reuse existing stub helper from this file
    await m._text_mode("x")
    ok = await stub.brain._confirm_decision("Bash", {"command": "rm x"}, "Bash: rm x")
    assert ok is True and prompts == ["Run Bash: rm x? [y/N] "]
```
(Adapt to the stub helper that already exists in `tests/test_main.py`; if none exists, build the same stub inline as the other `_text_mode` tests do.)

`tests/test_menubar.py`: `app._on_state("confirming"); app._refresh(None); assert app.title == "V ?"`.

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_orchestrator.py tests/test_main.py tests/test_menubar.py -q` → FAIL.

- [ ] **Step 3: Implement**

`veronica/orchestrator.py` `confirm()`:
```python
    async def confirm(self, summary: str) -> bool:
        if self.muted:
            log.info("confirm skipped (muted): %s", summary)
            return False
        prev = self.state
        self._set("confirming")
        try:
            async with self._speech_lock:
                self.player.reset()
                await self._say_unlocked(f"Run {summary}?")
                pcm = await self.recorder.capture(max_s=max(1, self.s.confirm_listen_s))
                if pcm is None:
                    return False
                heard = await self.stt.atranscribe(pcm)
        finally:
            self._set(prev)
        ok = self.is_confirmation(heard)
        log.info("confirm heard=%r -> %s", heard, ok)
        return ok
```
`run_forever`: after `await asyncio.sleep(self.s.wake_retry_s)` add `self._set("idle")`.

`veronica/ui/menubar.py`: `ICONS["confirming"] = "?"`.

`veronica/__main__.py`: the text-mode confirm becomes classify-aware. `Brain` already auto-allows "allow"-class tools before calling `_confirm`, so `_confirm` is only reached for confirm-class tools. Implement:
```python
def _ask_stdin(prompt: str) -> str:
    return input(prompt)


async def _text_mode(text: str) -> None:
    orch = build_orchestrator(settings, audio=False)

    async def confirm(summary: str) -> bool:
        answer = await asyncio.to_thread(_ask_stdin, f"Run {summary}? [y/N] ")
        ok = answer.strip().lower() in ("y", "yes")
        print(f"[tool] {summary} -> {'allowed' if ok else 'declined'}")
        return ok

    orch.brain._confirm = confirm
    try:
        print("[text mode] safe tools run automatically; risky tools ask y/N on this terminal")
        for sent in await orch.handle_text(text):
            print(sent)
    finally:
        await orch.brain.close()
```
Update the existing text-mode tests' banner assertion to the new banner string, and the `[tool] Bash: ls -> allowed` expectation: since the stub calls `brain._confirm("Bash: ls")` directly, fake `_ask_stdin` to return "y" in that test.

`README.md` — add under Setup/Run:
```
- macOS will ask for Microphone access for your terminal app on first run (System Settings → Privacy & Security → Microphone).
- First run downloads the whisper `small.en` model (~470 MB).
- Say the wake word while Veronica is talking to interrupt her (barge-in).
- Risky actions (writing files, shell commands that change things, AppleScript, clipboard writes) ask "Run …?" — answer "yes" or "no".
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest -q` → all PASS, pristine. `uv run pytest tests/test_menubar.py -v --log-cli-level=ERROR` → no ERROR records.

- [ ] **Step 5: Live text smoke**

Run: `uv run python -m veronica --text "what is on my clipboard"` → auto-allowed `clipboard_read`, spoken answer. Run: `uv run python -m veronica --text "copy the word hello to my clipboard"` → prompt `Run Copy to clipboard: hello? [y/N] ` → `y` → done.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: confirming state, muted confirm, wake-retry idle, y/N in text mode, README"
```

---

## Self-review

- **Spec coverage:** §1 classifier → T1, gate wiring → T2. §2 mac tools + registration + summaries → T3 (+T2 summaries). §3 barge-in → T6 (stop/threshold, dual-input check) + T7; spec's follow-up-window barge dropped by plan ruling (stated in Global Constraints); self-trigger guard implemented as constant 0.8 (ruling stated in T7). §4 small.en/VAD/chime → T4; pre-warm + pipelined TTS → T5. §5 backlog → T8 (wake-retry idle, muted confirm, y/N, confirming state, README). §6 tests → each task; live checks: T3 (Finder, Safari), T4 (small.en), T6 (dual input), T7 (manual barge), T8 (clipboard).
- **Placeholders:** none.
- **Type consistency:** `WakeWord.wait(threshold) -> bool` used identically in T6/T7 fakes; `Brain.interrupt()` T2 ↔ T7; `classify` T1 ↔ T2 ↔ T8 (text mode relies on Brain auto-allow, doesn't call classify directly); `Orchestrator.chime(freq_hz, ms)` T4 ↔ T7; `player.played/stops/resets` counters as in Phase 1 test fakes.
