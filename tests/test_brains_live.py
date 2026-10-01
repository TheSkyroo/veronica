"""Real-CLI tests for the external brains (`-m live`; deselected by
default). They need the vendor CLI installed and logged in, spawn it for
real, and — for the hook tests — merge our PreToolUse entry into the
user's own hook file (restored afterwards) and register our MCP servers
with the CLI. They also refresh the captured fixtures under
tests/fixtures/brains/ that the hermetic tests parse."""
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from veronica.brain.backends import BACKENDS, check_backend, make_brain
from veronica.brain.backends.antigravity import AntigravityBrain
from veronica.brain.backends.claude import ClaudeBrain
from veronica.brain.backends.codex import CodexBrain
from veronica.brain.backends.copilot import CopilotBrain
from veronica.brain.gate import GateServer, ToolGate
from veronica.tools import registry, screen
from veronica.brain.switch import BrainSwitcher
from veronica.config import Settings
from veronica.orchestrator import ConfirmResult

pytestmark = pytest.mark.live
FIX = Path(__file__).parent / "fixtures" / "brains"
needs_agy = pytest.mark.skipif(shutil.which("agy") is None, reason="agy not installed")
needs_codex = pytest.mark.skipif(shutil.which("codex") is None, reason="codex not installed")
needs_copilot = pytest.mark.skipif(shutil.which("copilot") is None, reason="copilot not installed")


@pytest.fixture
def short_home():
    """AF_UNIX socket paths are capped at ~104 bytes on macOS and the hook /
    MCP children get the gate socket as an absolute path, so the test
    home lives directly under /tmp rather than pytest's long tmp_path."""
    d = Path(tempfile.mkdtemp(prefix="vb-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def agy_hooks_file():
    """Back up ~/.gemini/config/hooks.json (which may not exist) and put it
    back exactly as it was."""
    f = Path.home() / ".gemini" / "config" / "hooks.json"
    before = f.read_text() if f.exists() else None
    try:
        yield f
    finally:
        if before is None:
            if f.exists():
                f.unlink()
        else:
            f.write_text(before)


@pytest.fixture
def copilot_hooks_file():
    """Back up ~/.copilot/hooks/veronica.json (which may not exist) and put
    it back exactly as it was."""
    f = Path.home() / ".copilot" / "hooks" / "veronica.json"
    before = f.read_text() if f.exists() else None
    try:
        yield f
    finally:
        if before is None:
            f.unlink(missing_ok=True)
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(before)


def _capture(name, argv, cwd):
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=180)
    (FIX / f"{name}.jsonl").write_text(p.stdout)
    return p


class _Recording:
    """Keeps the raw stream so a run can be saved as a fixture."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.raw: list[str] = []

    def parse(self, line):
        self.raw.append(line)
        return super().parse(line)


class RecordingAntigravity(_Recording, AntigravityBrain):
    pass


class RecordingCodex(_Recording, CodexBrain):
    pass


class RecordingCopilot(_Recording, CopilotBrain):
    pass


async def _gate_server(home, answers, cards, seen=None):
    """A ToolGate served on home/gate.sock whose confirm pops `answers`;
    every tool name that reaches `decide` (from the socket or in-process)
    is appended to `seen`."""
    s = Settings(home=home)

    async def confirm(summary, detail=""):
        cards.append(("confirm", summary))
        a = answers.pop(0)
        return a if isinstance(a, ConfirmResult) else ConfirmResult("approved" if a else "denied")

    gate = ToolGate(s, confirm, on_tool=lambda su, d: cards.append((su, d)))
    if seen is not None:
        decide = gate.decide

        async def recording_decide(tool, inp):
            seen.append(tool)
            return await decide(tool, inp)

        gate.decide = recording_decide
    srv = GateServer(gate, s.gate_socket, run_tool=registry.call_tool)
    await srv.start()
    return s, gate, srv


async def _brain_with_gate(home, answers, cards, brain_cls=RecordingAntigravity):
    """A brain whose gate is served on home/gate.sock and whose confirm
    pops `answers`."""
    s, gate, srv = await _gate_server(home, answers, cards)
    return brain_cls(s, gate, on_tool=lambda su, d: cards.append((su, d))), srv


@needs_agy
def test_agy_plain_turn(tmp_path):
    p = _capture("antigravity-plain", ["agy", "-p", "Reply with exactly: pineapple. Nothing else.",
                                      "--output-format", "stream-json"], tmp_path)
    assert p.returncode == 0 and "pineapple" in p.stdout.lower()


@needs_agy
async def test_agy_persistent_plain_turn(short_home, agy_hooks_file):
    b, srv = await _brain_with_gate(short_home, [], [])
    try:
        out = [x async for x in b.ask("Reply with exactly: pineapple. Nothing else.")]
        assert any("pineapple" in x.lower() for x in out), out
        assert b._proc is not None and b.s.session_file_for("antigravity").read_text()
        assert b.scope_file.read_text() == b.s.session_file_for("antigravity").read_text()
        # second turn on the same child
        out2 = [x async for x in b.ask("Now reply with exactly: kiwi. Nothing else.")]
        assert any("kiwi" in x.lower() for x in out2), out2
    finally:
        await b.close()
        await srv.stop()


@needs_agy
async def test_agy_native_shell_gated_and_allowed(short_home, agy_hooks_file):
    proof = short_home / "canary-ok.txt"
    cards = []
    b, srv = await _brain_with_gate(short_home, [True] * 5, cards)
    try:
        # `touch` is confirm-class, so this proves the whole chain: hook -> gate -> confirm -> ran
        out = [x async for x in b.ask(f"Run the shell command `touch {proof} && echo canary-ok` with your "
                                      "run_command tool and reply with its output only.")]
        assert any("canary-ok" in x for x in out), out
        assert proof.exists()
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert any(e["call"] == "run_command" and "canary-ok" in e["key"] for e in entries), entries
        assert any(c[0] == "confirm" for c in cards), cards       # the gate was really asked
        assert b._proc is not None                                # canary did not trip
        merged = json.loads(agy_hooks_file.read_text())
        assert any(b.hook_command() in json.dumps(e) for e in merged["hooks"]["PreToolUse"])
    finally:
        (FIX / "antigravity-tool.jsonl").write_text("\n".join(b.raw) + "\n")
        await b.close()
        await srv.stop()


@needs_agy
async def test_agy_native_shell_denied_does_not_run(short_home, agy_hooks_file):
    proof = short_home / "denied-proof.txt"
    cards = []
    b, srv = await _brain_with_gate(short_home, [False] * 5, cards)
    try:
        out = [x async for x in b.ask(f"Run the shell command `touch {proof}` with your run_command tool, "
                                      "then reply with one short sentence saying whether it ran.")]
        assert not proof.exists(), out
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert any(str(proof) in e["key"] for e in entries), entries
        assert any(c[0] == "confirm" for c in cards)
    finally:
        await b.close()
        await srv.stop()


@needs_agy
async def test_agy_mcp_tool_through_serve(short_home, agy_hooks_file):
    cards = []
    b, srv = await _brain_with_gate(short_home, [True] * 5, cards)
    try:
        out = [x async for x in b.ask("Call the veronica-mac MCP server's volume_get tool and tell me "
                                      "the volume level in one short sentence.")]
        assert any(any(ch.isdigit() for ch in x) for x in out), out
        # ours are unwrapped from call_mcp_tool and let through by the hook (tools.serve gates them)
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert not any(e["call"] == "call_mcp_tool" for e in entries), entries
        assert ("volume_get", "auto") in cards, cards
    finally:
        await b.close()
        await srv.stop()


# -- codex --------------------------------------------------------------------
@needs_codex
def test_codex_plain_turn(tmp_path):
    p = _capture("codex-plain", ["codex", "exec", "--json", "--skip-git-repo-check", "-C", str(tmp_path),
                                 "Reply with exactly: pineapple. Nothing else."], tmp_path)
    assert p.returncode == 0 and "pineapple" in p.stdout.lower()


@needs_codex
async def test_codex_brain_plain_turn_and_resume(short_home):
    b, srv = await _brain_with_gate(short_home, [], [], RecordingCodex)
    try:
        out = [x async for x in b.ask("Reply with exactly: pineapple. Nothing else.")]
        assert any("pineapple" in x.lower() for x in out), out
        sid = b.s.session_file_for("codex").read_text()
        assert sid and b._proc is None
        out2 = [x async for x in b.ask("What fruit did you just name? One word.")]
        assert any("pineapple" in x.lower() for x in out2), out2       # resumed the thread
        assert b.s.session_file_for("codex").read_text() == sid
    finally:
        await b.close()
        await srv.stop()


@needs_codex
async def test_codex_native_shell_gated_and_allowed(short_home):
    proof = short_home / "canary-ok.txt"
    cards = []
    b, srv = await _brain_with_gate(short_home, [True] * 5, cards, RecordingCodex)
    try:
        out = [x async for x in b.ask(f"Run the shell command `touch {proof} && echo canary-ok` and reply "
                                      "with its output only.")]
        assert any("canary-ok" in x for x in out), out
        assert proof.exists()
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert any(e["call"] == "Bash" and "canary-ok" in e["key"] for e in entries), entries
        assert any(c[0] == "confirm" for c in cards), cards       # the gate was really asked
        assert b.s.codex_native_tools is True                     # canary did not trip
    finally:
        (FIX / "codex-shell.jsonl").write_text("\n".join(b.raw) + "\n")
        await b.close()
        await srv.stop()


@needs_codex
async def test_codex_native_shell_denied_does_not_run(short_home):
    proof = short_home / "denied-proof.txt"
    cards = []
    b, srv = await _brain_with_gate(short_home, [False] * 5, cards, RecordingCodex)
    try:
        out = [x async for x in b.ask(f"Run the shell command `touch {proof}`, then reply with one short "
                                      "sentence saying whether it ran.")]
        assert not proof.exists(), out
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert any(str(proof) in e["key"] for e in entries), entries
        assert any(c[0] == "confirm" for c in cards)
        assert b.s.codex_native_tools is True
    finally:
        await b.close()
        await srv.stop()


@needs_codex
async def test_codex_mcp_tool_through_serve(short_home):
    cards = []
    b, srv = await _brain_with_gate(short_home, [True] * 5, cards, RecordingCodex)
    try:
        out = [x async for x in b.ask("Call the veronica-mac MCP server's volume_get tool and tell me "
                                      "the volume level in one short sentence.")]
        assert any(any(ch.isdigit() for ch in x) for x in out), out
        assert any('"mcp_tool_call"' in l and "volume_get" in l for l in b.raw), b.raw
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert not any("volume_get" in json.dumps(e) for e in entries), entries   # hook lets ours through
        assert ("volume_get", "auto") in cards, cards                           # tools.serve gated it
    finally:
        (FIX / "codex-mcp.jsonl").write_text("\n".join(b.raw) + "\n")
        await b.close()
        await srv.stop()


@needs_codex
async def test_codex_screenshot_is_captured_by_this_process(short_home):
    """The TCC bug: the capture must happen wherever the gate lives — in the
    app, not in the `tools.serve` child the CLI spawned, which is a
    different binary as far as macOS Screen Recording is concerned. Here
    "the app" is pytest, so what this proves is that the capture ran on
    THIS side of the socket and its image came back through Codex."""
    cards, seen = [], []
    here = os.getpid()
    took = []
    real = screen.capture_screenshot

    def capture(region):
        took.append(os.getpid())
        return real(region)

    s, gate, srv = await _gate_server(short_home, [True] * 5, cards, seen)
    screen.capture_screenshot = capture
    b = RecordingCodex(s, gate, on_tool=lambda su, d: cards.append((su, d)))
    try:
        [x async for x in b.ask("Call the veronica-screen MCP server's screenshot tool with "
                                "region 'screen', then say in one short sentence what you see.")]
        assert seen.count("mcp__screen__screenshot") == 1, seen
        assert took == [here], took                       # captured in the gate's process
        shots = [l for l in b.raw if "mcp_tool_call" in l and "screenshot" in l]
        assert shots and "couldn't capture" not in "".join(shots).lower(), shots
    finally:
        # No fixture for this one: the stream carries a picture of the
        # user's screen, which has no business in the repo.
        screen.capture_screenshot = real
        await b.close()
        await srv.stop()


# -- copilot ------------------------------------------------------------------
@needs_copilot
def test_copilot_plain_turn(tmp_path):
    p = _capture("copilot-plain", ["copilot", "-p", "Reply with exactly: pineapple. Nothing else.",
                                   "--output-format", "json", "--silent", "--no-ask-user", "--disable-builtin-mcps"],
                 tmp_path)
    assert p.returncode == 0 and "pineapple" in p.stdout.lower()
    assert json.loads(p.stdout.splitlines()[-1])["type"] == "result"


@needs_copilot
async def test_copilot_brain_plain_turn_and_resume(short_home, copilot_hooks_file):
    b, srv = await _brain_with_gate(short_home, [], [], RecordingCopilot)
    try:
        out = [x async for x in b.ask("Reply with exactly: pineapple. Nothing else.")]
        assert any("pineapple" in x.lower() for x in out), out
        sid = b.s.session_file_for("copilot").read_text()
        assert sid == b._new_session_id and b._proc is None
        assert copilot_hooks_file.exists()
        out2 = [x async for x in b.ask("What fruit did you just name? One word.")]
        assert any("pineapple" in x.lower() for x in out2), out2       # resumed the session
        assert b.s.session_file_for("copilot").read_text() == sid
    finally:
        await b.close()
        await srv.stop()
    assert not copilot_hooks_file.exists()                              # close() removes our hook file


@needs_copilot
async def test_copilot_native_shell_gated_and_allowed(short_home, copilot_hooks_file):
    proof = short_home / "canary-ok.txt"
    cards = []
    b, srv = await _brain_with_gate(short_home, [True] * 5, cards, RecordingCopilot)
    try:
        out = [x async for x in b.ask(f"Run the shell command `touch {proof} && echo canary-ok` with your bash "
                                      "tool and reply with its output only.")]
        assert any("canary-ok" in x for x in out), out
        assert proof.exists()
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert any(e["call"] == "bash" and "canary-ok" in e["key"] for e in entries), entries
        assert any(c[0] == "confirm" for c in cards), cards       # the gate was really asked
        assert b.s.copilot_native_tools is True                   # canary did not trip
    finally:
        (FIX / "copilot-shell.jsonl").write_text("\n".join(b.raw) + "\n")
        await b.close()
        await srv.stop()


@needs_copilot
async def test_copilot_native_shell_denied_does_not_run(short_home, copilot_hooks_file):
    proof = short_home / "denied-proof.txt"
    cards = []
    b, srv = await _brain_with_gate(short_home, [False] * 5, cards, RecordingCopilot)
    try:
        out = [x async for x in b.ask(f"Run the shell command `touch {proof}` with your bash tool, then reply "
                                      "with one short sentence saying whether it ran.")]
        assert not proof.exists(), out
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert any(str(proof) in e["key"] for e in entries), entries
        assert any(c[0] == "confirm" for c in cards)
        assert b.s.copilot_native_tools is True
    finally:
        await b.close()
        await srv.stop()


@needs_copilot
async def test_copilot_mcp_tool_through_serve(short_home, copilot_hooks_file):
    cards = []
    b, srv = await _brain_with_gate(short_home, [True] * 5, cards, RecordingCopilot)
    try:
        out = [x async for x in b.ask("Call the veronica-mac MCP server's volume_get tool and tell me "
                                      "the volume level in one short sentence.")]
        assert any(any(ch.isdigit() for ch in x) for x in out), out
        assert any('"tool.execution_start"' in l and "volume_get" in l for l in b.raw), b.raw
        entries = [json.loads(l) for l in b.hook_log.read_text().splitlines()]
        assert not any("volume_get" in json.dumps(e) for e in entries), entries   # hook lets ours through
        assert ("volume_get", "auto") in cards, cards                           # tools.serve gated it
        assert cards.count(("volume_get", "auto")) == 1                         # and gated it once
    finally:
        (FIX / "copilot-mcp.jsonl").write_text("\n".join(b.raw) + "\n")
        await b.close()
        await srv.stop()


# -- end to end through the registry (every brain, the shared gate) -----------
VOLUME_PROMPT = "Use the volume_get tool and tell me the result in one short sentence."


@pytest.mark.parametrize("name", list(BACKENDS))
async def test_every_brain_answers_through_the_gate(name, short_home, agy_hooks_file, copilot_hooks_file):
    """`make_brain(name)` on a real GateServer: one turn that calls our
    `volume_get` (auto-allowed) speaks a sentence, and the gate saw exactly
    one request for it — over the socket from tools.serve for the CLI
    brains, in-process (can_use_tool) for Claude."""
    avail = check_backend(name)
    if not avail.ok:
        pytest.skip(f"{name}: {avail.reason}")
    cards, seen = [], []
    s, gate, srv = await _gate_server(short_home, [True] * 5, cards, seen)
    b = make_brain(name, s, gate=gate, on_tool=lambda su, d: cards.append((su, d)))
    assert isinstance(b, BACKENDS[name].cls) and b.name == name
    try:
        out = [x async for x in b.ask(VOLUME_PROMPT)]
        assert out and any(x.strip() for x in out), out                    # a sentence was spoken
        assert seen.count("mcp__mac__volume_get") == 1, seen              # gated exactly once
        assert ("volume_get", "auto") in cards, cards
        assert not any(c[0] == "confirm" for c in cards), cards            # read-only: no question asked
    finally:
        await b.close()
        await srv.stop()


async def test_claude_brain_in_process_through_the_gate(short_home):
    """The Claude backend runs our MCP servers in-process (no tools.serve
    child, no hook): the same prompt reaches the gate via can_use_tool."""
    avail = check_backend("claude")
    if not avail.ok:
        pytest.skip(f"claude: {avail.reason}")
    cards, seen = [], []
    s, gate, srv = await _gate_server(short_home, [True] * 5, cards, seen)
    b = ClaudeBrain(s, gate=gate, on_tool=lambda su, d: cards.append((su, d)))
    try:
        out = [x async for x in b.ask(VOLUME_PROMPT)]
        assert out and any(x.strip() for x in out), out
        assert seen == ["mcp__mac__volume_get"], seen
        assert ("volume_get", "auto") in cards, cards
        assert s.session_file.read_text().strip()                          # session saved for resume
        assert not (short_home / "backends" / "claude").exists()           # no CLI workspace / hook log
    finally:
        await b.close()
        await srv.stop()


async def test_switcher_starts_on_codex(short_home):
    """`BrainSwitcher.start()` with the default preference activates Codex
    on a Mac where it is installed and logged in — no stand-in."""
    avail = check_backend("codex")
    if not avail.ok:
        pytest.skip(f"codex: {avail.reason}")
    said = []

    async def say(text):
        said.append(text)

    s, gate, srv = await _gate_server(short_home, [], [])
    assert s.brain_backend == "codex"
    sw = BrainSwitcher(s, gate=gate, say=say)
    try:
        await sw.start()
        assert sw.brain.name == "codex" and isinstance(sw.brain, CodexBrain)
        assert sw.status_label() == "Codex"
        assert sw.standing_in is False and said == []
    finally:
        await sw.brain.close()
        await srv.stop()
