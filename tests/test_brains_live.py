"""Real-CLI tests for the external brains (`-m live`; deselected by
default). They need the vendor CLI installed and logged in, spawn it for
real, and — for the hook tests — merge our PreToolUse entry into the
user's own hook file (restored afterwards) and register our MCP servers
with the CLI. They also refresh the captured fixtures under
tests/fixtures/brains/ that the hermetic tests parse."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from veronica.brain.backends.antigravity import AntigravityBrain
from veronica.brain.gate import GateServer, ToolGate
from veronica.config import Settings
from veronica.orchestrator import ConfirmResult

pytestmark = pytest.mark.live
FIX = Path(__file__).parent / "fixtures" / "brains"
needs_agy = pytest.mark.skipif(shutil.which("agy") is None, reason="agy not installed")


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


def _capture(name, argv, cwd):
    p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=180)
    (FIX / f"{name}.jsonl").write_text(p.stdout)
    return p


class RecordingAntigravity(AntigravityBrain):
    """Keeps the raw stream so a run can be saved as a fixture."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.raw: list[str] = []

    def parse(self, line):
        self.raw.append(line)
        return super().parse(line)


async def _brain_with_gate(home, answers, cards):
    """An AntigravityBrain whose gate is served on home/gate.sock and whose
    confirm pops `answers`."""
    s = Settings(home=home)

    async def confirm(summary, detail=""):
        cards.append(("confirm", summary))
        a = answers.pop(0)
        return a if isinstance(a, ConfirmResult) else ConfirmResult("approved" if a else "denied")

    gate = ToolGate(s, confirm, on_tool=lambda su, d: cards.append((su, d)))
    srv = GateServer(gate, s.gate_socket)
    await srv.start()
    return RecordingAntigravity(s, gate, on_tool=lambda su, d: cards.append((su, d))), srv


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
