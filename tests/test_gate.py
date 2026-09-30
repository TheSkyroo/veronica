import time
from pathlib import Path

import pytest

from veronica.brain import gate as gate_mod
from veronica.brain.gate import ToolGate
from veronica.config import Settings
from veronica.orchestrator import ConfirmResult
from veronica.tools.computer_events import Front

FINDER = Front(app="Finder", bundle_id="com.apple.finder", window_title="Desktop", pid=1)


def make(answers, *, front=FINDER, now=None, said=None, **settings):
    calls, cards = [], []
    # Nothing auto-allowed unless a test says so: clipboard_write is on the
    # shipped list, and it's the stand-in confirm-class tool here.
    settings.setdefault("auto_allow_tools", [])

    async def confirm(summary, detail="", *, question=None):
        calls.append((summary, detail, question))
        a = answers.pop(0)
        return a if isinstance(a, ConfirmResult) else ConfirmResult("approved" if a else "denied")

    async def say(text):
        if said is not None:
            said.append(text)

    clock = (lambda: now[0]) if now is not None else time.monotonic
    g = ToolGate(Settings(**settings), confirm, on_tool=lambda s, d: cards.append((s, d)),
                 frontmost=lambda: front, clock=clock, say=say)
    return g, calls, cards


async def test_allow_class_is_auto_without_asking():
    g, calls, cards = make([])
    d = await g.decide("mcp__mac__volume_get", {})
    assert d.allow and d.kind == "auto" and calls == [] and cards == [("volume_get", "auto")]


async def test_confirm_class_asks_and_yes_allows():
    g, calls, _ = make([True])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert d.allow and d.kind == "approved" and calls[0][0] == "Copy to clipboard: hi"


async def test_no_denies_with_user_declined():
    g, _, _ = make([False])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert not d.allow and d.kind == "denied" and d.message == "user declined"


async def test_other_answer_becomes_redirect():
    g, _, _ = make([ConfirmResult("other", "open it in the other profile")])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
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
    # a second confirm is needed after clear_trust; answers list is empty -> IndexError proves it asked
    with pytest.raises(IndexError):
        await g.decide("mcp__computer__computer_click", {"x": 5, "y": 6})
    assert calls[-1][0] == "Click (5, 6)"


async def test_preapproval_covers_first_confirm_call_only():
    now = [10.0]
    g, calls, cards = make([True], now=now)
    g.begin_turn(7)
    g.preapprove(7, until=30.0)
    d1 = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert d1.allow and d1.kind == "preapproved" and calls == [] and cards[-1] == ("Copy to clipboard: hi", "preapproved")
    d2 = await g.decide("mcp__mac__clipboard_write", {"text": "yo"})
    assert d2.allow and d2.kind == "approved" and len(calls) == 1


async def test_preapproval_never_for_always_confirm():
    now = [10.0]
    g, calls, _ = make([True], now=now)
    g.begin_turn(1); g.preapprove(1, until=30.0)
    d = await g.decide("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"})
    assert d.kind == "approved" and len(calls) == 1


async def test_allowlisted_shortcut_runs_without_asking():
    g, calls, cards = make([], shortcut_allowlist=["Morning", "Pay Rent"])
    d = await g.decide("mcp__mac__run_shortcut", {"name": "morning"})
    assert d.allow and d.kind == "auto" and calls == []
    assert cards[-1] == ("Run the shortcut 'morning'", "auto")


async def test_unlisted_shortcut_asks():
    g, calls, _ = make([True], shortcut_allowlist=["Morning"])
    d = await g.decide("mcp__mac__run_shortcut", {"name": "Wipe Disk"})
    assert d.allow and d.kind == "approved"
    assert calls[0][0] == "Run the shortcut 'Wipe Disk'"


async def test_shortcuts_ask_by_default():
    g, calls, _ = make([True])
    assert (await g.decide("mcp__mac__run_shortcut", {"name": "Morning"})).kind == "approved"
    assert len(calls) == 1


async def test_message_send_is_asked_even_when_preapproved():
    now = [10.0]
    g, calls, _ = make([True], now=now)
    g.begin_turn(1); g.preapprove(1, until=30.0)
    d = await g.decide("mcp__pim__message_send", {"to": "Priya", "body": "on my way"})
    assert d.kind == "approved" and calls[0][0] == "Message Priya: on my way"


# -- GateServer: the socket front for out-of-process callers ------------------
import asyncio
import json

from veronica.brain.gate import GateServer


@pytest.fixture
def sock(tmp_path, monkeypatch):
    """AF_UNIX paths are capped at ~104 bytes and pytest's tmp_path on macOS
    is longer, so bind relative to it."""
    monkeypatch.chdir(tmp_path)
    return Path("gate.sock")


async def _roundtrip(path, req):
    r, w = await asyncio.open_unix_connection(str(path))
    w.write((json.dumps(req) + "\n").encode())
    await w.drain()
    line = await r.readline()
    w.close()
    await w.wait_closed()
    return json.loads(line)


async def test_gate_server_allow_and_deny(sock):
    g, _, _ = make([True, False])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        assert (sock).stat().st_mode & 0o777 == 0o600
        ok = await _roundtrip(sock,
                              {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "a"}, "origin": "mcp", "backend": "codex"})
        assert ok == {"allow": True, "kind": "approved", "reason": ""}
        no = await _roundtrip(sock,
                              {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "b"}, "origin": "hook", "backend": "codex"})
        assert no == {"allow": False, "kind": "denied", "reason": "user declined"}
    finally:
        await srv.stop()
    assert not (sock).exists()


async def test_gate_server_allow_class_needs_no_confirm(sock):
    g, calls, _ = make([])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        ok = await _roundtrip(sock,
                              {"v": 1, "tool": "mcp__mac__volume_get", "input": {}, "origin": "mcp", "backend": "codex"})
        assert ok == {"allow": True, "kind": "auto", "reason": ""} and calls == []
    finally:
        await srv.stop()


async def test_gate_server_malformed_request_is_denied(sock):
    g, _, _ = make([])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        r, w = await asyncio.open_unix_connection(str(sock))
        w.write(b"not json\n")
        await w.drain()
        assert json.loads(await r.readline()) == {"allow": False, "kind": "denied", "reason": "bad request"}
        w.close()
        await w.wait_closed()
    finally:
        await srv.stop()


async def test_gate_server_serializes_confirms(sock):
    order = []

    async def confirm(summary, detail="", *, question=None):
        order.append(("start", summary))
        await asyncio.sleep(0.05)
        order.append(("end", summary))
        return True

    g = ToolGate(Settings(auto_allow_tools=[]), confirm)
    srv = GateServer(g, sock)
    await srv.start()
    try:
        await asyncio.gather(
            _roundtrip(sock, {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "A"}, "origin": "mcp", "backend": "x"}),
            _roundtrip(sock, {"v": 1, "tool": "mcp__mac__clipboard_write", "input": {"text": "B"}, "origin": "mcp", "backend": "x"}),
        )
    finally:
        await srv.stop()
    assert [o[0] for o in order] == ["start", "end", "start", "end"]


# -- GateServer: running the tool in the app process ("op": "call") -----------


def runner(log_to, content=None, *, delay=0.0):
    """A stand-in for registry.call_tool that records what it was asked."""
    async def run_tool(tool, args):
        log_to.append((tool, args))
        if delay:
            await asyncio.sleep(delay)
        return content if content is not None else ([{"type": "text", "text": "ran"}], False)
    return run_tool


CALL = {"v": 1, "op": "call", "tool": "mcp__mac__clipboard_write",
        "input": {"text": "a"}, "origin": "mcp", "backend": "codex"}


async def test_call_runs_the_tool_here_after_a_yes(sock):
    g, _, cards = make([True])
    ran = []
    srv = GateServer(g, sock, run_tool=runner(ran))
    await srv.start()
    try:
        resp = await _roundtrip(sock, CALL)
    finally:
        await srv.stop()
    assert resp == {"allow": True, "kind": "approved", "reason": "",
                    "content": [{"type": "text", "text": "ran"}], "is_error": False}
    assert ran == [("mcp__mac__clipboard_write", {"text": "a"})]


async def test_call_shows_exactly_one_hud_card(sock):
    """One call, one decide, one card — running the tool must not add a second."""
    g, _, cards = make([])
    srv = GateServer(g, sock, run_tool=runner([]))
    await srv.start()
    try:
        await _roundtrip(sock, dict(CALL, tool="mcp__mac__volume_get", input={}))
    finally:
        await srv.stop()
    assert cards == [("volume_get", "auto")]


async def test_call_denied_never_runs_the_tool(sock):
    g, _, _ = make([False])
    ran = []
    srv = GateServer(g, sock, run_tool=runner(ran))
    await srv.start()
    try:
        resp = await _roundtrip(sock, CALL)
    finally:
        await srv.stop()
    assert resp == {"allow": False, "kind": "denied", "reason": "user declined"}
    assert ran == []


async def test_call_without_a_runner_is_denied(sock):
    g, calls, _ = make([])
    srv = GateServer(g, sock)
    await srv.start()
    try:
        resp = await _roundtrip(sock, CALL)
    finally:
        await srv.stop()
    assert resp["allow"] is False and resp["kind"] == "denied" and calls == []


async def test_call_returns_image_content(sock):
    g, _, _ = make([])
    shot = ([{"type": "image", "data": "QUJD", "mimeType": "image/png"},
             {"type": "text", "text": "Screenshot of the screen."}], False)
    srv = GateServer(g, sock, run_tool=runner([], shot))
    await srv.start()
    try:
        resp = await _roundtrip(sock, dict(CALL, tool="mcp__screen__screenshot", input={"region": "screen"}))
    finally:
        await srv.stop()
    assert resp["allow"] and resp["content"] == shot[0] and resp["is_error"] is False


async def test_a_tool_that_blows_up_is_an_error_not_a_dropped_call(sock):
    g, _, _ = make([])

    async def boom(tool, args):
        raise RuntimeError("kaboom")

    srv = GateServer(g, sock, run_tool=boom)
    await srv.start()
    try:
        resp = await _roundtrip(sock, dict(CALL, tool="mcp__mac__volume_get", input={}))
    finally:
        await srv.stop()
    assert resp["allow"] and resp["is_error"] and "kaboom" in resp["content"][0]["text"]


async def test_a_slow_tool_does_not_block_an_unrelated_confirm(sock):
    """Confirms are serialized because the user can only be asked one thing
    at a time; running the tool is not, or one long capture would wedge the
    gate for everything behind it."""
    order = []

    async def confirm(summary, detail="", *, question=None):
        order.append(f"confirm {summary}")
        return True

    async def run_tool(tool, args):
        order.append("run start")
        await asyncio.sleep(0.1)
        order.append("run end")
        return [{"type": "text", "text": "ran"}], False

    g = ToolGate(Settings(auto_allow_tools=[]), confirm)
    srv = GateServer(g, sock, run_tool=run_tool)
    await srv.start()
    try:
        await asyncio.gather(
            _roundtrip(sock, CALL),
            _roundtrip(sock, {"v": 1, "tool": "mcp__mac__clipboard_write",
                              "input": {"text": "B"}, "origin": "mcp", "backend": "x"}),
        )
    finally:
        await srv.stop()
    assert order.index("confirm Copy to clipboard: B") < order.index("run end")


async def test_unknown_op_is_a_bad_request(sock):
    g, calls, _ = make([])
    srv = GateServer(g, sock, run_tool=runner([]))
    await srv.start()
    try:
        resp = await _roundtrip(sock, dict(CALL, op="whatever"))
    finally:
        await srv.stop()
    assert resp == {"allow": False, "kind": "denied", "reason": "bad request"} and calls == []


# -- auto-allow: "yes, and stop asking" ---------------------------------------


@pytest.fixture
def saved(monkeypatch):
    """What the gate wrote to prefs.json (the on-disk half of the setting)."""
    written: list[tuple] = []
    monkeypatch.setattr(gate_mod.prefs, "save_settings_override",
                        lambda field, value: written.append((field, value)))
    return written


async def test_an_auto_allowed_tool_runs_without_asking():
    g, calls, cards = make([], auto_allow_tools=["mcp__mac__clipboard_write"])
    d = await g.decide("mcp__mac__clipboard_write", {"text": "hi"})
    assert d.allow and d.kind == "auto" and calls == []
    assert cards == [("Copy to clipboard: hi", "auto")]   # HUD wire value unchanged


async def test_the_shipped_default_auto_allows_clipboard_write():
    calls = []

    async def confirm(summary, detail="", *, question=None):
        calls.append(summary)
        return True

    g = ToolGate(Settings(), confirm)
    assert (await g.decide("mcp__mac__clipboard_write", {"text": "hi"})).kind == "auto"
    assert calls == []


async def test_eligible_tool_is_offered_the_option_in_the_question():
    g, calls, _ = make([True])
    await g.decide("mcp__pim__reminder_create", {"title": "milk"})
    assert calls[0][2] and calls[0][2].endswith(gate_mod.ALWAYS_HINT)


async def test_the_option_is_not_offered_for_a_tool_that_cannot_be_auto_allowed():
    g, calls, _ = make([True])
    await g.decide("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"})
    assert calls[0][2] is None


async def test_always_approves_and_remembers_an_eligible_tool(saved):
    g, calls, _ = make([ConfirmResult("approved", "yes, dont ask again", always=True)])
    d = await g.decide("mcp__pim__reminder_create", {"title": "milk"})
    assert d.allow and d.kind == "approved"
    assert g.s.auto_allow_tools == ["mcp__pim__reminder_create"]
    assert saved == [("auto_allow_tools", ["mcp__pim__reminder_create"])]
    # ...and the next one never reaches the question.
    d2 = await g.decide("mcp__pim__reminder_create", {"title": "eggs"})
    assert d2.kind == "auto" and len(calls) == 1


async def test_always_keeps_what_is_already_on_the_list(saved):
    g, _, _ = make([ConfirmResult("approved", "always", always=True)],
                   auto_allow_tools=["mcp__mac__clipboard_write"])
    await g.decide("mcp__memory__fact_add", {"text": "x"})
    assert g.s.auto_allow_tools == ["mcp__mac__clipboard_write", "mcp__memory__fact_add"]
    assert saved[-1] == ("auto_allow_tools", ["mcp__mac__clipboard_write", "mcp__memory__fact_add"])


async def test_a_plain_yes_remembers_nothing(saved):
    g, _, _ = make([True])
    await g.decide("mcp__pim__reminder_create", {"title": "milk"})
    assert g.s.auto_allow_tools == [] and saved == []


@pytest.mark.parametrize("tool, inp", [
    ("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"}),
    ("mcp__mac__run_shortcut", {"name": "Wipe Disk"}),
    ("Bash", {"command": "rm -rf /tmp/x"}),
])
async def test_always_on_an_ineligible_tool_approves_once_and_says_so(saved, tool, inp):
    said = []
    g, calls, _ = make([ConfirmResult("approved", "always", always=True),
                        ConfirmResult("approved", "yes")], said=said)
    d = await g.decide(tool, inp)
    assert d.allow and d.kind == "approved"      # the single call still goes ahead
    assert g.s.auto_allow_tools == [] and saved == []
    assert said == [gate_mod.ALWAYS_ASK]
    # and it is asked again next time
    assert (await g.decide(tool, inp)).kind == "approved" and len(calls) == 2


async def test_always_on_a_screen_action_approves_once_and_says_so(saved):
    said = []
    g, calls, _ = make([ConfirmResult("approved", "always", always=True)], said=said)
    d = await g.decide("mcp__computer__computer_click", {"x": 5, "y": 6})
    assert d.allow and d.kind == "approved"
    assert g.s.auto_allow_tools == [] and saved == []
    assert said == [gate_mod.ALWAYS_ASK]


@pytest.mark.parametrize("tool, inp", [
    ("mcp__pim__mail_send", {"to": "a@b.c", "subject": "x", "body": "y"}),
    ("mcp__pim__message_send", {"to": "Priya", "body": "hi"}),
    ("mcp__mac__applescript", {"script": "delete everything"}),
    ("mcp__computer__computer_click", {"x": 1, "y": 2}),
    ("mcp__mac__run_shortcut", {"name": "Wipe Disk"}),
])
async def test_a_hand_typed_ineligible_tool_is_asked_every_single_time(tool, inp):
    # trust window off, so nothing but the auto-allow list is under test
    g, calls, _ = make([True, True, True], auto_allow_tools=[tool], computer_trust_s=0)
    for _ in range(3):
        assert (await g.decide(tool, inp)).kind == "approved"
    assert len(calls) == 3


# -- busy: the brain's silence clock stops while the gate works ---------------


async def test_the_gate_is_busy_through_the_confirm_and_the_tool_run(sock):
    seen = []

    async def confirm(summary, detail="", *, question=None):
        seen.append(("confirm", g.busy))
        return True

    async def run_tool(tool, args):
        seen.append(("run", g.busy))
        return [{"type": "text", "text": "ran"}], False

    g = ToolGate(Settings(auto_allow_tools=[]), confirm)
    srv = GateServer(g, sock, run_tool=run_tool)
    await srv.start()
    try:
        assert not g.busy
        await _roundtrip(sock, CALL)
    finally:
        await srv.stop()
    assert seen == [("confirm", True), ("run", True)]
    assert not g.busy


async def test_wait_quiet_restarts_the_clock_when_the_gate_goes_idle():
    g, _, _ = make([])

    async def line_after(s):
        await asyncio.sleep(s)
        return b"line"

    async def busy_for(s):
        with g.working():
            await asyncio.sleep(s)

    t = asyncio.create_task(busy_for(0.2))
    await asyncio.sleep(0)
    # 0.25 s of quiet, but only 0.05 of it with the gate idle
    assert await g.wait_quiet(line_after(0.25), 0.1) == b"line"
    await t
    with pytest.raises(TimeoutError):
        await g.wait_quiet(line_after(1), 0.05)


# -- a slow answer: the gate answers before its caller gives up ---------------


async def test_a_confirm_past_the_callers_budget_is_denied_and_said(sock, caplog):
    """The caller (the CLI's hook, tools.serve) stops waiting at its budget
    and reads that as a deny. The gate answers just before that instead:
    a deny, never an approval — and she says why, rather than the step
    quietly vanishing."""
    said, cancelled = [], []

    async def confirm(summary, detail="", *, question=None):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(summary)
            raise
        return True

    async def say(text):
        said.append(text)

    ran = []
    g = ToolGate(Settings(auto_allow_tools=[]), confirm, say=say)
    srv = GateServer(g, sock, run_tool=runner(ran))
    srv.reply_margin_s = 0.1
    await srv.start()
    try:
        resp = await asyncio.wait_for(_roundtrip(sock, dict(CALL, budget=0.4)), 2)
    finally:
        await srv.stop()
    assert resp["allow"] is False and resp["kind"] == "denied"
    assert ran == [] and cancelled == ["Copy to clipboard: a"]
    assert said == ["You took a while to answer, so I skipped that step."]
    assert "reason=gate_timeout" in caplog.text
    assert not g.busy


async def test_a_request_that_never_got_its_turn_to_ask_is_denied_quietly(sock, caplog):
    """Queued behind another confirm until its caller's budget ran out: it
    was never asked, so there is no answer to blame — deny, log, no line."""
    said = []
    release = asyncio.Event()

    async def confirm(summary, detail="", *, question=None):
        await release.wait()
        return True

    async def say(text):
        said.append(text)

    g = ToolGate(Settings(auto_allow_tools=[]), confirm, say=say)
    srv = GateServer(g, sock, run_tool=runner([]))
    srv.reply_margin_s = 0.1
    await srv.start()
    try:
        first = asyncio.ensure_future(_roundtrip(sock, dict(CALL, budget=30)))
        await asyncio.sleep(0.05)
        second = await asyncio.wait_for(_roundtrip(sock, dict(CALL, input={"text": "b"}, budget=0.4)), 2)
    finally:
        release.set()
        await first
        await srv.stop()
    assert second["allow"] is False
    assert said == []
    assert "reason=gate_busy" in caplog.text
