import asyncio
import contextlib
import json
import logging
import os
import shlex
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from veronica import prefs
from veronica.brain.agent import (
    COMPUTER_PREFIX,
    Confirm,
    confirm_prompt,
    summarize_detail,
    summarize_tool,
)
from veronica.brain.base import Decision
from veronica.brain.policy import (
    AUTO_ALLOWABLE,
    TRUST_EXCLUDED_BUNDLES,
    always_confirm,
    classify,
)
from veronica.config import Settings
from veronica.tools.computer_events import Front, frontmost, is_system_dialog

log = logging.getLogger("veronica.brain")

# (content blocks, is_error) for one allowed tool call, run in this process.
RunTool = Callable[[str, dict], Awaitable[tuple[list[dict], bool]]]
NO_RUNNER = "Veronica can't run tools right now"
# Tacked onto the spoken question, but only for a tool the user could
# switch off for good (policy.AUTO_ALLOWABLE).
ALWAYS_HINT = " Say always and I'll stop asking."
# What she says when "always" lands on something that can never be
# auto-allowed. The call itself still goes ahead.
ALWAYS_ASK = "That one I'll always ask about."
# How long a brain may sit on one tool call (or wait on the gate) before the
# turn is given up on anyway. Only a wedged tool gets near it: the gate's
# own call budget is well under it.
BUSY_CEILING_S = 600.0


class ToolStall(TimeoutError):
    """wait_quiet gave up because a tool (or the gate) never finished, not
    because the model went quiet."""


def _confirm_outcome(result) -> tuple[str, str, bool]:
    """(outcome, heard, always) of a Confirm result; a bare bool is
    approved/denied. `always` is a yes that also said stop asking."""
    outcome = getattr(result, "outcome", None)
    if outcome is None:
        return ("approved" if result else "denied"), "", False
    return outcome, getattr(result, "heard", "") or "", bool(getattr(result, "always", False))


class ToolGate:
    """The one place a tool call is allowed or refused, for every backend.
    In-process for Claude (ClaudeBrain._can_use_tool wraps decide()), over
    the gate socket for external CLIs (GateServer)."""

    def __init__(
        self,
        settings: Settings,
        confirm: Confirm,
        on_tool: Callable[[str, str], None] | None = None,
        frontmost: Callable[[], Front] = frontmost,
        clock: Callable[[], float] = time.monotonic,
        say: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.s = settings
        self._confirm = confirm
        self._on_tool = on_tool
        self._frontmost = frontmost
        self._clock = clock
        # One spoken line, when "always" lands on a tool that can never be
        # auto-allowed. Wired to Orchestrator.say; None in text mode.
        self._say = say
        # Trust window (spec E4): after the user approves one confirm-class
        # screen action, further ones in the same app are auto-allowed until
        # `_trust_until` (monotonic seconds). Cleared on barge, "that's all",
        # or a "no".
        self._trust_until = 0.0
        self._trust_app: str | None = None
        # Pre-approval by request wording ("copy this, just do it"): the
        # orchestrator numbers its turns (begin_turn) and, when the request
        # itself said go ahead, pre-approves that turn for a few seconds.
        # It covers the FIRST confirm-class call of that turn only, is
        # consumed on use, and never applies to policy.always_confirm tools.
        self._current_turn = 0
        self._asked_this_turn = 0          # confirm-class calls seen this turn
        self._preapproved_turn: int | None = None
        self._preapproved_until = 0.0
        # Set by the gate when a confirmation was answered with something
        # other than yes/no: the orchestrator picks it up after the turn
        # and runs it as the next request. Cleared when a turn starts.
        self.pending_redirect: str | None = None
        # Confirms and tool runs in flight (working()), and an event that
        # fires on every change, so a brain waiting on its stream can stop
        # counting silence while the gate is the one keeping it quiet.
        self._busy = 0
        self._changed = asyncio.Event()
        self.busy_ceiling_s = BUSY_CEILING_S

    # -- busy: the silence clock ---------------------------------------------
    @contextlib.contextmanager
    def working(self):
        """Held around a confirm and a tool run. While any is held, a
        brain's stream going quiet is the gate's doing, not the model's."""
        self._busy += 1
        self._pulse()
        try:
            yield
        finally:
            self._busy -= 1
            self._pulse()

    @property
    def busy(self) -> bool:
        return self._busy > 0

    def _pulse(self) -> None:
        ev, self._changed = self._changed, asyncio.Event()
        ev.set()

    async def wait_quiet(self, aw, silence_s: float, *, tool_running: bool = False):
        """Await `aw` (a brain's next stream line), raising TimeoutError only
        after `silence_s` of silence nothing accounts for. While the gate is
        busy, or `tool_running` (the stream itself says a call is in
        flight), the clock stops; it starts again from zero when the gate
        goes idle. The model gets its full allowance after every tool
        result. A wait that stays busy past busy_ceiling_s raises ToolStall."""
        task = asyncio.ensure_future(aw)
        loop = asyncio.get_running_loop()
        give_up = loop.time() + self.busy_ceiling_s
        try:
            while True:
                busy = tool_running or self.busy
                changed = asyncio.ensure_future(self._changed.wait())
                timeout = give_up - loop.time() if busy else silence_s
                try:
                    done, _ = await asyncio.wait({task, changed}, timeout=max(0.0, timeout),
                                                 return_when=asyncio.FIRST_COMPLETED)
                finally:
                    changed.cancel()
                if task in done:
                    return task.result()
                if changed in done:
                    continue
                raise ToolStall() if busy else TimeoutError()
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    def _card(self, summary: str, decision: str) -> None:
        if self._on_tool:
            self._on_tool(summary, decision)

    # -- permission gate ------------------------------------------------------
    # Shell commands that only "work" under a different TCC identity than the
    # app (screencapture run by the Claude CLI child process needs its own
    # Screen Recording grant) — redirect the brain to the in-process tool.
    _REDIRECT_BASH = {
        "screencapture": "Use the screenshot tool instead of screencapture — it runs inside Veronica, which has the Screen Recording permission.",
    }

    def _bash_redirect(self, tool_name: str, input: dict) -> str | None:
        if tool_name != "Bash":
            return None
        try:
            argv = shlex.split(str(input.get("command", "")))
        except ValueError:
            return None
        for tok in argv:
            base = os.path.basename(tok)
            if base in self._REDIRECT_BASH:
                return self._REDIRECT_BASH[base]
        return None

    async def decide(self, tool_name: str, input: dict) -> Decision:
        summary = summarize_tool(tool_name, input)
        redirect = self._bash_redirect(tool_name, input)
        if redirect is not None:
            log.info("tool redirected: %s -> %s", summary, redirect)
            return Decision(False, "redirect", redirect)
        if classify(tool_name, input, self.s.shortcut_allowlist, self.s.auto_allow_tools) == "allow":
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
        outcome, heard, always = _confirm_outcome(await self._ask(tool_name, summary, input))
        if outcome == "approved":
            if always:
                await self._remember(tool_name)
            return Decision(True, "approved")
        return self._deny(outcome, heard)

    # -- auto-allow ("yes, and stop asking") ----------------------------------
    def _ask(self, tool_name: str, summary: str, input: dict):
        """The confirm call. A tool the user could switch off for good is
        asked with the option said out loud; everything else is asked the
        way it always was."""
        detail = summarize_detail(tool_name, input)
        if tool_name in AUTO_ALLOWABLE:
            return self._confirm(summary, detail, question=confirm_prompt(summary) + ALWAYS_HINT)
        return self._confirm(summary, detail)

    async def _remember(self, tool_name: str) -> None:
        """Persist a "yes, and stop asking" — to the live Settings and to
        prefs.json, so it survives a restart. Only for AUTO_ALLOWABLE tools:
        anything else keeps its yes/no, and she says so. The approval itself
        stands either way."""
        if tool_name not in AUTO_ALLOWABLE:
            log.info("always refused for %s (not auto-allowable)", tool_name)
            if self._say is not None:
                await self._say(ALWAYS_ASK)
            return
        if tool_name in self.s.auto_allow_tools:
            return
        self.s.auto_allow_tools = [*self.s.auto_allow_tools, tool_name]
        prefs.save_settings_override("auto_allow_tools", list(self.s.auto_allow_tools))
        log.info("auto-allow added: %s", tool_name)

    def _deny(self, outcome: str, heard: str) -> Decision:
        """A "no" is a plain decline; anything else the user said instead is
        handed to the brain in the deny message (it usually re-plans right
        away) and kept in `pending_redirect` for the orchestrator."""
        if outcome == "other":
            self.pending_redirect = heard
            return Decision(False, "other", f"user declined and said: {heard!r}", heard)
        return Decision(False, "denied", "user declined")

    # -- pre-approval by request wording ---------------------------------------
    def begin_turn(self, turn_id: int) -> None:
        """Called by the orchestrator at the top of every brain turn. A
        pre-approval that was for some other turn is dropped here."""
        self._current_turn = turn_id
        self._asked_this_turn = 0
        if self._preapproved_turn != turn_id:
            self._preapproved_turn = None

    def preapprove(self, turn_id: int, until: float) -> None:
        """Skip the yes/no for the first confirm-class call of `turn_id`,
        if it comes before `until` (monotonic seconds)."""
        self._preapproved_turn = turn_id
        self._preapproved_until = until
        log.info("pre-approval armed for turn %d", turn_id)

    def _preapproved(self, tool_name: str, input: dict, front: Front | None) -> bool:
        """One-shot: true once, for the first confirm-class call of the
        pre-approved turn, and only while the setting is on. Never for an
        always-confirm tool — and that call still uses up the slot, so a
        "just do it" can't slide onto whatever comes next."""
        first = self._asked_this_turn == 0
        self._asked_this_turn += 1
        if not (
            first
            and self.s.preapprove_by_wording
            and self._preapproved_turn is not None
            and self._preapproved_turn == self._current_turn
            and self._clock() < self._preapproved_until
        ):
            return False
        self._preapproved_turn = None
        return not always_confirm(tool_name, input, front)

    # -- trust window (E4) ----------------------------------------------------
    def clear_trust(self) -> None:
        self._trust_until = 0.0
        self._trust_app = None

    @staticmethod
    def _trustable(front: Front) -> bool:
        """Can a trust window belong to `front` at all? Never for a system
        dialog or a terminal, and never without a bundle id."""
        return bool(front.bundle_id) and not is_system_dialog(front) and front.bundle_id not in TRUST_EXCLUDED_BUNDLES

    def _trusted(self, front: Front, now: float) -> bool:
        return (
            self.s.computer_trust_s > 0          # setting it to 0 closes an open window
            and self._trust_app is not None
            and now < self._trust_until
            and front.bundle_id == self._trust_app
            and self._trustable(front)
        )

    async def _gate_computer(self, tool_name: str, input: dict, summary: str, front: Front) -> Decision:
        """Confirm gate for confirm-class `mcp__computer__*` tools. `front`
        is the frontmost app as looked up at gate time; a system permission
        dialog (`is_system_dialog`, keyed on bundle id — those windows have
        empty titles) or a terminal never gets the trust exemption, and
        neither does anything that presses Enter (`always_confirm`). After
        a "yes" the frontmost app and clock are read again: the user may
        have switched apps while being asked, and the window belongs to
        what is in front now, from now."""
        if not always_confirm(tool_name, input, front) and self._trusted(front, self._clock()):
            log.info("trusted: %s", summary)
            self._card(summary, "auto")     # HUD wire value unchanged
            return Decision(True, "trusted")
        log.info("tool request: %s", summary)
        outcome, heard, always = _confirm_outcome(await self._ask(tool_name, summary, input))
        if outcome == "approved":
            if always:
                await self._remember(tool_name)   # never eligible: she says so
            front = self._frontmost()
            now = self._clock()
            window = self.s.computer_trust_s
            if window > 0 and self._trustable(front):
                self._trust_until = now + window
                self._trust_app = front.bundle_id
                log.info("trust window opened for %s (%ss)", front.bundle_id, window)
            return Decision(True, "approved")
        self.clear_trust()
        return self._deny(outcome, heard)


class GateServer:
    """Unix-socket front for ToolGate.decide, for the out-of-process
    callers (tools.serve, brain.hook). One JSON line per request:
    {"v": 1, "op", "tool", "input", "origin": "mcp"|"hook", "backend"} in,
    {"allow", "kind", "reason"} out.

    Two ops. "decide" (the default, and all brain.hook ever asks) answers
    the permission question and leaves the caller to act on it. "call"
    also RUNS the tool here, in the app process, and returns its MCP
    content blocks — that is how an external brain's tools.serve child
    gets a screenshot without macOS attributing the capture to the CLI
    that spawned it. It needs `run_tool`; without one it fails closed.

    Confirms are serialized because the orchestrator can only ask one
    question at a time. Running a tool is not: a capture that waits a
    minute on the user would otherwise wedge every other call behind it."""

    def __init__(self, gate: ToolGate, path: Path, run_tool: RunTool | None = None) -> None:
        self.gate, self.path = gate, path
        self._run_tool = run_tool
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

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            try:
                req = json.loads(line)
                tool, inp = str(req["tool"]), dict(req.get("input") or {})
                op = str(req.get("op") or "decide")
                if op not in ("decide", "call"):
                    raise ValueError(op)
            except Exception:
                resp = {"allow": False, "kind": "denied", "reason": "bad request"}
            else:
                log.info("gate %s from %s/%s: %s", op, req.get("backend"), req.get("origin"), tool)
                if op == "call" and self._run_tool is None:
                    resp = {"allow": False, "kind": "denied", "reason": NO_RUNNER}
                else:
                    # The asking brain's stream is quiet from here until it
                    # hears back: its silence clock stops (ToolGate.working).
                    with self.gate.working():
                        async with self._lock:              # the confirm, and only the confirm
                            d = await self.gate.decide(tool, inp)
                        resp = {"allow": d.allow, "kind": d.kind, "reason": d.message}
                        if op == "call" and d.allow:
                            content, is_error = await self._run(tool, inp)
                            resp["content"], resp["is_error"] = content, is_error
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()
        except Exception:
            log.exception("gate request failed")
        finally:
            writer.close()

    async def _run(self, tool: str, inp: dict) -> tuple[list[dict], bool]:
        """Run an allowed tool. A runner that raises is reported to the
        brain as an error result, never as a dropped call."""
        try:
            return await self._run_tool(tool, inp)
        except Exception as exc:
            log.exception("tool %s failed", tool)
            return [{"type": "text", "text": f"error: {exc}"}], True
