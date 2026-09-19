import logging
import os
import shlex
import time
from collections.abc import Callable

from veronica.brain.agent import (
    COMPUTER_PREFIX,
    Confirm,
    summarize_detail,
    summarize_tool,
)
from veronica.brain.base import Decision
from veronica.brain.policy import TRUST_EXCLUDED_BUNDLES, always_confirm, classify
from veronica.config import Settings
from veronica.tools.computer_events import Front, frontmost, is_system_dialog

log = logging.getLogger("veronica.brain")


def _confirm_outcome(result) -> tuple[str, str]:
    """(outcome, heard) of a Confirm result; a bare bool is approved/denied."""
    outcome = getattr(result, "outcome", None)
    if outcome is None:
        return ("approved" if result else "denied"), ""
    return outcome, getattr(result, "heard", "") or ""


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
    ) -> None:
        self.s = settings
        self._confirm = confirm
        self._on_tool = on_tool
        self._frontmost = frontmost
        self._clock = clock
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
        outcome, heard = _confirm_outcome(await self._confirm(summary, summarize_detail(tool_name, input)))
        if outcome == "approved":
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
