"""`BrainSwitcher`: which brain is active, manual switching, and automatic
failover when one hits its usage limit.

The *preferred* brain is `settings.brain_backend`. The active one can differ
at runtime — a stand-in — for two reasons: the preferred brain hit a usage
limit (time-based cooldown, `limited_until`) or it isn't installed/logged in
(re-checked every minute). Either way the preference itself only changes on
a manual switch. Nothing here spawns a CLI: brains are built lazily and
`check` only looks at local markers.

A third reason to stand in: there is no internet. Every brain but `local`
needs a vendor host, so `maybe_offline` (called once per turn) puts the
local model in when the probe in `veronica.net` says the wire is dead, and
`maybe_return` goes back — silently — when it is alive again."""
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from veronica import net, prefs
from veronica.brain.backends import BACKENDS, Availability, check_backend, make_brain
from veronica.brain.base import Brain
from veronica.brain.gate import ToolGate
from veronica.config import Settings

log = logging.getLogger("veronica.brain")

NO_BRAIN_LINE = "No brain is ready — log into Codex, Antigravity or Claude."
NO_BRAIN_LABEL = "No brain"
RECHECK_S = 60.0     # how often a stand-in for availability reasons re-checks the preferred brain
OFFLINE_LINE = "No internet — switching to the local model."
LOCAL = "local"      # the one backend that needs no network


class NoBrain:
    """Placeholder when no backend is installed and logged in: every ask
    answers with the login hint; the gate proxies keep the orchestrator's
    calls working."""

    name = "none"

    def __init__(self, gate: ToolGate) -> None:
        self.gate = gate

    async def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]:
        yield NO_BRAIN_LINE

    async def interrupt(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def clear_trust(self) -> None:
        self.gate.clear_trust()

    def begin_turn(self, turn_id: int) -> None:
        self.gate.begin_turn(turn_id)

    def preapprove(self, turn_id: int, until: float) -> None:
        self.gate.preapprove(turn_id, until)

    @property
    def pending_redirect(self) -> str | None:
        return self.gate.pending_redirect

    @pending_redirect.setter
    def pending_redirect(self, value: str | None) -> None:
        self.gate.pending_redirect = value


def _label(name: str) -> str:
    info = BACKENDS.get(name)
    return info.label if info else NO_BRAIN_LABEL


class BrainSwitcher:
    def __init__(
        self,
        settings: Settings,
        *,
        gate: ToolGate,
        on_tool: Callable[[str, str], None] | None = None,
        memory=None,
        factory=make_brain,
        check: Callable[[str], Availability] = check_backend,
        clock: Callable[[], float] = time.monotonic,
        say: Callable[[str], object] | None = None,
        on_backend: Callable[[str, bool], None] | None = None,
        is_online: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self.s = settings
        self.gate = gate
        self._on_tool = on_tool
        self._memory = memory
        self._factory = factory
        self._check = check
        #: name -> is that brain's vendor reachable? Takes the backend name
        #: so the probe hits the host that brain actually needs.
        self._is_online = is_online or (lambda name: net.online(net.VENDOR_HOSTS.get(name)))
        self._clock = clock
        self._say = say
        self._on_backend = on_backend
        self.preferred: str = settings.brain_backend
        self.brain: Brain = NoBrain(gate)
        self.standing_in = False
        # backend name -> monotonic time its usage-limit cooldown ends
        self.limited_until: dict[str, float] = {}
        # Why the active brain is a stand-in: "limit" | "not installed" |
        # "not logged in" | None. Availability reasons are re-checked on a
        # timer; the limit reason waits for `limited_until[preferred]`.
        self._standin_reason: str | None = None
        self._next_recheck = 0.0

    # -- helpers ----------------------------------------------------------------
    async def _speak(self, text: str) -> None:
        log.info("brain: %s", text)
        if self._say is not None:
            await self._say(text)

    def _order(self) -> list[str]:
        """Preferred first, then the configured failover order, then the rest."""
        out = [self.preferred]
        configured = [x.strip() for x in self.s.brain_failover_order.split(",")]
        for n in [*configured, *BACKENDS]:
            if n in BACKENDS and n not in out:
                out.append(n)
        return out

    def _candidates(self) -> list[str]:
        """`_order()` without the local model. Standing in for a brain that
        isn't logged in, or moving off one that hit its usage limit, must
        never drop the user onto the 3B model: that is `maybe_offline`'s
        job alone, and `switch("local")` is still there for asking."""
        return [n for n in self._order() if n != LOCAL]

    def _cooling(self, name: str) -> bool:
        return self.limited_until.get(name, 0.0) > self._clock()

    def _notify(self) -> None:
        if self._on_backend is not None:
            self._on_backend(self.status_label(), self.standing_in)

    async def _activate(self, name: str | None) -> None:
        """Close the active brain (its session file stays, so switching
        back resumes) and build `name`; None = NoBrain."""
        old = self.brain
        new = NoBrain(self.gate) if name is None else self._factory(
            name, self.s, gate=self.gate, on_tool=self._on_tool, memory=self._memory)
        try:
            await old.close()
        except Exception:
            log.exception("closing %s brain failed", old.name)
        self.brain = new
        self.standing_in = new.name != self.preferred
        if not self.standing_in:
            self._standin_reason = None
        log.info("brain: %s active (preferred %s)", new.name, self.preferred)

    async def _stand_in(self, why: str, *, announce: bool) -> None:
        """Preferred is unavailable for `why`: activate the first available
        alternative (or NoBrain) and schedule the next availability check."""
        for n in self._candidates():
            if n != self.preferred and self._check(n).ok:
                await self._activate(n)
                if announce:
                    # why is "not installed" / "not logged in" -> "isn't installed"
                    await self._speak(f"{_label(self.preferred)} isn't {why.removeprefix('not ')}, "
                                      f"so I'm on {_label(n)} for now.")
                break
        else:
            await self._activate(None)
            if announce:
                await self._speak(NO_BRAIN_LINE)
        self.standing_in = True
        self._standin_reason = why
        self._next_recheck = self._clock() + RECHECK_S
        self._notify()

    # -- public -----------------------------------------------------------------
    def status_label(self) -> str:
        """"Codex", or "Claude (for Codex)" while standing in."""
        label = _label(self.brain.name)
        if self.standing_in and self.brain.name in BACKENDS:
            return f"{label} (for {_label(self.preferred)})"
        return label

    async def start(self) -> None:
        """Activate the preferred brain, or a stand-in when it isn't ready
        (said once). Called before the first turn; nothing is spawned."""
        avail = self._check(self.preferred)
        if avail.ok:
            await self._activate(self.preferred)
            self._notify()
            return
        await self._stand_in(avail.reason, announce=True)

    async def switch(self, name: str, *, manual: bool = True) -> Availability:
        """Switch to `name` if it's ready. Manual switches make it the
        preferred brain (saved to prefs) and clear its limit cooldown."""
        if name not in BACKENDS:
            return Availability(False, "not installed", f"I don't know a brain called {name}.")
        avail = self._check(name)
        if not avail.ok:
            return avail
        if manual:
            self.preferred = name
            self.s.brain_backend = name
            prefs.save_settings_override("brain_backend", name)
            self.limited_until.pop(name, None)
        await self._activate(name)
        if self.standing_in:
            # A programmatic stand-in: return on the preferred brain's cooldown
            # if it has one, else on its next availability check.
            self._standin_reason = "limit" if self._cooling(self.preferred) else self._check(self.preferred).reason
        self._notify()
        return avail

    async def failover(self, reason: str) -> str | None:
        """The active brain hit its usage limit: cool it down and move to
        the next ready brain in order. Returns the new name, or None when
        nothing else is ready (or failover is off). Does not re-run the
        user's request — the orchestrator does."""
        current = self.brain.name
        label = _label(current)
        if not self.s.brain_failover:
            await self._speak(f"{label} hit its usage limit.")
            return None
        log.warning("brain: %s usage limit: %s", current, reason)
        self.limited_until[current] = self._clock() + self.s.brain_limit_cooldown_min * 60
        for n in self._candidates():
            if n == current or self._cooling(n) or not self._check(n).ok:
                continue
            await self._speak(f"{label} hit its usage limit — switching to {_label(n)}.")
            await self._activate(n)
            self.standing_in = True
            self._standin_reason = "limit"
            self._notify()
            return n
        await self._speak(f"{label} hit its usage limit and no other brain is ready.")
        return None

    async def unavailable(self, reason: str) -> str | None:
        """The active brain couldn't run at all (the local model's server
        wouldn't start): cool it down like a usage limit and move to the
        next ready brain. Returns the new name, or None when nothing else
        is ready (or failover is off)."""
        current = self.brain.name
        label = "The local model" if current == LOCAL else _label(current)
        log.warning("brain: %s wouldn't start: %s", current, reason)
        if not self.s.brain_failover:
            await self._speak(f"{label} wouldn't start.")
            return None
        # Cooling also keeps maybe_offline from putting a dead local model
        # straight back in on the next turn.
        self.limited_until[current] = self._clock() + self.s.brain_limit_cooldown_min * 60
        for n in self._candidates():
            if n == current or self._cooling(n) or not self._check(n).ok:
                continue
            await self._speak(f"{label} wouldn't start — switching to {_label(n)}.")
            await self._activate(n)
            if self.standing_in:
                self._standin_reason = "down"
            self._notify()
            return n
        await self._speak(f"{label} wouldn't start and no other brain is ready.")
        return None

    async def maybe_return(self) -> None:
        """Before each turn: go back to the preferred brain silently once
        its cooldown has passed / it has become available."""
        if not self.standing_in:
            return
        now = self._clock()
        if self._standin_reason == "offline":
            # net.online is itself cached, so this costs nothing most turns.
            if not await self._is_online(self.preferred):
                return
        elif self._standin_reason in ("limit", "down"):
            if self._cooling(self.preferred):
                return
        elif now < self._next_recheck:
            return
        self._next_recheck = now + RECHECK_S
        avail = self._check(self.preferred)
        if avail.ok:
            await self._activate(self.preferred)
            self._notify()
            return
        # Cooldown over but the preferred brain is (now) not installed /
        # logged in: keep the stand-in and re-check on the timer instead.
        self._standin_reason = avail.reason
        if self.brain.name == "none":
            # Nothing was ready at start; maybe something is now.
            await self._stand_in(avail.reason, announce=False)

    async def maybe_offline(self) -> None:
        """Before each turn: if the active brain needs a vendor host and
        there is no way to reach it, stand the local model in — the same
        machinery as a usage limit, announced once. Off with
        `brain_offline_fallback`; a no-op when we're already local."""
        if not self.s.brain_offline_fallback:
            return
        current = self.brain.name
        if current == LOCAL or current not in BACKENDS:
            return
        if await self._is_online(current):
            return
        if not self._check(LOCAL).ok:
            log.warning("brain: offline and the local model isn't set up")
            return
        if self._cooling(LOCAL):
            log.warning("brain: offline, but the local model failed to start recently")
            return
        await self._speak(OFFLINE_LINE)
        await self._activate(LOCAL)
        self.standing_in = True
        self._standin_reason = "offline"
        self._notify()

    def online_candidate(self) -> str | None:
        """The first ready brain that isn't the local one — what "go online"
        goes back to."""
        for n in self._candidates():
            if self._check(n).ok:
                return n
        return None
