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


class BrainUnavailable(Exception):
    """The brain couldn't run this turn at all (the local model's server
    wouldn't start). Raised out of ask() so the switcher can move to
    another brain; the message is what to say when it can't."""


class Brain(Protocol):
    """What the orchestrator needs from any backend: sentences streamed
    from ask(), an interrupt, a close, and the shared gate."""
    name: str
    gate: "ToolGate"  # noqa: F821  (veronica.brain.gate; string to avoid the import cycle)

    def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]: ...
    async def interrupt(self) -> None: ...
    async def close(self) -> None: ...
