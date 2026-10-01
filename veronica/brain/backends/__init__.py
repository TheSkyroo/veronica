"""The brains Veronica can run on: registry, availability checks, factory.

Every backend uses the vendor CLI's own login — no API keys. `check_backend`
only looks at cheap local markers (binary on PATH, a file the login writes)
so the menu bar and the switcher can poll it; nothing here spawns a CLI.

`local` is the odd one out: no vendor, no login, just a llama.cpp server and
a .gguf on this PC, so its availability is "are those two files there"."""
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from veronica.brain.backends.antigravity import AntigravityBrain
from veronica.brain.backends.claude import ClaudeBrain
from veronica.brain.backends.codex import CodexBrain
from veronica.brain.backends.copilot import CopilotBrain
from veronica.brain.backends.local import LocalBrain
from veronica.brain.base import Brain
from veronica.brain.gate import ToolGate
from veronica.config import Settings

log = logging.getLogger("veronica.brain")

Reason = Literal["ok", "not installed", "not logged in"]


@dataclass(frozen=True)
class BackendInfo:
    name: str
    label: str
    binary: str
    install_cmd: str
    login_cmd: str
    # Relative to home: files/dirs whose existence means "logged in". Empty
    # = nothing checkable, login is assumed (the backend errors loudly itself).
    login_markers: tuple[str, ...]
    cls: type


@dataclass(frozen=True)
class Availability:
    ok: bool
    reason: Reason
    hint: str = ""     # spoken sentence when not ok


BACKENDS: dict[str, BackendInfo] = {b.name: b for b in (
    BackendInfo("codex", "Codex", "codex", "npm i -g @openai/codex", "codex login",
                (".codex/auth.json",), CodexBrain),
    BackendInfo("antigravity", "Antigravity", "agy",
                "the Antigravity CLI installer from antigravity.google", "agy",
                (".gemini/antigravity-cli/conversations",), AntigravityBrain),
    # `claude` is not spawned by us (claude_agent_sdk finds it); listed for
    # the availability check only. The SDK reports a missing login itself.
    BackendInfo("claude", "Claude", "claude", "npm i -g @anthropic-ai/claude-code", "claude",
                (), ClaudeBrain),
    BackendInfo("copilot", "Copilot", "copilot", "npm i -g @github/copilot", "copilot login",
                (".copilot/config.json",), CopilotBrain),
    # Offline: `binary` is a path from Settings, not a name on PATH, and
    # there is nothing to log into — see `_check_local`.
    BackendInfo("local", "Local", "llama-server", "", "", (), LocalBrain),
)}

# Antigravity keeps its Google credentials in the OS credential store —
# Windows Credential Manager — under a generic credential whose target
# names "antigravity"; the conversations directory only appears after the
# first chat, so fall back to the credential when the directory is missing.
_ANTIGRAVITY_CREDENTIAL = "antigravity"


def _win_credential_targets(filter_: str) -> list[str]:
    """Target names of the user's stored Windows credentials matching
    `filter_` (CredEnumerateW wildcards). Only names are read, never the
    secrets. Lazy: pywin32 is a Windows-only dependency."""
    import win32cred  # type: ignore[import-not-found]
    try:
        creds = win32cred.CredEnumerate(filter_, 0) or ()
    except Exception:   # pywintypes.error ERROR_NOT_FOUND: no match
        return []
    return [str(c.get("TargetName") or "") for c in creds]


def _credential_has(lookup: Callable[[str], list[str]], name: str) -> bool:
    """Is there a stored credential whose target mentions `name`? Any
    failure (no pywin32, no Credential Manager) reads as "not logged in"."""
    try:
        return any(name in t.lower() for t in lookup(f"*{name}*"))
    except Exception as e:  # noqa: BLE001
        log.debug("credential lookup failed: %s", e)
        return False


def _check_local(settings: Settings, exists: Callable[[Path], bool]) -> Availability:
    """The local brain is ready when both its files are on disk. No PATH
    lookup (the binary is an absolute path) and no login at all."""
    if not exists(Path(settings.local_server_bin)):
        return Availability(False, "not installed",
                            "The local model server isn't there — set its path in Settings.")
    if not exists(Path(settings.local_model)):
        return Availability(False, "not installed",
                            "The local model file isn't there — pick one in Settings.")
    return Availability(True, "ok")


def check_backend(
    name: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[Path], bool] | None = None,
    home: Path | None = None,
    credentials: Callable[[str], list[str]] = _win_credential_targets,
    settings: Settings | None = None,
) -> Availability:
    """Is `name` installed and logged in? Cheap and local: PATH + marker
    files (+ Credential Manager for Antigravity). `hint` reads naturally
    aloud."""
    info = BACKENDS.get(name)
    if info is None:
        return Availability(False, "not installed", f"I don't know a brain called {name}.")
    exists = Path.exists if exists is None else exists
    if name == "local":
        from veronica import config       # late: config imports nothing from here
        return _check_local(settings or config.settings, exists)
    if not which(info.binary):
        return Availability(False, "not installed",
                            f"{info.label} isn't installed — run {info.install_cmd}, then {info.login_cmd}.")
    home = Path.home() if home is None else home
    logged_in = (not info.login_markers or any(exists(home / m) for m in info.login_markers)
                 or (name == "antigravity" and _credential_has(credentials, _ANTIGRAVITY_CREDENTIAL)))
    if not logged_in:
        return Availability(False, "not logged in",
                            f"{info.label} isn't logged in — run {info.login_cmd} in a terminal.")
    return Availability(True, "ok")


def make_brain(name: str, settings: Settings, *, gate: ToolGate, on_tool=None, memory=None) -> Brain:
    """Build (not start) the backend `name` on the shared gate."""
    if name not in BACKENDS:
        raise KeyError(f"unknown brain backend {name!r}")
    cls = BACKENDS[name].cls
    if cls is ClaudeBrain:
        return ClaudeBrain(settings, on_tool=on_tool, memory=memory, gate=gate)
    return cls(settings, gate, on_tool=on_tool, memory=memory)
