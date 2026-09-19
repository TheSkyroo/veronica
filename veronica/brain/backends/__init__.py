"""The brains Veronica can run on: registry, availability checks, factory.

Every backend uses the vendor CLI's own login — no API keys. `check_backend`
only looks at cheap local markers (binary on PATH, a file the login writes)
so the menu bar and the switcher can poll it; nothing here spawns a CLI."""
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from veronica.brain.backends.antigravity import AntigravityBrain
from veronica.brain.backends.claude import ClaudeBrain
from veronica.brain.backends.codex import CodexBrain
from veronica.brain.backends.copilot import CopilotBrain
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
                "curl -fsSL https://antigravity.google/cli/install.sh | bash", "agy",
                (".gemini/antigravity-cli/conversations",), AntigravityBrain),
    # `claude` is not spawned by us (claude_agent_sdk finds it); listed for
    # the availability check only. The SDK reports a missing login itself.
    BackendInfo("claude", "Claude", "claude", "npm i -g @anthropic-ai/claude-code", "claude",
                (), ClaudeBrain),
    BackendInfo("copilot", "Copilot", "copilot", "npm i -g @github/copilot", "copilot login",
                (".copilot/config.json",), CopilotBrain),
)}

# Antigravity keeps its Google credentials in the macOS Keychain; the
# conversations directory only appears after the first chat, so fall back
# to the Keychain item when the directory is missing.
_ANTIGRAVITY_KEYCHAIN = ["security", "find-generic-password", "-a", "antigravity"]


def _keychain_has(run: Callable, argv: list[str]) -> bool:
    try:
        return run(argv, capture_output=True, timeout=5).returncode == 0
    except Exception as e:  # noqa: BLE001  (missing `security`, timeout: treat as not logged in)
        log.debug("keychain lookup failed: %s", e)
        return False


def check_backend(
    name: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
    exists: Callable[[Path], bool] | None = None,
    home: Path | None = None,
    run: Callable = subprocess.run,
) -> Availability:
    """Is `name` installed and logged in? Cheap and local: PATH + marker
    files (+ the Keychain for Antigravity). `hint` reads naturally aloud."""
    info = BACKENDS.get(name)
    if info is None:
        return Availability(False, "not installed", f"I don't know a brain called {name}.")
    if not which(info.binary):
        return Availability(False, "not installed",
                            f"{info.label} isn't installed — run {info.install_cmd}, then {info.login_cmd}.")
    home = Path.home() if home is None else home
    exists = Path.exists if exists is None else exists
    logged_in = (not info.login_markers or any(exists(home / m) for m in info.login_markers)
                 or (name == "antigravity" and _keychain_has(run, _ANTIGRAVITY_KEYCHAIN)))
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
