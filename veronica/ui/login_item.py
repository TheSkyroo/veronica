"""Manage the "Start at Login" LaunchAgent for the built .app bundle.

Writes/removes ~/Library/LaunchAgents/io.manik.veronica.plist, and best-effort
(un)registers it with launchd via `launchctl bootstrap`/`bootout` — failures
there are ignored (e.g. already bootstrapped, or launchd unavailable in a
sandboxed test), since the plist file itself is the source of truth for
`is_enabled()` and RunAtLoad picks it up on the next login regardless.
"""
from __future__ import annotations

import os
import plistlib
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

LABEL = "io.manik.veronica"
APP_EXEC_SUFFIX = "Veronica.app/Contents/MacOS/Veronica"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def log_dir() -> Path:
    return Path.home() / ".veronica" / "logs"


def is_enabled() -> bool:
    return plist_path().exists()


def bundle_app_path(
    argv0: str | None = None,
    env: Mapping[str, str] | None = None,
    exists: Callable[[Path], bool] = Path.exists,
) -> Path | None:
    """The .app bundle's path (…/Veronica.app) when running from one (i.e.
    launched via the built .app's launcher, as opposed to
    `uv run python -m veronica` / a bare interpreter), else None. This is
    also the "are we running from a bundle" check "Start at Login" needs,
    since it must point the LaunchAgent at a real .app path.

    The launcher script runs `python -m veronica`, so sys.argv[0] is
    `…/veronica/__main__.py` rather than the bundle executable; the launcher
    therefore exports `VERONICA_APP_BUNDLE` (checked first; must end in .app
    and exist), and as a fallback the bundle is derived from
    `VERONICA_BUNDLE_BUILD` (…/Veronica.app/Contents/Resources/build.json).
    The argv[0]-ends-with-Contents/MacOS/Veronica heuristic is kept last.
    `env`/`exists` are injectable for tests."""
    import sys

    env = os.environ if env is None else env

    explicit = (env.get("VERONICA_APP_BUNDLE") or "").strip()
    if explicit:
        p = Path(explicit)
        if p.suffix == ".app" and exists(p):
            return p

    build_json = (env.get("VERONICA_BUNDLE_BUILD") or "").strip()
    if build_json:
        parents = Path(build_json).parents
        if len(parents) >= 3:
            p = parents[2]
            if p.suffix == ".app" and exists(p):
                return p

    candidate = argv0 if argv0 is not None else sys.argv[0]
    if not candidate.endswith(APP_EXEC_SUFFIX):
        return None
    # candidate == "<app>/Contents/MacOS/Veronica" -> strip 3 path segments
    return Path(candidate).parent.parent.parent


def _launchctl(*args: str) -> None:
    try:
        subprocess.run(["launchctl", *args], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # best-effort: the plist file is the source of truth


def enable(app_path: Path) -> None:
    """Write the LaunchAgent plist for `app_path` (the .app bundle, i.e. the
    dir ending in Veronica.app) and register it with launchd."""
    app_path = Path(app_path)
    exec_path = app_path / "Contents" / "MacOS" / "Veronica"

    agents_dir = plist_path().parent
    agents_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = log_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "Label": LABEL,
        "ProgramArguments": [str(exec_path)],
        "RunAtLoad": True,
        "KeepAlive": False,
        "StandardOutPath": str(logs_dir / "launchd.log"),
        "StandardErrorPath": str(logs_dir / "launchd.log"),
    }
    with open(plist_path(), "wb") as f:
        plistlib.dump(data, f)

    uid = os.getuid()
    _launchctl("bootstrap", f"gui/{uid}", str(plist_path()))


def disable() -> None:
    path = plist_path()
    if not path.exists():
        return
    uid = os.getuid()
    _launchctl("bootout", f"gui/{uid}/{LABEL}")
    path.unlink(missing_ok=True)
