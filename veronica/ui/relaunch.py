"""Restart the app after an update.

A second Veronica.exe started while this one is still running would find the
instance lock held and exit, so we spawn a detached, windowless PowerShell
helper that waits for our pid to exit (`Wait-Process`) and only then starts
the exe — a slow shutdown never yields two instances, nor none.

If the update couldn't replace the running build in place (Windows locks a
running exe and its DLLs), scripts/build_app.py left the new one next to it
as `<dist>/Veronica.new`; the helper swaps that in once we've exited, before
starting it.

The pid and paths reach the helper through environment variables, never
interpolated into the `-Command` text, so no path can inject PowerShell. If
we're not running from the built exe (dev run) there's nothing to restart —
we just quit and let the developer start it again.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

# CreateProcess flags (the subprocess constants only exist on Windows).
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

STAGED_SUFFIX = ".new"

WAIT_THEN_START = (
    "$ErrorActionPreference = 'SilentlyContinue'; "
    "Wait-Process -Id ([int]$env:VERONICA_RELAUNCH_PID); "
    "$exe = $env:VERONICA_RELAUNCH_EXE; "
    "$staged = $env:VERONICA_RELAUNCH_STAGED; "
    "if ($staged -and (Test-Path -LiteralPath $staged)) { "
    "$dir = Split-Path -LiteralPath $exe -Parent; "
    # File handles can outlive the process by a moment: retry the swap.
    "for ($i = 0; $i -lt 50; $i++) { "
    "Remove-Item -LiteralPath $dir -Recurse -Force; "
    "if (-not (Test-Path -LiteralPath $dir)) { break }; Start-Sleep -Milliseconds 200 }; "
    "Move-Item -LiteralPath $staged -Destination $dir }; "
    "Start-Process -FilePath $exe"
)


def staged_dir(exe_path: Path) -> Path:
    """Where build_app stages a build it couldn't swap in: <dist>/Veronica.new."""
    folder = Path(exe_path).parent
    return folder.with_name(folder.name + STAGED_SUFFIX)


def relaunch(
    exe_path: Path | None,
    quit: Callable[[], None],
    popen=subprocess.Popen,
    pid: int | None = None,
    env: dict | None = None,
) -> bool:
    """Schedule a restart of `exe_path` (if any) and quit. Returns True if a
    restart was scheduled. If scheduling fails we don't quit — better a
    stale app than no app. `pid` is the process the helper waits on
    (default: this one)."""
    if exe_path is None:
        quit()
        return False
    pid = os.getpid() if pid is None else pid
    child_env = dict(os.environ if env is None else env)
    child_env.update({
        "VERONICA_RELAUNCH_PID": str(pid),
        "VERONICA_RELAUNCH_EXE": str(exe_path),
        "VERONICA_RELAUNCH_STAGED": str(staged_dir(Path(exe_path))),
        # The new exe must set itself up from scratch, not as a child of
        # this (PyInstaller-frozen) process.
        "PYINSTALLER_RESET_ENVIRONMENT": "1",
    })
    try:
        popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-WindowStyle", "Hidden", "-Command", WAIT_THEN_START],
            env=child_env, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,   # no console, own Ctrl+C group
        )
    except (OSError, ValueError):
        return False
    quit()
    return True
