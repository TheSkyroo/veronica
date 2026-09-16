"""Restart the app after an update.

A process can't `open` itself while it's still running (LaunchServices just
activates the existing instance), so we spawn a detached shell that polls
`kill -0 <our pid>` until we've actually exited and only then `open -n`s the
bundle — so a slow shutdown never yields two instances. The bundle path and
pid are passed as positional shell args ($1/$2), never interpolated into the
`-c` string. If we're not running from a bundle (dev run) there's nothing to
reopen — we just quit and let the developer start it again.
"""
from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

WAIT_THEN_OPEN = 'while kill -0 "$2" 2>/dev/null; do sleep 0.2; done; open -n "$1"'


def relaunch(
    bundle_path: Path | None,
    quit: Callable[[], None],
    popen=subprocess.Popen,
    pid: int | None = None,
) -> bool:
    """Schedule a relaunch of `bundle_path` (if any) and quit. Returns True
    if a relaunch was scheduled. If scheduling fails we don't quit — better a
    stale app than no app. `pid` is the process the helper waits on
    (default: this one)."""
    if bundle_path is None:
        quit()
        return False
    pid = os.getpid() if pid is None else pid
    try:
        popen(["/bin/sh", "-c", WAIT_THEN_OPEN, "sh", str(bundle_path), str(pid)], start_new_session=True)
    except (OSError, ValueError):
        return False
    quit()
    return True
