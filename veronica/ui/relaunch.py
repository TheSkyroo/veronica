"""Restart the app after an update.

A process can't `open` itself while it's still running (LaunchServices just
activates the existing instance), so we spawn a detached shell that waits a
second for us to exit and then `open -n`s the bundle, and only then quit. If
we're not running from a bundle (dev run) there's nothing to reopen — we just
quit and let the developer start it again.
"""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path


def relaunch(bundle_path: Path | None, quit: Callable[[], None], popen=subprocess.Popen) -> bool:
    """Schedule a relaunch of `bundle_path` (if any) and quit. Returns True
    if a relaunch was scheduled. If scheduling fails we don't quit — better a
    stale app than no app."""
    if bundle_path is None:
        quit()
        return False
    try:
        popen(["/bin/sh", "-c", f'sleep 1; open -n "{bundle_path}"'], start_new_session=True)
    except (OSError, ValueError):
        return False
    quit()
    return True
