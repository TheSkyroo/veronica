"""Tiny on-disk store for runtime UI preferences (HUD mode, position) that
should survive across restarts, kept separate from Settings (env-driven
config) since these are toggled at runtime by voice/menu, not configured."""
import json
import logging
import os
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("veronica.prefs")

_PREFS_PATH = Path.home() / ".veronica" / "prefs.json"
# Every load-modify-save runs under this: the menubar's main thread, the
# orchestrator's loop thread and the settings bridge all write prefs.json,
# and a lost update would silently drop someone's setting.
_LOCK = threading.RLock()


def _read() -> dict:
    try:
        return json.loads(_PREFS_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        log.warning("prefs file unreadable; ignoring", exc_info=True)
        return {}


def _write(data: dict) -> None:
    """Atomic replace: write a sibling temp file, fsync, os.replace() over
    prefs.json — a crash mid-write never leaves a truncated file behind."""
    _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".prefs.", suffix=".tmp", dir=_PREFS_PATH.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _PREFS_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load() -> dict:
    """Read prefs.json, tolerating a missing/corrupt file (returns {})."""
    with _LOCK:
        return _read()


def save(prefs: dict) -> None:
    """Write prefs.json, merging over whatever's already on disk so a
    caller that only knows about one key doesn't clobber the others."""
    with _LOCK:
        try:
            current = _read()
            current.update(prefs)
            _write(current)
        except OSError:
            log.warning("failed to save prefs", exc_info=True)


def get(key, default=None):
    """Convenience accessor: load() then dict.get(key, default)."""
    return load().get(key, default)


def save_settings_override(field: str, value) -> None:
    """Persist one Settings-field override, merged under the "settings"
    dict in prefs.json (leaving other overrides and other prefs alone)."""
    with _LOCK:
        current_settings = dict(_read().get("settings", {}))
        current_settings[field] = value
        save({"settings": current_settings})


def clear_settings_override(field: str) -> None:
    """Remove one Settings-field override, if present."""
    with _LOCK:
        current_settings = dict(_read().get("settings", {}))
        if field in current_settings:
            del current_settings[field]
            save({"settings": current_settings})
