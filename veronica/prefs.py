"""Tiny on-disk store for runtime UI preferences (HUD mode, position) that
should survive across restarts, kept separate from Settings (env-driven
config) since these are toggled at runtime by voice/menu, not configured."""
import json
import logging
from pathlib import Path

log = logging.getLogger("veronica.prefs")

_PREFS_PATH = Path.home() / ".veronica" / "prefs.json"


def load() -> dict:
    """Read prefs.json, tolerating a missing/corrupt file (returns {})."""
    try:
        return json.loads(_PREFS_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        log.warning("prefs file unreadable; ignoring", exc_info=True)
        return {}


def save(prefs: dict) -> None:
    """Write prefs.json, merging over whatever's already on disk so a
    caller that only knows about one key doesn't clobber the others."""
    try:
        current = load()
        current.update(prefs)
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PREFS_PATH.write_text(json.dumps(current))
    except OSError:
        log.warning("failed to save prefs", exc_info=True)


def get(key, default=None):
    """Convenience accessor: load() then dict.get(key, default)."""
    return load().get(key, default)


def save_settings_override(field: str, value) -> None:
    """Persist one Settings-field override, merged under the "settings"
    dict in prefs.json (leaving other overrides and other prefs alone)."""
    current_settings = load().get("settings", {})
    current_settings = dict(current_settings)
    current_settings[field] = value
    save({"settings": current_settings})


def clear_settings_override(field: str) -> None:
    """Remove one Settings-field override, if present."""
    current_settings = dict(load().get("settings", {}))
    if field in current_settings:
        del current_settings[field]
        save({"settings": current_settings})
