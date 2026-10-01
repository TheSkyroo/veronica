"""The orb, drawn with Pillow: the system tray icon (tinted per state) and the
fallback for scripts/make_icon.py's assets/Veronica.ico when the HUD page
can't be rendered headlessly."""
from __future__ import annotations

import numpy as np

# Tray tint per orchestrator state (plus "muted"), roughly the HUD orb's own
# palette so the two read as the same thing.
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle": (122, 208, 255),
    "listening": (110, 231, 183),
    "followup": (94, 234, 212),
    "thinking": (167, 139, 250),
    "speaking": (125, 211, 252),
    "confirming": (240, 195, 108),
    "warming": (148, 163, 184),
    "error": (255, 107, 107),
    "muted": (100, 108, 122),
}
DEFAULT_COLOR = STATE_COLORS["idle"]

CORE_R = 0.66       # core radius, as a fraction of the half-size
GLOW_W = 0.16       # glow falloff width beyond the core
SUPERSAMPLE = 4


def orb_image(size: int = 64, rgb: tuple[int, int, int] = DEFAULT_COLOR, core: float = CORE_R):
    """A size x size RGBA PIL image of a glowing orb in `rgb`; `core` is the
    solid part's radius (the rest is halo)."""
    from PIL import Image

    n = size * SUPERSAMPLE
    c = (n - 1) / 2
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dx, dy = (xx - c) / (n / 2), (yy - c) / (n / 2)
    r = np.sqrt(dx * dx + dy * dy)

    base = np.array(rgb, dtype=np.float32) / 255.0
    # Core: darker towards the rim, with a soft highlight up and to the left.
    shade = np.clip(1.0 - r / core, 0.0, 1.0)
    highlight = np.exp(-(((dx + 0.22) ** 2 + (dy + 0.26) ** 2) / 0.09))
    color = base[None, None, :] * (0.45 + 0.55 * shade[..., None])
    color = color + (1.0 - color) * (0.75 * highlight[..., None])
    # Alpha: opaque core with a 1.5% anti-aliased edge, then a fading halo.
    edge = np.clip((core - r) / 0.015 + 0.5, 0.0, 1.0)
    glow = 0.55 * np.exp(-(((r - core) / GLOW_W) ** 2)) * (r > core)
    alpha = np.clip(edge + glow, 0.0, 1.0)
    halo = (r > core)[..., None]
    color = np.where(halo, base[None, None, :], color)

    rgba = np.dstack([np.clip(color, 0.0, 1.0), alpha]) * 255.0
    img = Image.fromarray(rgba.round().astype(np.uint8), "RGBA")
    return img.resize((size, size), Image.LANCZOS)


# The tray is 16-32 px: mostly orb, a thin halo.
TRAY_CORE_R = 0.84


def state_image(state: str, *, muted: bool = False, size: int = 64):
    """The tray icon for `state` (grey while muted)."""
    rgb = STATE_COLORS["muted"] if muted else STATE_COLORS.get(state, DEFAULT_COLOR)
    return orb_image(size, rgb, core=TRAY_CORE_R)
