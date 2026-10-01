"""Build assets/Veronica.ico (the exe/window icon) from the HUD orb.

Steps:
1. Render the orb: headless Chromium (Playwright) against index.html?icon=1 —
   hud.js detects that query param, hides the card/text/caption and scales
   the orb canvas to fill the viewport (see hud.css's `body.icon-mode`
   rules) — screenshotted at 1024x1024 with a transparent background. If
   Playwright (or its Chromium) isn't available, fall back to the orb Pillow
   draws for the tray (veronica.ui.icon), so the build never goes iconless.
2. Save a multi-resolution .ico (16..256 px) with Pillow.

Run directly:

    uv run python scripts/make_icon.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HUD_DIR = REPO / "veronica" / "ui" / "hud"
ASSETS_DIR = REPO / "assets"
ICON_PNG = 1024

# Every size Windows asks an .ico for (taskbar, Alt-Tab, Explorer views,
# high-DPI scalings of each).
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]


def render_orb_png(dest: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("make_icon: playwright not installed — drawing the orb with Pillow instead")
        return False

    url = (HUD_DIR / "index.html").resolve().as_uri() + "?icon=1"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                print(f"make_icon: could not launch chromium ({e}) — drawing the orb with Pillow instead")
                return False
            try:
                page = browser.new_page(
                    viewport={"width": ICON_PNG, "height": ICON_PNG},
                    device_scale_factor=1,  # hud.js renders a 1024 px backing store in icon mode
                    base_url=None,
                )
                page.goto(url)
                page.wait_for_timeout(2200)  # transition settled + spoke history filled
                page.screenshot(path=str(dest), omit_background=True)
            finally:
                browser.close()
    except Exception as e:
        print(f"make_icon: render failed ({e}) — drawing the orb with Pillow instead")
        return False
    return dest.exists()


def drawn_orb(size: int = 256):
    sys.path.insert(0, str(REPO))
    from veronica.ui.icon import orb_image

    return orb_image(size)


def write_ico(img, dest: Path) -> Path:
    """Save `img` (RGBA, square, >= 256 px) as a multi-size .ico."""
    from PIL import Image

    img = img.convert("RGBA")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.LANCZOS)
    img.save(dest, format="ICO", sizes=[(s, s) for s in ICO_SIZES])
    return dest


def main() -> int:
    from PIL import Image

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = ASSETS_DIR / "_orb_icon_1024.png"
    try:
        if render_orb_png(png_path):
            img = Image.open(png_path).convert("RGBA")
            print("make_icon: rendered the HUD orb")
        else:
            img = drawn_orb()
        ico = write_ico(img, ASSETS_DIR / "Veronica.ico")
    finally:
        png_path.unlink(missing_ok=True)
    print(f"make_icon: wrote {ico}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
