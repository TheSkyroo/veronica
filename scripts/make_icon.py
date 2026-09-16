"""Render the HUD orb (veronica/ui/hud) headlessly and build assets/Veronica.icns.

Steps:
1. Launch headless Chromium (Playwright) against index.html?icon=1 — hud.js
   detects that query param, hides the card/text/caption and scales the orb
   canvas to fill the viewport (see hud.css's `body.icon-mode` rules) —
   and screenshot at 1024x1024 with a transparent background.
2. Build a .iconset directory (16..512 @1x/@2x) from that PNG, via Pillow if
   available, else `sips -z`.
3. `iconutil -c icns` the iconset into assets/Veronica.icns.

If Playwright (or its Chromium browser) isn't available, this prints a
message and exits 0 rather than failing the build.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HUD_DIR = REPO / "veronica" / "ui" / "hud"
ASSETS_DIR = REPO / "assets"
ICON_PNG = 1024

# (size, scale) pairs iconutil expects inside a .iconset, per Apple's naming
# convention (icon_16x16.png, icon_16x16@2x.png, ...).
ICONSET_SIZES = [16, 32, 128, 256, 512]


def render_orb_png(dest: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("make_icon: playwright not installed — skipping icon build")
        return False

    url = (HUD_DIR / "index.html").resolve().as_uri() + "?icon=1"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as e:
                print(f"make_icon: could not launch chromium ({e}) — skipping icon build")
                return False
            try:
                page = browser.new_page(
                    viewport={"width": ICON_PNG, "height": ICON_PNG},
                    device_scale_factor=6,  # canvas backing store ~1020px at SIZE=170
                    base_url=None,
                )
                page.goto(url)
                page.wait_for_timeout(300)  # let the first animation frame(s) paint
                page.screenshot(path=str(dest), omit_background=True)
            finally:
                browser.close()
    except Exception as e:
        print(f"make_icon: render failed ({e}) — skipping icon build")
        return False
    return dest.exists()


def build_iconset_pillow(src: Path, iconset: Path) -> None:
    from PIL import Image

    img = Image.open(src).convert("RGBA")
    for size in ICONSET_SIZES:
        img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
        img.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")


def build_iconset_sips(src: Path, iconset: Path) -> None:
    for size in ICONSET_SIZES:
        for suffix, px in ((f"icon_{size}x{size}.png", size), (f"icon_{size}x{size}@2x.png", size * 2)):
            dest = iconset / suffix
            shutil.copy(src, dest)
            subprocess.run(["sips", "-z", str(px), str(px), str(dest)], check=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = ASSETS_DIR / "_orb_icon_1024.png"

    if not render_orb_png(png_path):
        return 0

    iconset = ASSETS_DIR / "Veronica.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir(parents=True)

    try:
        build_iconset_pillow(png_path, iconset)
        print("make_icon: built iconset with Pillow")
    except ImportError:
        if shutil.which("sips") is None:
            print("make_icon: neither Pillow nor sips available — skipping icon build")
            shutil.rmtree(iconset, ignore_errors=True)
            png_path.unlink(missing_ok=True)
            return 0
        build_iconset_sips(png_path, iconset)
        print("make_icon: built iconset with sips")

    if shutil.which("iconutil") is None:
        print("make_icon: iconutil not available — leaving Veronica.iconset in place, skipping .icns")
        return 0

    icns_path = ASSETS_DIR / "Veronica.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)], check=True)
    print(f"make_icon: wrote {icns_path}")

    shutil.rmtree(iconset, ignore_errors=True)
    png_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
