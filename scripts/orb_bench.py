"""Benchmark the HUD hologram orb: average canvas draw time per state.

Loads veronica/ui/hud/index.html in headless Chromium (Playwright), sets the
speck count (--particles) via window.hud.configure(), drives each state the way the
orchestrator would (mic/voice events), and reads the renderer's
window.__hud hook (frameMs / frames / totalMs) over 300 drawn frames.
Prints ms/frame per state. A script, not a test.

    uv run python scripts/orb_bench.py [--particles 400] [--frames 300] [--mini]
"""
import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HUD = REPO / "veronica" / "ui" / "hud" / "index.html"
STATES = ["idle", "listening", "thinking", "speaking", "confirming", "error", "warming"]
VOICE = [0.2, 0.5, 0.8, 1, 0.9, 0.6, 0.3, 0.7, 1, 0.5, 0.2, 0.6, 0.9, 0.4, 0.1] * 20


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--dpr", type=float, default=2.0)
    args = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("orb_bench: playwright not installed", file=sys.stderr)
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 540, "height": 300}, device_scale_factor=args.dpr)
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined && window.__hud !== undefined")
        if args.mini:
            page.evaluate("window.hud.setMode('mini')")
        page.evaluate("n => window.hud.configure({particles: n})", args.particles)
        active = page.evaluate("window.hud.state().particles")
        print(f"orb_bench: {active} specks, dpr={args.dpr}, {'mini' if args.mini else 'full'} mode, "
              f"{args.frames} frames per state")
        for state in STATES:
            page.evaluate("s => window.hud.push({kind:'state', payload:s})", state)
            if state == "speaking":
                page.evaluate("v => window.hud.push({kind:'voice', payload:{step_ms:50, levels:v}})", VOICE)
            if state == "confirming":
                page.evaluate("window.hud.push({kind:'tool', payload:{summary:'x', decision:'ask', timeout_ms:60000}})")
            page.wait_for_timeout(450)          # let the ~400 ms transition settle
            page.evaluate("window.__hud.reset()")
            # keep the mic meter moving while listening so ripples spawn
            frames = 0
            while frames < args.frames:
                if state == "listening":
                    page.evaluate("window.hud.push({kind:'mic', payload: 0.3 + 0.6 * Math.random()})")
                page.wait_for_timeout(50)
                frames = page.evaluate("window.__hud.frames")
            total, frames = page.evaluate("[window.__hud.totalMs, window.__hud.frames]")
            print(f"  {state:<11} {total / frames:6.2f} ms/frame  ({frames} frames)")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
